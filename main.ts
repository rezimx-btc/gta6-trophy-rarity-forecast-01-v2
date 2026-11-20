// main.ts — GTA VI trophy rarity forecaster (count-consistent dilution model).
//
// Forecasts, for every PlayStation 5 trophy of Grand Theft Auto VI, its earn
// rate on Exophase at a case's target. The model is built for the way the
// page actually moves: Exophase refreshes its counts on its own schedule, so
// a refresh jumps the tracked-player count up and dilutes every rate, and the
// rates then recover as the new cohort plays. The forecast therefore fits
// the RATE with a saturating curve (it approaches a per-trophy ceiling
// instead of growing without bound), keeps earners equal to rate times
// players so the forecast stays count-consistent, and widens the spread
// (and shifts the median down) when the display has been frozen long enough
// that a refresh is overdue. All fitting is robust, so one bad read cannot
// bend a line. Node 22, standard library only, no network, deterministic.
//
// Reads $MAKEFUN_INPUT_DIR (else /input), writes one forecast per case to
// $MAKEFUN_OUTPUT_DIR (else /output).

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

export interface TrophyInfo {
  id: number;
  name: string;
  description: string;
  tier: string;
  points: number;
  secret: boolean;
}

export interface HistoryRead {
  slot: string;
  at: string;
  players_tracked: number;
  trophies_earned: number;
  rates: (number | null)[];
}

export interface CaseInput {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  game: unknown;
  trophies: TrophyInfo[];
  history: HistoryRead[];
  notes?: string;
}

export interface ForecastParams {
  shiftAlphaFlat: number;
  shiftAlphaObs: number;
  spreadBeta: number;
  spreadFloor: number;
  priorIntervalH: number;
  priorJump: number;
}

export const DEFAULT_PARAMS: ForecastParams = {
  shiftAlphaFlat: 1.0,
  shiftAlphaObs: 0,
  spreadBeta: 0,
  spreadFloor: 0.05,
  priorIntervalH: 5,
  priorJump: 6.06,
};

const EPOCH_MS = Date.parse("2026-11-19T00:00:00Z");
const Z_BY_LEVEL: Record<string, number> = {
  "0.05": -1.6448536269514722,
  "0.25": -0.6744897501960817,
  "0.5": 0,
  "0.75": 0.6744897501960817,
  "0.95": 1.6448536269514722,
};

function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

function sigmoid(z: number): number {
  return 1 / (1 + Math.exp(-z));
}

// Logit of a rate given in percent (0-100), clamped so 0% and 100% stay finite.
function logitPct(p: number): number {
  const q = clamp(p, 0.01, 99.99) / 100;
  return Math.log(q / (1 - q));
}

function hoursSinceEpoch(iso: string): number {
  return Math.max((Date.parse(iso) - EPOCH_MS) / 3600000, 0);
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// Robust straight-line fit (ordinary least squares, then Huber IRLS rounds),
// so one outlying read cannot bend the line. Returns intercept, slope and
// the robust residual scale (0 when the points sit exactly on a line).
function robustLine(
  xs: number[],
  ys: number[],
): { a: number; b: number; scale: number } {
  const n = xs.length;
  let mx = 0;
  let my = 0;
  for (let i = 0; i < n; i++) {
    mx += xs[i];
    my += ys[i];
  }
  mx /= n;
  my /= n;
  let sxx = 0;
  let sxy = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx;
    sxx += dx * dx;
    sxy += dx * (ys[i] - my);
  }
  let b = sxx > 1e-12 ? sxy / sxx : 0;
  let a = my - b * mx;
  for (let round = 0; round < 6; round++) {
    const res = ys.map((y, i) => y - (a + b * xs[i]));
    const med = median(res);
    const mad = median(res.map((r) => Math.abs(r - med)));
    const s = 1.4826 * mad + 1e-9;
    const k = 1.345 * s;
    let sw = 0;
    let sx = 0;
    let sy = 0;
    let sxx2 = 0;
    let sxy2 = 0;
    for (let i = 0; i < n; i++) {
      const w = Math.abs(res[i]) <= k ? 1 : k / Math.abs(res[i]);
      sw += w;
      sx += w * xs[i];
      sy += w * ys[i];
      sxx2 += w * xs[i] * xs[i];
      sxy2 += w * xs[i] * ys[i];
    }
    const den = sw * sxx2 - sx * sx;
    if (Math.abs(den) < 1e-12) break;
    const b2 = (sw * sxy2 - sx * sy) / den;
    const a2 = (sy - b2 * sx) / sw;
    if (!isFinite(a2) || !isFinite(b2)) break;
    a = a2;
    b = b2;
  }
  const res = ys.map((y, i) => y - (a + b * xs[i]));
  const med = median(res);
  const scale = 1.4826 * median(res.map((r) => Math.abs(r - med)));
  return { a, b, scale: Math.max(scale, 0) };
}

// Quantiles of a forecast centred on a logit median with a logit spread.
function quantilesFromLogit(medLogit: number, sig: number, levels: number[]): number[] {
  return levels.map((lv) => {
    const z = Z_BY_LEVEL[String(lv)] ?? 0;
    const v = clamp(100 * sigmoid(medLogit + z * sig), 0.01, 99.99);
    return Math.round(v * 10000) / 10000;
  });
}

export function forecastCase(
  c: CaseInput,
  params: ForecastParams = DEFAULT_PARAMS,
): { id: string; trophies: { id: number; q: number[] }[] } {
  const checkpointMs = Date.parse(c.at);
  const targetMs = Date.parse(c.target);
  const horizonDays = Math.max((targetMs - checkpointMs) / 86400000, 0);
  const horizonH = horizonDays * 24;
  const xTarget = Math.log(1 + hoursSinceEpoch(c.target));
  const levels = c.levels && c.levels.length ? c.levels : [0.05, 0.25, 0.5, 0.75, 0.95];

  const reads = c.history
    .filter((r) => Date.parse(r.at) <= checkpointMs)
    .sort((a, b) => Date.parse(a.at) - Date.parse(b.at));
  const lastRead = reads.length ? reads[reads.length - 1] : null;
  const lastPlayers = lastRead ? lastRead.players_tracked : 1000;

  // --- How the page refreshes: change events in the history ---------------
  // A change is a read where the player count or any rate moved. From the
  // gaps between changes we learn the refresh interval; from the player
  // jumps we learn how much a refresh dilutes the rates. With no change at
  // all in the history the page has been frozen since the first read, so we
  // fall back to the prior the task itself states: the first refresh took
  // the tracked count from 638 to 3,868 within about five hours.
  const changes: number[] = [];
  for (let j = 1; j < reads.length; j++) {
    const prev = reads[j - 1];
    const cur = reads[j];
    let changed = cur.players_tracked !== prev.players_tracked;
    if (!changed) {
      for (let i = 0; i < c.trophies.length; i++) {
        const a = prev.rates[i];
        const b2 = cur.rates[i];
        if (a === null || b2 === null) {
          if (a !== b2) changed = true;
          continue;
        }
        if (Math.abs(a - b2) > 0.005) {
          changed = true;
          break;
        }
      }
    }
    if (changed) changes.push(j);
  }
  let intervalH = params.priorIntervalH;
  let jump = params.priorJump;
  if (changes.length >= 1) {
    const gaps: number[] = [];
    const jumps: number[] = [];
    for (let k = 0; k < changes.length; k++) {
      const j = changes[k];
      const prevMs =
        k === 0 ? Date.parse(reads[0].at) : Date.parse(reads[changes[k - 1]].at);
      gaps.push((Date.parse(reads[j].at) - prevMs) / 3600000);
      const before = reads[j - 1].players_tracked;
      if (before > 0 && reads[j].players_tracked > before) {
        jumps.push(reads[j].players_tracked / before);
      }
    }
    if (gaps.length) intervalH = Math.max(median(gaps), 0.5);
    const jm = jumps.length ? median(jumps) : 0;
    if (jm > 1) jump = jm;
  }
  const lastChangeMs = changes.length
    ? Date.parse(reads[changes[changes.length - 1]].at)
    : reads.length
      ? Date.parse(reads[0].at)
      : checkpointMs;
  const frozenH = Math.max((checkpointMs - lastChangeMs) / 3600000, 0);
  const expectedRefreshes = Math.min(
    (frozenH + horizonH) / Math.max(intervalH, 0.5),
    2.5,
  );
  const logJump = Math.log(Math.max(jump, 1));

  // --- Tracked players at the target ---------------------------------------
  // Robust line of log players against log-time; players never shrink, so a
  // negative slope is clipped to flat.
  let playersTarget = lastPlayers;
  {
    const px: number[] = [];
    const py: number[] = [];
    for (const r of reads) {
      if (r.players_tracked > 0) {
        px.push(Math.log(1 + hoursSinceEpoch(r.at)));
        py.push(Math.log(r.players_tracked));
      }
    }
    if (px.length >= 3) {
      const fit = robustLine(px, py);
      playersTarget = Math.exp(fit.a + Math.max(fit.b, 0) * xTarget);
      playersTarget = Math.max(playersTarget, lastPlayers);
    }
  }

  // --- One forecast row per trophy -----------------------------------------
  const rows: { id: number; q: number[] }[] = [];
  for (let i = 0; i < c.trophies.length; i++) {
    const tid = c.trophies[i].id;
    const pts: { x: number; r: number; P: number }[] = [];
    for (const rd of reads) {
      const r = rd.rates[i];
      if (r !== null && r !== undefined && r > 0 && rd.players_tracked > 0) {
        pts.push({
          x: Math.log(1 + hoursSinceEpoch(rd.at)),
          r,
          P: rd.players_tracked,
        });
      }
    }
    const lastR = pts.length ? pts[pts.length - 1].r : 1;

    // A trophy every tracked player has earned stays earned: forecast it
    // at the top with a narrow spread.
    if (lastR >= 99.995) {
      rows.push({ id: tid, q: quantilesFromLogit(logitPct(99.99), 0.05, levels) });
      continue;
    }

    // Too little history to fit anything: the house baseline (last rate,
    // random-walk spread) is the honest fallback.
    if (pts.length < 3) {
      const sig = 0.5 * Math.sqrt(horizonDays);
      rows.push({
        id: tid,
        q: quantilesFromLogit(logitPct(lastR), Math.max(sig, 0.08), levels),
      });
      continue;
    }

    // Earners trend: robust line of log earners against log-time. Earners
    // only grow, so a negative slope is clipped to flat.
    let earnersTarget = 0;
    {
      const ex = pts.map((p) => p.x);
      const ey = pts.map((p) => Math.log(Math.max((p.r / 100) * p.P, 0.5)));
      const fit = robustLine(ex, ey);
      earnersTarget = Math.exp(fit.a + Math.max(fit.b, 0) * xTarget);
    }

    // Saturating rate fit: the rate approaches a per-trophy ceiling instead
    // of growing without bound. For each candidate ceiling, the logit of
    // rate-over-ceiling is linear in log-time; the ceiling with the
    // tightest robust fit wins. On flat history every ceiling fits equally
    // and the forecast is just the last rate, as it should be.
    const maxR = Math.max(...pts.map((p) => p.r));
    const ceilings = [...new Set(
      [1.05, 1.3, 1.7, 2.5, 4].map((f) => Math.min(100, maxR * f)).concat([100]),
    )].filter((v) => v > maxR || v === 100);
    let best: { C: number; a: number; b: number; scale: number } | null = null;
    for (const C of ceilings) {
      const xs: number[] = [];
      const ys: number[] = [];
      for (const p of pts) {
        if (p.r >= 0.98 * C) continue;
        xs.push(p.x);
        ys.push(logitPct((p.r / C) * 100));
      }
      if (xs.length < 3) continue;
      const fit = robustLine(xs, ys);
      if (!best || fit.scale < best.scale) {
        best = { C, a: fit.a, b: fit.b, scale: fit.scale };
      }
    }
    let rateSat: number;
    let resid: number;
    if (best) {
      rateSat = best.C * sigmoid(best.a + best.b * xTarget);
      resid = best.scale;
    } else {
      rateSat = lastR;
      resid = 0.1;
    }
    rateSat = clamp(rateSat, 0.01, 99.99);

    // Count-consistency: the earners the saturating rate implies must stay
    // near the earners trend, or the rate is clipped back to it.
    const implied = (rateSat / 100) * playersTarget;
    if (implied > 0 && earnersTarget > 0) {
      const ratio = implied / earnersTarget;
      if (ratio > 4) rateSat = clamp(rateSat * (4 / ratio), 0.01, 99.99);
      else if (ratio < 0.25) rateSat = clamp(rateSat * (0.25 / ratio), 0.01, 99.99);
    }

    // No persistence shrinkage: backtesting rejected it — pulling the
    // median toward the last observed rate undershoots, because the truth
    // keeps growing behind a frozen display while the last rate stands
    // still. The median is the saturating fit's extrapolation.
    let medLogit = logitPct(rateSat);

    // Overdue refresh: when the display has been frozen and refreshes are
    // expected before the target, the next refresh dilutes the rates, so
    // the median moves down by the expected log-dilution. The shift is
    // strong only when the history has never seen a refresh at all (a flat
    // history cannot anticipate the first refresh — a structural break);
    // once refreshes appear in the history, the trend fit already prices
    // their average effect, so the shift nearly vanishes.
    const shiftAlphaEff =
      changes.length === 0 ? params.shiftAlphaFlat : params.shiftAlphaObs;
    medLogit -= shiftAlphaEff * logJump * expectedRefreshes;

    // Spread: the fit's own residual, plus a random-walk widening with the
    // horizon, plus a term for the refreshes expected before the target.
    // Never degenerate: a zero-width forecast is a claim of certainty the
    // model does not have.
    const sigma = Math.sqrt(
      Math.max(resid, 0.02) * Math.max(resid, 0.02) +
        0.25 * horizonDays +
        (params.spreadBeta * logJump) ** 2 * expectedRefreshes,
    );
    const sig = clamp(Math.max(sigma, params.spreadFloor), 0.01, 3);
    rows.push({ id: tid, q: quantilesFromLogit(medLogit, sig, levels) });
  }

  rows.sort((a, b) => a.id - b.id);
  return { id: c.id, trophies: rows };
}

// --- Entry point -----------------------------------------------------------
// Reads every *.in case in the input directory (name order) and writes one
// .out forecast per case to the output directory. Runs only when this file
// is executed directly; importing it (for tests) runs nothing.
function main(): void {
  const inDir = process.env.MAKEFUN_INPUT_DIR || "/input";
  const outDir = process.env.MAKEFUN_OUTPUT_DIR || "/output";
  fs.mkdirSync(outDir, { recursive: true });
  const files = fs
    .readdirSync(inDir)
    .filter((f) => f.endsWith(".in"))
    .sort();
  for (const f of files) {
    const c = JSON.parse(
      fs.readFileSync(path.join(inDir, f), "utf8"),
    ) as CaseInput;
    const fc = forecastCase(c);
    const out = JSON.stringify({ trophies: fc.trophies }) + "\n";
    fs.writeFileSync(path.join(outDir, f.replace(/\.in$/, ".out")), out);
  }
}

const isMain =
  process.argv[1] &&
  path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) main();
