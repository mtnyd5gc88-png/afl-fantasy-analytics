"use client";

import { useCallback, useEffect, useMemo, useState } from "react";
import { evaluateTeam, fetchPlayers, recommendTrades } from "@/lib/api";
import type { PlayerProjection, TradeRecommendation } from "@/lib/types";

function SparkBar({ low, mid, high }: { low: number; mid: number; high: number }) {
  const min = Math.min(low, mid, high) - 5;
  const max = Math.max(low, mid, high) + 5;
  const span = max - min || 1;
  const p = (v: number) => `${((v - min) / span) * 100}%`;
  return (
    <div className="relative h-2 w-full rounded-full bg-slate-800">
      <div
        className="absolute top-0 h-2 rounded-full bg-emerald-500/30"
        style={{ left: p(low), width: `calc(${p(high)} - ${p(low)})` }}
      />
      <div
        className="absolute -top-1 h-4 w-1 rounded bg-ball"
        style={{ left: p(mid), transform: "translateX(-50%)" }}
      />
    </div>
  );
}

export default function Home() {
  const [players, setPlayers] = useState<PlayerProjection[]>([]);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [bank, setBank] = useState(120_000);
  const [teamEval, setTeamEval] = useState<Awaited<ReturnType<typeof evaluateTeam>> | null>(null);
  const [trades, setTrades] = useState<TradeRecommendation[]>([]);
  const [q, setQ] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setErr(null);
    try {
      const p = await fetchPlayers();
      setPlayers(p);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Failed to load");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    if (players.length && selected.size === 0) {
      setSelected(new Set(players.slice(0, 8).map((x) => x.player_id)));
    }
  }, [players, selected.size]);

  const filtered = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return players;
    return players.filter((p) => p.name.toLowerCase().includes(s) || p.position.toLowerCase().includes(s));
  }, [players, q]);

  const toggle = (id: number) => {
    setSelected((prev) => {
      const n = new Set(prev);
      if (n.has(id)) n.delete(id);
      else n.add(id);
      return n;
    });
  };

  const runEval = async () => {
    setErr(null);
    try {
      const ids = Array.from(selected);
      const ev = await evaluateTeam(ids);
      setTeamEval(ev);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Evaluate failed");
    }
  };

  const runTrades = async () => {
    setErr(null);
    try {
      const ids = Array.from(selected);
      const t = await recommendTrades({ field_player_ids: ids, bank });
      setTrades(t);
    } catch (e) {
      setErr(e instanceof Error ? e.message : "Trade recommend failed");
    }
  };

  return (
    <main className="mx-auto flex max-w-7xl flex-col gap-8 px-4 py-10">
      <header className="flex flex-col gap-2 border-b border-slate-800 pb-6">
        <h1 className="text-3xl font-semibold tracking-tight text-white">AFL Fantasy Analytics</h1>
        <p className="max-w-2xl text-slate-400">
          Model-driven projections with uncertainty bands, value vs price, and PuLP + Monte Carlo trade
          recommendations over a configurable horizon.
        </p>
      </header>

      {err && (
        <div className="rounded-lg border border-rose-900/60 bg-rose-950/40 px-4 py-3 text-rose-200">{err}</div>
      )}

      <section className="grid gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-4 rounded-xl border border-slate-800 bg-slate-900/40 p-4">
          <div className="flex flex-wrap items-end justify-between gap-3">
            <div>
              <h2 className="text-lg font-medium text-white">Team input</h2>
              <p className="text-sm text-slate-500">Select field players for evaluation and trade search.</p>
            </div>
            <div className="flex flex-wrap gap-2">
              <input
                className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm"
                placeholder="Search name / position"
                value={q}
                onChange={(e) => setQ(e.target.value)}
              />
              <button
                type="button"
                onClick={() => void runEval()}
                className="rounded-md bg-emerald-600 px-4 py-2 text-sm font-medium text-white hover:bg-emerald-500"
              >
                Evaluate team
              </button>
              <button
                type="button"
                onClick={() => void load()}
                className="rounded-md border border-slate-600 px-4 py-2 text-sm text-slate-200 hover:bg-slate-800"
              >
                Refresh data
              </button>
            </div>
          </div>
          <div className="max-h-64 overflow-auto rounded-lg border border-slate-800">
            <table className="w-full text-left text-sm">
              <thead className="sticky top-0 bg-slate-950/95 text-xs uppercase text-slate-500">
                <tr>
                  <th className="px-3 py-2">On</th>
                  <th className="px-3 py-2">Player</th>
                  <th className="px-3 py-2">Pos</th>
                  <th className="px-3 py-2 text-right">Proj</th>
                  <th className="px-3 py-2 text-right">Risk (σ)</th>
                  <th className="px-3 py-2 text-right">Value</th>
                </tr>
              </thead>
              <tbody>
                {loading && (
                  <tr>
                    <td colSpan={6} className="px-3 py-6 text-center text-slate-500">
                      Loading…
                    </td>
                  </tr>
                )}
                {!loading &&
                  filtered.map((p) => (
                    <tr key={p.player_id} className="border-t border-slate-800/80 hover:bg-slate-800/30">
                      <td className="px-3 py-2">
                        <input
                          type="checkbox"
                          checked={selected.has(p.player_id)}
                          onChange={() => toggle(p.player_id)}
                        />
                      </td>
                      <td className="px-3 py-2 font-medium text-slate-100">{p.name}</td>
                      <td className="px-3 py-2 text-slate-400">{p.position}</td>
                      <td className="px-3 py-2 text-right tabular-nums">{p.predicted_points.toFixed(1)}</td>
                      <td className="px-3 py-2 text-right tabular-nums text-amber-200/90">
                        {p.variance_proxy.toFixed(1)}
                      </td>
                      <td className="px-3 py-2 text-right tabular-nums text-emerald-300/90">
                        {p.value_vs_price.toFixed(2)}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="space-y-4 rounded-xl border border-slate-800 bg-slate-900/40 p-4">
          <h2 className="text-lg font-medium text-white">Team projection</h2>
          {!teamEval && <p className="text-sm text-slate-500">Run evaluate to aggregate selected players.</p>}
          {teamEval && (
            <div className="space-y-3">
              <div>
                <div className="text-xs uppercase text-slate-500">Expected total</div>
                <div className="text-3xl font-semibold text-white">{teamEval.expected_total.toFixed(1)}</div>
                <div className="text-xs text-slate-500">
                  90% band ≈ {teamEval.ci_low.toFixed(1)} – {teamEval.ci_high.toFixed(1)}
                </div>
              </div>
              <div className="space-y-2">
                {teamEval.per_player.map((p) => (
                  <div key={p.player_id} className="rounded-lg border border-slate-800/80 p-2">
                    <div className="flex justify-between text-sm">
                      <span className="font-medium text-slate-100">{p.name}</span>
                      <span className="tabular-nums text-slate-300">{p.predicted_points.toFixed(1)} pts</span>
                    </div>
                    <SparkBar low={p.ci_low} mid={p.predicted_points} high={p.ci_high} />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </section>

      <section className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
          <h2 className="text-lg font-medium text-white">Projections</h2>
          <p className="mb-3 text-sm text-slate-500">Full squad table with confidence intervals and value metric.</p>
          <div className="max-h-[420px] overflow-auto rounded-lg border border-slate-800">
            <table className="w-full text-left text-sm">
              <thead className="sticky top-0 bg-slate-950/95 text-xs uppercase text-slate-500">
                <tr>
                  <th className="px-3 py-2">Player</th>
                  <th className="px-3 py-2">Pos</th>
                  <th className="px-3 py-2 text-right">Proj</th>
                  <th className="px-3 py-2 text-right">Low</th>
                  <th className="px-3 py-2 text-right">High</th>
                  <th className="px-3 py-2 text-right">Price</th>
                  <th className="px-3 py-2 text-right">Value</th>
                </tr>
              </thead>
              <tbody>
                {players.map((p) => (
                  <tr key={p.player_id} className="border-t border-slate-800/80">
                    <td className="px-3 py-2 text-slate-100">{p.name}</td>
                    <td className="px-3 py-2 text-slate-400">{p.position}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{p.predicted_points.toFixed(1)}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-slate-400">{p.ci_low.toFixed(1)}</td>
                    <td className="px-3 py-2 text-right tabular-nums text-slate-400">{p.ci_high.toFixed(1)}</td>
                    <td className="px-3 py-2 text-right tabular-nums">${(p.price / 1000).toFixed(0)}k</td>
                    <td className="px-3 py-2 text-right tabular-nums text-emerald-300/90">
                      {p.value_vs_price.toFixed(2)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="rounded-xl border border-slate-800 bg-slate-900/40 p-4">
          <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
            <div>
              <h2 className="text-lg font-medium text-white">Trade recommendations</h2>
              <p className="text-sm text-slate-500">MIP on candidates + Monte Carlo team deltas.</p>
            </div>
            <div className="flex items-center gap-2">
              <label className="text-xs text-slate-500">Bank</label>
              <input
                type="number"
                className="w-32 rounded-md border border-slate-700 bg-slate-950 px-2 py-1 text-sm"
                value={bank}
                onChange={(e) => setBank(Number(e.target.value))}
              />
              <button
                type="button"
                onClick={() => void runTrades()}
                className="rounded-md bg-amber-500 px-4 py-2 text-sm font-semibold text-slate-950 hover:bg-amber-400"
              >
                Run optimizer
              </button>
            </div>
          </div>
          {!trades.length && <p className="text-sm text-slate-500">No recommendations yet.</p>}
          <ul className="space-y-3">
            {trades.map((t, i) => (
              <li key={i} className="rounded-lg border border-slate-800 p-3">
                <div className="flex flex-wrap justify-between gap-2 text-sm">
                  <span className="text-slate-200">
                    Out <span className="font-mono text-rose-300">{t.trade_out_id}</span> → In{" "}
                    <span className="font-mono text-emerald-300">{t.trade_in_id}</span>
                  </span>
                  <span className="tabular-nums text-slate-300">ΔEV ≈ {t.expected_value_gain.toFixed(1)}</span>
                </div>
                <div className="mt-2 grid gap-1 text-xs text-slate-500 sm:grid-cols-2">
                  <span>MC mean Δ (team): {t.mc_mean.toFixed(1)}</span>
                  <span>Risk (std Δ): {t.mc_std.toFixed(1)}</span>
                  <span>Budget delta: ${(t.net_budget_delta / 1000).toFixed(1)}k</span>
                  <span>Horizon-weighted objective uses multi-round simulation.</span>
                </div>
              </li>
            ))}
          </ul>
        </div>
      </section>
    </main>
  );
}
