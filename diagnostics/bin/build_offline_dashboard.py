#!/usr/bin/env python3
"""Build a single-flight HTML dashboard for an offline H_E correction run.

Reads one ``diagnostics/runs/<run>/`` directory produced by
``run_causal_he_capture_controller.py --save-series`` and writes:

  * ``<output>/dashboard.html``         standalone HTML, loads ``__DATA__`` inline
  * ``<output>/dashboard_data.json``    raw payload (also embeds every iteration's
                                       time series so the page works from ``file://``)

Layout: four tabs (Overview / Plateaux / Decisions / Run config).
Aesthetic: dark glass-cockpit theme, FL-numbered altitudes, monospace numerics,
custom-Plotly so zoom and pan via the modebar still work.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def _f(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return f


def _jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray,)):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_metrics(run_dir: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    metrics = pd.read_csv(run_dir / "metrics.csv")
    cfg_path = run_dir / "run_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
    return metrics, cfg


def _per_plateau_scorecards(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = pd.read_csv(run_dir / "baseline_scorecard.csv")
    ctrl = pd.read_csv(run_dir / "controlled_scorecard.csv")
    return base, ctrl


def _decisions(run_dir: Path) -> pd.DataFrame:
    return pd.read_csv(run_dir / "controller_decisions.csv")


def _kpis(metrics: pd.DataFrame, decisions: pd.DataFrame) -> dict[str, object]:
    base_row = metrics.loc[metrics["variant"] == "baseline"].iloc[0]
    ctrl_row = metrics.loc[metrics["variant"] == "controlled"].iloc[0]
    n_plat = int(base_row["plateaux_evaluated"])
    base_share = float(base_row["closure_share"])
    ctrl_share = float(ctrl_row["closure_share"])
    applied = decisions.loc[decisions["status"] == "applied"]
    return {
        "n_plateaux_evaluated": n_plat,
        "n_plateaux_closed_baseline": int(base_row["plateaux_closed"]),
        "n_plateaux_closed_controlled": int(ctrl_row["plateaux_closed"]),
        "closure_share_baseline": base_share,
        "closure_share_controlled": ctrl_share,
        "closure_share_delta": ctrl_share - base_share,
        "median_abs_closure_error_baseline_ft": float(base_row["median_abs_closure_error_ft"]),
        "median_abs_closure_error_controlled_ft": float(ctrl_row["median_abs_closure_error_ft"]),
        "fullflight_mae_baseline_ft": float(base_row["fullflight_mae_ft"]),
        "fullflight_mae_controlled_ft": float(ctrl_row["fullflight_mae_ft"]),
        "mae_delta_ft": float(ctrl_row["fullflight_mae_ft"] - base_row["fullflight_mae_ft"]),
        "median_abs_capture_timing_error_baseline_s": float(base_row["median_abs_capture_timing_error_s"]),
        "median_abs_capture_timing_error_controlled_s": float(ctrl_row["median_abs_capture_timing_error_s"]),
        "n_applied_iterations": int(len(applied)),
        "n_accepted": int((decisions["status"] == "accepted").sum()),
        "n_skipped_bad_sensitivity": int((decisions["status"] == "skipped_bad_sensitivity").sum()),
        "n_iteration_limit": int((decisions["status"] == "iteration_limit").sum()),
        "n_no_eligible_segment": int((decisions["status"] == "no_eligible_segment").sum()),
    }


def _convergence_payload(decisions: pd.DataFrame) -> list[dict[str, object]]:
    applied = decisions.loc[decisions["status"].isin(["applied", "accepted"])].copy()
    applied = applied.sort_values(["event", "iteration"]).reset_index(drop=True)
    traces = []
    for event, grp in applied.groupby("event"):
        grp = grp.reset_index(drop=True)
        traces.append({
            "event": int(event),
            "h_sel_ft": float(grp["h_sel_ft"].iloc[0]),
            "x": list(range(len(grp))),
            "midpoint_error_ft": [_f(v) for v in grp["midpoint_error_ft"]],
            "post_midpoint_error_ft": [_f(v) for v in grp["post_midpoint_error_ft"]],
            "cumulative_delta_ft": [_f(v) for v in grp["cumulative_delta_he_ft"]],
            "sensitivity_ft_per_ft": [_f(v) for v in grp["sensitivity_ft_per_ft"]],
            "selection": grp["selection"].tolist(),
            "status": grp["status"].tolist(),
        })
    return traces


def _summary_rows(base_card: pd.DataFrame, ctrl_card: pd.DataFrame,
                  decisions: pd.DataFrame) -> list[dict[str, object]]:
    rows = []
    for _, base_row in base_card.iterrows():
        event = int(base_row["event"])
        ctrl_match = ctrl_card.loc[ctrl_card["event"] == event]
        if ctrl_match.empty:
            continue
        ctrl_row = ctrl_match.iloc[0]
        ev_decisions = decisions.loc[decisions["event"] == event]
        n_applied = int((ev_decisions["status"] == "applied").sum())
        controller_status = "fixed" if n_applied > 0 else (
            "no_op" if bool(base_row["replay_captured_at_tau_plateau_mid_250ft"]) else "open")
        rows.append({
            "event": event,
            "h_sel_ft": float(base_row["h_sel_ft"]),
            "plateau_min": float(base_row["time_min"]),
            "phase": str(base_row["mode"]),
            "baseline_abs_err_ft": float(base_row["abs_replay_error_to_target_ft"]),
            "controlled_abs_err_ft": float(ctrl_row["abs_replay_error_to_target_ft"]),
            "abs_err_delta_ft": float(ctrl_row["abs_replay_error_to_target_ft"]
                                      - base_row["abs_replay_error_to_target_ft"]),
            "baseline_captured": bool(base_row["replay_captured_at_tau_plateau_mid_250ft"]),
            "controlled_captured": bool(ctrl_row["replay_captured_at_tau_plateau_mid_250ft"]),
            "baseline_timing_s": (None if pd.isna(base_row["timing_error_min"])
                                  else float(base_row["timing_error_min"]) * 60.0),
            "controlled_timing_s": (None if pd.isna(ctrl_row["timing_error_min"])
                                    else float(ctrl_row["timing_error_min"]) * 60.0),
            "plateau_length_s": float(base_row["plateau_length_s"]),
            "controller_status": controller_status,
            "n_iterations": n_applied,
        })
    return rows


def _load_iteration_payload(run_dir: Path) -> list[dict[str, object]]:
    """Embed every iteration's ReplayArtefacts time series inline.

    Output rows include:
      - the same metadata as ``_iteration_index`` (sidebar buttons need it),
      - the full series (predicted / observed / target / phase / TAS / gamma / VZ),
      - the matching decision record so the card can show its applied delta.
    """
    series_dir = run_dir / "iteration_series"
    if not (series_dir / "meta.json").exists():
        return []
    meta = json.loads((series_dir / "meta.json").read_text())
    decisions_df = _decisions(run_dir)
    decisions_list = decisions_df.fillna(np.nan).to_dict("records")
    out = []
    for entry in meta["iterations"]:
        idx = int(entry["index"])
        parquet = series_dir / f"iteration_{idx:03d}.parquet"
        df = pd.read_parquet(parquet)
        item: dict[str, object] = {
            "index": idx,
            "label": str(entry["label"]),
            "event": (int(entry["event"]) if entry.get("event") is not None else None),
        }
        plateau = entry.get("plateau")
        if plateau is not None:
            item["h_sel_ft"] = float(plateau["h_sel_ft"])
            item["phase"] = str(plateau["phase"])
            item["plateau"] = {
                "h_sel_ft": float(plateau["h_sel_ft"]),
                "phase": str(plateau["phase"]),
                "mid_index": int(plateau["mid_index"]),
                "arrival_index": int(plateau["arrival_index"]),
                "start_index": int(plateau["start_index"]),
                "end_index": int(plateau["end_index"]),
                "time_min": float(plateau["time_min"]),
            }
        decision = entry.get("decision")
        if decision is not None:
            item["decision"] = decision
        item["series"] = {
            "time_min": [_f(v) for v in df["time_min"].to_numpy()],
            "predicted_altitude_ft": [_f(v) for v in df["predicted_altitude_ft"].to_numpy()],
            "observed_altitude_ft": [_f(v) for v in df["observed_altitude_ft"].to_numpy()],
            "inferred_target_ft": [_f(v) for v in df["inferred_target_ft"].to_numpy()],
            "phase": [str(v) for v in df["phase"].to_numpy()],
            "generated_tas_kt": [_f(v) for v in df["generated_tas_kt"].to_numpy()],
            "observed_tas_kt": [_f(v) for v in df["observed_tas_kt"].to_numpy()],
            "predicted_gamma_deg": [_f(v) for v in df["predicted_gamma_deg"].to_numpy()],
            "observed_gamma_deg": [_f(v) for v in df["observed_gamma_deg"].to_numpy()],
            "observed_vz_fpm": [_f(v) for v in df["observed_vz_fpm"].to_numpy()],
        }
        out.append(item)
    return out


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Offline H_E correction - __RUN_TITLE__</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {
      --bg: #f7f8fa;
      --panel: #ffffff;
      --panel-2: #f1f3f7;
      --panel-3: #fafbfc;
      --border: #e3e6eb;
      --border-strong: #c9cdd3;
      --text: #1f2937;
      --text-muted: #5a6577;
      --text-faint: #9aa3b3;
      --good: #047857;
      --good-bg: #ecfdf5;
      --good-border: #a7f3d0;
      --warn: #b45309;
      --warn-bg: #fffbeb;
      --warn-border: #fcd34d;
      --bad: #b91c1c;
      --bad-bg: #fef2f2;
      --bad-border: #fca5a5;
      --accent: #0369a1;
      --accent-bg: #f0f9ff;
      --accent-border: #7dd3fc;
      --neutral: #6b7280;
      --neutral-bg: #f3f4f6;
      --neutral-border: #d1d5db;
    }
    * { box-sizing: border-box; }
    html, body { background: var(--bg); color: var(--text); }
    body { background: var(--bg); }
    .topbar { background: var(--panel); }
    .tabs { background: var(--panel); }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Inter", "IBM Plex Sans",
                   "Segoe UI", Roboto, sans-serif;
      font-feature-settings: "tnum" 1;
      line-height: 1.5;
      min-height: 100vh;
    }
    code, .mono, .num { font-family: "JetBrains Mono", "SF Mono", Menlo, monospace; }

.topbar {
      padding: 18px 28px;
      border-bottom: 1px solid var(--border);
      display: flex; align-items: center; justify-content: space-between;
      flex-wrap: wrap; gap: 12px;
    }
    .topbar h1 {
      margin: 0; font-size: 19px; font-weight: 600; color: var(--text);
      letter-spacing: 0;
    }
    .topbar h1 .mono { color: var(--accent); font-weight: 600; }
    .topbar .meta { color: var(--text-faint); font-size: 12px; }
    .topbar .meta strong { color: var(--text-muted); font-weight: 500; }

    .tabs {
      display: flex; gap: 0; padding: 0 24px;
      border-bottom: 1px solid var(--border);
      position: sticky; top: 0; z-index: 10;
    }
    .tabs button {
      background: transparent; border: 0; color: var(--text-muted);
      padding: 14px 18px; cursor: pointer; font-size: 13px; font-weight: 500;
      letter-spacing: 0.04em; text-transform: uppercase;
      border-bottom: 2px solid transparent; transition: color 0.15s;
    }
    .tabs button:hover { color: var(--text); }
    .tabs button.active {
      color: var(--accent); border-bottom-color: var(--accent);
    }
    .tab-content { display: none; padding: 24px; max-width: 1400px; margin: 0 auto; }
    .tab-content.active { display: block; }

    .badges { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 20px; }
    .badge {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 5px 10px; border-radius: 4px; font-size: 11px;
      background: var(--panel-2); border: 1px solid var(--border);
      color: var(--text-muted); letter-spacing: 0.04em;
      text-transform: uppercase; font-weight: 600;
    }
    .badge .key { color: var(--text-faint); font-weight: 500; }
    .badge .val { color: var(--text); font-weight: 600; }

    .kpis {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 12px; margin-bottom: 20px;
    }
    .kpi {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 8px; padding: 14px 16px;
    }
    .kpi .label {
      color: var(--text-muted); font-size: 11px; letter-spacing: 0.08em;
      text-transform: uppercase; font-weight: 600;
    }
    .kpi .value {
      font-size: 26px; font-weight: 600; color: var(--text);
      margin-top: 6px; line-height: 1.1;
    }
    .kpi .delta { font-size: 13px; color: var(--text-muted); margin-top: 6px; }
    .kpi.good .value { color: var(--good); }
    .kpi.good { border-color: var(--good-border); background: var(--good-bg); }
    .kpi.warn .value { color: var(--warn); }
    .kpi.warn { border-color: var(--warn-border); background: var(--warn-bg); }
    .kpi.bad .value { color: var(--bad); }
    .kpi.bad { border-color: var(--bad-border); background: var(--bad-bg); }
    .kpi.muted .value { color: var(--text-muted); }

    .narrative {
      background: var(--panel); border: 1px solid var(--border);
      border-radius: 8px; padding: 16px 18px; color: var(--text-muted);
      margin-bottom: 20px; font-size: 14px;
    }
    .narrative strong { color: var(--text); font-weight: 600; }

    .plateaux-layout {
      display: grid;
      grid-template-columns: 280px minmax(0, 1fr);
      gap: 20px;
      align-items: start;
    }
    .event-tabs {
      display: flex;
      flex-direction: column;
      gap: 6px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px;
      position: sticky; top: 80px;
      z-index: 5;
    }
    .event-tabs .item {
      display: grid;
      grid-template-columns: auto 1fr auto;
      align-items: center;
      gap: 12px;
      padding: 12px 14px;
      border-radius: 6px;
      cursor: pointer;
      transition: background 0.12s, border-color 0.12s, transform 0.06s;
      user-select: none;
      background: var(--panel-2);
      border: 1px solid var(--border);
    }
    .event-tabs .item:hover {
      background: var(--panel-3);
      border-color: var(--border-strong);
    }
    .event-tabs .item:active { transform: translateY(1px); }
    .event-tabs .item.active {
      background: var(--accent-bg);
      border-color: var(--accent-border);
    }
    .event-tabs .item .dot {
      width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
      box-shadow: 0 0 0 3px var(--panel);
    }
    .event-tabs .item.fixed .dot { background: var(--good); }
    .event-tabs .item.no_op .dot { background: var(--accent); }
    .event-tabs .item.open .dot { background: var(--bad); }
    .event-tabs .item .body-col { display: flex; flex-direction: column; gap: 2px; min-width: 0; }
    .event-tabs .item .ev {
      font-size: 10px; color: var(--text-faint);
      letter-spacing: 0.08em; text-transform: uppercase; font-weight: 600;
    }
    .event-tabs .item .fl {
      font-size: 16px; font-weight: 600; color: var(--text);
      letter-spacing: 0.01em;
    }
    .event-tabs .item .status-text {
      font-size: 10px; color: var(--text-muted);
      letter-spacing: 0.08em; text-transform: uppercase; font-weight: 600;
      padding: 3px 7px; border-radius: 3px; background: var(--panel);
      border: 1px solid var(--border);
    }
    .event-tabs .item.fixed .status-text {
      color: var(--good); border-color: var(--good-border); background: var(--good-bg);
    }
    .event-tabs .item.no_op .status-text {
      color: var(--accent); border-color: var(--accent-border); background: var(--accent-bg);
    }
    .event-tabs .item.open .status-text {
      color: var(--bad); border-color: var(--bad-border); background: var(--bad-bg);
    }
    .event-tabs .item .iter-tag {
      font-size: 10px; color: var(--text-faint);
      font-family: "JetBrains Mono", monospace;
    }

    .view-main {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 22px;
      min-height: 740px;
    }
    .view-main .placeholder {
      color: var(--text-faint); font-size: 14px; text-align: center;
      padding: 80px 0;
    }

    .event-header {
      display: flex; align-items: center; gap: 16px; flex-wrap: wrap;
      padding-bottom: 18px; margin-bottom: 20px;
      border-bottom: 1px solid var(--border);
    }
    .event-header .ev {
      font-size: 13px; color: var(--text-muted);
      letter-spacing: 0.06em; text-transform: uppercase; font-weight: 600;
    }
    .event-header .fl {
      font-size: 28px; font-weight: 600; color: var(--text);
      letter-spacing: 0.02em;
    }
    .event-header .phase {
      font-size: 11px; padding: 4px 10px; border-radius: 3px;
      background: var(--panel-2); border: 1px solid var(--border);
      color: var(--text-muted); letter-spacing: 0.06em;
      text-transform: uppercase; font-weight: 600;
    }
    .event-header .meta { color: var(--text-muted); font-size: 13px; }
    .event-header .badge {
      margin-left: auto; font-size: 11px; font-weight: 600;
      letter-spacing: 0.06em; text-transform: uppercase;
      padding: 4px 10px; border-radius: 3px;
    }
    .event-header .badge.fixed {
      background: var(--good-bg); color: var(--good);
      border: 1px solid var(--good-border);
    }
    .event-header .badge.no_op {
      background: var(--accent-bg); color: var(--accent);
      border: 1px solid var(--accent-border);
    }
    .event-header .badge.open {
      background: var(--bad-bg); color: var(--bad);
      border: 1px solid var(--bad-border);
    }

    .stats {
      display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px; margin-bottom: 20px;
    }
    .stat {
      padding: 14px 16px; background: var(--panel);
      border: 1px solid var(--border); border-radius: 8px;
    }
    .stat .label {
      color: var(--text-muted); font-size: 10px; letter-spacing: 0.08em;
      text-transform: uppercase; font-weight: 600;
    }
    .stat .value {
      font-size: 22px; font-weight: 600; color: var(--text);
      margin-top: 6px;
    }
    .stat .sub { font-size: 11px; color: var(--text-muted); margin-top: 2px; }
    .stat.good .value { color: var(--good); }
    .stat.bad .value { color: var(--bad); }

    .conv-chart, .replay-chart {
      background: transparent; border: 0;
      border-radius: 0; padding: 0; margin-bottom: 16px;
      width: 100%;
    }
    .conv-chart > .plot-container,
    .replay-chart > .plot-container,
    #conv-active, #replay-active { width: 100% !important; }

    .section-title {
      color: var(--text-muted); font-size: 11px; letter-spacing: 0.08em;
      text-transform: uppercase; font-weight: 600; margin-bottom: 6px;
    }

    .view-switcher {
      display: inline-flex;
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 3px;
      margin-bottom: 16px;
      gap: 0;
    }
    .view-switcher button {
      background: transparent;
      border: 0;
      padding: 7px 14px;
      border-radius: 4px;
      cursor: pointer;
      font-size: 12px;
      font-weight: 600;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--text-muted);
      transition: background 0.12s, color 0.12s, box-shadow 0.12s;
    }
    .view-switcher button:hover { color: var(--text); }
    .view-switcher button.active {
      background: var(--panel);
      color: var(--accent);
      box-shadow: 0 1px 2px rgba(0,0,0,0.06);
    }

    .iter-picker { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 12px; }
    .iter-picker button {
      background: var(--panel-2); border: 1px solid var(--border);
      color: var(--text-muted); padding: 5px 10px; border-radius: 4px;
      cursor: pointer; font-size: 12px;
      font-family: "JetBrains Mono", monospace;
    }
    .iter-picker button:hover { background: var(--panel-3); color: var(--text); }
    .iter-picker button.active {
      background: var(--accent-bg); border-color: var(--accent-border);
      color: var(--accent);
    }
    .iter-picker button .tag {
      display: inline-block; margin-left: 6px;
      font-size: 10px; padding: 1px 6px; border-radius: 2px;
      background: var(--panel); color: var(--text-muted);
      border: 1px solid var(--border);
    }

    .feature-tabs { display: flex; gap: 4px; margin-bottom: 6px; }
    .feature-tabs button {
      background: var(--panel-2); border: 1px solid var(--border);
      color: var(--text-muted); padding: 5px 12px; border-radius: 4px;
      cursor: pointer; font-size: 12px; font-weight: 500;
      letter-spacing: 0.04em; text-transform: uppercase;
    }
    .feature-tabs button:hover { background: var(--panel-3); color: var(--text); }
    .feature-tabs button.active {
      background: var(--accent); border-color: var(--accent); color: white;
    }

    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td {
      padding: 8px 10px; border-bottom: 1px solid var(--border); text-align: right;
    }
    th:first-child, td:first-child,
    th:nth-child(2), td:nth-child(2),
    th:nth-child(3), td:nth-child(3) { text-align: left; }
    th {
      color: var(--text-faint); font-weight: 600; font-size: 11px;
      letter-spacing: 0.06em; text-transform: uppercase;
    }
    tr:hover td { background: var(--panel-2); }

    .filterbar { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }
    .filterbar label {
      color: var(--text-faint); font-size: 11px; font-weight: 600;
      letter-spacing: 0.06em; text-transform: uppercase;
    }
    .filterbar select {
      background: var(--panel-2); color: var(--text);
      border: 1px solid var(--border); border-radius: 4px;
      padding: 6px 10px; font-size: 13px;
    }

    .def-list { display: grid; grid-template-columns: 200px 1fr; gap: 8px 16px; }
    .def-list dt {
      color: var(--text-faint); font-size: 11px; letter-spacing: 0.06em;
      text-transform: uppercase; font-weight: 600;
    }
    .def-list dd { margin: 0; color: var(--text); font-size: 13px; word-break: break-all; }

    .pill {
      display: inline-block; padding: 2px 8px; border-radius: 3px;
      font-size: 11px; font-weight: 600; letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .pill.good { background: var(--good-bg); color: var(--good); border: 1px solid var(--good-border); }
    .pill.bad { background: var(--bad-bg); color: var(--bad); border: 1px solid var(--bad-border); }
    .pill.warn { background: var(--warn-bg); color: var(--warn); border: 1px solid var(--warn-border); }
    .pill.neutral { background: var(--neutral-bg); color: var(--neutral); border: 1px solid var(--border); }
  </style>
</head>
<body>
  <div class="topbar">
    <h1>Offline H<sub>E</sub> correction &middot; <span class="mono">__FLIGHT_ID__</span></h1>
    <div class="meta">
      <strong>Flight</strong> __FLIGHT_ID__ &middot;
      <strong>Generated</strong> __GENERATED_AT__
    </div>
  </div>

  <div class="tabs">
    <button class="active" data-tab="overview">Overview</button>
    <button data-tab="plateaux">Plateaux</button>
    <button data-tab="decisions">Decisions</button>
    <button data-tab="config">Run config</button>
  </div>

  <section class="tab-content active" id="overview">
    <div class="badges" id="badges"></div>
    <div class="kpis" id="kpis"></div>
    <div class="narrative" id="narrative"></div>
    <div class="badges" id="phase-summary"></div>
  </section>

  <section class="tab-content" id="plateaux">
    <div class="plateaux-layout">
      <nav class="event-tabs" id="event-tabs"></nav>
      <main class="view-main" id="event-view">
        <div class="placeholder" id="event-placeholder">select an event from the list</div>
      </main>
    </div>
  </section>

  <section class="tab-content" id="decisions">
    <div class="filterbar" id="decisions-filter"></div>
    <div id="decisions-table"></div>
  </section>

  <section class="tab-content" id="config">
    <dl class="def-list" id="config-list"></dl>
  </section>

  <script>
    const D = __DATA__;
    const PAPER = "rgba(255,255,255,0)";
    const PLOT = "rgba(255,255,255,0)";
    const GRID = "#e3e6eb";
    const AXIS = "#9aa3b3";
    const TEXT = "#1f2937";
    const MUTED = "#5a6577";
    const OBS = "#6b7280";
    const REPLAY = "#0369a1";
    const TARGET = "#b45309";
    const GOOD = "#047857";
    const WARN = "#b45309";
    const BAD = "#b91c1c";

const BASE = {
      paper_bgcolor: PAPER, plot_bgcolor: PLOT,
      font: { color: TEXT, size: 12, family: "-apple-system, Inter, sans-serif" },
      margin: { l: 64, r: 24, t: 36, b: 56 },
      autosize: true,
      xaxis: { gridcolor: GRID, zerolinecolor: GRID, linecolor: AXIS, tickcolor: AXIS,
                zerolinewidth: 1, showline: true },
      yaxis: { gridcolor: GRID, zerolinecolor: GRID, linecolor: AXIS, tickcolor: AXIS,
                zerolinewidth: 1, showline: true },
      legend: { bgcolor: "rgba(255,255,255,0)", font: { color: TEXT, size: 11 },
                orientation: "h", x: 0, y: -0.18, xanchor: "left" },
      hoverlabel: { bgcolor: "#ffffff", bordercolor: "#c9cdd3",
                       font: { color: TEXT, family: "JetBrains Mono, monospace" } },
    };
    const CFG = {
      responsive: true, displaylogo: false,
      modeBarButtonsToAdd: ["zoom2d", "pan2d", "zoomIn2d", "zoomOut2d",
                              "autoScale2d", "resetScale2d"],
      modeBarButtonsToRemove: ["select2d", "lasso2d", "autoScale2d",
                                "toImage", "sendDataToCloud",
                                "toggleSpikelines", "hoverClosestCartesian",
                                "hoverCompareCartesian"],
      scrollZoom: true,
      displayModeBar: "hover",
    };

    function fmt(v, digits = 1) {
      if (v === null || v === undefined || Number.isNaN(v)) return "-";
      return Number(v).toFixed(digits);
    }
    function fmtInt(v) { return (v === null || v === undefined) ? "-" : Math.round(v); }
    function flStr(ft) {
      if (ft === null || ft === undefined) return "-";
      const fl = Math.round(ft / 100);
      if (Number.isNaN(fl)) return "-";
      return "FL" + String(fl).padStart(3, "0");
    }
    function numCls(v, goodThr, badThr) {
      if (v === null || v === undefined) return "";
      if (goodThr !== undefined && v <= goodThr) return "good";
      if (badThr !== undefined && v >= badThr) return "bad";
      return "";
    }

    function el(tag, attrs, children) {
      const e = document.createElement(tag);
      if (attrs) for (const k of Object.keys(attrs)) {
        if (k === "class") e.className = attrs[k];
        else if (k === "html") e.innerHTML = attrs[k];
        else if (k.startsWith("on")) e.addEventListener(k.slice(2).toLowerCase(), attrs[k]);
        else e.setAttribute(k, attrs[k]);
      }
      const append = (c) => {
        if (c === null || c === undefined || c === false) return;
        if (Array.isArray(c)) c.forEach(append);
        else if (c instanceof Node) e.appendChild(c);
        else e.appendChild(document.createTextNode(String(c)));
      };
      if (children) children.forEach(append);
      return e;
    }

    function drawOverview() {
      const k = D.kpis;
      const c = D.config;
      const badges = [
        { k: "sensitivity update", v: c.sensitivity_update || "finite_difference" },
        { k: "modes", v: (c.modes || ["CLIMB"]).join(", ") },
        { k: "inner band", v: `${fmtInt(c.inner_band_ft)} ft` },
        { k: "max |delta H_E|", v: `${fmtInt(c.max_delta_he_ft)} ft` },
        { k: "epsilon RDP", v: `${fmtInt(c.epsilon_ft)} ft` },
        { k: "alpha", v: fmt(c.alpha, 2) },
        { k: "activation gap", v: `${fmtInt(c.activation_gap_ft)} ft` },
      ];
      document.getElementById("badges").innerHTML = badges.map(b =>
        `<span class="badge"><span class="key">${b.k}</span> <span class="val">${b.v}</span></span>`
      ).join("");
      const closureDelta = k.closure_share_delta;
      const tiles = [
        { label: "Closure", value: `${Math.round(k.closure_share_controlled * 100)}%`,
          delta: `baseline ${Math.round(k.closure_share_baseline * 100)}% (+${Math.round(closureDelta * 100)}pp)`,
          klass: k.closure_share_controlled >= 0.99 ? "good" : k.closure_share_controlled >= 0.5 ? "warn" : "bad" },
        { label: "Plateaux closed", value: `${k.n_plateaux_closed_controlled}/${k.n_plateaux_evaluated}`,
          delta: `baseline ${k.n_plateaux_closed_baseline}/${k.n_plateaux_evaluated}`,
          klass: k.n_plateaux_closed_controlled === k.n_plateaux_evaluated ? "good" : "warn" },
        { label: "MAE", value: `${fmtInt(k.fullflight_mae_controlled_ft)} ft`,
          delta: `baseline ${fmtInt(k.fullflight_mae_baseline_ft)} ft, delta ${fmtInt(k.mae_delta_ft)} ft`,
          klass: numCls(k.fullflight_mae_controlled_ft, 150, 400) },
        { label: "Median |mid err|", value: `${fmtInt(k.median_abs_closure_error_controlled_ft)} ft`,
          delta: `baseline ${fmtInt(k.median_abs_closure_error_baseline_ft)} ft`,
          klass: numCls(k.median_abs_closure_error_controlled_ft, c.inner_band_ft || 80, 250) },
        { label: "Capture timing", value: `${fmtInt(k.median_abs_capture_timing_error_controlled_s)} s`,
          delta: `baseline ${fmtInt(k.median_abs_capture_timing_error_baseline_s)} s`,
          klass: numCls(k.median_abs_capture_timing_error_controlled_s, 12, 30) },
        { label: "Iterations", value: `${k.n_applied_iterations}`,
          delta: `${k.n_accepted} accepted, ${k.n_skipped_bad_sensitivity} bad-sens, ${k.n_iteration_limit} limit`,
          klass: "muted" },
      ];
      document.getElementById("kpis").innerHTML = tiles.map(t =>
        `<div class="kpi ${t.klass}"><div class="label">${t.label}</div><div class="value">${t.value}</div><div class="delta">${t.delta}</div></div>`
      ).join("");

      const closedCount = k.n_plateaux_closed_controlled;
      const narrative = `
        <strong>${k.n_applied_iterations} applied iterations</strong> on
        <strong>${closedCount}/${k.n_plateaux_evaluated} plateaus</strong>.
        Baseline closed ${k.n_plateaux_closed_baseline}/${k.n_plateaux_evaluated} plateaus
        with median |midpoint error| ${fmtInt(k.median_abs_closure_error_baseline_ft)} ft;
        the controlled replay closes ${closedCount} of them at ${fmtInt(k.median_abs_closure_error_controlled_ft)} ft median,
        well inside the ${fmtInt(c.inner_band_ft)} ft inner band.
        Full-flight MAE drops from ${fmtInt(k.fullflight_mae_baseline_ft)} ft to
        ${fmtInt(k.fullflight_mae_controlled_ft)} ft
        (delta <strong>${fmtInt(k.mae_delta_ft)} ft</strong>),
        capture-timing error from ${fmtInt(k.median_abs_capture_timing_error_baseline_s)} s to
        ${fmtInt(k.median_abs_capture_timing_error_controlled_s)} s.`;
      document.getElementById("narrative").innerHTML = narrative;

      const phaseList = (c.modes || ["CLIMB"]).map(m =>
        `<span class="badge"><span class="key">phase</span> <span class="val">${m}</span></span>`
      ).join("");
      document.getElementById("phase-summary").innerHTML =
        `<span class="badge"><span class="key">eligible phases</span></span> ${phaseList}`;
    }

    function drawConfig() {
      const c = D.config;
      const list = document.getElementById("config-list");
      const skip = new Set(["commands", "context", "output_dir", "model_path"]);
      const keys = Object.keys(c).filter(k => !skip.has(k)).sort();
      keys.forEach(h => {
        const dt = document.createElement("dt"); dt.textContent = h;
        const dd = document.createElement("dd");
        const v = c[h];
        dd.textContent = typeof v === "string" || typeof v === "number"
          ? v : JSON.stringify(v);
        list.appendChild(dt); list.appendChild(dd);
      });
      if (c.commands) list.appendChild(el("dt", {textContent: "commands"})),
                       list.appendChild(el("dd", {textContent: c.commands, class: "mono"}));
      if (c.context)  list.appendChild(el("dt", {textContent: "context"})),
                       list.appendChild(el("dd", {textContent: c.context, class: "mono"}));
      if (c.model_path) list.appendChild(el("dt", {textContent: "model_path"})),
                        list.appendChild(el("dd", {textContent: c.model_path, class: "mono"}));
      if (c.output_dir) list.appendChild(el("dt", {textContent: "output_dir"})),
                        list.appendChild(el("dd", {textContent: c.output_dir, class: "mono"}));
    }

    function drawPlateaux() {
      const list = document.getElementById("event-tabs");
      list.innerHTML = "";
      D.scorecard.forEach((p) => {
        const status = p.controller_status || (p.controlled_captured ? "no_op" : "open");
        const labelMap = { fixed: "fixed", no_op: "no-op", open: "open" };
        const item = el("div", {
          class: `item ${status}`,
          onclick: () => selectEvent(p.event),
        });
        item.appendChild(el("span", { class: "dot" }));
        const body = el("div", { class: "body-col" });
        body.appendChild(el("span", { class: "ev" }, [`Event ${p.event}`]));
        body.appendChild(el("span", { class: "fl" }, [flStr(p.h_sel_ft)]));
        body.appendChild(el("span", { class: "iter-tag" },
          [status === "fixed" ? `${p.n_iterations} iters`
           : status === "no_op" ? "no-op"
           : `${fmtInt(p.baseline_abs_err_ft)} ft open`]));
        item.appendChild(body);
        item.appendChild(el("span", { class: "status-text" }, [labelMap[status]]));
        item.dataset.event = String(p.event);
        list.appendChild(item);
      });
      if (D.scorecard.length) selectEvent(D.scorecard[0].event);
    }

    let currentEvent = null;
    let currentIterIdx = null;
    let currentFeature = "altitude";
    let currentView = "convergence";

    function selectEvent(event) {
      currentEvent = event;
      currentIterIdx = null;
      document.querySelectorAll("#event-tabs .item").forEach(b => {
        b.classList.toggle("active", Number(b.dataset.event) === event);
      });
      renderEventView();
    }

    function selectView(name) {
      currentView = name;
      document.querySelectorAll("#event-view .view-switcher button").forEach(b => {
        b.classList.toggle("active", b.dataset.view === name);
      });
      const convWrap = document.getElementById("view-conv");
      const replayWrap = document.getElementById("view-replay");
      if (convWrap) convWrap.style.display = (name === "convergence") ? "" : "none";
      if (replayWrap) replayWrap.style.display = (name === "replay") ? "" : "none";
    }

    function renderEventView() {
      const root = document.getElementById("event-view");
      root.innerHTML = "";
      const p = D.scorecard.find(x => x.event === currentEvent);
      if (!p) return;
      const eventIters = D.iterations.filter(it => it.event === currentEvent);

      const status = p.controller_status || (p.controlled_captured ? "no_op" : "open");
      const statusLabel = { fixed: "fixed", no_op: "no-op", open: "open" }[status];
      const header = el("div", { class: "event-header" });
      header.appendChild(el("span", { class: "ev" }, [`Event ${p.event}`]));
      header.appendChild(el("span", { class: "fl" }, [flStr(p.h_sel_ft)]));
      header.appendChild(el("span", { class: "phase" }, [p.phase]));
      header.appendChild(el("span", { class: "meta mono" },
        [`${fmt(p.plateau_min, 2)} min, plateau ${fmtInt(p.plateau_length_s)} s`]));
      header.appendChild(el("span", { class: `badge ${status}` }, [statusLabel]));
      root.appendChild(header);

      root.appendChild(buildStats(p));

      const switcher = el("div", { class: "view-switcher" });
      [
        { id: "convergence", label: `Convergence (${eventIters.length})` },
        { id: "replay", label: `Replay (${eventIters.length})` },
      ].forEach(v => {
        const b = el("button", {
          type: "button",
          onclick: () => selectView(v.id),
        }, [v.label]);
        b.dataset.view = v.id;
        switcher.appendChild(b);
      });
      root.appendChild(switcher);

      const convWrap = el("div", { id: "view-conv" });
      convWrap.appendChild(el("div", { class: "section-title" }, ["convergence trajectory"]));
      const convDiv = el("div", { class: "conv-chart", id: "conv-active",
                                     style: "min-height: 460px" });
      convWrap.appendChild(convDiv);
      root.appendChild(convWrap);

      const replayWrap = el("div", { id: "view-replay" });
      replayWrap.appendChild(el("div", { class: "section-title" },
        [`replay \u00b7 ${eventIters.length} iterations`]));
      const iterRow = el("div", { class: "iter-picker" });
      eventIters.forEach((it, j) => {
        const btn = el("button", {
          type: "button",
          onclick: () => selectIter(j),
        }, [
          `iter ${it.index}`,
          el("span", { class: "tag" }, [iterLabel(it)]),
        ]);
        btn.dataset.j = String(j);
        iterRow.appendChild(btn);
      });
      replayWrap.appendChild(iterRow);

      const featureTabs = el("div", { class: "feature-tabs" });
      ["altitude", "tas", "gamma", "v_z"].forEach(name => {
        const b = el("button", {
          type: "button",
          onclick: () => selectFeature(name),
        }, [name.replace("_", " ")]);
        b.dataset.feature = name;
        featureTabs.appendChild(b);
      });
      replayWrap.appendChild(featureTabs);

      const replayDiv = el("div", { class: "replay-chart", id: "replay-active",
                                        style: "min-height: 460px" });
      replayWrap.appendChild(replayDiv);
      root.appendChild(replayWrap);

      renderConvChart(currentEvent);
      if (currentIterIdx === null) currentIterIdx = Math.max(0, eventIters.length - 1);
      selectIter(currentIterIdx);
      selectFeature(currentFeature);
      selectView(currentView);
    }

    function buildStats(p) {
      const wrap = el("div", { class: "stats" });
      const closed = p.controlled_captured;
      const status = p.controller_status || (closed ? "no_op" : "open");
      const items = [
        { label: "Baseline |err|", value: fmtInt(p.baseline_abs_err_ft) + " ft",
          klass: "bad" },
        { label: "Controlled |err|", value: fmtInt(p.controlled_abs_err_ft) + " ft",
          klass: closed ? "good" : "bad" },
        { label: "Iterations", value: status === "fixed" ? `${p.n_iterations}`
                              : status === "no_op" ? "no-op" : "-",
          klass: status === "fixed" ? "good" : status === "no_op" ? "" : "bad" },
        { label: "Capture timing (ctrl)", value: p.controlled_timing_s === null ? "-"
          : (p.controlled_timing_s >= 0 ? "+" : "") + fmtInt(p.controlled_timing_s) + " s",
          klass: "" },
      ];
      items.forEach(it => {
        wrap.appendChild(el("div", { class: `stat ${it.klass}` }, [
          el("div", { class: "label" }, [it.label]),
          el("div", { class: "value" }, [it.value]),
        ]));
      });
      return wrap;
    }

    function renderConvChart(event) {
      const trace = D.convergence.find(t => t.event === event);
      if (!trace) return;
      const innerBand = D.config.inner_band_ft || 80;
      const xmax = trace.x.length;
      const traces = [{
        x: trace.x, y: trace.midpoint_error_ft.map(v => Math.abs(v || 0)),
        mode: "lines+markers", line: { color: REPLAY, width: 2 },
        marker: { color: REPLAY, size: 7 },
        name: "|mid err|",
        hovertemplate: `iter %{x}<br>|err| %{y:.0f} ft<extra></extra>`,
      }];
      traces.push({
        x: [0, xmax], y: [250, 250], mode: "lines",
        line: { color: WARN, width: 1, dash: "dash" }, hoverinfo: "skip",
        name: "closure 250 ft",
      });
      traces.push({
        x: [0, xmax], y: [innerBand, innerBand], mode: "lines",
        line: { color: GOOD, width: 1, dash: "dot" }, hoverinfo: "skip",
        name: `inner ${innerBand} ft`,
      });
      Plotly.newPlot("conv-active", traces, {
        ...BASE,
        height: 440, margin: { ...BASE.margin, t: 20, b: 60 },
        xaxis: { ...BASE.xaxis, title: "iteration", dtick: 1, range: [-0.5, xmax + 0.5] },
        yaxis: { ...BASE.yaxis, title: "|mid err| (ft)", rangemode: "tozero",
                 gridcolor: GRID },
        showlegend: true,
        legend: { ...BASE.legend, y: -0.45, yanchor: "top" },
      }, CFG);
    }

    function iterLabel(it) {
      if (it.label && it.label.includes("Baseline")) return "baseline";
      if (it.decision && it.decision.selection) return it.decision.selection.replace("_", " ");
      return "applied";
    }

    function selectIter(j) {
      currentIterIdx = j;
      const eventIters = D.iterations.filter(it => it.event === currentEvent);
      document.querySelectorAll("#event-view .iter-picker button").forEach((b, k) => {
        b.classList.toggle("active", k === j);
      });
      renderReplayChart(eventIters[j]);
    }

    function selectFeature(name) {
      currentFeature = name;
      document.querySelectorAll("#event-view .feature-tabs button").forEach(b => {
        b.classList.toggle("active", b.dataset.feature === name);
      });
      const eventIters = D.iterations.filter(it => it.event === currentEvent);
      if (currentIterIdx !== null && eventIters[currentIterIdx]) {
        renderReplayChart(eventIters[currentIterIdx]);
      }
    }

    function renderReplayChart(iter) {
      if (!iter) return;
      const s = iter.series;
      const x = s.time_min;
      const traces = [];
      let yTitle = "";
      let yRange = null;
      const annotations_ = [];

      if (currentFeature === "altitude") {
        traces.push({ x, y: s.observed_altitude_ft, mode: "lines", name: "observed",
          line: { color: OBS, width: 1.8 },
          hovertemplate: "t=%{x:.2f}m<br>obs %{y:.0f} ft<extra></extra>" });
        traces.push({ x, y: s.predicted_altitude_ft, mode: "lines", name: "replay",
          line: { color: REPLAY, width: 2 },
          hovertemplate: "t=%{x:.2f}m<br>replay %{y:.0f} ft<extra></extra>" });
        traces.push({ x, y: s.inferred_target_ft, mode: "lines", name: "target h_sel",
          line: { color: TARGET, width: 1.6, dash: "dash", shape: "hv" },
          hovertemplate: "t=%{x:.2f}m<br>target %{y:.0f} ft<extra></extra>" });
        yTitle = "altitude (ft)";
        if (iter.plateau) {
          const target = iter.plateau.h_sel_ft;
          const pad = 3000;
          let yMin = target - pad;
          let yMax = target + pad;
          const obs = s.predicted_altitude_ft.filter(v => v !== null);
          const minObs = Math.min(...obs);
          const maxObs = Math.max(...obs);
          if (minObs < yMin) yMin = Math.max(0, minObs - 200);
          if (maxObs > yMax) yMax = maxObs + 500;
          yRange = [yMin, yMax];
        } else {
          const predMax = Math.max(...s.predicted_altitude_ft.filter(v => v !== null));
          yRange = [0, Math.max(42000, predMax * 1.05)];
        }
        if (iter.plateau) {
          const t = s.time_min;
          annotations_.push({
            x: t[iter.plateau.mid_index], y: s.predicted_altitude_ft[iter.plateau.mid_index],
            xref: "x", yref: "y", text: "mid",
            showarrow: true, arrowhead: 2, ax: 0, ay: -30,
            font: { color: TEXT, size: 10 }, arrowcolor: TARGET,
          });
          annotations_.push({
            x: t[iter.plateau.arrival_index], y: s.inferred_target_ft[iter.plateau.arrival_index],
            xref: "x", yref: "y", text: "arrival",
            showarrow: true, arrowhead: 2, ax: 0, ay: 30,
            font: { color: GOOD, size: 10 }, arrowcolor: GOOD,
          });
        }
      } else if (currentFeature === "tas") {
        traces.push({ x, y: s.observed_tas_kt, mode: "lines", name: "observed",
          line: { color: OBS, width: 1.8 },
          hovertemplate: "t=%{x:.2f}m<br>obs %{y:.0f} kt<extra></extra>" });
        traces.push({ x, y: s.generated_tas_kt, mode: "lines", name: "replay",
          line: { color: REPLAY, width: 2 },
          hovertemplate: "t=%{x:.2f}m<br>replay %{y:.0f} kt<extra></extra>" });
        yTitle = "TAS (kt)";
      } else if (currentFeature === "gamma") {
        traces.push({ x, y: s.observed_gamma_deg, mode: "lines", name: "observed",
          line: { color: OBS, width: 1.8 },
          hovertemplate: "t=%{x:.2f}m<br>obs %{y:.2f} deg<extra></extra>" });
        traces.push({ x, y: s.predicted_gamma_deg, mode: "lines", name: "replay",
          line: { color: REPLAY, width: 2 },
          hovertemplate: "t=%{x:.2f}m<br>replay %{y:.2f} deg<extra></extra>" });
        yTitle = "gamma (deg)";
      } else if (currentFeature === "v_z") {
        traces.push({ x, y: s.observed_vz_fpm, mode: "lines", name: "observed",
          line: { color: OBS, width: 1.8 },
          hovertemplate: "t=%{x:.2f}m<br>obs %{y:.0f} fpm<extra></extra>" });
        yTitle = "vertical rate (ft/min)";
      }

      const xLast = x[x.length - 1] || 1;
      let xRange;
      if (iter.plateau) {
        const idxLo = Math.max(0, iter.plateau.start_index);
        const idxHi = Math.min(x.length - 1, iter.plateau.end_index);
        const tStart = x[idxLo];
        const tEnd = x[idxHi];
        const span = Math.max(0.5, tEnd - tStart);
        const margin = Math.max(0.5, span * 0.2);
        xRange = [Math.max(0, tStart - margin), Math.min(xLast, tEnd + margin)];
      } else {
        xRange = [0, xLast];
      }
      Plotly.newPlot("replay-active", traces, {
        ...BASE,
        height: 440,
        margin: { ...BASE.margin, t: 28, b: 60 },
        xaxis: { ...BASE.xaxis, title: "time (min)", range: xRange },
        yaxis: { ...BASE.yaxis, title: yTitle, range: yRange },
        annotations: annotations_,
        showlegend: true,
        legend: { ...BASE.legend, y: -0.16 },
      }, CFG);
    }

    function drawDecisions() {
      const rows = D.decisions;
      const events = [...new Set(rows.map(r => r.event))].sort((a, b) => a - b);
      const statuses = [...new Set(rows.map(r => r.status))].sort();
      const selections = [...new Set(rows.map(r => r.selection).filter(Boolean))].sort();
      const filterbar = document.getElementById("decisions-filter");
      filterbar.appendChild(makeSelect("filter-event", "Event", ["all", ...events], v => v, v =>
        v === "all" ? "all" : `event ${v}`));
      filterbar.appendChild(makeSelect("filter-status", "Status", ["all", ...statuses], v => v));
      filterbar.appendChild(makeSelect("filter-selection", "Selection", ["all", ...selections], v => v));
      ["filter-event", "filter-status", "filter-selection"].forEach(id =>
        document.getElementById(id).addEventListener("change", renderDecisionsTable));
      renderDecisionsTable();
    }
    function makeSelect(id, label, values, transform, labelFn) {
      const wrap = el("label", {}, [label + " "]);
      const s = el("select", { id });
      values.forEach(v => {
        const opt = document.createElement("option");
        opt.value = v;
        opt.textContent = labelFn ? labelFn(v) : transform(v);
        s.appendChild(opt);
      });
      wrap.appendChild(s);
      return wrap;
    }
    function renderDecisionsTable() {
      const ev = document.getElementById("filter-event").value;
      const st = document.getElementById("filter-status").value;
      const sel = document.getElementById("filter-selection").value;
      const rows = D.decisions.filter(r =>
        (ev === "all" || r.event === Number(ev)) &&
        (st === "all" || r.status === st) &&
        (sel === "all" || r.selection === sel));
      const cols = ["event", "h_sel_ft", "segment_start", "iteration",
        "selection", "midpoint_error_ft", "applied_delta_he_ft",
        "cumulative_delta_he_ft", "sensitivity_ft_per_ft", "status"];
      const head = cols.map(c => `<th>${c}</th>`).join("");
      const body = rows.map(r => {
        const cells = cols.map(c => {
          const v = r[c];
          if (v === null || v === undefined) return "<td>-</td>";
          if (typeof v === "number") return `<td class="mono">${fmt(v, 2)}</td>`;
          if (c === "status") {
            const klass = v === "accepted" ? "good" :
                            v === "applied" ? "" :
                            v === "skipped_bad_sensitivity" || v === "stalled_at_curve_minimum" ? "warn" :
                            v === "iteration_limit" || v === "no_eligible_segment" ? "bad" : "";
            return `<td><span class="pill ${klass}">${v.replace(/_/g, " ")}</span></td>`;
          }
          if (c === "selection") {
            const klass = v === "finite_difference" ? "" : v === "broyden" ? "good" : "neutral";
            return `<td><span class="pill ${klass}">${v}</span></td>`;
          }
          return `<td class="mono">${v}</td>`;
        }).join("");
        return `<tr>${cells}</tr>`;
      }).join("");
      document.getElementById("decisions-table").innerHTML =
        `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>` +
        `<div style="margin-top:8px;color:var(--text-faint);font-size:11px">${rows.length} of ${D.decisions.length} rows</div>`;
    }

    document.querySelectorAll(".tabs button").forEach(btn => btn.addEventListener("click", () => {
      document.querySelectorAll(".tabs button").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach(t => t.classList.remove("active"));
      btn.classList.add("active");
      const target = document.getElementById(btn.dataset.tab);
      if (target) target.classList.add("active");
    }));

    drawOverview();
    drawConfig();
    drawPlateaux();
    drawDecisions();
  </script>
</body>
</html>
"""


def build_payload(run_dir: Path) -> dict[str, object]:
    metrics, cfg = _read_metrics(run_dir)
    base_card, ctrl_card = _per_plateau_scorecards(run_dir)
    decisions = _decisions(run_dir)
    return {
        "run_id": run_dir.name,
        "generated_at": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "config": cfg,
        "kpis": _kpis(metrics, decisions),
        "scorecard": _summary_rows(base_card, ctrl_card, decisions),
        "decisions": _jsonable(decisions.fillna(np.nan).to_dict("records")),
        "convergence": _convergence_payload(decisions),
        "iterations": _load_iteration_payload(run_dir),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Diagnostics run directory produced by "
                         "run_causal_he_capture_controller.py --save-series")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Where to write dashboard.html and dashboard_data.json. "
                         "Defaults to <run-dir>.")
    args = ap.parse_args()
    if not (args.run_dir / "metrics.csv").exists():
        raise FileNotFoundError(f"No metrics.csv in {args.run_dir}; is this a controller run dir?")
    series_dir = args.run_dir / "iteration_series"
    if not series_dir.exists():
        raise FileNotFoundError(
            f"No iteration_series/ in {args.run_dir}. Re-run the controller with --save-series."
        )
    output_dir = args.output_dir or args.run_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = build_payload(args.run_dir)
    flight_id = ""
    cmds = payload.get("config", {}).get("commands", "")
    if cmds:
        stem = Path(cmds).stem
        flight_id = stem.rsplit("_eps", 1)[0] if "_eps" in stem else stem
    html = HTML_TEMPLATE
    html = html.replace("__DATA__", json.dumps(payload, separators=(",", ":")))
    html = html.replace("__RUN_TITLE__", str(payload["run_id"]))
    html = html.replace("__FLIGHT_ID__", flight_id or "-")
    html = html.replace("__GENERATED_AT__", str(payload["generated_at"]))
    (output_dir / "dashboard.html").write_text(html)
    (output_dir / "dashboard_data.json").write_text(json.dumps(payload, indent=2))

    print(f"dashboard_html={output_dir / 'dashboard.html'}")
    print(f"dashboard_data={output_dir / 'dashboard_data.json'}")
    print(f"inline_iterations={sum(1 for it in payload['iterations'])}")


if __name__ == "__main__":
    main()