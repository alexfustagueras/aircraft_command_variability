#!/usr/bin/env python3
"""Build interactive dataset explorer (HTML + JSON) from data/routes/."""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.logic import (  # noqa: E402
    FAMILY_MAP,
    gc_nm_to_bin,
    load_flight_metadata_table,
    typecode_to_family,
)
from pipeline.opendata import (  # noqa: E402
    accepted_command_flight_ids,
    list_routes,
    route_dataset_dir,
    route_gc_nm,
)

GC_NM_EDGES = [0.0, 500.0, 1000.0]
HTML_OUT = Path(__file__).with_name("dataset_explorer.html")
JSON_OUT = Path(__file__).with_name("dataset_explorer.data.json")
TEMPLATE = Path(__file__).with_name("dataset_explorer.template.html")


def _gc_nm_bin_label(gc_nm: float) -> str:
    b = gc_nm_to_bin(float(gc_nm), np.array(GC_NM_EDGES, dtype=float))
    lo, hi = GC_NM_EDGES[b], GC_NM_EDGES[b + 1]
    return f"{lo:.0f}–{hi:.0f} nm"


def _airport_table() -> pd.DataFrame:
    from traffic.data import airports

    return airports.data


def _route_stats(route: str) -> dict:
    ds = route_dataset_dir(route)
    dep, arr = route.split("_", 1)
    gcnm = float(route_gc_nm(route))
    n_accepted = len(accepted_command_flight_ids(route))
    n_manifest_done = 0
    n_manifest = 0
    mp = ds / "manifest.parquet"
    if mp.exists():
        man = pd.read_parquet(mp)
        n_manifest = len(man)
        if "status" in man.columns:
            n_manifest_done = int((man["status"] == "done").sum())
    qc_fail = 0
    qp = ds / "commands" / "command_qc.parquet"
    if qp.exists():
        qc = pd.read_parquet(qp)
        qc_fail = int((~qc["accepted"].astype(bool)).sum())
    return {
        "id": route,
        "dep": dep,
        "arr": arr,
        "gc_nm": round(gcnm, 1),
        "gc_nm_bin": _gc_nm_bin_label(gcnm),
        "n_accepted": n_accepted,
        "n_manifest": n_manifest,
        "n_manifest_done": n_manifest_done,
        "n_qc_rejected": qc_fail,
    }


def build_payload(routes: list[str]) -> dict:
    apt = _airport_table()
    route_rows = [_route_stats(r) for r in routes]
    meta = load_flight_metadata_table(routes)
    flights: list[dict] = []
    if not meta.empty:
        for _, r in meta.iterrows():
            flights.append(
                {
                    "route": str(r["route"]),
                    "flight_id": str(r["flight_id"]),
                    "typecode": str(r["typecode"] or ""),
                    "family": str(r["family"]),
                    "gc_nm": float(r["gc_nm"]),
                    "gc_nm_bin": _gc_nm_bin_label(float(r["gc_nm"])),
                }
            )

    icaos: set[str] = set()
    for rr in route_rows:
        icaos.add(rr["dep"])
        icaos.add(rr["arr"])

    airports: list[dict] = []
    for icao in sorted(icaos):
        sub = apt.loc[apt["icao"] == icao]
        if sub.empty:
            continue
        row = sub.iloc[0]
        airports.append(
            {
                "icao": icao,
                "name": str(row.get("name", icao)),
                "lat": float(row.latitude),
                "lon": float(row.longitude),
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gc_nm_edges": GC_NM_EDGES,
        "family_map": FAMILY_MAP,
        "routes": route_rows,
        "flights": flights,
        "airports": airports,
        "summary": {
            "n_routes": len(route_rows),
            "n_airports": len(airports),
            "n_accepted_flights": len(flights),
            "n_manifest_done": sum(r["n_manifest_done"] for r in route_rows),
        },
    }


def write_explorer(payload: dict) -> None:
    JSON_OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    html = TEMPLATE.read_text(encoding="utf-8")
    embedded = json.dumps(payload, separators=(",", ":"))
    html = html.replace("/*__DATA__*/", f"const DATA = {embedded};")
    HTML_OUT.write_text(html, encoding="utf-8")
    print(f"wrote {JSON_OUT}")
    print(f"wrote {HTML_OUT}")


def main() -> None:
    routes = list_routes()
    if not routes:
        print("No routes under data/routes/")
        return
    payload = build_payload(routes)
    write_explorer(payload)
    s = payload["summary"]
    print(
        f"routes={s['n_routes']} airports={s['n_airports']} "
        f"accepted={s['n_accepted_flights']}"
    )
    print(f"Open: file://{HTML_OUT}")
    print(f"Or:   cd {HTML_OUT.parent} && python -m http.server 8765")


if __name__ == "__main__":
    main()
