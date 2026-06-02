"""Empirical law fitting utilities."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from pipeline.opendata import (
    accepted_command_flight_ids,
    merge_event_position,
    phi_d_at_event,
    route_arrival_coords,
    route_dataset_dir,
    route_gc_nm,
)

PHI_BIN = 0.02
VZ_BIN_FPM = 100
H_BIN_FT = 500
CAS_BIN_KT = 5
MACH_BIN = 0.01
PHI_JOINT_BINS = 20

FAMILY_MAP: dict[str, list[str]] = {
    "A320 family": ["A319", "A320", "A321", "A20N", "A21N"],
    "A220": ["BCS1", "BCS3"],
    "B737": ["B738", "B737", "B739"],
    "E-Jet": ["E190", "E195", "E290", "E295", "E75L"],
}


def typecode_to_family(tc: str | float | None) -> str:
    if pd.isna(tc):
        return "Other"
    tc = str(tc).strip().upper()
    for fam, codes in FAMILY_MAP.items():
        if tc in codes:
            return fam
    return "Other"


def transition_matrix_bin(events: pd.DataFrame, bin_col: str) -> pd.DataFrame:
    rows = []
    for _, g in events.sort_values("start_timestamp").groupby("flight_id"):
        v = g[bin_col].to_numpy()
        for a, b in zip(v[:-1], v[1:]):
            rows.append({"from": a, "to": b})
    trans = pd.DataFrame(rows)
    if trans.empty:
        return pd.DataFrame()
    return pd.crosstab(trans["from"], trans["to"], normalize="index")


def sample_markov_profile(
    mat: pd.DataFrame, n_plateaus: int, start: float, rng: np.random.Generator) -> np.ndarray:
    cur = float(start)
    path = [cur]
    for _ in range(max(0, n_plateaus - 1)):
        if cur not in mat.index:
            break
        row = mat.loc[cur]
        if row.sum() <= 0:
            break
        nxt = rng.choice(row.index.to_numpy(), p=row.to_numpy())
        cur = float(nxt)
        path.append(cur)
    return np.asarray(path, dtype=float)


def _phase_window(
    route: str, flight_id: str, phase: str) -> tuple[pd.Timestamp, pd.Timestamp, float] | None:
    phase = phase.upper()
    cp = route_dataset_dir(route) / "commands" / f"{flight_id}.parquet"
    if not cp.exists():
        return None
    ph = pd.read_parquet(cp, columns=["timestamp", "phase"])
    ph["timestamp"] = pd.to_datetime(ph["timestamp"], utc=True)
    ph = ph.loc[ph["phase"].astype(str).str.upper() == phase]
    if len(ph) < 2:
        return None
    t0, t1 = ph["timestamp"].iloc[0], ph["timestamp"].iloc[-1]
    T_s = (t1 - t0).total_seconds()
    if T_s <= 30:
        return None
    return t0, t1, T_s


def enrich_events_with_phi(events: pd.DataFrame, phase: str, *, bin_col: str) -> pd.DataFrame:
    """Attach φ info to each segment: φ_mid and dφ=duration/T_phase, plus φ-bin."""
    phase = phase.upper()
    work = events
    if "phase" in work.columns:
        work = work.loc[work["phase"].astype(str).str.upper() == phase]
    rows = []
    for (route, fid), g in work.groupby(["route", "flight_id"]):
        win = _phase_window(route, fid, phase)
        if win is None:
            continue
        t0, _, T_s = win
        for _, seg in g.sort_values("start_timestamp").iterrows():
            ts = pd.Timestamp(seg["start_timestamp"])
            phi_mid = float(np.clip((ts - t0).total_seconds() / T_s, 0.0, 1.0))
            val = float(seg[bin_col])
            rows.append(
                {
                    "route": route,
                    "flight_id": fid,
                    "start_timestamp": ts,
                    "duration_s": float(seg["duration_s"]),
                    "phi_mid": phi_mid,
                    "dphi": float(seg["duration_s"]) / T_s,
                    bin_col: val,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    edges = np.linspace(0, 1, PHI_JOINT_BINS + 1)
    out["phi_bin"] = pd.cut(out["phi_mid"], bins=edges, labels=False, include_lowest=True).astype(
        int
    )
    return out


def build_vz_h_joint_library(vz_ev: pd.DataFrame, h_ev: pd.DataFrame, phase: str) -> pd.DataFrame:
    """φ-conditioned joint rows: (vz_bin, h_bin, dφ)."""
    phase = phase.upper()
    vz_sub = vz_ev.loc[vz_ev["phase"].astype(str).str.upper() == phase].copy()
    h_sub = h_ev.loc[h_ev["phase"].astype(str).str.upper() == phase].copy()
    if vz_sub.empty:
        return pd.DataFrame()
    vz_sub["vz_bin"] = (pd.to_numeric(vz_sub["value"], errors="coerce") / VZ_BIN_FPM).round() * VZ_BIN_FPM
    h_sub["h_bin"] = (pd.to_numeric(h_sub["value"], errors="coerce") / H_BIN_FT).round() * H_BIN_FT
    vz_sub = vz_sub.dropna(subset=["vz_bin"])
    rows = []
    for (route, fid), vz_g in vz_sub.groupby(["route", "flight_id"]):
        h_g = h_sub[(h_sub["route"] == route) & (h_sub["flight_id"] == fid)]
        if h_g.empty:
            continue
        h_asof = h_g[["start_timestamp", "h_bin"]].copy()
        h_asof["start_timestamp"] = pd.to_datetime(h_asof["start_timestamp"], utc=True)
        h_asof = h_asof.sort_values("start_timestamp")
        plateaus = enrich_events_with_phi(vz_g, phase, bin_col="vz_bin")
        if plateaus.empty:
            continue
        for _, row in plateaus.iterrows():
            ts = pd.Timestamp(row["start_timestamp"])
            m = pd.merge_asof(
                pd.DataFrame({"start_timestamp": [ts]}),
                h_asof,
                on="start_timestamp",
                direction="nearest",
                tolerance=pd.Timedelta("60s"),
            )
            if m["h_bin"].isna().iloc[0]:
                continue
            rows.append({**row.to_dict(), "h_bin": float(m["h_bin"].iloc[0])})
    out = pd.DataFrame(rows)
    return out.loc[out["dphi"] > 0] if not out.empty else out


def build_h_phi_library(h_ev: pd.DataFrame, phase: str) -> pd.DataFrame:
    """φ-conditioned h_sel plateaus."""
    phase = phase.upper()
    h_sub = h_ev.loc[h_ev["phase"].astype(str).str.upper() == phase].copy()
    if h_sub.empty:
        return pd.DataFrame()
    h_sub["h_bin"] = (pd.to_numeric(h_sub["value"], errors="coerce") / H_BIN_FT).round() * H_BIN_FT
    h_sub = h_sub.dropna(subset=["h_bin"])
    lib = enrich_events_with_phi(h_sub, phase, bin_col="h_bin")
    return lib.loc[lib["dphi"] > 0] if not lib.empty else lib


def build_cas_vz_joint_library(cas_ev: pd.DataFrame, vz_ev: pd.DataFrame, phase: str) -> pd.DataFrame:
    """φ-conditioned joint rows: (cas_bin, vz_bin, dφ)."""
    phase = phase.upper()
    cas_work = cas_ev.loc[cas_ev["phase"].astype(str).str.upper() == phase].copy()
    cas_work["cas_bin"] = (pd.to_numeric(cas_work["value"], errors="coerce") / CAS_BIN_KT).round() * CAS_BIN_KT
    cas_work = cas_work.dropna(subset=["cas_bin"])
    lib = enrich_events_with_phi(cas_work, phase, bin_col="cas_bin")
    if lib.empty:
        return lib
    vz_sub = vz_ev.loc[vz_ev["phase"].astype(str).str.upper() == phase].copy()
    vz_sub["vz_bin"] = (pd.to_numeric(vz_sub["value"], errors="coerce") / VZ_BIN_FPM).round() * VZ_BIN_FPM
    vz_sub = vz_sub.dropna(subset=["vz_bin"])
    rows = []
    for (route, fid), g in lib.groupby(["route", "flight_id"]):
        vz_g = vz_sub[(vz_sub["route"] == route) & (vz_sub["flight_id"] == fid)]
        if vz_g.empty:
            continue
        vz_asof = vz_g[["start_timestamp", "vz_bin"]].sort_values("start_timestamp")
        vz_asof["start_timestamp"] = pd.to_datetime(vz_asof["start_timestamp"], utc=True)
        for _, row in g.iterrows():
            ts = pd.Timestamp(row["start_timestamp"])
            m = pd.merge_asof(
                pd.DataFrame({"start_timestamp": [ts]}),
                vz_asof,
                on="start_timestamp",
                direction="nearest",
                tolerance=pd.Timedelta("60s"),
            )
            if m["vz_bin"].isna().iloc[0]:
                continue
            rows.append({**row.to_dict(), "vz_bin": float(m["vz_bin"].iloc[0])})
    out = pd.DataFrame(rows)
    return out.loc[out["dphi"] > 0] if not out.empty else out


@dataclass
class SampleContext:
    """Conditioning for one synthetic draw."""

    gc_nm: float
    typecode_family: str
    rng: np.random.Generator
    gc_nm_bin: int = 0
    route: str | None = None
    phi_d: float | None = None
    n_mach: int = 1
    typecode: str | None = None


@dataclass
class ConditioningSelection:
    """Resolved typecode / gc_nm filter for laws + trajectory pools."""

    typecode: str
    gc_nm: float
    routes: list[str]
    flights: pd.DataFrame
    family: str

    @property
    def n_flights(self) -> int:
        return len(self.flights)


@dataclass
class PhaseLaws:
    vz_markov: pd.DataFrame = field(default_factory=pd.DataFrame)
    cas_markov: pd.DataFrame = field(default_factory=pd.DataFrame)
    vz_phi: pd.DataFrame = field(default_factory=pd.DataFrame)
    cas_phi: pd.DataFrame = field(default_factory=pd.DataFrame)
    cas_markov_hi: pd.DataFrame = field(default_factory=pd.DataFrame)
    cas_markov_lo: pd.DataFrame = field(default_factory=pd.DataFrame)
    vz_h_joint: pd.DataFrame = field(default_factory=pd.DataFrame)
    h_phi: pd.DataFrame = field(default_factory=pd.DataFrame)
    cas_vz_joint: pd.DataFrame = field(default_factory=pd.DataFrame)
    vz_seg_counts: pd.Series = field(default_factory=pd.Series)
    cas_seg_counts: pd.Series = field(default_factory=pd.Series)
    cas_seg_counts_hi: pd.Series = field(default_factory=pd.Series)
    cas_seg_counts_lo: pd.Series = field(default_factory=pd.Series)
    vz_starts: pd.Series = field(default_factory=pd.Series)
    cas_starts: pd.Series = field(default_factory=pd.Series)
    cas_starts_hi: pd.Series = field(default_factory=pd.Series)
    cas_starts_lo: pd.Series = field(default_factory=pd.Series)


@dataclass
class EmpiricalLaws:
    """Pooled empirical laws keyed by (gc_nm_bin, typecode_family)."""

    phase_laws: dict[tuple[int, str, str], PhaseLaws] = field(default_factory=dict)
    mach_spatial: pd.DataFrame = field(default_factory=pd.DataFrame)
    mach_level_by_gc: dict[int, pd.DataFrame] = field(default_factory=dict)
    phi_d_by_gc_bin: dict[int, np.ndarray] = field(default_factory=dict)
    gc_nm_edges: np.ndarray = field(default_factory=lambda: np.array([0.0, 500.0, 1000.0]))
    routes: list[str] = field(default_factory=list)

    def get_phase(self, gc_nm_bin: int, family: str, phase: str) -> PhaseLaws:
        key = (int(gc_nm_bin), family, phase.upper())
        if key not in self.phase_laws:
            return PhaseLaws()
        return self.phase_laws[key]

    def _spatial_subset(
        self,
        ctx: SampleContext,
        filters: list[tuple[str, float, float]],
        *,
        min_n: int = 5,
    ) -> pd.DataFrame:
        df = self.mach_spatial
        if df.empty:
            return df
        sub = df.loc[df["gc_bin"] == ctx.gc_nm_bin]
        if len(sub) < min_n:
            sub = df
        for col, val, width in filters:
            if col not in sub.columns:
                continue
            if col == "n_mach":
                hit = sub.loc[sub["n_mach"].astype(int) == int(val)]
            elif np.isfinite(val):
                b = float(round(val / width) * width)
                hit = sub.loc[sub[col] == b]
            else:
                continue
            if len(hit) >= min_n:
                sub = hit
        return sub if len(sub) >= min_n else (df if len(df) >= min_n else df)

    def draw_n_mach(self, ctx: SampleContext, *, h_pre_max: float) -> int:
        hbin = float(round(h_pre_max / H_BIN_FT) * H_BIN_FT) if np.isfinite(h_pre_max) else np.nan
        sub = self._spatial_subset(ctx, [("h_pre_bin", hbin, H_BIN_FT)])
        if sub.empty or "n_mach" not in sub.columns:
            return 1
        return max(1, int(ctx.rng.choice(sub["n_mach"].astype(int).values)))

    def draw_phi_up(
        self, ctx: SampleContext, *, h_pre_max: float, cas_pre_last: float
    ) -> float:
        hbin = float(round(h_pre_max / H_BIN_FT) * H_BIN_FT) if np.isfinite(h_pre_max) else np.nan
        cbin = float(round(cas_pre_last / CAS_BIN_KT) * CAS_BIN_KT) if np.isfinite(cas_pre_last) else np.nan
        sub = self._spatial_subset(
            ctx,
            [("h_pre_bin", hbin, H_BIN_FT), ("cas_pre_bin", cbin, CAS_BIN_KT)],
        )
        pool = sub["phi_up"].dropna().to_numpy(dtype=float) if not sub.empty else np.array([])
        if len(pool) == 0:
            pool = self.mach_spatial["phi_up"].dropna().to_numpy(dtype=float)
        if len(pool) == 0:
            return 0.18
        return float(ctx.rng.choice(pool))

    def draw_phi_dn(
        self,
        ctx: SampleContext,
        *,
        phi_tod: float,
        mach_last: float,
        n_mach: int,
    ) -> float:
        ptod = float(round(phi_tod / PHI_BIN) * PHI_BIN)
        mbin = float(round(mach_last / MACH_BIN) * MACH_BIN) if np.isfinite(mach_last) else np.nan
        sub = self._spatial_subset(
            ctx,
            [
                ("phi_tod_bin", ptod, PHI_BIN),
                ("mach_bin", mbin, MACH_BIN),
                ("n_mach", float(n_mach), 1.0),
            ],
        )
        pool = sub["phi_dn"].dropna().to_numpy(dtype=float) if not sub.empty else np.array([])
        if len(pool) == 0:
            sub2 = self._spatial_subset(ctx, [("phi_tod_bin", ptod, PHI_BIN)])
            pool = sub2["phi_dn"].dropna().to_numpy(dtype=float) if not sub2.empty else np.array([])
        if len(pool) == 0:
            pool = self.mach_spatial["phi_dn"].dropna().to_numpy(dtype=float)
        if len(pool) == 0:
            return float(np.clip(phi_tod + 0.05, 0.0, 1.0))
        return float(ctx.rng.choice(pool))

    def draw_phi_d(self, ctx: SampleContext) -> float:
        if ctx.phi_d is not None:
            return float(ctx.phi_d)
        pool = self.phi_d_by_gc_bin.get(int(ctx.gc_nm_bin))
        if pool is None or len(pool) == 0:
            pools = [p for p in self.phi_d_by_gc_bin.values() if len(p)]
            if not pools:
                raise ValueError(
                    f"No phi_d samples for gc_nm_bin={ctx.gc_nm_bin} (gc_nm={ctx.gc_nm:.0f} nm). "
                    "Run enrich (--enrich-all-routes) so top_of_descent_events.parquet has phi_d."
                )
            pool = np.concatenate(pools)
        return float(ctx.rng.choice(pool))


def gc_nm_to_bin(gc_nm: float, edges: np.ndarray) -> int:
    i = int(np.digitize([gc_nm], edges)[0] - 1)
    return max(0, min(i, len(edges) - 2))


def make_sample_context(
    *,
    gc_nm: float,
    typecode: str | None = None,
    seed: int | None = None,
    laws: EmpiricalLaws | None = None,
    route: str | None = None) -> SampleContext:
    """One synthetic draw's conditioning + RNG."""
    laws = laws or EmpiricalLaws()
    rng = np.random.default_rng(seed)
    gcnm = float(gc_nm)
    tc = str(typecode).strip().upper() if typecode else None
    fam = typecode_to_family(tc)
    return SampleContext(
        gc_nm=gcnm,
        typecode_family=fam,
        route=route,
        rng=rng,
        gc_nm_bin=gc_nm_to_bin(gcnm, laws.gc_nm_edges),
        typecode=tc,
    )


def _load_events_table(routes: list[str]) -> pd.DataFrame:
    parts = []
    for route in routes:
        p = route_dataset_dir(route) / "commands" / "command_events.parquet"
        if not p.exists():
            continue
        ev = pd.read_parquet(p)
        ev["route"] = route
        acc = set(accepted_command_flight_ids(route))
        ev = ev[ev["flight_id"].astype(str).isin(acc)]
        parts.append(ev)
    if not parts:
        raise FileNotFoundError("No command_events.parquet in routes")
    return pd.concat(parts, ignore_index=True)


def _attach_phase_events(events: pd.DataFrame) -> pd.DataFrame:
    """Operational phase at segment start."""
    if events.empty:
        return events
    events = events.copy()
    events["start_timestamp"] = pd.to_datetime(events["start_timestamp"], utc=True)
    aligned = []
    for (route, fid), g in events.groupby(["route", "flight_id"]):
        cp = route_dataset_dir(route) / "commands" / f"{fid}.parquet"
        if not cp.exists():
            continue
        ph = pd.read_parquet(cp, columns=["timestamp", "phase"])
        ph["timestamp"] = pd.to_datetime(ph["timestamp"], utc=True)
        ph = ph.sort_values("timestamp")
        g = g.sort_values("start_timestamp")
        if "phase" in g.columns:
            g = g.drop(columns=["phase"])
        m = pd.merge_asof(
            g,
            ph,
            left_on="start_timestamp",
            right_on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("2s"),
        )
        if "phase" in m.columns:
            aligned.append(m)
    return pd.concat(aligned, ignore_index=True) if aligned else events.iloc[0:0]


def _phi_d_samples_from_tod(tod: pd.DataFrame) -> np.ndarray:
    """TOD table samples for generative phi_d."""
    if tod.empty:
        return np.array([], dtype=float)
    phi = (
        pd.to_numeric(tod["phi_d"], errors="coerce")
        if "phi_d" in tod.columns
        else pd.Series(np.nan, index=tod.index)
    )
    if "flight_progress" in tod.columns:
        fp = pd.to_numeric(tod["flight_progress"], errors="coerce")
        phi = phi.where(phi.notna(), fp)
    return phi.dropna().to_numpy(dtype=float)


def _events_before(events: pd.DataFrame, t_cut: pd.Timestamp, command: str) -> pd.DataFrame:
    sub = events[events["command"] == command].copy()
    sub["start_timestamp"] = pd.to_datetime(sub["start_timestamp"], utc=True)
    return sub[sub["start_timestamp"] < pd.Timestamp(t_cut)]


def _fit_mach_spatial_flights(events: pd.DataFrame, routes: list[str], gc_edges: np.ndarray) -> pd.DataFrame:
    """Per-flight φ_up, φ_dn, n_mach and pre-mach command stats."""
    mach = events[events["command"] == "mach_sel"].copy()
    mach["start_timestamp"] = pd.to_datetime(mach["start_timestamp"], utc=True)
    cas = events[events["command"] == "cas_sel"].copy()
    cas["start_timestamp"] = pd.to_datetime(cas["start_timestamp"], utc=True)

    tod_by_key: dict[tuple[str, str], float] = {}
    for route in routes:
        p = route_dataset_dir(route) / "metadata" / "top_of_descent_events.parquet"
        if not p.exists():
            continue
        tod = pd.read_parquet(p)
        tc = "timestamp" if "timestamp" in tod.columns else "start_timestamp"
        for _, tr in tod.iterrows():
            if "phi_d" in tr and pd.notna(tr["phi_d"]):
                tod_by_key[(route, str(tr["flight_id"]))] = float(tr["phi_d"])

    ades_cache = {r: route_arrival_coords(r) for r in routes}
    rows: list[dict[str, Any]] = []
    route_set = set(routes)
    for (route, fid), mg in mach.groupby(["route", "flight_id"]):
        if route not in route_set:
            continue
        mg = mg.sort_values("start_timestamp")
        ades = ades_cache[route]
        gcnm = route_gc_nm(route)
        gc_bin = max(0, min(int(np.digitize([gcnm], gc_edges)[0] - 1), len(gc_edges) - 2))
        ap = route_dataset_dir(route) / "data" / "adsb" / f"{fid}.parquet"
        if not ap.exists():
            continue
        adsb = pd.read_parquet(ap)
        adsb["timestamp"] = pd.to_datetime(adsb["timestamp"], utc=True)

        t_up = pd.Timestamp(mg.iloc[0]["start_timestamp"])
        t_last_m = pd.Timestamp(mg.iloc[-1]["start_timestamp"])
        ev = events[(events["route"] == route) & (events["flight_id"] == fid)]

        ev_up = merge_event_position({"timestamp": t_up}, adsb)
        phi_up = phi_d_at_event(ev_up, adsb, ades_lat=ades[0], ades_lon=ades[1])

        c_after = cas[
            (cas["route"] == route)
            & (cas["flight_id"] == fid)
            & (cas["start_timestamp"] > t_last_m)
        ].sort_values("start_timestamp")
        phi_dn = np.nan
        if not c_after.empty:
            t_dn = pd.Timestamp(c_after.iloc[0]["start_timestamp"])
            ev_dn = merge_event_position({"timestamp": t_dn}, adsb)
            phi_dn = phi_d_at_event(ev_dn, adsb, ades_lat=ades[0], ades_lon=ades[1])

        h_pre = _events_before(ev, t_up, "h_sel")
        c_pre = _events_before(ev, t_up, "cas_sel")
        h_vals = pd.to_numeric(h_pre["value"], errors="coerce")
        c_vals = pd.to_numeric(c_pre["value"], errors="coerce")
        h_pre_max = float(h_vals.max()) if h_vals.notna().any() else np.nan
        cas_pre_last = float(c_vals.iloc[-1]) if c_vals.notna().any() else np.nan

        rows.append(
            {
                "route": route,
                "flight_id": fid,
                "gc_nm": gcnm,
                "gc_bin": gc_bin,
                "n_mach": len(mg),
                "mach_last": float(pd.to_numeric(mg.iloc[-1]["value"], errors="coerce")),
                "phi_up": float(phi_up) if np.isfinite(phi_up) else np.nan,
                "phi_dn": float(phi_dn) if np.isfinite(phi_dn) else np.nan,
                "phi_tod": tod_by_key.get((route, str(fid)), np.nan),
                "h_pre_max": h_pre_max,
                "cas_pre_last": cas_pre_last,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["h_pre_bin"] = (df["h_pre_max"] / H_BIN_FT).round() * H_BIN_FT
    df["cas_pre_bin"] = (df["cas_pre_last"] / CAS_BIN_KT).round() * CAS_BIN_KT
    df["phi_tod_bin"] = (df["phi_tod"] / PHI_BIN).round() * PHI_BIN
    df["mach_bin"] = (df["mach_last"] / MACH_BIN).round() * MACH_BIN
    return df


def load_flight_metadata_table(routes: list[str]) -> pd.DataFrame:
    """Accepted flights with ``route``, ``flight_id``, ``typecode``, ``gc_nm``, ``family``."""
    rows: list[dict[str, Any]] = []
    for route in routes:
        gcnm = route_gc_nm(route)
        meta_p = route_dataset_dir(route) / "metadata" / "flight_metadata.parquet"
        tc_by_fid: dict[str, str] = {}
        if meta_p.exists():
            m = pd.read_parquet(meta_p, columns=["flight_id", "typecode"])
            tc_by_fid = {str(r["flight_id"]): str(r["typecode"]) for _, r in m.iterrows()}
        for fid in accepted_command_flight_ids(route):
            tc = tc_by_fid.get(fid, "")
            rows.append(
                {
                    "route": route,
                    "flight_id": fid,
                    "typecode": tc,
                    "gc_nm": gcnm,
                    "family": typecode_to_family(tc),
                }
            )
    return pd.DataFrame(rows)


def routes_for_gc_nm(routes: list[str], gc_nm: float, *, atol_nm: float = 150.0) -> list[str]:
    """Keep routes whose filed ``gc_nm`` is near ``gc_nm`` (nm)."""
    return [r for r in routes if abs(route_gc_nm(r) - float(gc_nm)) <= atol_nm]


def select_conditioning(
    routes: list[str],
    *,
    typecode: str,
    gc_nm: float,
    gc_nm_atol_nm: float = 150.0,
    min_ops_flights: int | None = None) -> ConditioningSelection:
    """Filter routes/flights to ``typecode`` and ``gc_nm`` band; warn if too few ops flights."""
    if gc_nm is None:
        raise ValueError("gc_nm is required (set GC_NM in notebook / CLI).")
    routes = routes_for_gc_nm(routes, float(gc_nm), atol_nm=gc_nm_atol_nm)
    if not routes:
        warnings.warn(
            f"No routes within {gc_nm_atol_nm} nm of gc_nm={gc_nm}. "
            "Widen atol or check ROUTES.",
            stacklevel=2,
        )
    meta = load_flight_metadata_table(routes)
    tc = str(typecode).strip().upper()
    flights = meta.loc[meta["typecode"].astype(str).str.upper() == tc].copy()
    if flights.empty:
        warnings.warn(
            f"No accepted flights with typecode={typecode!r} on routes {routes}.",
            stacklevel=2,
        )
    if min_ops_flights is not None and len(flights) < min_ops_flights:
        warnings.warn(
            f"Only {len(flights)} operational flights for typecode={typecode!r} "
            f"(gc_nm filter={gc_nm}); need at least {min_ops_flights}.",
            UserWarning,
            stacklevel=2,
        )
    return ConditioningSelection(
        typecode=tc,
        gc_nm=float(gc_nm),
        routes=list(routes),
        flights=flights,
        family=typecode_to_family(tc),
    )


def fit_empirical_laws(
    routes: list[str],
    *,
    events: pd.DataFrame | None = None,
    tod_paths: bool = True,
    conditioning: ConditioningSelection | None = None) -> EmpiricalLaws:
    """Fit pooled laws from route command events."""
    if events is None:
        events = _load_events_table(routes)
    if conditioning is not None and not conditioning.flights.empty:
        events = events.merge(
            conditioning.flights[["route", "flight_id"]],
            on=["route", "flight_id"],
            how="inner",
        )
    events = _attach_phase_events(events)
    gc_vals = np.array([route_gc_nm(r) for r in routes], dtype=float)
    gc_edges = np.quantile(gc_vals, [0.0, 1 / 3, 2 / 3, 1.0])
    gc_edges[0] = max(0.0, gc_edges[0] - 1.0)
    gc_edges[-1] = gc_edges[-1] + 1.0

    return build_empirical_laws_from_events(
        routes=routes,
        events=events,
        gc_edges=gc_edges,
        tod_paths=tod_paths,
    )


def build_empirical_laws_from_events(
    *,
    routes: list[str],
    events: pd.DataFrame,
    gc_edges: np.ndarray,
    tod_paths: bool) -> EmpiricalLaws:
    """Core fitter: build EmpiricalLaws from an already-loaded events table."""
    laws = EmpiricalLaws(routes=list(routes), gc_nm_edges=np.asarray(gc_edges, dtype=float))
    laws.mach_spatial = _fit_mach_spatial_flights(events, routes, laws.gc_nm_edges)

    # Mach on LEVEL
    mach = events[events["command"] == "mach_sel"].copy()
    mach["mach_bin"] = (pd.to_numeric(mach["value"], errors="coerce") / MACH_BIN).round() * MACH_BIN
    mach_parts: dict[int, list[pd.DataFrame]] = {}
    for route in routes:
        gcnm = route_gc_nm(route)
        bin_i = int(np.digitize([gcnm], laws.gc_nm_edges)[0] - 1)
        bin_i = max(0, min(bin_i, len(laws.gc_nm_edges) - 2))
        sub = mach[(mach["route"] == route) & (mach["phase"].astype(str).str.upper() == "LEVEL")]
        if not sub.empty:
            mach_parts.setdefault(bin_i, []).append(sub[["mach_bin", "duration_s"]])
    laws.mach_level_by_gc = {k: pd.concat(v, ignore_index=True) for k, v in mach_parts.items()}

    # phi_d by gc_nm bin
    if tod_paths:
        phi_parts: dict[int, list[np.ndarray]] = {}
        for route in routes:
            p = route_dataset_dir(route) / "metadata" / "top_of_descent_events.parquet"
            if not p.exists():
                continue
            tod = pd.read_parquet(p)
            samples = _phi_d_samples_from_tod(tod)
            if len(samples) == 0:
                continue
            gcnm = route_gc_nm(route)
            bin_i = gc_nm_to_bin(gcnm, laws.gc_nm_edges)
            phi_parts.setdefault(bin_i, []).append(samples)
        laws.phi_d_by_gc_bin = {
            k: np.concatenate(v) for k, v in phi_parts.items()
        }

    vz_ev = events[events["command"] == "vz_sel"]
    h_ev = events[events["command"] == "h_sel"]
    cas_ev = events[events["command"] == "cas_sel"]

    # Resolve typecode family per flight
    meta_rows = []
    for route in routes:
        meta_p = route_dataset_dir(route) / "metadata" / "flight_metadata.parquet"
        if meta_p.exists():
            m = pd.read_parquet(meta_p, columns=["flight_id", "typecode"])
            m["route"] = route
            m["gc_nm"] = route_gc_nm(route)
            meta_rows.append(m)
    meta = pd.concat(meta_rows, ignore_index=True) if meta_rows else pd.DataFrame()
    tc_map = meta.set_index(["route", "flight_id"])["typecode"].to_dict() if not meta.empty else {}

    for route in routes:
        gcnm = route_gc_nm(route)
        gc_bin = int(np.digitize([gcnm], laws.gc_nm_edges)[0] - 1)
        gc_bin = max(0, min(gc_bin, len(laws.gc_nm_edges) - 2))

        route_fids = (
            meta.loc[meta["route"] == route, "flight_id"].astype(str).tolist()
            if not meta.empty
            else accepted_command_flight_ids(route)
        )
        families = sorted({typecode_to_family(tc_map.get((route, fid))) for fid in route_fids} | {"Other"})
        for phase in ("CLIMB", "DESCENT"):
            for family in families:
                fids = [
                    fid for fid in route_fids if typecode_to_family(tc_map.get((route, fid))) == family
                ]
                if not fids:
                    continue
                pl = PhaseLaws()

                vz_p = vz_ev[(vz_ev["route"] == route) & (vz_ev["flight_id"].isin(fids))]
                h_p = h_ev[(h_ev["route"] == route) & (h_ev["flight_id"].isin(fids))]
                cas_p = cas_ev[(cas_ev["route"] == route) & (cas_ev["flight_id"].isin(fids))]

                vz_p = vz_p[vz_p["phase"].astype(str).str.upper() == phase]
                h_p = h_p[h_p["phase"].astype(str).str.upper() == phase]
                cas_p = cas_p[cas_p["phase"].astype(str).str.upper() == phase]

                # φ libraries
                if not vz_p.empty:
                    vz_p = vz_p.copy()
                    vz_p["vz_bin"] = (
                        pd.to_numeric(vz_p["value"], errors="coerce") / VZ_BIN_FPM
                    ).round() * VZ_BIN_FPM
                    vz_p = vz_p.dropna(subset=["vz_bin"])
                    pl.vz_markov = transition_matrix_bin(vz_p, "vz_bin")
                    if not vz_p.empty:
                        pl.vz_starts = (
                            vz_p.sort_values("start_timestamp")
                            .groupby("flight_id")
                            .first()["vz_bin"]
                        )
                        pl.vz_seg_counts = (
                            vz_p.groupby("flight_id").size().astype(float)
                        )
                    pl.vz_phi = enrich_events_with_phi(vz_p, phase, bin_col="vz_bin")

                if not h_p.empty:
                    pl.h_phi = build_h_phi_library(h_p, phase)

                if not cas_p.empty:
                    cas_p = cas_p.copy()
                    cas_p["cas_bin"] = (
                        pd.to_numeric(cas_p["value"], errors="coerce") / CAS_BIN_KT
                    ).round() * CAS_BIN_KT
                    cas_p = cas_p.dropna(subset=["cas_bin"])
                    pl.cas_markov = transition_matrix_bin(cas_p, "cas_bin")
                    if not cas_p.empty:
                        pl.cas_starts = (
                            cas_p.sort_values("start_timestamp")
                            .groupby("flight_id")
                            .first()["cas_bin"]
                        )
                        pl.cas_seg_counts = (
                            cas_p.groupby("flight_id").size().astype(float)
                        )
                    pl.cas_phi = enrich_events_with_phi(cas_p, phase, bin_col="cas_bin")

                # Joint libraries
                pl.vz_h_joint = build_vz_h_joint_library(vz_p, h_p, phase)
                pl.cas_vz_joint = build_cas_vz_joint_library(cas_p, vz_p, phase)

                laws.phase_laws[(gc_bin, family, phase)] = pl

    return laws

