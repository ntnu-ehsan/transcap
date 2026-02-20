
#%% 
# Importing transformer data
import pandas as pd
from typing import Dict, Iterable, Tuple, Optional

# Works with both legacy and PyPSA-style inputs.
INPUT_CSV = "data/pypsa_eur_transformers.csv"

# %%
import math
import pandas as pd
from typing import Dict, Iterable, Tuple, Optional
import re


# ---- Catalogs (MVA per transformer) ----
CATALOG: Dict[Tuple[int, int], Iterable[int]] = {
    (400, 220): [100, 160, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500],
    (220, 132): [100, 160, 200, 250, 315, 400, 500, 630],
    (225, 220): [100, 160, 200, 250, 315, 400, 500],
    (400, 320): [250, 315, 400, 500, 630, 800, 1000, 1250, 1600]
}


# ---- HV-side reactance lookup (per-unit on single-transformer base) ----
HV_XPU = {
    400: 0.14,
    225: 0.12,
    220: 0.12,
}

# Voltage bucketing for mixed-grid datasets (e.g., PyPSA-Europe).
# Values are snapped to these nominal levels before catalog lookup.
def snap_voltage_kv(v_kv: float) -> int:
    v = float(v_kv)
    if 360 <= v <= 420:
        return 400
    if 210 <= v <= 240:
        return 220
    if 290 <= v <= 340:
        return 320
    return int(round(v))

def _parse_quality_score(df: pd.DataFrame) -> float:
    """Higher score means columns look correctly aligned and numeric where expected."""
    score = 0.0
    for c in ("voltage_bus0", "voltage_bus1"):
        if c in df.columns:
            score += float(pd.to_numeric(df[c], errors="coerce").notna().mean())

    for c in ("s_max", "s_nom", "Max_Apparent_Power", "Max_Active_Power"):
        if c in df.columns:
            score += float(pd.to_numeric(df[c], errors="coerce").notna().mean())

    # Penalize obvious mis-parse pattern where geometry text spills into power columns.
    for c in ("s_max", "s_nom", "Max_Apparent_Power"):
        if c in df.columns:
            as_str = df[c].astype(str)
            if as_str.str.contains("LINESTRING", na=False).any():
                score -= 5.0
    return score

def read_transformer_csv(path: str) -> pd.DataFrame:
    """
    Robust CSV reader for both:
      - normal CSV (default parser),
      - single-quoted geometry fields that contain commas.
    """
    df_default = pd.read_csv(path)
    df_single_quote = pd.read_csv(path, quotechar=chr(39), engine="python")
    return df_single_quote if _parse_quality_score(df_single_quote) > _parse_quality_score(df_default) else df_default

def infer_smax_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a unified apparent-power column named `Smax_MVA_input` exists.
    Priority:
      1) s_max
      2) s_nom
      3) Max_Apparent_Power
      4) sqrt(Max_Active_Power^2 + Max_Reactive_Power^2)
    """
    out = df.copy()
    cols = set(out.columns)

    if "s_max" in cols:
        out["Smax_MVA_input"] = pd.to_numeric(out["s_max"], errors="coerce")
        return out

    if "s_nom" in cols:
        out["Smax_MVA_input"] = pd.to_numeric(out["s_nom"], errors="coerce")
        return out

    if "Max_Apparent_Power" in cols:
        out["Smax_MVA_input"] = pd.to_numeric(out["Max_Apparent_Power"], errors="coerce")
        return out

    if {"Max_Active_Power", "Max_Reactive_Power"}.issubset(cols):
        p = pd.to_numeric(out["Max_Active_Power"], errors="coerce")
        q = pd.to_numeric(out["Max_Reactive_Power"], errors="coerce")
        out["Smax_MVA_input"] = (p.pow(2) + q.pow(2)).pow(0.5)
        return out

    raise ValueError(
        "Input file must contain one of: `s_max`, `s_nom`, `Max_Apparent_Power`, "
        "or both `Max_Active_Power` and `Max_Reactive_Power`."
    )

def normalize_voltage_pair(v0_kv: float, v1_kv: float) -> Optional[Tuple[int, int]]:
    """Normalize to (HV, LV) integer kV and check against catalog keys."""
    hv, lv = sorted((snap_voltage_kv(v0_kv), snap_voltage_kv(v1_kv)), reverse=True)
    key = (int(hv), int(lv))
    return key if key in CATALOG else None

def parse_linestring_endpoints(geometry: object) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Parse a WKT LINESTRING and return (x0, y0, x1, y1) for first and last points.
    Returns Nones when geometry is missing/invalid.
    """
    if geometry is None or (isinstance(geometry, float) and pd.isna(geometry)):
        return None, None, None, None

    text = str(geometry).strip()
    if not text.upper().startswith("LINESTRING"):
        return None, None, None, None

    pairs = re.findall(r"(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s+(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", text)
    if len(pairs) < 2:
        return None, None, None, None

    x0, y0 = map(float, pairs[0])
    x1, y1 = map(float, pairs[-1])
    return x0, y0, x1, y1

def min_units_for_rating(Smax_MVA: float, R: float, k_emer: float) -> int:
    """Minimal n (>=2) satisfying n*R >= Smax and (n-1)*k_emer*R >= Smax."""
    if R <= 0 or Smax_MVA <= 0:
        return math.inf
    n_normal = math.ceil(Smax_MVA / R)
    n_n1 = math.ceil(1 + Smax_MVA / (k_emer * R))
    return max(2, n_normal, n_n1)

def required_rating_for_n(Smax_MVA: float, n: int, k_emer: float) -> float:
    """Per-unit rating R needed for a given n to satisfy both constraints."""
    if n < 2:
        return math.inf
    return max(Smax_MVA / n, Smax_MVA / ((n - 1) * k_emer))

def choose_by_rating_first(
    Smax_MVA: float,
    catalog_MVA: Iterable[int],
    k_emer: float = 1.25,
    max_n: int = 3,
) -> dict:
    """Select feasible option with minimal n first, then minimal installed MVA."""
    feasible_best = None
    cat_sorted = sorted(catalog_MVA)

    for R in cat_sorted:
        n = min_units_for_rating(Smax_MVA, R, k_emer)
        if n <= max_n:
            installed = n * R
            candidate = {
                "status": "OK",
                "n": n,
                "per_unit_MVA": R,
                "installed_MVA": installed,
                "required_per_unit_MVA": R,
                "normal_margin_MVA": installed - Smax_MVA,
                "n1_margin_MVA": (n - 1) * k_emer * R - Smax_MVA,
            }
            if (feasible_best is None or
                candidate["n"] < feasible_best["n"] or
                (candidate["n"] == feasible_best["n"] and candidate["installed_MVA"] < feasible_best["installed_MVA"]) or
                (candidate["n"] == feasible_best["n"] and candidate["installed_MVA"] == feasible_best["installed_MVA"] and candidate["per_unit_MVA"] < feasible_best["per_unit_MVA"])):
                feasible_best = candidate

    if feasible_best is not None:
        return feasible_best

    # infeasible diagnostic at n = max_n
    R_need_at_max_n = required_rating_for_n(Smax_MVA, max_n, k_emer)
    biggest_catalog = cat_sorted[-1] if cat_sorted else None
    short = (R_need_at_max_n - biggest_catalog) if biggest_catalog is not None else None
    return {
        "status": "INFEASIBLE_MAX_N",
        "n": None,
        "per_unit_MVA": None,
        "installed_MVA": None,
        "required_per_unit_MVA": R_need_at_max_n,
        "normal_margin_MVA": None,
        "n1_margin_MVA": None,
        "note": (
            f"Need per-unit >= {R_need_at_max_n:.3f} MVA for n={max_n} but catalog max is {biggest_catalog} "
            f"(short by {short:.3f} MVA)." if short is not None else
            f"Need per-unit >= {R_need_at_max_n:.3f} MVA for n={max_n}."
        ),
    }

def size_substations_for_Nminus1(
    substations: pd.DataFrame,
    k_emer: float = 1.25,
    use_column: str = "Smax_MVA_input",  # MVA
    max_n: int = 3,                           # ≤ 3 transformers
    system_base_MVA: float = 1000.0,           # load-flow system base
) -> pd.DataFrame:
    """
    Rating-first N-1 sizing (n <= max_n) + equivalent short-circuit reactance.
    Adds columns:
      - X_pu_on_installed_base
      - X_pu_on_system_base (system_base_MVA)
      - X_ohm_HV
    """
    out = []
    for _, r in substations.iterrows():
        v0 = float(r["voltage_bus0"])
        v1 = float(r["voltage_bus1"])
        hv_kv = int(round(max(v0, v1)))
        hv_kv_lookup = snap_voltage_kv(max(v0, v1))
        lv_kv = min(v0, v1)
        key = normalize_voltage_pair(v0, v1)
        Smax = float(r[use_column]) if pd.notna(r[use_column]) else None
        x0, y0, x1, y1 = parse_linestring_endpoints(r.get("geometry"))
        hv_on_bus0 = v0 >= v1
        hv_bus = r["bus0"] if hv_on_bus0 else r["bus1"]
        lv_bus = r["bus1"] if hv_on_bus0 else r["bus0"]
        hv_x = x0 if hv_on_bus0 else x1
        hv_y = y0 if hv_on_bus0 else y1
        lv_x = x1 if hv_on_bus0 else x0
        lv_y = y1 if hv_on_bus0 else y0

        base = {
            "transformer_id": r["transformer_id"],
            "bus0": r["bus0"],
            "bus1": r["bus1"],
            "voltage_bus0": v0,
            "voltage_bus1": v1,
            "voltage_pair_key": key,
            "HV_kV": hv_kv,
            "HV_kV_lookup": hv_kv_lookup,
            "Smax_MVA": Smax,
            "k_emer": k_emer,
            "max_units": max_n,
            "system_base_MVA": system_base_MVA,
            "geometry": r.get("geometry"),
            "bus0_x": x0,
            "bus0_y": y0,
            "bus1_x": x1,
            "bus1_y": y1,
            "hv_bus": hv_bus,
            "lv_bus": lv_bus,
            "hv_x": hv_x,
            "hv_y": hv_y,
            "lv_x": lv_x,
            "lv_y": lv_y,
        }

        if key is None or Smax is None or Smax <= 0:
            base.update({
                "catalog_used": list(CATALOG.get(key, [])) if key else None,
                "n_recommended": None,
                "per_unit_MVA": None,
                "installed_MVA": None,
                "installed_to_Smax_ratio": None,
                "required_per_unit_MVA": None,
                "normal_margin_MVA": None,
                "n1_margin_MVA": None,
                "status": "BAD_INPUT" if (Smax is None or Smax <= 0) else "UNSUPPORTED_VOLTAGE",
                "note": None,
                "X_pu_on_installed_base": None,
                "X_pu_on_system_base": None,
                "X_ohm_HV": None,
                "X_ohm_LV": None
            })
            out.append(base)
            continue

        choice = choose_by_rating_first(
            Smax_MVA=Smax,
            catalog_MVA=CATALOG[key],
            k_emer=k_emer,
            max_n=max_n,
        )

        if choice["status"] != "OK":
            base.update({
                "catalog_used": list(CATALOG[key]),
                "n_recommended": None,
                "per_unit_MVA": None,
                "installed_MVA": None,
                "installed_to_Smax_ratio": None,
                "required_per_unit_MVA": choice["required_per_unit_MVA"],
                "normal_margin_MVA": None,
                "n1_margin_MVA": None,
                "status": choice["status"],
                "note": choice.get("note"),
                "X_pu_on_installed_base": None,
                "X_pu_on_system_base": None,
                "X_ohm_HV": None,
                "X_ohm_LV": None
            })
            out.append(base)
            continue

        # Sizing details
        n = choice["n"]
        R = choice["per_unit_MVA"]
        installed = choice["installed_MVA"]
        ratio = installed / Smax if Smax else None

        # HV-side unit reactance (p.u. on single-transformer base)
        X_unit_pu = HV_XPU.get(hv_kv_lookup, None)

        if X_unit_pu is None:
            # Unknown HV voltage → no reactance estimate
            X_pu_installed = None
            X_pu_system = None
            X_ohm_HV = None
            X_ohm_LV = None
            note_extra = f"No Xpu rule for HV={hv_kv} kV (lookup class {hv_kv_lookup} kV)."
        else:
            # Parallel identical units:
            # - On installed base (n*R): X_pu is same as unit p.u.
            X_pu_installed = X_unit_pu
            # - On system base: scale by 100 MVA base (or user base)
            X_pu_system = X_unit_pu * (system_base_MVA / (n * R))
            # - In ohms on HV side: X = X_pu * (V_kV^2 / S_MVA) with n in parallel
            #   -> equivalent ohms = (X_unit_pu * V^2 / R) / n
            X_ohm_HV = X_unit_pu * (hv_kv ** 2) / (R * n)

            # Compute LV-side reactance (ohms)
            X_ohm_LV = X_ohm_HV * (lv_kv / hv_kv) ** 2
            note_extra = None

        base.update({
            "catalog_used": list(CATALOG[key]),
            "n_recommended": n,
            "per_unit_MVA": R,
            "installed_MVA": installed,
            "installed_to_Smax_ratio": ratio,
            "required_per_unit_MVA": choice["required_per_unit_MVA"],
            "normal_margin_MVA": choice["normal_margin_MVA"],
            "n1_margin_MVA": choice["n1_margin_MVA"],
            "status": "OK",
            "note": note_extra,
            # Reactance outputs
            "X_pu_on_installed_base": X_pu_installed,
            "X_pu_on_system_base": X_pu_system,
            "X_ohm_HV": X_ohm_HV,
            "X_ohm_LV": X_ohm_LV
        })
        out.append(base)

    return pd.DataFrame(out)




# %%
substations = read_transformer_csv(INPUT_CSV)
substations = infer_smax_column(substations)

# %%
result = size_substations_for_Nminus1(
    substations,
    k_emer=1.25,                 # your emergency loading factor
    use_column="Smax_MVA_input",  # MVA
    max_n=3                      # <= 3 transformers
)
# %%
result.to_csv('data/transformers_reactance.csv', index=False)

# %%

import pandas as pd

df = pd.read_csv('data/transformers_reactance.csv')
# assuming df is your DataFrame
df["X_ohm_LV"] = df.apply(
    lambda r: r["X_ohm_HV"] * (min(r["voltage_bus0"], r["voltage_bus1"]) / r["HV_kV"]) ** 2
    if pd.notna(r["X_ohm_HV"]) and pd.notna(r["voltage_bus0"]) and pd.notna(r["voltage_bus1"])
    else None,
    axis=1
)

try:
    df.to_csv("data/pypsa_transformers_reactance_HL.csv", index=False)
except PermissionError:
    # Common when file is open in Excel/IDE preview.
    df.to_csv("data/pypsa_transformers_reactance_HL_new.csv", index=False)

# %%
