#%%
import math
import pandas as pd

def estimate_transformers_flexible(
    df,
    S_base_MVA=100.0,
    pf=0.95,
    use_n_minus_one=False,
    bank_phases=None,
    xr_default=10.0,
    transformer_options=None
):
    """
    Estimate per-unit transformer impedances dynamically for 400/220 and 220/132 autotransformers.
    - Automatically handles parallel units.
    - Optionally enforces N–1.
    - Optionally accounts for 3 single-phase banks.
    - Allows user-specified transformer options.

    Required columns:
        ['transformer_id','bus0','bus1','voltage_bus0','voltage_bus1',
         'Max_Active_Power','Max_Apparent_Power']
    """

    # Default data (Typical Standard Ratings (IEC range))
    DEFAULT_OPTIONS = {
        (400, 220): {'X_percent': 14.0, 'unit_sizes': [100, 160, 250, 315, 400, 500, 630, 800]},
        (220, 132): {'X_percent': 12.0, 'unit_sizes': [100, 160, 200, 250, 315]}
    }
    options = transformer_options or DEFAULT_OPTIONS

    def normalize_pair(v0, v1):
        """Snap voltage levels to nominal 400/220/132."""
        vhi, vlo = (v0, v1) if v0 >= v1 else (v1, v0)
        def bucket(v):
            if v >= 300: return 400
            if v >= 170: return 220
            if v >= 100: return 132
            return int(round(v))
        return bucket(vhi), bucket(vlo)

    results = []

    for _, r in df.iterrows():
        v0, v1 = float(r['voltage_bus0']), float(r['voltage_bus1'])
        vhi, vlo = normalize_pair(v0, v1)
        kv_pair = (vhi, vlo)

        # Transformer specs
        spec = options.get(kv_pair, options.get((vhi, vlo), None))
        if spec is None:
            # fallback generic
            spec = {'X_percent': 12.0, 'unit_sizes': [300, 500, 800, 1000]}
        X_percent = spec['X_percent']
        unit_sizes = spec['unit_sizes']

        # Determine apparent power
        if pd.notna(r['Max_Apparent_Power']) and r['Max_Apparent_Power'] > 0:
            S_req = float(r['Max_Apparent_Power'])
        elif pd.notna(r['Max_Active_Power']) and r['Max_Active_Power'] > 0:
            S_req = float(r['Max_Active_Power']) / pf
        else:
            S_req = 0.0

        # Account for banks (if given)
        bank_factor = bank_phases if bank_phases else 1.0

        # Try each available unit size and pick the minimal total capacity
        best = None
        for S_unit_raw in unit_sizes:
            S_unit = S_unit_raw * bank_factor  # total MVA per bank or 3-phase unit
            if S_unit <= 0:
                continue

            # choose number of units
            if use_n_minus_one:
                n_units = math.ceil(S_req / S_unit) + 1
            else:
                n_units = math.ceil(S_req / S_unit)
            n_units = max(1, n_units)

            total_capacity = n_units * S_unit
            if total_capacity >= S_req:
                best = (n_units, S_unit_raw)
                break
        if best is None:
            # if even largest unit too small, take the largest
            best = (math.ceil(S_req / (unit_sizes[-1]*bank_factor)), unit_sizes[-1])

        n_units, S_unit_raw = best
        S_unit_total = S_unit_raw * bank_factor

        # Per-unit impedance
        X_pu = (X_percent / 100.0) * (S_base_MVA / (n_units * S_unit_total))
        R_pu = X_pu / xr_default
        tap = (v0 / v1) if v1 > 0 else 1.0

        out = dict(r)
        out.update({
            'type': 'autotransformer',
            'kv_high': vhi,
            'kv_low': vlo,
            'X_percent_nameplate': X_percent,
            'n_units': int(n_units),
            'unit_MVA': float(S_unit_raw),
            'bank_phases': bank_phases if bank_phases else 3,
            'X_pu': float(X_pu),
            'R_pu': float(R_pu),
            'tap': float(tap),
            'S_req': float(S_req),
        })
        results.append(out)

    return pd.DataFrame(results)

#%% 
# Importing transformer data
if __name__ == "__main__":
    substations = pd.read_csv('data/transformers_with_max_flows.csv')
    transformers = estimate_transformers_flexible(
        substations,
        S_base_MVA=1000.0,
        pf=0.95,
        use_n_minus_one=True,
        bank_phases=3
    )

# %%
import pandas as pd
from typing import Dict, Iterable, Tuple, Optional

# ---- Configuration (your catalogs) ----
CATALOG: Dict[Tuple[int, int], Iterable[int]] = {
    (400, 220): [100, 160, 250, 315, 400, 500, 630, 800,1000],
    (220, 132): [100, 160, 200, 250, 315],
}

def normalize_voltage_pair(v0: float, v1: float) -> Optional[Tuple[int, int]]:
    """ 
    Return a normalized (HV, LV) integer kV tuple that matches the catalog keys,
    or None if unsupported. Works regardless of which side is bus0/bus1.
    """
    hv, lv = sorted((round(v0), round(v1)), reverse=True)
    key = (int(hv), int(lv))
    return key if key in CATALOG else None

def choose_transformers_for_Smax(
    Smax_MVA: float,
    catalog_MVA: Iterable[int],
    n_candidates: Iterable[int] = (2, 3),
    k_emer: float = 1.25,
) -> Optional[dict]:
    """
    Given a peak apparent power Smax (MVA), a catalog of per-unit ratings (MVA),
    and N-1 emergency factor k_emer, return the lowest-total-MVA feasible choice.

    Constraint for identical units:
        Normal:   n * R >= Smax
        N-1:     (n-1) * k_emer * R >= Smax
    => R_req(n) = max(Smax/n, Smax/((n-1)*k_emer)), n >= 2
    """
    best = None

    for n in n_candidates:
        if n < 2:
            continue  # N-1 requires at least 2 units

        # Required per-unit rating to satisfy both constraints
        R_req = max(Smax_MVA / n, Smax_MVA / ((n - 1) * k_emer))

        # Pick the smallest catalog rating meeting R_req
        feasible = [R for R in catalog_MVA if R >= R_req]
        if not feasible:
            continue
        R_pick = min(feasible)

        # Calculate simple margins
        normal_margin = n * R_pick - Smax_MVA
        n1_margin = (n - 1) * k_emer * R_pick - Smax_MVA

        candidate = {
            "n": n,
            "per_unit_MVA": R_pick,
            "installed_MVA": n * R_pick,
            "required_per_unit_MVA": R_req,
            "normal_margin_MVA": normal_margin,
            "n1_margin_MVA": n1_margin,
        }

        # Choose the candidate with the smallest installed MVA;
        # tie-breaker: fewer units, then smaller per-unit rating.
        if (best is None or
            candidate["installed_MVA"] < best["installed_MVA"] or
            (candidate["installed_MVA"] == best["installed_MVA"] and candidate["n"] < best["n"]) or
            (candidate["installed_MVA"] == best["installed_MVA"] and candidate["n"] == best["n"] and candidate["per_unit_MVA"] < best["per_unit_MVA"])
        ):
            best = candidate

    return best  # None means no catalog rating can satisfy N-1 for this Smax

def size_substations_for_Nminus1(
    substations: pd.DataFrame,
    n_candidates: Iterable[int] = (2, 3),
    k_emer: float = 1.25,
    use_column: str = "Max_Apparent_Power",
) -> pd.DataFrame:
    """
    Adds recommended N-1 configuration columns for each substation row.

    Parameters
    ----------
    substations : DataFrame with columns
        ['transformer_id','bus0','bus1','voltage_bus0','voltage_bus1',
         'Max_Active_Power','Max_Apparent_Power']
    n_candidates : iterable of ints, e.g., (2,3)
    k_emer : float, emergency short-time loading multiplier
    use_column : 'Max_Apparent_Power' (default) or 'Max_Active_Power' if you must

    Returns
    -------
    DataFrame with appended columns:
        - voltage_pair_key
        - catalog_used
        - n_recommended
        - per_unit_MVA
        - installed_MVA
        - required_per_unit_MVA
        - normal_margin_MVA
        - n1_margin_MVA
        - status  (OK / UNSUPPORTED_VOLTAGE / INFEASIBLE_NEED_HIGHER_RATING)
    """
    rows = []
    for _, r in substations.iterrows():
        key = normalize_voltage_pair(r["voltage_bus0"], r["voltage_bus1"])
        Smax = float(r[use_column])

        out = {
            "transformer_id": r["transformer_id"],
            "bus0": r["bus0"],
            "bus1": r["bus1"],
            "voltage_bus0": r["voltage_bus0"],
            "voltage_bus1": r["voltage_bus1"],
            "Smax_MVA": Smax,
        }

        if key is None:
            out.update({
                "voltage_pair_key": None,
                "catalog_used": None,
                "n_recommended": None,
                "per_unit_MVA": None,
                "installed_MVA": None,
                "required_per_unit_MVA": None,
                "normal_margin_MVA": None,
                "n1_margin_MVA": None,
                "status": "UNSUPPORTED_VOLTAGE",
            })
        else:
            choice = choose_transformers_for_Smax(
                Smax_MVA=Smax,
                catalog_MVA=CATALOG[key],
                n_candidates=n_candidates,
                k_emer=k_emer,
            )
            if choice is None:
                out.update({
                    "voltage_pair_key": key,
                    "catalog_used": list(CATALOG[key]),
                    "n_recommended": None,
                    "per_unit_MVA": None,
                    "installed_MVA": None,
                    "required_per_unit_MVA": max(
                        Smax / min(n_candidates),
                        Smax / ((min(n_candidates) - 1) * k_emer)
                    ) if min(n_candidates) >= 2 else None,
                    "normal_margin_MVA": None,
                    "n1_margin_MVA": None,
                    "status": "INFEASIBLE_NEED_HIGHER_RATING",
                })
            else:
                out.update({
                    "voltage_pair_key": key,
                    "catalog_used": list(CATALOG[key]),
                    "n_recommended": choice["n"],
                    "per_unit_MVA": choice["per_unit_MVA"],
                    "installed_MVA": choice["installed_MVA"],
                    "required_per_unit_MVA": choice["required_per_unit_MVA"],
                    "normal_margin_MVA": choice["normal_margin_MVA"],
                    "n1_margin_MVA": choice["n1_margin_MVA"],
                    "status": "OK",
                })
        rows.append(out)

    return pd.DataFrame(rows)


# %%
result = size_substations_for_Nminus1(substations, n_candidates=(2,3,4), k_emer=1.25)
print(result)
# %%
