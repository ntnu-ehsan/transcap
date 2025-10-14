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

    # Default data (if user doesn't supply)
    DEFAULT_OPTIONS = {
        (400, 220): {'X_percent': 14.0, 'unit_sizes': [400, 600, 800, 1000, 1200, 1500]},
        (220, 132): {'X_percent': 12.0, 'unit_sizes': [200, 300, 400, 500, 600, 800]}
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
