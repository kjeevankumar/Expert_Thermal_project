# -*- coding: utf-8 -*-
"""
Parameter Sweep and Dataset Generation for Heat Sink Thermal Model
Property of Expert Thermal / XThermal

This script performs a full factorial grid sweep over:
  - TDP: 30 W to 250 W (23 values, 10 W increments)
  - V_air: 0.5 m/s to 15 m/s (30 values, 0.5 m/s increments)
  - k_tim: 1 to 12 W/m.K (12 integer values)
Total combinations: 23 * 30 * 12 = 8,280

Calculations are evaluated using `calculate_heat_sink` from `heat_sink_model.py`.
The results are stored in a Pandas DataFrame, validated, and saved to
`02_Dataset/thermal_dataset.csv`.
"""

import os
import itertools
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

from heat_sink_model import calculate_heat_sink


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def generate_parameter_grids() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates parameter arrays using NumPy as per specification:
      1. TDP: 30 W to 250 W (23 evenly spaced values, 10 W increments)
      2. Air velocity: 0.5 m/s to 15 m/s (30 evenly spaced values, 0.5 m/s increments)
      3. TIM conductivity: 1 to 12 W/m.K (12 evenly spaced integer values)
    """
    tdp_values = np.linspace(30.0, 250.0, 23)
    v_air_values = np.linspace(0.5, 15.0, 30)
    k_tim_values = np.linspace(1, 12, 12, dtype=int)

    return tdp_values, v_air_values, k_tim_values


def run_parameter_sweep() -> pd.DataFrame:
    """
    Executes the full factorial parameter sweep over all combinations
    using itertools.product and calculates heat sink thermal metrics.
    """
    tdp_vals, v_air_vals, k_tim_vals = generate_parameter_grids()

    expected_combinations = len(tdp_vals) * len(v_air_vals) * len(k_tim_vals)
    print(f"Generating parameter grid: {len(tdp_vals)} TDP x {len(v_air_vals)} V_air x {len(k_tim_vals)} k_tim = {expected_combinations:,} combinations")

    records: List[Dict[str, Any]] = []

    # Use itertools.product() to generate all combinations
    for tdp, v_air, k_tim in itertools.product(tdp_vals, v_air_vals, k_tim_vals):
        # Call parameterized physics model
        result = calculate_heat_sink(
            TDP=float(tdp),
            V_air=float(v_air),
            k_tim=float(k_tim)
        )
        records.append(result)

    df = pd.DataFrame(records)
    return df


def validate_dataset(df: pd.DataFrame) -> bool:
    """
    Comprehensive validation suite for the generated dataset.
    Checks:
      - Exact row count (8,280)
      - Parameter bounds
      - Missing or infinite values
      - Duplicate input combinations
      - Physical validity of thermal outputs
      - Separate and internal testing of reference scenario
    """
    print("\n" + "=" * 75)
    print("DATASET VALIDATION CHECKS")
    print("=" * 75)
    all_passed = True

    # 1. Row count check
    expected_rows = 8280
    if len(df) == expected_rows:
        print(f"[PASSED] Row count check: exactly {len(df):,} rows.")
    else:
        print(f"[FAILED] Row count check: expected {expected_rows}, got {len(df)}")
        all_passed = False

    # 2. Input parameter range checks
    tdp_min, tdp_max = df["TDP"].min(), df["TDP"].max()
    v_min, v_max = df["V_air"].min(), df["V_air"].max()
    ktim_min, ktim_max = df["k_tim"].min(), df["k_tim"].max()

    tdp_ok = (tdp_min == 30.0) and (tdp_max == 250.0) and (df["TDP"].nunique() == 23)
    v_ok = np.isclose(v_min, 0.5) and np.isclose(v_max, 15.0) and (df["V_air"].nunique() == 30)
    ktim_ok = (ktim_min == 1) and (ktim_max == 12) and (df["k_tim"].nunique() == 12)

    if tdp_ok:
        print(f"[PASSED] TDP range: [{tdp_min:.1f}, {tdp_max:.1f}] W with {df['TDP'].nunique()} distinct values (10 W steps).")
    else:
        print(f"[FAILED] TDP range invalid: min={tdp_min}, max={tdp_max}, distinct={df['TDP'].nunique()}")
        all_passed = False

    if v_ok:
        print(f"[PASSED] Air velocity range: [{v_min:.1f}, {v_max:.1f}] m/s with {df['V_air'].nunique()} distinct values (0.5 m/s steps).")
    else:
        print(f"[FAILED] Air velocity range invalid: min={v_min}, max={v_max}, distinct={df['V_air'].nunique()}")
        all_passed = False

    if ktim_ok:
        print(f"[PASSED] TIM conductivity range: [{ktim_min}, {ktim_max}] W/(m*K) with {df['k_tim'].nunique()} distinct integer values.")
    else:
        print(f"[FAILED] TIM conductivity range invalid: min={ktim_min}, max={ktim_max}, distinct={df['k_tim'].nunique()}")
        all_passed = False

    # 3. Missing and infinite value checks
    num_missing = df.isnull().sum().sum()
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    num_infinite = np.isinf(df[numeric_cols]).sum().sum()

    if num_missing == 0:
        print(f"[PASSED] Missing values check: 0 null/missing entries across all columns.")
    else:
        print(f"[FAILED] Found {num_missing} missing values in dataset.")
        all_passed = False

    if num_infinite == 0:
        print(f"[PASSED] Infinite values check: 0 infinite values across all numeric columns.")
    else:
        print(f"[FAILED] Found {num_infinite} infinite values in numeric columns.")
        all_passed = False

    # 4. Duplicate input combinations check
    duplicates = df.duplicated(subset=["TDP", "V_air", "k_tim"]).sum()
    if duplicates == 0:
        print(f"[PASSED] Uniqueness check: 0 duplicate (TDP, V_air, k_tim) combinations.")
    else:
        print(f"[FAILED] Found {duplicates} duplicate input combinations.")
        all_passed = False

    # 5. Physical bounds and sanity check
    r_total_finite_pos = (df["R_total"] > 0).all() and np.isfinite(df["R_total"]).all()
    t_j_above_ambient = (df["T_j"] > df["T_ambient"]).all() and np.isfinite(df["T_j"]).all()

    if r_total_finite_pos and t_j_above_ambient:
        print(f"[PASSED] Physical validity check: All R_total > 0 and all T_j > T_ambient (25 °C).")
        print(f"         R_total span : [{df['R_total'].min():.6f}, {df['R_total'].max():.6f}] °C/W")
        print(f"         T_j span     : [{df['T_j'].min():.2f}, {df['T_j'].max():.2f}] °C")
    else:
        print(f"[FAILED] Physical validity violation detected in thermal outputs.")
        all_passed = False

    # 6. Reference scenario verification (Separate call & Dataset lookup)
    # Reference condition: TDP = 150 W, V_air = 1.0 m/s, k_tim = 4 W/(m*K)
    ref_tdp = 150.0
    ref_v_air = 1.0
    ref_k_tim = 4.0

    # Expected values directly from the original script Heat_Sink_calculation.py
    orig_expected = {
        "Re": 73.41034222174244,
        "Nu": 2.048014084892021,
        "h": 23.278089503132986,
        "R_total": 0.3692049352073151,
        "T_j": 80.38074028109727
    }

    # Separate invocation test
    separate_res = calculate_heat_sink(TDP=ref_tdp, V_air=ref_v_air, k_tim=ref_k_tim)
    sep_match = all(
        np.isclose(separate_res[k], orig_expected[k], atol=1e-12, rtol=1e-12)
        for k in orig_expected
    )

    # In-grid lookup test
    grid_matches = df[
        (df["TDP"] == ref_tdp) &
        np.isclose(df["V_air"], ref_v_air) &
        (df["k_tim"] == ref_k_tim)
    ]

    grid_match_found = len(grid_matches) == 1
    if grid_match_found:
        grid_row = grid_matches.iloc[0]
        grid_vals_match = all(
            np.isclose(grid_row[k], orig_expected[k], atol=1e-12, rtol=1e-12)
            for k in orig_expected
        )
    else:
        grid_vals_match = False

    if sep_match:
        print(f"[PASSED] Separate reference scenario test: outputs match original Heat_Sink_calculation.py exactly.")
    else:
        print(f"[FAILED] Separate reference scenario test did not match original outputs.")
        all_passed = False

    if grid_match_found and grid_vals_match:
        print(f"[PASSED] In-grid reference scenario check: found in grid (row {grid_matches.index[0]}), exact match to original.")
        print(f"         Re = {grid_row['Re']:.6f}, Nu = {grid_row['Nu']:.6f}, h = {grid_row['h']:.6f} W/m^2.K")
        print(f"         R_total = {grid_row['R_total']:.6f} °C/W, T_j = {grid_row['T_j']:.4f} °C")
    else:
        print(f"[FAILED] In-grid reference scenario matching failed.")
        all_passed = False

    print("=" * 75)
    return all_passed


def main():
    # 1. Sweep parameter space and generate dataset
    df = run_parameter_sweep()

    # 2. Perform comprehensive data validation
    is_valid = validate_dataset(df)
    if not is_valid:
        raise ValueError("Dataset validation failed!")

    # 3. Create output directory and save CSV
    output_dir = os.path.join(PROJECT_ROOT, "02_Dataset")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "thermal_dataset.csv")

    df.to_csv(output_file, index=False)
    file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"\nSuccessfully saved dataset to: {output_file} ({file_size_mb:.2f} MB)")

    # 4. Display dataset shape and columns
    print("\n" + "=" * 75)
    print("DATASET OVERVIEW")
    print("=" * 75)
    print(f"Dataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"\nColumn Names ({len(df.columns)}):")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:2d}. {col:<20} ({df[col].dtype})")

    # 5. Display descriptive statistics for key features and targets
    key_columns = [
        "TDP", "V_air", "k_tim", "Re", "Nu", "h",
        "R_conv", "R_tim", "R_hs", "R_total", "T_j"
    ]
    print("\n" + "=" * 75)
    print("DESCRIPTIVE STATISTICS (Key Variables & Targets)")
    print("=" * 75)
    stats_df = df[key_columns].describe().T[["count", "mean", "std", "min", "25%", "50%", "75%", "max"]]
    print(stats_df.to_string())
    print("=" * 75)


if __name__ == "__main__":
    main()
