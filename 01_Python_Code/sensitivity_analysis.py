# -*- coding: utf-8 -*-
"""
Correlation and Sensitivity Analysis for Heat Sink Thermal Performance
Expert Thermal / XThermal Assignment

This script implements Task 1, Part D:
  Step 1: Correlation Analysis (Pearson & Spearman)
  Step 2: Parameter Sensitivity (Controlled One-At-A-Time around Reference & Full Spans)
  Step 3: ML Feature Importance (Permutation Importance for RF & GB on Test Set)
  Step 4: Comprehensive Visualizations Generation
  Step 5: Tabular Data Exports & Engineering Verification
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, root_mean_squared_error, mean_absolute_error

from heat_sink_model import calculate_heat_sink


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "02_Dataset", "thermal_dataset.csv")
MODELS_PATH = os.path.join(PROJECT_ROOT, "03_ML_Models")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "04_Results")
PLOTS_PATH = os.path.join(RESULTS_PATH, "plots")


# =============================================================================
# STEP 1: CORRELATION ANALYSIS
# =============================================================================

def perform_correlation_analysis(csv_path: str = DATASET_PATH, output_dir: str = RESULTS_PATH):
    """
    Computes Pearson and Spearman correlation matrices for TDP, V_air, k_tim, R_total, T_j.
    Saves matrices to CSV files.
    """
    print("=" * 80)
    print("STEP 1: CORRELATION ANALYSIS (Pearson & Spearman)")
    print("=" * 80)

    df = pd.read_csv(csv_path)
    analysis_cols = ["TDP", "V_air", "k_tim", "R_total", "T_j"]
    sub_df = df[analysis_cols].copy()

    # Calculate Pearson and Spearman correlation matrices
    pearson_corr = sub_df.corr(method="pearson")
    spearman_corr = sub_df.corr(method="spearman")

    os.makedirs(output_dir, exist_ok=True)
    pearson_file = os.path.join(output_dir, "correlation_pearson.csv")
    spearman_file = os.path.join(output_dir, "correlation_spearman.csv")

    pearson_corr.to_csv(pearson_file)
    spearman_corr.to_csv(spearman_file)

    print(f"Saved Pearson correlation to : {pearson_file}")
    print(f"Saved Spearman correlation to: {spearman_file}")

    print("\n--- Pearson Correlation Matrix (Linear Dependency) ---")
    print(pearson_corr.round(4).to_string())

    print("\n--- Spearman Rank Correlation Matrix (Monotonic Dependency) ---")
    print(spearman_corr.round(4).to_string())
    print("\nCorrelation Analysis Step 1 Completed.\n")

    return sub_df, pearson_corr, spearman_corr


# =============================================================================
# STEP 2: PARAMETER SENSITIVITY (ONE-AT-A-TIME & ELASTICITY)
# =============================================================================

def perform_sensitivity_analysis():
    """
    Performs controlled One-At-A-Time (OAT) sensitivity around reference:
      TDP = 150 W, V_air = 1.0 m/s, k_tim = 4.0 W/(m*K)
    Calculates:
      - Local finite-difference derivatives (dy/dx)
      - Normalized elasticity ( (x0 / y0) * (dy/dx) )
      - Full-span dynamic responses and percentage variations
    """
    print("=" * 80)
    print("STEP 2: PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)

    # Reference Operating Point
    ref_tdp = 150.0
    ref_v_air = 1.0
    ref_k_tim = 4.0

    ref_out = calculate_heat_sink(ref_tdp, ref_v_air, ref_k_tim)
    r0 = ref_out["R_total"]
    tj0 = ref_out["T_j"]

    print(f"Reference Operating Point: TDP = {ref_tdp} W, V_air = {ref_v_air} m/s, k_tim = {ref_k_tim} W/(m*K)")
    print(f"Reference Outputs        : R_total = {r0:.6f} °C/W, T_j = {tj0:.4f} °C\n")

    # 1. Local Finite-Difference Sensitivity around Reference
    # TDP perturbation: delta = 1.0 W
    p_plus = calculate_heat_sink(ref_tdp + 1.0, ref_v_air, ref_k_tim)
    p_minus = calculate_heat_sink(ref_tdp - 1.0, ref_v_air, ref_k_tim)
    dr_dtdp = (p_plus["R_total"] - p_minus["R_total"]) / 2.0
    dtj_dtdp = (p_plus["T_j"] - p_minus["T_j"]) / 2.0
    elast_r_tdp = (ref_tdp / r0) * dr_dtdp
    elast_tj_tdp = (ref_tdp / tj0) * dtj_dtdp

    # V_air perturbation: delta = 0.05 m/s
    v_plus = calculate_heat_sink(ref_tdp, ref_v_air + 0.05, ref_k_tim)
    v_minus = calculate_heat_sink(ref_tdp, ref_v_air - 0.05, ref_k_tim)
    dr_dv = (v_plus["R_total"] - v_minus["R_total"]) / 0.10
    dtj_dv = (v_plus["T_j"] - v_minus["T_j"]) / 0.10
    elast_r_v = (ref_v_air / r0) * dr_dv
    elast_tj_v = (ref_v_air / tj0) * dtj_dv

    # k_tim perturbation: delta = 0.10 W/(m*K)
    k_plus = calculate_heat_sink(ref_tdp, ref_v_air, ref_k_tim + 0.10)
    k_minus = calculate_heat_sink(ref_tdp, ref_v_air, ref_k_tim - 0.10)
    dr_dk = (k_plus["R_total"] - k_minus["R_total"]) / 0.20
    dtj_dk = (k_plus["T_j"] - k_minus["T_j"]) / 0.20
    elast_r_k = (ref_k_tim / r0) * dr_dk
    elast_tj_k = (ref_k_tim / tj0) * dtj_dk

    local_sens_df = pd.DataFrame([
        {
            "Parameter": "TDP",
            "Units": "W",
            "Nominal_Value": ref_tdp,
            "dR_total_dx": dr_dtdp,
            "dR_total_dx_units": "(°C/W)/W",
            "Elasticity_R_total": elast_r_tdp,
            "dT_j_dx": dtj_dtdp,
            "dT_j_dx_units": "°C/W",
            "Elasticity_T_j": elast_tj_tdp
        },
        {
            "Parameter": "V_air",
            "Units": "m/s",
            "Nominal_Value": ref_v_air,
            "dR_total_dx": dr_dv,
            "dR_total_dx_units": "(°C/W)/(m/s)",
            "Elasticity_R_total": elast_r_v,
            "dT_j_dx": dtj_dv,
            "dT_j_dx_units": "°C/(m/s)",
            "Elasticity_T_j": elast_tj_v
        },
        {
            "Parameter": "k_tim",
            "Units": "W/(m*K)",
            "Nominal_Value": ref_k_tim,
            "dR_total_dx": dr_dk,
            "dR_total_dx_units": "(°C/W)/(W/m.K)",
            "Elasticity_R_total": elast_r_k,
            "dT_j_dx": dtj_dk,
            "dT_j_dx_units": "°C/(W/m.K)",
            "Elasticity_T_j": elast_tj_k
        }
    ])

    print("--- Local Finite-Difference Sensitivity at Reference ---")
    print(local_sens_df.to_string(index=False))

    # 2. Full-Span One-At-A-Time (OAT) Sweeps
    # Sweep TDP: [30, 250] W (fixed V_air = 1.0, k_tim = 4.0)
    tdp_sweep = np.linspace(30.0, 250.0, 45)
    tdp_results = [calculate_heat_sink(t, ref_v_air, ref_k_tim) for t in tdp_sweep]
    df_tdp = pd.DataFrame(tdp_results)

    # Sweep V_air: [0.5, 15.0] m/s (fixed TDP = 150.0, k_tim = 4.0)
    v_sweep = np.linspace(0.5, 15.0, 50)
    v_results = [calculate_heat_sink(ref_tdp, v, ref_k_tim) for v in v_sweep]
    df_v = pd.DataFrame(v_results)

    # Sweep k_tim: [1.0, 12.0] W/(m*K) (fixed TDP = 150.0, V_air = 1.0)
    k_sweep = np.linspace(1.0, 12.0, 45)
    k_results = [calculate_heat_sink(ref_tdp, ref_v_air, k) for k in k_sweep]
    df_k = pd.DataFrame(k_results)

    # Summary of full-span impacts relative to reference
    span_summary = pd.DataFrame([
        {
            "Parameter": "TDP (30 to 250 W)",
            "Output": "T_j [°C]",
            "Min_Value": df_tdp["T_j"].min(),
            "Max_Value": df_tdp["T_j"].max(),
            "Total_Delta": df_tdp["T_j"].max() - df_tdp["T_j"].min(),
            "Percent_Change_vs_Ref": ((df_tdp["T_j"].max() - df_tdp["T_j"].min()) / tj0) * 100,
            "Linearity": "Strictly Linear (R² = 1.0000)"
        },
        {
            "Parameter": "V_air (0.5 to 15 m/s)",
            "Output": "R_total [°C/W]",
            "Min_Value": df_v["R_total"].min(),
            "Max_Value": df_v["R_total"].max(),
            "Total_Delta": df_v["R_total"].min() - df_v["R_total"].max(),
            "Percent_Change_vs_Ref": ((df_v["R_total"].max() - df_v["R_total"].min()) / r0) * 100,
            "Linearity": "Nonlinear (V^(-1/3) power law, diminishing returns)"
        },
        {
            "Parameter": "k_tim (1 to 12 W/m.K)",
            "Output": "R_total [°C/W]",
            "Min_Value": df_k["R_total"].min(),
            "Max_Value": df_k["R_total"].max(),
            "Total_Delta": df_k["R_total"].min() - df_k["R_total"].max(),
            "Percent_Change_vs_Ref": ((df_k["R_total"].max() - df_k["R_total"].min()) / r0) * 100,
            "Linearity": "Nonlinear (1/k hyperbolic decay, asymptotic saturation)"
        }
    ])

    print("\n--- Full-Span Dynamic Range & Impact Summary ---")
    print(span_summary.to_string(index=False))

    # Save sensitivity summary CSV
    os.makedirs(RESULTS_PATH, exist_ok=True)
    local_sens_df.to_csv(os.path.join(RESULTS_PATH, "sensitivity_local_derivatives.csv"), index=False)
    span_summary.to_csv(os.path.join(RESULTS_PATH, "sensitivity_full_span.csv"), index=False)
    print("\nSaved sensitivity CSV tables to 04_Results/")
    print("Sensitivity Step 2 Completed.\n")

    return local_sens_df, span_summary, df_tdp, df_v, df_k, ref_out


# =============================================================================
# STEP 3: ML FEATURE IMPORTANCE (PERMUTATION IMPORTANCE)
# =============================================================================

def perform_feature_importance():
    """
    Evaluates permutation importance for Random Forest and Gradient Boosting
    separately on held-out test data for both targets (R_total and T_j).
    """
    print("=" * 80)
    print("STEP 3: ML PERMUTATION FEATURE IMPORTANCE")
    print("=" * 80)

    # Load models
    rf_model = joblib.load(os.path.join(MODELS_PATH, "random_forest_model.joblib"))
    gb_model = joblib.load(os.path.join(MODELS_PATH, "gradient_boosting_model.joblib"))

    # Load dataset and prepare test split identically
    df = pd.read_csv(DATASET_PATH)
    feature_cols = ["TDP", "V_air", "k_tim"]
    target_cols = ["R_total", "T_j"]

    X = df[feature_cols]
    y = df[target_cols]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Define target-specific scoring functions (drop in R2 score upon feature shuffling)
    def score_r_total(estimator, X_val, y_val):
        preds = estimator.predict(X_val)
        return r2_score(y_val["R_total"], preds[:, 0])

    def score_tj(estimator, X_val, y_val):
        preds = estimator.predict(X_val)
        return r2_score(y_val["T_j"], preds[:, 1])

    # Permutation importance on held-out test data
    n_repeats = 10
    random_state = 42

    print(f"Calculating permutation importance on held-out test set ({len(X_test)} samples, {n_repeats} repeats)...")

    # Random Forest
    rf_perm_r = permutation_importance(rf_model, X_test, y_test, scoring=score_r_total, n_repeats=n_repeats, random_state=random_state)
    rf_perm_t = permutation_importance(rf_model, X_test, y_test, scoring=score_tj, n_repeats=n_repeats, random_state=random_state)

    # Gradient Boosting
    gb_perm_r = permutation_importance(gb_model, X_test, y_test, scoring=score_r_total, n_repeats=n_repeats, random_state=random_state)
    gb_perm_t = permutation_importance(gb_model, X_test, y_test, scoring=score_tj, n_repeats=n_repeats, random_state=random_state)

    records = []
    for model_name, perm_r, perm_t in [("Random Forest", rf_perm_r, rf_perm_t), ("Gradient Boosting", gb_perm_r, gb_perm_t)]:
        for i, feat in enumerate(feature_cols):
            records.append({
                "Model": model_name,
                "Feature": feat,
                "R_total_Drop_R2_Mean": perm_r.importances_mean[i],
                "R_total_Drop_R2_Std": perm_r.importances_std[i],
                "T_j_Drop_R2_Mean": perm_t.importances_mean[i],
                "T_j_Drop_R2_Std": perm_t.importances_std[i],
            })

    perm_df = pd.DataFrame(records)

    print("\n--- Permutation Feature Importance Summary (Drop in Test R² Score) ---")
    print(f"{'Model':<18} | {'Feature':<8} | {'R_total Drop R² (mean ± std)':<32} | {'T_j Drop R² (mean ± std)':<32}")
    print("-" * 98)
    for _, r in perm_df.iterrows():
        r_str = f"{r['R_total_Drop_R2_Mean']:+.6f} ± {r['R_total_Drop_R2_Std']:.6f}"
        t_str = f"{r['T_j_Drop_R2_Mean']:+.6f} ± {r['T_j_Drop_R2_Std']:.6f}"
        print(f"{r['Model']:<18} | {r['Feature']:<8} | {r_str:<32} | {t_str:<32}")
    print("-" * 98)

    perm_file = os.path.join(RESULTS_PATH, "permutation_importance.csv")
    perm_df.to_csv(perm_file, index=False)
    print(f"Saved permutation importance table to: {perm_file}")
    print("Feature Importance Step 3 Completed.\n")

    return perm_df, rf_perm_r, rf_perm_t, gb_perm_r, gb_perm_t


# =============================================================================
# STEP 4: VISUALIZATION GENERATION
# =============================================================================

def generate_all_plots(
    sub_df: pd.DataFrame,
    pearson_corr: pd.DataFrame,
    spearman_corr: pd.DataFrame,
    df_tdp: pd.DataFrame,
    df_v: pd.DataFrame,
    df_k: pd.DataFrame,
    local_sens_df: pd.DataFrame,
    perm_df: pd.DataFrame,
    plots_dir: str = PLOTS_PATH
):
    """
    Creates and saves all 8 required visualizations with high fidelity and proper units:
      1. Pearson correlation heatmap
      2. Spearman correlation heatmap (and combined correlation_heatmap.png)
      3. TDP vs Junction Temperature
      4. Air Velocity vs Thermal Resistance
      5. TIM Conductivity vs Thermal Resistance
      6. Parameter Sensitivity Comparison (Normalized Elasticities)
      7. Random Forest Feature Importance
      8. Gradient Boosting Feature Importance
    """
    os.makedirs(plots_dir, exist_ok=True)
    print("=" * 80)
    print("STEP 4: GENERATING PLOTS")
    print("=" * 80)

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10

    cols_order = ["TDP", "V_air", "k_tim", "R_total", "T_j"]

    # 1. Pearson Correlation Heatmap
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    cax = ax.matshow(pearson_corr.loc[cols_order, cols_order], cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Pearson Correlation ($r$)")
    ax.set_xticks(range(len(cols_order)))
    ax.set_yticks(range(len(cols_order)))
    ax.set_xticklabels(cols_order, fontweight="bold")
    ax.set_yticklabels(cols_order, fontweight="bold")
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    for i in range(len(cols_order)):
        for j in range(len(cols_order)):
            val = pearson_corr.loc[cols_order[i], cols_order[j]]
            text_color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=10)
    ax.set_title("Pearson Correlation Matrix (Linear Relationships)", fontweight="bold", pad=14, fontsize=12)
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "correlation_heatmap_pearson.png")
    fig.savefig(p1)
    # Also save as primary correlation_heatmap.png as per prompt
    fig.savefig(os.path.join(plots_dir, "correlation_heatmap.png"))
    plt.close(fig)
    print(f"Saved: {p1}")

    # 2. Spearman Correlation Heatmap
    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=300)
    cax = ax.matshow(spearman_corr.loc[cols_order, cols_order], cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04, label="Spearman Rank Correlation ($\\rho$)")
    ax.set_xticks(range(len(cols_order)))
    ax.set_yticks(range(len(cols_order)))
    ax.set_xticklabels(cols_order, fontweight="bold")
    ax.set_yticklabels(cols_order, fontweight="bold")
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)
    for i in range(len(cols_order)):
        for j in range(len(cols_order)):
            val = spearman_corr.loc[cols_order[i], cols_order[j]]
            text_color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=text_color, fontweight="bold", fontsize=10)
    ax.set_title("Spearman Rank Correlation Matrix (Monotonic Relationships)", fontweight="bold", pad=14, fontsize=12)
    plt.tight_layout()
    p2 = os.path.join(plots_dir, "correlation_heatmap_spearman.png")
    fig.savefig(p2)
    plt.close(fig)
    print(f"Saved: {p2}")

    # 3. TDP vs Junction Temperature
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(df_tdp["TDP"], df_tdp["T_j"], color="#d62728", lw=2.5, label="Physics Model Response")
    # Mark reference point
    ax.scatter([150.0], [80.3807], color="black", s=70, zorder=5, label="Reference Point (150 W, 80.38 °C)")
    ax.axhline(80.3807, color="gray", linestyle=":", alpha=0.6)
    ax.axvline(150.0, color="gray", linestyle=":", alpha=0.6)
    ax.set_xlabel("Processor TDP [W]", fontweight="bold")
    ax.set_ylabel("Processor Junction Temperature ($T_j$) [°C]", fontweight="bold")
    ax.set_title("TDP vs. Junction Temperature ($V_{air}=1.0$ m/s, $k_{tim}=4.0$ W/m·K)", fontweight="bold", pad=12)
    ax.annotate(r"Strictly Linear Response" "\n" r"Slope = $R_{total} = 0.3692$ °C/W" "\n" r"$\Delta T_j = 81.2$ °C across range",
                xy=(70, 50.8), xytext=(50, 85),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    ax.legend(loc="lower right")
    plt.tight_layout()
    p3 = os.path.join(plots_dir, "tdp_vs_junction_temperature.png")
    fig.savefig(p3)
    plt.close(fig)
    print(f"Saved: {p3}")

    # 4. Air Velocity vs Thermal Resistance
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(df_v["V_air"], df_v["R_total"], color="#1f77b4", lw=2.5, label="Total Resistance $R_{total}$")
    ax.plot(df_v["V_air"], df_v["R_conv"], color="#aec7e8", linestyle="--", lw=1.8, label="Convective Resistance $R_{conv}$")
    ax.scatter([1.0], [0.3692], color="black", s=70, zorder=5, label="Reference (1.0 m/s, 0.3692 °C/W)")
    ax.set_xlabel("Inlet Air Velocity ($V_{air}$) [m/s]", fontweight="bold")
    ax.set_ylabel("Thermal Resistance [°C/W]", fontweight="bold")
    ax.set_title("Air Velocity vs. Thermal Resistance (TDP=150 W, $k_{tim}=4.0$ W/m·K)", fontweight="bold", pad=12)
    ax.annotate(r"Steep Convective Benefit at Low Speeds" "\n" r"Diminishing Returns Beyond 5 m/s" "\n" r"$R_{conv} \propto V_{air}^{-1/3}$",
                xy=(2.0, 0.33), xytext=(4.5, 0.37),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    ax.legend(loc="upper right")
    plt.tight_layout()
    p4 = os.path.join(plots_dir, "v_air_vs_thermal_resistance.png")
    fig.savefig(p4)
    plt.close(fig)
    print(f"Saved: {p4}")

    # 5. TIM Conductivity vs Thermal Resistance
    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    ax.plot(df_k["k_tim"], df_k["R_total"], color="#2ca02c", lw=2.5, label="Total Resistance $R_{total}$")
    ax.plot(df_k["k_tim"], df_k["R_tim"], color="#98df8a", linestyle="--", lw=1.8, label="TIM Resistance $R_{tim}$")
    ax.scatter([4.0], [0.3692], color="black", s=70, zorder=5, label="Reference (4.0 W/m·K, 0.3692 °C/W)")
    ax.set_xlabel("TIM Thermal Conductivity ($k_{tim}$) [W/(m·K)]", fontweight="bold")
    ax.set_ylabel("Thermal Resistance [°C/W]", fontweight="bold")
    ax.set_title("TIM Conductivity vs. Thermal Resistance (TDP=150 W, $V_{air}=1.0$ m/s)", fontweight="bold", pad=12)
    ax.annotate("Hyperbolic $1/k_{tim}$ Decay\nRapid Saturation: Beyond 4 W/m·K,\nfurther TIM gains are minimal ($< 0.007$ °C/W)",
                xy=(2.0, 0.385), xytext=(4.0, 0.388),
                arrowprops=dict(arrowstyle="->", color="#333333", lw=1.2),
                bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    ax.legend(loc="upper right")
    plt.tight_layout()
    p5 = os.path.join(plots_dir, "k_tim_vs_thermal_resistance.png")
    fig.savefig(p5)
    plt.close(fig)
    print(f"Saved: {p5}")

    # 6. Parameter Sensitivity Comparison (Normalized Elasticity)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

    # Elasticity for R_total
    params = ["TDP", "V_air", "k_tim"]
    elast_r = [
        local_sens_df.loc[local_sens_df["Parameter"] == p, "Elasticity_R_total"].values[0]
        for p in params
    ]
    colors_r = ["#7f7f7f", "#1f77b4", "#2ca02c"]
    bars1 = ax1.bar(params, elast_r, color=colors_r, width=0.45, edgecolor="#333333")
    ax1.axhline(0, color="black", lw=1.0)
    ax1.set_ylabel("Normalized Sensitivity / Elasticity [$-$]", fontweight="bold")
    ax1.set_title("Sensitivity on Thermal Resistance ($R_{total}$)\n$\\varepsilon = (x_0 / R_0) \\cdot (\\partial R / \\partial x)$", fontweight="bold")
    ax1.set_ylim(-0.20, 0.05)
    for bar in bars1:
        h = bar.get_height()
        va = "bottom" if h < 0 else "top"
        ax1.text(bar.get_x() + bar.get_width() / 2, h - 0.015, f"{h:.4f}", ha="center", va=va, fontweight="bold", fontsize=9)

    # Elasticity for T_j
    elast_t = [
        local_sens_df.loc[local_sens_df["Parameter"] == p, "Elasticity_T_j"].values[0]
        for p in params
    ]
    colors_t = ["#d62728", "#1f77b4", "#2ca02c"]
    bars2 = ax2.bar(params, elast_t, color=colors_t, width=0.45, edgecolor="#333333")
    ax2.axhline(0, color="black", lw=1.0)
    ax2.set_ylabel("Normalized Sensitivity / Elasticity [$-$]", fontweight="bold")
    ax2.set_title("Sensitivity on Junction Temperature ($T_j$)\n$\\varepsilon = (x_0 / T_{j0}) \\cdot (\\partial T_j / \\partial x)$", fontweight="bold")
    ax2.set_ylim(-0.15, 0.80)
    for bar in bars2:
        h = bar.get_height()
        va = "bottom" if h > 0 else "top"
        offset = 0.02 if h > 0 else -0.02
        ax2.text(bar.get_x() + bar.get_width() / 2, h + offset, f"{h:+.4f}", ha="center", va=va, fontweight="bold", fontsize=9)

    plt.suptitle("Local Normalized Elasticity at Reference Operating Point (150 W, 1 m/s, 4 W/m·K)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    p6 = os.path.join(plots_dir, "parameter_sensitivity_comparison.png")
    fig.savefig(p6)
    plt.close(fig)
    print(f"Saved: {p6}")

    # 7. Random Forest Feature Importance (Permutation)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    rf_data = perm_df[perm_df["Model"] == "Random Forest"]

    # Target 1: R_total
    ax1.barh(rf_data["Feature"], rf_data["R_total_Drop_R2_Mean"], xerr=rf_data["R_total_Drop_R2_Std"],
             color="#1f77b4", alpha=0.85, edgecolor="#333333", height=0.45, capsize=4)
    ax1.set_xlabel("Drop in Test Set $R^2$ Score upon Permutation", fontweight="bold")
    ax1.set_title("RF Feature Importance: $R_{total}$", fontweight="bold")
    for i, (_, row) in enumerate(rf_data.iterrows()):
        val = row["R_total_Drop_R2_Mean"]
        ax1.text(max(val, 0) + 0.05, i, f"{val:.4f}", va="center", fontweight="bold", fontsize=9)

    # Target 2: T_j
    ax2.barh(rf_data["Feature"], rf_data["T_j_Drop_R2_Mean"], xerr=rf_data["T_j_Drop_R2_Std"],
             color="#d62728", alpha=0.85, edgecolor="#333333", height=0.45, capsize=4)
    ax2.set_xlabel("Drop in Test Set $R^2$ Score upon Permutation", fontweight="bold")
    ax2.set_title("RF Feature Importance: $T_j$", fontweight="bold")
    for i, (_, row) in enumerate(rf_data.iterrows()):
        val = row["T_j_Drop_R2_Mean"]
        ax2.text(max(val, 0) + 0.05, i, f"{val:.4f}", va="center", fontweight="bold", fontsize=9)

    plt.suptitle("Random Forest Permutation Feature Importance (Held-Out Test Set)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    p7 = os.path.join(plots_dir, "feature_importance_random_forest.png")
    fig.savefig(p7)
    plt.close(fig)
    print(f"Saved: {p7}")

    # 8. Gradient Boosting Feature Importance (Permutation)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    gb_data = perm_df[perm_df["Model"] == "Gradient Boosting"]

    # Target 1: R_total
    ax1.barh(gb_data["Feature"], gb_data["R_total_Drop_R2_Mean"], xerr=gb_data["R_total_Drop_R2_Std"],
             color="#2ca02c", alpha=0.85, edgecolor="#333333", height=0.45, capsize=4)
    ax1.set_xlabel("Drop in Test Set $R^2$ Score upon Permutation", fontweight="bold")
    ax1.set_title("Gradient Boosting Feature Importance: $R_{total}$", fontweight="bold")
    for i, (_, row) in enumerate(gb_data.iterrows()):
        val = row["R_total_Drop_R2_Mean"]
        ax1.text(max(val, 0) + 0.05, i, f"{val:.4f}", va="center", fontweight="bold", fontsize=9)

    # Target 2: T_j
    ax2.barh(gb_data["Feature"], gb_data["T_j_Drop_R2_Mean"], xerr=gb_data["T_j_Drop_R2_Std"],
             color="#ff7f0e", alpha=0.85, edgecolor="#333333", height=0.45, capsize=4)
    ax2.set_xlabel("Drop in Test Set $R^2$ Score upon Permutation", fontweight="bold")
    ax2.set_title("Gradient Boosting Feature Importance: $T_j$", fontweight="bold")
    for i, (_, row) in enumerate(gb_data.iterrows()):
        val = row["T_j_Drop_R2_Mean"]
        ax2.text(max(val, 0) + 0.05, i, f"{val:.4f}", va="center", fontweight="bold", fontsize=9)

    plt.suptitle("Gradient Boosting Permutation Feature Importance (Held-Out Test Set)", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    p8 = os.path.join(plots_dir, "feature_importance_gradient_boosting.png")
    fig.savefig(p8)
    plt.close(fig)
    print(f"Saved: {p8}")

    print("Step 4 Plot Generation Completed.\n")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("\n" + "#" * 80)
    print("TASK 1, PART D: CORRELATION AND SENSITIVITY ANALYSIS PIPELINE")
    print("#" * 80 + "\n")

    # Step 1: Correlation Analysis
    sub_df, pearson_corr, spearman_corr = perform_correlation_analysis()

    # Step 2: Sensitivity Analysis
    local_sens_df, span_summary, df_tdp, df_v, df_k, ref_out = perform_sensitivity_analysis()

    # Step 3: Feature Importance
    perm_df, rf_perm_r, rf_perm_t, gb_perm_r, gb_perm_t = perform_feature_importance()

    # Step 4: Visualizations
    generate_all_plots(sub_df, pearson_corr, spearman_corr, df_tdp, df_v, df_k, local_sens_df, perm_df)

    print("#" * 80)
    print("ALL CORRELATION AND SENSITIVITY CALCULATIONS COMPLETED SUCCESSFULLY!")
    print("#" * 80)


if __name__ == "__main__":
    main()
