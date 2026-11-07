# -*- coding: utf-8 -*-
"""
ML Surrogate Model Development and Evaluation for Heat Sink Thermal Performance
Expert Thermal / XThermal Assignment

This script implements Task 1, Part C:
  Step 1: Data Preparation & Validation
  Step 2: Train-Test Split (80/20, random_state=42)
  Step 3: Train Random Forest & Gradient Boosting Regressors
  Step 4: Model Evaluation (MAE, RMSE, R² per target)
  Step 5: Visualization Generation (Parity plots & Metric comparisons)
  Step 6: Model & Metadata Persistence (Joblib & JSON)
  Step 7: Additional Validations (Physical constraints, Off-grid Interpolation,
          T_j = 25 + TDP * R_total Consistency, Benchmark Latency)
"""

import os
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from heat_sink_model import calculate_heat_sink


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(PROJECT_ROOT, "02_Dataset", "thermal_dataset.csv")
MODELS_PATH = os.path.join(PROJECT_ROOT, "03_ML_Models")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "04_Results")
PLOTS_PATH = os.path.join(RESULTS_PATH, "plots")


# =============================================================================
# STEP 1: DATA PREPARATION & VALIDATION
# =============================================================================

def load_and_validate_data(csv_path: str = DATASET_PATH):
    """
    Loads dataset, enforces feature and target isolation, and validates data integrity.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}. Please run generate_dataset.py first.")

    df = pd.read_csv(csv_path)

    # Required inputs (ML features only - excluding all physics intermediates)
    feature_cols = ["TDP", "V_air", "k_tim"]
    target_cols = ["R_total", "T_j"]

    print("=" * 80)
    print("STEP 1: DATA PREPARATION & VALIDATION")
    print("=" * 80)
    print(f"Loaded dataset: {csv_path}")
    print(f"Dataset Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Confirm expected row count
    assert df.shape[0] == 8280, f"Expected 8,280 rows, got {df.shape[0]}"

    # Validate feature and target columns exist
    for col in feature_cols + target_cols:
        assert col in df.columns, f"Missing required column: {col}"

    # Verify no missing or infinite values
    assert df[feature_cols + target_cols].isnull().sum().sum() == 0, "Missing values detected!"
    assert np.all(np.isfinite(df[feature_cols + target_cols])), "Infinite values detected!"

    # Verify no duplicate input combinations
    dup_count = df.duplicated(subset=feature_cols).sum()
    assert dup_count == 0, f"Found {dup_count} duplicate feature rows!"

    # Extract X and y
    X = df[feature_cols].copy()
    y = df[target_cols].copy()

    print(f"Features strictly isolated to 3 engineering inputs: {feature_cols}")
    print(f"Targets isolated to 2 regression outputs: {target_cols}")
    print("Zero missing, infinite, or duplicate entries detected.")
    print("Validation Step 1 PASSED.\n")

    return X, y


# =============================================================================
# STEP 2: TRAIN-TEST SPLIT
# =============================================================================

def split_dataset(X: pd.DataFrame, y: pd.DataFrame, test_size: float = 0.20, random_state: int = 42):
    """
    Splits features and targets into train and test sets using train_test_split.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("=" * 80)
    print("STEP 2: TRAIN-TEST SPLIT")
    print("=" * 80)
    print(f"Test size: {test_size * 100:.0f}% (random_state={random_state})")
    print(f"Training set: {X_train.shape[0]:,} rows (80.0%)")
    print(f"Testing set : {X_test.shape[0]:,} rows (20.0%)")
    print(f"Feature matrix shape: Train {X_train.shape}, Test {X_test.shape}")
    print(f"Target matrix shape : Train {y_train.shape}, Test {y_test.shape}")
    print("Train-Test Split Step 2 PASSED.\n")

    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 3: TRAIN TWO ML MODELS
# =============================================================================

def train_surrogate_models(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """
    Trains Model A (RandomForestRegressor) and Model B (MultiOutputRegressor(GradientBoostingRegressor)).
    """
    print("=" * 80)
    print("STEP 3: TRAIN SURROGATE ML MODELS")
    print("=" * 80)

    # Model A: Random Forest Regressor
    print("Training Model A: RandomForestRegressor (n_estimators=200, random_state=42, n_jobs=-1)...")
    t0 = time.time()
    rf_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    t_rf_train = time.time() - t0
    print(f"Model A (Random Forest) trained in {t_rf_train:.3f} s.")

    # Model B: Gradient Boosting Regressor wrapped in MultiOutputRegressor
    print("Training Model B: MultiOutputRegressor(GradientBoostingRegressor(n_estimators=200, lr=0.05, max_depth=3, random_state=42))...")
    t0 = time.time()
    gb_base = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    gb_model = MultiOutputRegressor(gb_base)
    gb_model.fit(X_train, y_train)
    t_gb_train = time.time() - t0
    print(f"Model B (Gradient Boosting) trained in {t_gb_train:.3f} s.")
    print("Model Training Step 3 PASSED.\n")

    return {"Random Forest": rf_model, "Gradient Boosting": gb_model}


# =============================================================================
# STEP 4: MODEL EVALUATION
# =============================================================================

def evaluate_models(models: dict, X_test: pd.DataFrame, y_test: pd.DataFrame, output_dir: str = RESULTS_PATH):
    """
    Evaluates both models on held-out test set and saves results to
    04_Results/model_comparison.csv.
    """
    print("=" * 80)
    print("STEP 4: MODEL EVALUATION ON HELD-OUT TEST SET (1,656 samples)")
    print("=" * 80)

    target_info = [
        {"name": "R_total", "label": "Total Thermal Resistance", "unit": "°C/W"},
        {"name": "T_j", "label": "Junction Temperature", "unit": "°C"}
    ]

    records = []
    predictions = {}

    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        predictions[model_name] = pd.DataFrame(y_pred, columns=y_test.columns, index=y_test.index)

        for i, tinfo in enumerate(target_info):
            t_name = tinfo["name"]
            unit = tinfo["unit"]
            y_true_col = y_test[t_name].to_numpy()
            y_pred_col = y_pred[:, i]

            mae = mean_absolute_error(y_true_col, y_pred_col)
            rmse = root_mean_squared_error(y_true_col, y_pred_col)
            r2 = r2_score(y_true_col, y_pred_col)

            records.append({
                "Model": model_name,
                "Target": tinfo["label"],
                "Target_Variable": t_name,
                "Unit": unit,
                "MAE": mae,
                "RMSE": rmse,
                "R2_Score": r2
            })

    results_df = pd.DataFrame(records)

    # Format table for terminal display
    print(f"{'Model':<20} | {'Target':<28} | {'Unit':<6} | {'MAE':<12} | {'RMSE':<12} | {'R² Score':<10}")
    print("-" * 100)
    for _, row in results_df.iterrows():
        print(f"{row['Model']:<20} | {row['Target']:<28} | {row['Unit']:<6} | {row['MAE']:<12.6f} | {row['RMSE']:<12.6f} | {row['R2_Score']:<10.6f}")
    print("-" * 100)

    # Save to 04_Results/model_comparison.csv
    os.makedirs(output_dir, exist_ok=True)
    csv_file = os.path.join(output_dir, "model_comparison.csv")
    results_df.to_csv(csv_file, index=False)
    print(f"Saved evaluation metrics to: {csv_file}")
    print("Model Evaluation Step 4 PASSED.\n")

    return results_df, predictions


# =============================================================================
# STEP 5: VISUALIZATIONS
# =============================================================================

def generate_visualizations(y_test: pd.DataFrame, predictions: dict, results_df: pd.DataFrame, plots_dir: str = PLOTS_PATH):
    """
    Creates and saves parity plots and model comparison bar charts.
    """
    os.makedirs(plots_dir, exist_ok=True)
    print("=" * 80)
    print("STEP 5: VISUALIZATION GENERATION")
    print("=" * 80)

    # Set matplotlib style
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 10

    # 1. Actual vs Predicted: R_total - Random Forest
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    y_true = y_test["R_total"]
    y_pred = predictions["Random Forest"]["R_total"]
    r2_val = r2_score(y_true, y_pred)
    mae_val = mean_absolute_error(y_true, y_pred)
    rmse_val = root_mean_squared_error(y_true, y_pred)

    ax.scatter(y_true, y_pred, alpha=0.35, color="#1f77b4", edgecolors="none", s=20, label="Test Predictions")
    min_v, max_v = min(y_true.min(), y_pred.min()) * 0.98, max(y_true.max(), y_pred.max()) * 1.02
    ax.plot([min_v, max_v], [min_v, max_v], "r--", lw=1.8, label="Ideal Parity (y = x)")
    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_xlabel("Physics Model $R_{total}$ [°C/W]", fontweight="bold")
    ax.set_ylabel("Random Forest Predicted $R_{total}$ [°C/W]", fontweight="bold")
    ax.set_title("Random Forest: Actual vs. Predicted Thermal Resistance ($R_{total}$)", fontweight="bold", pad=12)
    stats_text = f"$R^2$: {r2_val:.6f}\nMAE: {mae_val:.6f} °C/W\nRMSE: {rmse_val:.6f} °C/W"
    ax.text(0.05, 0.92, stats_text, transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    ax.legend(loc="lower right")
    plt.tight_layout()
    p1 = os.path.join(plots_dir, "actual_vs_predicted_r_total_rf.png")
    fig.savefig(p1)
    plt.close(fig)
    print(f"Saved: {p1}")

    # 2. Actual vs Predicted: R_total - Gradient Boosting
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    y_true = y_test["R_total"]
    y_pred = predictions["Gradient Boosting"]["R_total"]
    r2_val = r2_score(y_true, y_pred)
    mae_val = mean_absolute_error(y_true, y_pred)
    rmse_val = root_mean_squared_error(y_true, y_pred)

    ax.scatter(y_true, y_pred, alpha=0.35, color="#2ca02c", edgecolors="none", s=20, label="Test Predictions")
    ax.plot([min_v, max_v], [min_v, max_v], "r--", lw=1.8, label="Ideal Parity (y = x)")
    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_xlabel("Physics Model $R_{total}$ [°C/W]", fontweight="bold")
    ax.set_ylabel("Gradient Boosting Predicted $R_{total}$ [°C/W]", fontweight="bold")
    ax.set_title("Gradient Boosting: Actual vs. Predicted Thermal Resistance ($R_{total}$)", fontweight="bold", pad=12)
    stats_text = f"$R^2$: {r2_val:.6f}\nMAE: {mae_val:.6f} °C/W\nRMSE: {rmse_val:.6f} °C/W"
    ax.text(0.05, 0.92, stats_text, transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    ax.legend(loc="lower right")
    plt.tight_layout()
    p2 = os.path.join(plots_dir, "actual_vs_predicted_r_total_gb.png")
    fig.savefig(p2)
    plt.close(fig)
    print(f"Saved: {p2}")

    # 3. Actual vs Predicted: T_j - Random Forest
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    y_true = y_test["T_j"]
    y_pred = predictions["Random Forest"]["T_j"]
    r2_val = r2_score(y_true, y_pred)
    mae_val = mean_absolute_error(y_true, y_pred)
    rmse_val = root_mean_squared_error(y_true, y_pred)

    ax.scatter(y_true, y_pred, alpha=0.35, color="#1f77b4", edgecolors="none", s=20, label="Test Predictions")
    min_v, max_v = min(y_true.min(), y_pred.min()) - 2, max(y_true.max(), y_pred.max()) + 2
    ax.plot([min_v, max_v], [min_v, max_v], "r--", lw=1.8, label="Ideal Parity (y = x)")
    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_xlabel("Physics Model $T_j$ [°C]", fontweight="bold")
    ax.set_ylabel("Random Forest Predicted $T_j$ [°C]", fontweight="bold")
    ax.set_title("Random Forest: Actual vs. Predicted Junction Temperature ($T_j$)", fontweight="bold", pad=12)
    stats_text = f"$R^2$: {r2_val:.6f}\nMAE: {mae_val:.4f} °C\nRMSE: {rmse_val:.4f} °C"
    ax.text(0.05, 0.92, stats_text, transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    ax.legend(loc="lower right")
    plt.tight_layout()
    p3 = os.path.join(plots_dir, "actual_vs_predicted_tj_rf.png")
    fig.savefig(p3)
    plt.close(fig)
    print(f"Saved: {p3}")

    # 4. Actual vs Predicted: T_j - Gradient Boosting
    fig, ax = plt.subplots(figsize=(7, 6), dpi=300)
    y_true = y_test["T_j"]
    y_pred = predictions["Gradient Boosting"]["T_j"]
    r2_val = r2_score(y_true, y_pred)
    mae_val = mean_absolute_error(y_true, y_pred)
    rmse_val = root_mean_squared_error(y_true, y_pred)

    ax.scatter(y_true, y_pred, alpha=0.35, color="#2ca02c", edgecolors="none", s=20, label="Test Predictions")
    ax.plot([min_v, max_v], [min_v, max_v], "r--", lw=1.8, label="Ideal Parity (y = x)")
    ax.set_xlim(min_v, max_v)
    ax.set_ylim(min_v, max_v)
    ax.set_xlabel("Physics Model $T_j$ [°C]", fontweight="bold")
    ax.set_ylabel("Gradient Boosting Predicted $T_j$ [°C]", fontweight="bold")
    ax.set_title("Gradient Boosting: Actual vs. Predicted Junction Temperature ($T_j$)", fontweight="bold", pad=12)
    stats_text = f"$R^2$: {r2_val:.6f}\nMAE: {mae_val:.4f} °C\nRMSE: {rmse_val:.4f} °C"
    ax.text(0.05, 0.92, stats_text, transform=ax.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc", alpha=0.9))
    ax.legend(loc="lower right")
    plt.tight_layout()
    p4 = os.path.join(plots_dir, "actual_vs_predicted_tj_gb.png")
    fig.savefig(p4)
    plt.close(fig)
    print(f"Saved: {p4}")

    # 5. Model Comparison Bar Charts (MAE, RMSE, R²)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=300)
    models = ["Random Forest", "Gradient Boosting"]
    colors = ["#1f77b4", "#2ca02c"]

    # Target 1: R_total metrics
    r_total_df = results_df[results_df["Target_Variable"] == "R_total"].set_index("Model")
    axes[0, 0].bar(models, [r_total_df.loc[m, "MAE"] for m in models], color=colors, width=0.45)
    axes[0, 0].set_title("$R_{total}$ Mean Absolute Error (MAE)", fontweight="bold")
    axes[0, 0].set_ylabel("MAE [°C/W]")
    for i, m in enumerate(models):
        val = r_total_df.loc[m, "MAE"]
        axes[0, 0].text(i, val * 1.05, f"{val:.5f}", ha="center", fontweight="bold", fontsize=9)

    axes[0, 1].bar(models, [r_total_df.loc[m, "RMSE"] for m in models], color=colors, width=0.45)
    axes[0, 1].set_title("$R_{total}$ Root Mean Squared Error (RMSE)", fontweight="bold")
    axes[0, 1].set_ylabel("RMSE [°C/W]")
    for i, m in enumerate(models):
        val = r_total_df.loc[m, "RMSE"]
        axes[0, 1].text(i, val * 1.05, f"{val:.5f}", ha="center", fontweight="bold", fontsize=9)

    axes[0, 2].bar(models, [r_total_df.loc[m, "R2_Score"] for m in models], color=colors, width=0.45)
    axes[0, 2].set_title("$R_{total}$ Coefficient of Determination ($R^2$)", fontweight="bold")
    axes[0, 2].set_ylabel("$R^2$ Score [-]")
    axes[0, 2].set_ylim(0.99, 1.0005)
    for i, m in enumerate(models):
        val = r_total_df.loc[m, "R2_Score"]
        axes[0, 2].text(i, val - 0.0008, f"{val:.6f}", ha="center", fontweight="bold", fontsize=9, color="white")

    # Target 2: T_j metrics
    tj_df = results_df[results_df["Target_Variable"] == "T_j"].set_index("Model")
    axes[1, 0].bar(models, [tj_df.loc[m, "MAE"] for m in models], color=colors, width=0.45)
    axes[1, 0].set_title("$T_j$ Mean Absolute Error (MAE)", fontweight="bold")
    axes[1, 0].set_ylabel("MAE [°C]")
    for i, m in enumerate(models):
        val = tj_df.loc[m, "MAE"]
        axes[1, 0].text(i, val * 1.05, f"{val:.3f}", ha="center", fontweight="bold", fontsize=9)

    axes[1, 1].bar(models, [tj_df.loc[m, "RMSE"] for m in models], color=colors, width=0.45)
    axes[1, 1].set_title("$T_j$ Root Mean Squared Error (RMSE)", fontweight="bold")
    axes[1, 1].set_ylabel("RMSE [°C]")
    for i, m in enumerate(models):
        val = tj_df.loc[m, "RMSE"]
        axes[1, 1].text(i, val * 1.05, f"{val:.3f}", ha="center", fontweight="bold", fontsize=9)

    axes[1, 2].bar(models, [tj_df.loc[m, "R2_Score"] for m in models], color=colors, width=0.45)
    axes[1, 2].set_title("$T_j$ Coefficient of Determination ($R^2$)", fontweight="bold")
    axes[1, 2].set_ylabel("$R^2$ Score [-]")
    axes[1, 2].set_ylim(0.99, 1.0005)
    for i, m in enumerate(models):
        val = tj_df.loc[m, "R2_Score"]
        axes[1, 2].text(i, val - 0.0008, f"{val:.6f}", ha="center", fontweight="bold", fontsize=9, color="white")

    plt.suptitle("Model Performance Comparison: Random Forest vs. Gradient Boosting", fontsize=14, fontweight="bold", y=1.00)
    plt.tight_layout()
    p5 = os.path.join(plots_dir, "model_comparison_metrics.png")
    fig.savefig(p5)
    plt.close(fig)
    print(f"Saved: {p5}")
    print("Visualization Generation Step 5 PASSED.\n")


# =============================================================================
# STEP 6: MODEL PERSISTENCE
# =============================================================================

def save_models_and_metadata(models: dict, output_dir: str = "models"):
    """
    Saves trained models via Joblib and writes metadata.json with configuration details.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print("STEP 6: MODEL PERSISTENCE")
    print("=" * 80)

    # Save Random Forest
    rf_path = os.path.join(output_dir, "random_forest_model.joblib")
    joblib.dump(models["Random Forest"], rf_path, compress=3)
    print(f"Saved Random Forest model to: {rf_path} ({os.path.getsize(rf_path)/(1024*1024):.2f} MB)")

    # Save Gradient Boosting
    gb_path = os.path.join(output_dir, "gradient_boosting_model.joblib")
    joblib.dump(models["Gradient Boosting"], gb_path, compress=3)
    print(f"Saved Gradient Boosting model to: {gb_path} ({os.path.getsize(gb_path)/1024:.2f} KB)")

    # Save Model Metadata
    metadata = {
        "project": "Expert Thermal / XThermal Heat Sink Surrogate Model",
        "date_created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "features": {
            "names": ["TDP", "V_air", "k_tim"],
            "units": ["W", "m/s", "W/(m*K)"],
            "descriptions": [
                "Processor Thermal Design Power",
                "Inlet cooling air velocity",
                "Thermal Interface Material thermal conductivity"
            ]
        },
        "targets": {
            "names": ["R_total", "T_j"],
            "units": ["°C/W", "°C"],
            "descriptions": [
                "Total junction-to-ambient thermal resistance",
                "Predicted processor junction temperature"
            ]
        },
        "models": {
            "Random_Forest": {
                "algorithm": "RandomForestRegressor",
                "parameters": {
                    "n_estimators": 200,
                    "random_state": 42,
                    "n_jobs": -1
                },
                "file": "random_forest_model.joblib"
            },
            "Gradient_Boosting": {
                "algorithm": "MultiOutputRegressor(GradientBoostingRegressor)",
                "parameters": {
                    "n_estimators": 200,
                    "learning_rate": 0.05,
                    "max_depth": 3,
                    "random_state": 42
                },
                "file": "gradient_boosting_model.joblib"
            }
        },
        "training_data": {
            "dataset": "02_Dataset/thermal_dataset.csv",
            "total_samples": 8280,
            "train_samples": 6624,
            "test_samples": 1656,
            "split_ratio": "80/20",
            "random_state": 42
        }
    }

    meta_path = os.path.join(output_dir, "model_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
    print(f"Saved model metadata to: {meta_path}")
    print("Model Persistence Step 6 PASSED.\n")


# =============================================================================
# STEP 7: ADDITIONAL VALIDATION & INTERPOLATION TESTING
# =============================================================================

def perform_additional_validations(models: dict, X_test: pd.DataFrame, y_test: pd.DataFrame):
    """
    Carries out 4 required additional validations:
      1. Physical validity of test set predictions.
      2. Off-grid interpolation testing using independent physics samples.
      3. Physical consistency check: T_j = 25 + TDP * R_total.
      4. Repeated benchmark of prediction latency (Physics vs RF vs GB).
    """
    print("=" * 80)
    print("STEP 7: ADDITIONAL VALIDATIONS & BENCHMARKING")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # 7.1 Verify Physical Bounds on Held-Out Test Set
    # -------------------------------------------------------------------------
    print("7.1 Checking physical bounds on held-out test predictions...")
    for model_name, model in models.items():
        preds = model.predict(X_test)
        r_pred = preds[:, 0]
        t_pred = preds[:, 1]

        assert np.all(np.isfinite(preds)), f"{model_name} produced non-finite predictions!"
        assert np.all(r_pred > 0), f"{model_name} predicted non-positive thermal resistance!"
        assert np.all(t_pred > 25.0), f"{model_name} predicted T_j <= ambient temperature!"
        print(f"  [PASSED] {model_name}: All R_total > 0 and all T_j > 25.0 °C (finite and positive).")

    # -------------------------------------------------------------------------
    # 7.2 Independent Off-Grid Interpolation Testing
    # -------------------------------------------------------------------------
    print("\n7.2 Generating independent off-grid interpolation test samples...")
    # Generate 500 points strictly BETWEEN grid points using continuous sampling
    rng = np.random.RandomState(999)
    n_interp = 500

    # Ensure off-grid by generating random continuous values within ranges
    # TDP in [35, 245], V_air in [0.7, 14.7], k_tim in [1.2, 11.8]
    interp_tdp = rng.uniform(35.0, 245.0, size=n_interp)
    interp_v_air = rng.uniform(0.7, 14.7, size=n_interp)
    interp_k_tim = rng.uniform(1.2, 11.8, size=n_interp)

    interp_features = pd.DataFrame({
        "TDP": interp_tdp,
        "V_air": interp_v_air,
        "k_tim": interp_k_tim
    })

    # Ground truth evaluation directly from physics simulator
    interp_records = []
    for _, row in interp_features.iterrows():
        sim_res = calculate_heat_sink(
            TDP=row["TDP"],
            V_air=row["V_air"],
            k_tim=row["k_tim"]
        )
        interp_records.append({
            "R_total": sim_res["R_total"],
            "T_j": sim_res["T_j"]
        })
    interp_y_true = pd.DataFrame(interp_records)

    print(f"Generated {n_interp} off-grid evaluation points. Evaluating surrogate interpolation...")

    interp_metrics = []
    for model_name, model in models.items():
        y_interp_pred = model.predict(interp_features)

        # R_total
        r_true = interp_y_true["R_total"].to_numpy()
        r_pred = y_interp_pred[:, 0]
        mae_r = mean_absolute_error(r_true, r_pred)
        rmse_r = root_mean_squared_error(r_true, r_pred)
        r2_r = r2_score(r_true, r_pred)

        # T_j
        t_true = interp_y_true["T_j"].to_numpy()
        t_pred = y_interp_pred[:, 1]
        mae_t = mean_absolute_error(t_true, t_pred)
        rmse_t = root_mean_squared_error(t_true, t_pred)
        r2_t = r2_score(t_true, t_pred)

        interp_metrics.append({
            "Model": model_name,
            "R_total_MAE": mae_r,
            "R_total_RMSE": rmse_r,
            "R_total_R2": r2_r,
            "T_j_MAE": mae_t,
            "T_j_RMSE": rmse_t,
            "T_j_R2": r2_t
        })

    interp_df = pd.DataFrame(interp_metrics)
    print("\nOff-Grid Interpolation Accuracy:")
    print(f"{'Model':<20} | {'R_tot MAE [°C/W]':<18} | {'R_tot R²':<12} | {'T_j MAE [°C]':<14} | {'T_j R²':<12}")
    print("-" * 85)
    for _, row in interp_df.iterrows():
        print(f"{row['Model']:<20} | {row['R_total_MAE']:<18.6f} | {row['R_total_R2']:<12.6f} | {row['T_j_MAE']:<14.4f} | {row['T_j_R2']:<12.6f}")
    print("-" * 85)

    # -------------------------------------------------------------------------
    # 7.3 Physical Consistency Check: T_j = 25 + TDP * R_total
    # -------------------------------------------------------------------------
    print("\n7.3 Checking physical consistency: T_j = 25 + TDP * R_total...")
    consistency_results = {}

    for model_name, model in models.items():
        preds = model.predict(X_test)
        pred_r_total = preds[:, 0]
        pred_tj = preds[:, 1]

        # Calculate theoretical T_j from predicted R_total
        expected_tj_from_r = 25.0 + (X_test["TDP"].to_numpy() * pred_r_total)

        # Discrepancy between independently predicted T_j and (25 + TDP * R_pred)
        discrepancy = np.abs(pred_tj - expected_tj_from_r)

        mean_disc = np.mean(discrepancy)
        max_disc = np.max(discrepancy)
        std_disc = np.std(discrepancy)
        p95_disc = np.percentile(discrepancy, 95)

        consistency_results[model_name] = {
            "mean": mean_disc,
            "max": max_disc,
            "std": std_disc,
            "p95": p95_disc
        }

        print(f"  {model_name}:")
        print(f"    Mean absolute discrepancy : {mean_disc:.4f} °C")
        print(f"    Max discrepancy          : {max_disc:.4f} °C")
        print(f"    95th percentile          : {p95_disc:.4f} °C")

    # -------------------------------------------------------------------------
    # 7.4 Prediction Latency & Throughput Benchmark
    # -------------------------------------------------------------------------
    print("\n7.4 Fair repeated prediction latency benchmark...")

    # Single-sample latency (50 iterations)
    sample_df = X_test.iloc[[0]]
    single_tdp = float(sample_df.iloc[0, 0])
    single_v_air = float(sample_df.iloc[0, 1])
    single_k_tim = float(sample_df.iloc[0, 2])

    n_single_runs = 50

    # Physics model single sample
    t0 = time.perf_counter()
    for _ in range(n_single_runs):
        calculate_heat_sink(single_tdp, single_v_air, single_k_tim)
    t_phys_single = (time.perf_counter() - t0) / n_single_runs

    # Random Forest single sample
    rf_m = models["Random Forest"]
    t0 = time.perf_counter()
    for _ in range(n_single_runs):
        rf_m.predict(sample_df)
    t_rf_single = (time.perf_counter() - t0) / n_single_runs

    # Gradient Boosting single sample
    gb_m = models["Gradient Boosting"]
    t0 = time.perf_counter()
    for _ in range(n_single_runs):
        gb_m.predict(sample_df)
    t_gb_single = (time.perf_counter() - t0) / n_single_runs

    # Batch latency (1000 samples, 10 iterations)
    batch_df = X_test.iloc[:1000]
    batch_X_vals = batch_df.to_numpy()
    n_batch_runs = 10

    # Physics model batch loop
    t0 = time.perf_counter()
    for _ in range(n_batch_runs):
        for r in batch_X_vals:
            calculate_heat_sink(r[0], r[1], r[2])
    t_phys_batch = (time.perf_counter() - t0) / n_batch_runs

    # Random Forest batch
    t0 = time.perf_counter()
    for _ in range(n_batch_runs):
        rf_m.predict(batch_df)
    t_rf_batch = (time.perf_counter() - t0) / n_batch_runs

    # Gradient Boosting batch
    t0 = time.perf_counter()
    for _ in range(n_batch_runs):
        gb_m.predict(batch_df)
    t_gb_batch = (time.perf_counter() - t0) / n_batch_runs

    print("\nBenchmark Results Summary:")
    print(f"{'Method / Model':<22} | {'Single Sample (ms)':<20} | {'1,000 Samples Batch (ms)':<26} | {'Throughput (samples/s)':<22}")
    print("-" * 98)
    print(f"{'Physics Simulator':<22} | {t_phys_single * 1e3:<20.4f} | {t_phys_batch * 1e3:<26.2f} | {1000.0 / t_phys_batch:<22.0f}")
    print(f"{'Random Forest':<22} | {t_rf_single * 1e3:<20.4f} | {t_rf_batch * 1e3:<26.2f} | {1000.0 / t_rf_batch:<22.0f}")
    print(f"{'Gradient Boosting':<22} | {t_gb_single * 1e3:<20.4f} | {t_gb_batch * 1e3:<26.2f} | {1000.0 / t_gb_batch:<22.0f}")
    print("-" * 98)

    timing_summary = {
        "physics": {"single_ms": t_phys_single * 1e3, "batch_ms": t_phys_batch * 1e3, "throughput": 1000.0 / t_phys_batch},
        "rf": {"single_ms": t_rf_single * 1e3, "batch_ms": t_rf_batch * 1e3, "throughput": 1000.0 / t_rf_batch},
        "gb": {"single_ms": t_gb_single * 1e3, "batch_ms": t_gb_batch * 1e3, "throughput": 1000.0 / t_gb_batch}
    }

    print("Step 7 Additional Validations PASSED.\n")
    return interp_df, consistency_results, timing_summary


# =============================================================================
# MAIN PIPELINE EXECUTION
# =============================================================================

def main():
    print("\n" + "#" * 80)
    print("TASK 1, PART C: ML SURROGATE MODEL DEVELOPMENT AND EVALUATION")
    print("#" * 80 + "\n")

    # Step 1: Data Preparation
    X, y = load_and_validate_data(DATASET_PATH)

    # Step 2: Train-Test Split
    X_train, X_test, y_train, y_test = split_dataset(X, y, test_size=0.20, random_state=42)

    # Step 3: Train Models
    models = train_surrogate_models(X_train, y_train)

    # Step 4: Model Evaluation
    results_df, predictions = evaluate_models(models, X_test, y_test, output_dir=RESULTS_PATH)

    # Step 5: Visualizations
    generate_visualizations(y_test, predictions, results_df, plots_dir=PLOTS_PATH)

    # Step 6: Model Persistence
    save_models_and_metadata(models, output_dir=MODELS_PATH)

    # Step 7: Additional Validations
    interp_df, consistency_results, timing_summary = perform_additional_validations(models, X_test, y_test)

    print("#" * 80)
    print("ALL ML SURROGATE PIPELINE TASKS COMPLETED SUCCESSFULLY!")
    print("#" * 80)


if __name__ == "__main__":
    main()
