# -*- coding: utf-8 -*-
"""
Validation script for heat_sink_model.py against Heat_Sink_calculation.py
Expert Thermal / XThermal
"""

import math
import sys
import heat_sink_model


def run_validation():
    print("Running automated validation suite...")

    # 1. Test Reference Scenario (150 W, 1 m/s, 4 W/m.K)
    results = heat_sink_model.calculate_heat_sink(TDP=150.0, V_air=1.0, k_tim=4.0)

    # Reference values directly produced by Heat_Sink_calculation.py
    expected_values = {
        "Re": 73.41034222174244,
        "Nu": 2.048014084892021,
        "h": 23.278089503132986,
        "R_total": 0.3692049352073151,
        "T_j": 80.38074028109727,
        "s_f": 0.001152542372881356,
        "A_fin": 0.2647037288135593,
        "A_base_convection": 0.00612,
        "A_total": 0.2708237288135593,
        "R_conv": 0.1586229246253045,
        "R_tim": 0.010582010582010583,
        "R_hs": 0.16920493520731508,
    }

    all_passed = True
    for key, expected in expected_values.items():
        actual = results[key]
        if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12):
            print(f"FAILED: {key} expected {expected}, got {actual}")
            all_passed = False
        else:
            print(f"PASSED: {key:<18} = {actual:<20.8f} (exact match)")

    # 2. Check Dictionary Return Structure
    required_keys = [
        "TDP", "V_air", "k_tim", "T_ambient", "L_die", "W_die", "Thickness_die",
        "L", "W", "H", "Fin_Thickness", "N_fins", "Base_Thickness", "t_tim",
        "R_jc", "k_air", "nu_air", "Pr_air", "s_f", "Re", "flow_regime", "Nu",
        "h", "h_fin", "A_fin", "A_Total_base", "A_base_convection", "A_total",
        "A_die", "R_conv", "R_tim", "R_hs", "R_total", "T_j"
    ]
    missing_keys = [k for k in required_keys if k not in results]
    if missing_keys:
        print(f"FAILED: Missing keys in results dictionary: {missing_keys}")
        all_passed = False
    else:
        print(f"PASSED: All {len(required_keys)} required dictionary keys present.")

    # 3. Test Parameterization (variations of inputs)
    res_high_vel = heat_sink_model.calculate_heat_sink(TDP=150.0, V_air=3.0, k_tim=4.0)
    assert res_high_vel["Re"] > results["Re"], "Higher velocity should increase Reynolds number"
    assert res_high_vel["h"] > results["h"], "Higher velocity should increase heat transfer coefficient"
    assert res_high_vel["T_j"] < results["T_j"], "Higher velocity should decrease junction temperature"
    print("PASSED: Velocity sensitivity test (V_air = 3.0 m/s cools more effectively).")

    res_turbulent = heat_sink_model.calculate_heat_sink(TDP=150.0, V_air=35.0, k_tim=4.0)
    assert res_turbulent["flow_regime"] == "turbulent", "Re >= 2300 should enter turbulent regime"
    print(f"PASSED: Turbulent transition test (Re = {res_turbulent['Re']:.1f} >= 2300 -> turbulent).")

    res_high_tdp = heat_sink_model.calculate_heat_sink(TDP=200.0, V_air=1.0, k_tim=4.0)
    assert res_high_tdp["T_j"] > results["T_j"], "Higher TDP should increase junction temperature"
    print("PASSED: TDP sensitivity test (TDP = 200 W raises T_j proportionally).")

    if all_passed:
        print("\nALL VALIDATION CHECKS COMPLETED SUCCESSFULLY!")
        return 0
    else:
        print("\nVALIDATION FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(run_validation())
