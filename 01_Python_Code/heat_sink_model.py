# -*- coding: utf-8 -*-
"""
Engineering AI/ML Physics Model for Heat Sink Thermal Calculation
Property of Expert Thermal / XThermal

This module provides a parameterized, reusable function for calculating
the thermal performance of a forced-convection finned heat sink.

The model calculates:
  - Hydraulic and aerodynamic parameters (fin spacing, Reynolds number)
  - Heat transfer parameters (Nusselt number, convective heat transfer coefficient)
  - Geometry and effective convective areas (fin area, base convection area)
  - Thermal resistance network (R_conv, R_tim, R_hs, R_total)
  - Resulting processor junction temperature (T_j)
"""

from typing import Dict, Any


def calculate_heat_sink(
    TDP: float = 150.0,
    V_air: float = 1.0,
    k_tim: float = 4.0
) -> Dict[str, Any]:
    """
    Calculates thermal resistance and processor junction temperature for a finned heat sink.

    Parameters
    ----------
    TDP : float
        Thermal Design Power of the processor in Watts [W].
    V_air : float
        Inlet air flow velocity in meters per second [m/s].
    k_tim : float
        Thermal conductivity of the Thermal Interface Material (TIM) in [W/(m*K)].

    Returns
    -------
    Dict[str, Any]
        Dictionary containing all inputs, intermediate geometric/aerodynamic/thermal
        calculations, and final outputs (resistances and junction temperature).
    """

    # -------------------------------------------------------------------------
    # Fixed Engineering Parameters & Physical Properties (Preserved from original)
    # -------------------------------------------------------------------------
    # Processor Die Geometry
    L_die: float = 0.0525          # Die length in meters [m]
    W_die: float = 0.045           # Die width in meters [m]
    Thickness_die: float = 0.0022  # Die thickness in meters [m]

    # Ambient & System Operating Conditions
    T_ambient: float = 25.0        # Ambient temperature in Celsius [°C]

    # Heat Sink Dimensions
    L: float = 90e-3               # Heat sink length in meters [m] (90 mm)
    W: float = 116e-3              # Heat sink width in meters [m] (116 mm)
    H: float = 27e-3               # Heat sink overall height in meters [m] (27 mm)
    Fin_Thickness: float = 0.8e-3  # Fin thickness in meters [m] (0.8 mm)
    N_fins: int = 60               # Total number of fins
    Base_Thickness: float = 2.5e-3 # Base thickness in meters [m] (2.5 mm)

    # Thermal Interface Material (TIM)
    t_tim: float = 0.1e-3          # TIM layer thickness in meters [m] (0.1 mm)

    # Package Thermal Resistance
    # Junction-to-case thermal resistance [°C/W] (preserved choice from original script)
    R_jc: float = 0.2

    # Air Properties at 25 °C (1 atm)
    k_air: float = 0.0262          # Thermal conductivity of air in [W/(m*K)]
    nu_air: float = 1.57e-5        # Kinematic viscosity of air in [m^2/s]
    Pr_air: float = 0.71           # Prandtl number of air [-]

    # -------------------------------------------------------------------------
    # Step 1: Fin Spacing Calculation
    # -------------------------------------------------------------------------
    # Free spacing between adjacent fins across the heat sink width:
    # s_f = (W - N_fins * Fin_Thickness) / (N_fins - 1)
    s_f: float = (W - (N_fins * Fin_Thickness)) / (N_fins - 1)

    # -------------------------------------------------------------------------
    # Step 2: Reynolds Number Calculation
    # -------------------------------------------------------------------------
    # Flow within the inter-fin channels using fin spacing s_f as characteristic length:
    # Re = (V_air * s_f) / nu_air
    Re: float = (V_air * s_f) / nu_air

    # -------------------------------------------------------------------------
    # Step 3: Nusselt Number Calculation (Laminar & Turbulent Regimes)
    # -------------------------------------------------------------------------
    # For Re < 2300: Developing laminar channel flow (Sieder-Tate correlation)
    # For Re >= 2300: Turbulent channel flow (Dittus-Boelter correlation)
    if Re < 2300:
        flow_regime = "laminar"
        Nu: float = 1.86 * ((Re * Pr_air * (2 * s_f) / L) ** (1 / 3))
    else:
        flow_regime = "turbulent"
        Nu: float = 0.023 * (Re ** 0.8) * (Pr_air ** 0.3)

    # -------------------------------------------------------------------------
    # Step 4: Convective Heat Transfer Coefficient
    # -------------------------------------------------------------------------
    # Derived from Nusselt number using hydraulic diameter D_h = 2 * s_f:
    # h = (Nu * k_air) / (2 * s_f)  [W/(m^2*K)]
    h: float = (Nu * k_air) / (2 * s_f)

    # -------------------------------------------------------------------------
    # Step 5: Geometry & Effective Convection Area Calculation
    # -------------------------------------------------------------------------
    # Fin active height excluding the base plate:
    h_fin: float = H - Base_Thickness

    # Fin convection area: both sides of all fins plus one channel base spacing
    # (preserved exact equation from original script):
    A_fin: float = N_fins * (2 * h_fin * L) + (s_f * L)

    # Total heat sink footprint base area:
    A_Total_base: float = L * W

    # Area of base exposed to airflow between fins:
    A_base_convection: float = A_Total_base - (Fin_Thickness * N_fins * L)

    # Total effective convective surface area:
    A_total: float = A_fin + A_base_convection

    # -------------------------------------------------------------------------
    # Step 6: Convective Thermal Resistance
    # -------------------------------------------------------------------------
    # Thermal resistance between heat sink surface and cooling air stream:
    # R_conv = 1 / (h * A_total)  [°C/W]
    R_conv: float = 1.0 / (h * A_total)

    # -------------------------------------------------------------------------
    # Step 7: Thermal Interface Material (TIM) Resistance
    # -------------------------------------------------------------------------
    # Die contact area:
    A_die: float = L_die * W_die

    # 1D conduction resistance across the TIM layer:
    # R_tim = t_tim / (k_tim * A_die)  [°C/W]
    R_tim: float = t_tim / (k_tim * A_die)

    # -------------------------------------------------------------------------
    # Step 8: Total Heat Sink Thermal Resistance
    # -------------------------------------------------------------------------
    # In original script: R_hs = R_conv + R_tim (no separate base conduction term)
    R_hs: float = R_conv + R_tim

    # -------------------------------------------------------------------------
    # Step 9: Total Thermal Resistance (Junction-to-Ambient)
    # -------------------------------------------------------------------------
    # Full resistance network: R_total = R_jc + R_hs
    R_total: float = R_jc + R_hs

    # -------------------------------------------------------------------------
    # Step 10: Processor Junction Temperature Calculation
    # -------------------------------------------------------------------------
    # T_j = T_ambient + TDP * R_total  [°C]
    T_j: float = T_ambient + (TDP * R_total)

    # -------------------------------------------------------------------------
    # Compilation of All Calculations into Result Dictionary
    # -------------------------------------------------------------------------
    results: Dict[str, Any] = {
        # Inputs & Variable Parameters
        "TDP": TDP,
        "V_air": V_air,
        "k_tim": k_tim,
        # Fixed Geometric & Operating Parameters
        "L_die": L_die,
        "W_die": W_die,
        "Thickness_die": Thickness_die,
        "T_ambient": T_ambient,
        "L": L,
        "W": W,
        "H": H,
        "Fin_Thickness": Fin_Thickness,
        "N_fins": N_fins,
        "Base_Thickness": Base_Thickness,
        "t_tim": t_tim,
        "R_jc": R_jc,
        "k_air": k_air,
        "nu_air": nu_air,
        "Pr_air": Pr_air,
        # Intermediate Aerodynamic & Convective Calculations
        "s_f": s_f,
        "Re": Re,
        "flow_regime": flow_regime,
        "Nu": Nu,
        "h": h,
        "h_fin": h_fin,
        "A_fin": A_fin,
        "A_Total_base": A_Total_base,
        "A_base_convection": A_base_convection,
        "A_total": A_total,
        "A_die": A_die,
        # Intermediate & Final Thermal Resistances
        "R_conv": R_conv,
        "R_tim": R_tim,
        "R_hs": R_hs,
        "R_total": R_total,
        # Final Output
        "T_j": T_j,
    }

    return results


# Maintain backward compatibility with the original function name
heat_sink_design = calculate_heat_sink
heat_sink_model = calculate_heat_sink


if __name__ == "__main__":
    print("=" * 70)
    print("EXPERT THERMAL / XTHERMAL - HEAT SINK MODEL VALIDATION")
    print("=" * 70)

    # Reference scenario: TDP = 150 W, V_air = 1.0 m/s, k_tim = 4.0 W/(m*K)
    ref_tdp = 150.0
    ref_v_air = 1.0
    ref_k_tim = 4.0

    print(f"\n--- Testing Reference Scenario ---")
    print(f"TDP           : {ref_tdp} W")
    print(f"Air Velocity  : {ref_v_air} m/s")
    print(f"TIM Conductivity: {ref_k_tim} W/(m*K)")

    res = calculate_heat_sink(TDP=ref_tdp, V_air=ref_v_air, k_tim=ref_k_tim)

    print("\n--- Output Results Dictionary ---")
    print(f"Fin Spacing (s_f)                      : {res['s_f'] * 1e3:.4f} mm")
    print(f"Reynolds Number (Re)                  : {res['Re']:.4f} ({res['flow_regime']})")
    print(f"Nusselt Number (Nu)                   : {res['Nu']:.4f}")
    print(f"Heat Transfer Coeff (h)               : {res['h']:.4f} W/(m^2*K)")
    print(f"Total Convective Area (A_total)       : {res['A_total']:.6f} m^2")
    print(f"Convective Resistance (R_conv)        : {res['R_conv']:.6f} °C/W")
    print(f"TIM Resistance (R_tim)                : {res['R_tim']:.6f} °C/W")
    print(f"Heat Sink Resistance (R_hs)           : {res['R_hs']:.6f} °C/W")
    print(f"Package Resistance (R_jc)             : {res['R_jc']:.4f} °C/W")
    print(f"Total Thermal Resistance (R_total)    : {res['R_total']:.6f} °C/W")
    print(f"Predicted Junction Temp (T_j)         : {res['T_j']:.4f} °C")

    # Comparison against original script values
    # Reference outputs directly from original Heat_Sink_calculation.py:
    # Re: 73.41034222174244
    # Nu: 2.048014084892021
    # h: 23.278089503132986 W/(m^2*K)
    # R_total: 0.3692049352073151 °C/W
    # T_j: 80.38074028109727 °C
    orig_re = 73.41034222174244
    orig_nu = 2.048014084892021
    orig_h = 23.278089503132986
    orig_r_total = 0.3692049352073151
    orig_tj = 80.38074028109727

    print("\n" + "=" * 70)
    print("COMPARISON WITH ORIGINAL SCRIPT (Heat_Sink_calculation.py)")
    print("=" * 70)
    print(f"{'Quantity':<35} | {'Original':<15} | {'Modified':<15} | {'Difference':<15}")
    print("-" * 85)

    comparisons = [
        ("Reynolds Number (Re)", orig_re, res["Re"]),
        ("Nusselt Number (Nu)", orig_nu, res["Nu"]),
        ("Convective Coeff h [W/m^2.K]", orig_h, res["h"]),
        ("Total Resistance R_total [°C/W]", orig_r_total, res["R_total"]),
        ("Junction Temp T_j [°C]", orig_tj, res["T_j"]),
    ]

    all_matched = True
    for name, orig_val, mod_val in comparisons:
        diff = abs(orig_val - mod_val)
        is_exact = diff < 1e-12
        if not is_exact:
            all_matched = False
        status = "MATCH (Exact)" if is_exact else f"DIFF: {diff:.2e}"
        print(f"{name:<35} | {orig_val:<15.6f} | {mod_val:<15.6f} | {status}")

    print("-" * 85)
    if all_matched:
        print("[SUCCESS] All outputs match the original script EXACTLY within machine precision.")
    else:
        print("[WARNING] Differences detected between original and modified outputs.")
    print("=" * 70)
