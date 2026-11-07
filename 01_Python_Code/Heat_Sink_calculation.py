# -*- coding: utf-8 -*-
"""
Property of Expert Thermal/XThermal
"""
import numpy as np

L_die = 0.0525 #Die length in m
W_die = 0.045 #Die Width in m
Thickness_die = 0.0022 #die Thickness in m


TDP = 150  # Thermal Design Power in Watts
T_ambient = 25  # Ambient temperature in Celsius
L = 90e-3  # Heat sink length in meters
W = 116e-3  # Heat sink width in meters
H = 27e-3  # Heat sink height in meters
Fin_Thickness = 0.8e-3  # Fin thickness in meters
N_fins = 60  # Number of fins
Base_Thickness = 2.5e-3  # Base thickness in meters
V_air = 1  # Air velocity in m/s
k_tim = 4.0  # Thermal conductivity of thermal interface material (TIM) in W/m.K
t_tim = 0.1e-3  # TIM thickness in meters
R_jc = 0.2  # Junction-to-case thermal resistance in °C/W

def heat_sink_design():
   
    # Given constants for air at 25 degree C
    k_air = 0.0262  # Thermal conductivity of air (W/m.K)
    nu_air = 1.57e-5  # Kinematic viscosity of air (m²/s)
    Pr_air = 0.71  # Prandtl number

    # Fin Spacing Calculation
    s_f = (W - (N_fins * Fin_Thickness)) / (N_fins - 1)

    # Reynolds Number Calculation
    Re = (V_air * s_f) / nu_air

    # Nusselt Number Calculation (Laminar & Turbulent cases)
    if Re < 2300:  # Laminar flow assumption
        Nu = 1.86 * ((Re * Pr_air * (2 * s_f) / L) ** (1/3))
    else:  # Turbulent flow
        Nu = 0.023 * (Re**0.8) * (Pr_air**0.3)

    # Convective Heat Transfer Coefficient
    h = (Nu * k_air) / (2 * s_f)
    
    #fin Height
    
    h_fin = H-Base_Thickness

    # Effective Surface Area Calculation
    A_fin = N_fins * (2 * h_fin * L)+ (s_f * L)
    A_Total_base = (L * W)
    
    A_base_convection = A_Total_base-(Fin_Thickness*N_fins*L) #only this area is exposed for convectionn
   
    A_total = A_fin + A_base_convection

    # Convective Thermal Resistance
    R_conv = 1 / (h * A_total)

    # TIM Thermal Resistance
    A_die = L_die*W_die # die area where thermal insulation is done
    R_tim = t_tim / (k_tim * A_die)

    # Total Heat Sink Thermal Resistance
    R_hs = R_conv + R_tim

    # Total Thermal Resistance
    R_total = R_jc + R_hs

    # Junction Temperature Calculation
    T_j = T_ambient + (TDP * R_total)

    print("The Reynolds Number is: ", Re)
    print("The Nusselt Number is: ", Nu)
    print("The Convective Heat Transfer Coefficient (W/m².K) is: ", h)
    print("Total Thermal Resistance (°C/W): ", R_total)
    print("Predicted Processor Junction Temperature (°C): ", T_j)
       
heat_sink_design()
