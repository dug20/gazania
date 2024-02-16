"""
Assumptions:
- Turbofan has seperate nozzle for bypass and core flow
- No turbines are present in the bypass flow (the script does not bother to look for any available turbine work in the bypass flow)

Notes:
- No checks are made to make ensure the exit pressures of Nozzle objects (p_amb or p_e) are the same as the ambient pressure given to a Turbofan object (p_in). The former are
  used as nozzle exit boundary conditions. The latter is used to calculate the pressure component of thrust.
"""

import cantera as ct
import gazania as gaz
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


class Turbofan:
    def __init__(self,
                 mdot_in,
                 BPR,
                 bypass_components,
                 core_components,
                 p_in = 101325,
                 T_in = 298.15,
                 V_in = 0.0,
                 model = "cantera"):
        """Object to represent and solve a turbofan engine cycle.

        Args:
            mdot_in (float): Total mass flow rate in (kg/s), includes bypass and core flow.
            BPR (float): Bypass ratio (mdot_bypass / mdot_core)
            bypass_components (list): List of bypass components, including the fan (represented by a compressor). e.g. [Compressor, Nozzle]
            core_components (list): List of core cycle components, including the fan (represented by a compressor). e.g. [Compressor, Compressor, Combustor, Turbine, Nozzle]
            p_in (float, optional): Inlet static pressure. This is also the ambient pressure used to calculate thrust. Defaults to 101325.
            T_in (float, optional): Inlet static temperature. Defaults to 298.15.
            V_in (float, optional): Inlet velocity. Defaults to 0.0.
            model (str, optional): Gas model to use, can be either "perfect" or "cantera". Defaults to "cantera", in which case gri30.yaml is used.

        Attributes:
            bypass_cycle (gazania.Cycle): Solved object representing the bypass cycle.  
            core_cycle (gazania.Cycle): Solved object representing the core cycle.   
            M_in (float): Inlet Mach number.
            p0_in (float): Inlet stagnation pressure (Pa).
            T0_in (float): Inlet stagnation temperature (K).
            bypass_compressor_powers (list): List of power inputs to each bypass compressor (W). Includes the fan. Positive means power input.
            core_compressor_powers (list): List of power inputs to each core cycle compressor (W), in the order that they appear in the cycle. Positive means power input.
            core_turbine_powers (list): List of power outputs from each core cycle turbine (W), in the order that they appear in the cycle. Negative means power output.
            net_power (float): Imbalance of power between the compressors and turbines (W). Positive means there is insufficient turbine power for the compressors.
            A_e_b (float): Bypass nozzle exit area (m2).
            A_e_c (float): Core nozzle exit area (m2).
            mdot_c (float): Core mass flow rate (kg/s)
            mdot_b (float): Bypass mass flow rate (kg/s).
            thrust_b (float): Thrust from the bypass (i.e. fan) cycle (N).
            thrust_c (float): Thrust from the core cycle (N).
            thrust (float): Total thrust from the engine (N).
        """
        self.mdot_in = mdot_in                          # Total inlet mass flow rate of air (kg/s)
        self.p_in = p_in                                # Inlet static pressure (Pa). Also used as the ambient pressure to calculate thrust.
        self.T_in = T_in                                # Inlet static temperature (K)
        self.BPR = BPR                                  # Bypass ratio (mdot_ban / mdot_core)
        self.bypass_components = bypass_components      # List of objects to represent the bypass cycle (including fan) - will often just be [Compressor, Nozzle]
        self.core_components = core_components          # List objects to represent the core cycle (including fan) - e.g. [Compressor, Compressor, Combustor, Turbine, Nozzle]
        self.V_in = V_in                                # Inlet velocity (m/s). Used to calculate the thrust.
        self.model = model
        self.solved = False

    def __str__(self):
        if self.solved:
            string = "<gazania.Turbofan object> \n"

            string += "Inlet conditions: \n"
            string += f"M = {self.M_in} \n"
            string += f"p0 = {self.p0_in} Pa (p = {self.p_in} Pa)\n"
            string += f"T0 = {self.T0_in} K (T = {self.T_in} Pa)\n"

            string += "\nPower: \n"
            string += f"Compressor powers (bypass) = {self.bypass_compressor_powers} W\n"
            string += f"Compressor powers (core) = {self.core_compressor_powers} W\n"
            string += f"Turbine powers (core) = {self.core_turbine_powers} W (negative means net power out)\n"
            string += f"Net power in = {self.net_power} W (positive means insufficent turbine power)\n"

            string += "\nThrust: \n"
            string += f"Core nozzle area = {self.A_e_c} m2 \n"
            string += f"Bypass nozzle area = {self.A_e_b} m2 \n\n"

            string += f"Total thrust = {self.thrust} N \n"
            string += f"Core thrust = {self.thrust_c} N \n"
            string += f"Bypass thrust = {self.thrust_b} N"
            return string
        else:
            return "<gazania.Turbofan object>: Unsolved"

    def solve(self):
        # Get inlet stagnation conditions
        if self.model == "cantera":
            gas_inlet = ct.Solution('gri30.yaml')         # Cantera Solution object that will be edited to calculate properties
            gas_inlet.TPY = self.T_in, self.p_in, gaz.AIR_COMP
            s_in = gas_inlet.s
            h_in = gas_inlet.h
            h0_in = h_in + 0.5 * self.V_in**2
            gamma_in = gas_inlet.cp / gas_inlet.cv
            R_in = gas_inlet.cp * (1 - 1/gamma_in)
            self.M_in = self.V_in / (gamma_in * R_in * self.T_in)**0.5

            # Solve until entropy is unchanged
            p0_guess = self.p_in * ( 1 + (gamma_in - 1)/2 * self.M_in**2 )**(gamma_in / (gamma_in - 1))

            def s_error(p0):
                gas_inlet.HP = h0_in, p0
                return s_in - gas_inlet.s
            
            # gas_inlet will be updated until scipy converges on the root, at which point gas_inlet will be at stagnation conditions
            scipy.optimize.root_scalar(s_error, bracket = [0.9 * p0_guess, 1.1 * p0_guess] )

            self.p0_in = gas_inlet.P                        # Inlet stagnation pressure (Pa)
            self.T0_in = gas_inlet.T                        # Inlet stagnation temperature (K)

        elif self.model == "perfect":
            # Isentropic flow relations
            self.M_in = self.V_in / (gaz.GAMMA_AIR * gaz.R_AIR * self.T_in)**0.5 
            self.p0_in = self.p_in * ( 1 + (gaz.GAMMA_AIR - 1)/2 * self.M_in**2 )**(gaz.GAMMA_AIR / (gaz.GAMMA_AIR - 1))
            self.T0_in = self.T_in * ( 1 + (gaz.GAMMA_AIR - 1)/2 * self.M_in**2 )

        self.mdot_c = self.mdot_in / (self.BPR + 1)     # Core mass flow rate (kg/s)
        self.mdot_b = self.mdot_c * self.BPR            # Bypass mass flow rate (kg/s)   

        self.bypass_cycle = gaz.Cycle(mdot_in = self.mdot_b, 
                                     components = self.bypass_components,
                                     p0_in = self.p0_in,
                                     T0_in = self.T0_in,
                                     model = self.model)

        self.core_cycle = gaz.Cycle(mdot_in = self.mdot_c, 
                                     components = self.core_components,
                                     p0_in = self.p0_in,
                                     T0_in = self.T0_in,
                                     model = self.model)
        
        self.bypass_cycle.solve()
        self.core_cycle.solve()

        # Calculate power output from compressors and turbines
        self.bypass_compressor_powers = []
        self.core_compressor_powers = []
        self.core_turbine_powers = []

        for i in range(len(self.bypass_cycle.components)):
            if type(self.bypass_cycle.components[i]) is gaz.Compressor:
                self.bypass_compressor_powers.append( (self.bypass_cycle.h[i+1] - self.bypass_cycle.h[i]) * self.bypass_cycle.mdot[i] )

        for i in range(len(self.core_cycle.components)):
            if type(self.core_cycle.components[i]) is gaz.Compressor:
                self.core_compressor_powers.append( (self.core_cycle.h[i+1] - self.core_cycle.h[i]) * self.core_cycle.mdot[i] )

        for i in range(len(self.core_cycle.components)):
            if type(self.core_cycle.components[i]) is gaz.Turbine:
                self.core_turbine_powers.append( (self.core_cycle.h[i+1] - self.core_cycle.h[i]) * self.core_cycle.mdot[i] )

        self.compressor_power = sum(self.bypass_compressor_powers) + sum(self.core_compressor_powers)
        self.turbine_power = sum(self.core_turbine_powers)

        self.net_power = self.compressor_power + self.turbine_power

        # Calculate thrust (taking into account pressure difference at nozzle exit)
        self.A_e_b = self.bypass_cycle.mdot[-1] / (self.bypass_cycle.rho[-1] * self.bypass_cycle.V[-1])      # Bypass nozzle exit area
        self.A_e_c = self.core_cycle.mdot[-1] / (self.core_cycle.rho[-1] * self.core_cycle.V[-1])            # Core nozzle exit area

        self.thrust_b = (self.mdot_b * self.bypass_cycle.V[-1]) + (self.bypass_cycle.p[-1] - self.p_in) * self.A_e_b - (self.mdot_b * self.V_in)
        self.thrust_c = (self.mdot_c * self.core_cycle.V[-1]) + (self.core_cycle.p[-1] - self.p_in) * self.A_e_c - (self.mdot_c * self.V_in)

        self.thrust = self.thrust_b + self.thrust_c

        # Update status
        self.solved = True
    