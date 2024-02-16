"""
Gazania is a Python module for the modelling and evaluation of gas turbine cycles.

Assumptions:
 - M = 0 at all points except in a Nozzle object.

Notes:
 - The 'gri30.yaml' model is used with Cantera - this means all species are modelled as IDEAL (but not perfect) gases.
 - When using a perfect gas, the enthalpy is defined at each point by h = cp * T.
 - Specific enthalpies cannot be compared before and after a ChangeProperties component.
 - If I understand cantera correctly, frozen flow is assumed in nozzles (as opposed to equilibrium flow).

 - When using cantera's Solution.SP or Solution.HP (or Quantity.HP or Quantity.SP), the 'H' and 'S' are actually specific quantities (normally written h and s).
 - If you equate a new variable to an already established cantera.Quantity of cantera.Solution, it will reference that object instead of copying it (you must use e.g. gas2.TPY = gas1.TPY to copy gas1 to gas2).

"""

import cantera as ct
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from tabulate import tabulate


GAMMA_AIR = 1.4                             # cp/cv
CP_AIR = 1005                               # J/kg/K
R_AIR = CP_AIR * (1 - 1/GAMMA_AIR)          # J/kg/K
AIR_COMP = "O2:23.14, N2:75.52, Ar:1.29"    # Air composition
N_POLY_INTEGRATION = 100                    # Number of discrete points used for a polytropic efficiency integration

# Compressible flow relations
def m_bar(M, gamma):    
    """Get the non-dimensional mass flow rate from Mach number. Defined as m_bar = mdot * sqrt(cp*T0)/(A*p0). A is the flow area (m2).

    Args:
        M (float): Mach number
        gamma (float): Ratio of specific heats cp/cv

    Returns:
        float: Non-dimensional mass flow rate
    """

    return gamma/(gamma-1)**0.5 * M * (1 + M**2 * (gamma-1)/2)**( -0.5*(gamma+1)/(gamma-1) )

def M_sub(m_bar_desired, gamma):
    """Get Mach number from non-dimensional mass flow rate. Returns the subsonic solution.

    Args:
        m_bar (float): Non-dimensional mass flow rate, defined as mdot * sqrt(cp*T0)/(A*p0). A is the flow area (m2).
        gamma (float): Ratio of specific heats cp/cv

    Returns:
        tuple: Subsonic Mach number.
    """
    # Check if solution is possible
    if m_bar_desired > m_bar(M = 1.0, gamma = gamma):
        raise ValueError("M_sub(): No solution for Mach number. If using a Nozzle, the area is likely too small for the given mass flow rate.")

    #if abs(m_bar_desired - m_bar(M = 1, gamma = gamma)) < 1e-12:
    #    return 1.0

    # Calculate subsonic solution
    subsonic_sol = scipy.optimize.root_scalar(lambda M : m_bar(M = M, gamma = gamma) - m_bar_desired, bracket = [0.0, 1.0])
    M_sub = subsonic_sol.root

    return M_sub

def compressible_T(T0, M, gamma):
    """Get static temperature from the Mach number and stagnation temperature.

    Args:
        T0 (float): Stagnation temperature (K)
        M (float): Mach number
        gamma (float): Ratio of specific heats cp/cv

    Returns:
        float: Static temperature (K)
    """
    return T0 * (1 + (gamma-1)/2 * M**2)**(-1)

def compressible_p(p0, M, gamma):
    """Get static pressure from the Mach number and stagnation pressure.

    Args:
        p0 (float): Stagnation pressure (Pa)
        M (float): Mach number
        gamma (float): Ratio of specific heats cp/cv

    Returns:
        float: Static pressure (Pa)
    """
    return p0 * (1 + (gamma-1)/2 * M**2)**(-gamma/(gamma-1))


def show():
    """Show plots (just runs matplotlib.pyplot.show() implicitly)."""
    plt.show()

class Cycle:
    """Object to represent and evaluate gas turbine or jet engine cycles.

    Args:
        T0_in (float, optional): Inlet stagnation temperature. Defaults to 298.15.
        p0_in (int, optional): Inlet stagnation pressure. Defaults to 101325.
        mdot_in (float, optional): Inlet mass flow rate of air. Defaults to 1.0.
        components (list, optional): List of cycle components (e.g. [Compressor, Combustor, Turbine, Nozzle]). Defaults to None.
        model (str, optional): Gas model to use, can be either "perfect" or "cantera". Defaults to "cantera", in which case gri30.yaml is used.

    Attributes:
        p (np.ndarray): Static pressure at each station (Pa)
        T (np.ndarray): Static temperature at each station (K)
        h (np.ndarray): Specific enthalpy at each station (J/kg/K)
        p0 (np.ndarray): Stagnation pressure at each station (Pa)
        T0 (np.ndarray): Stagnation temperature at each station (K)
        h0 (np.ndarray): Stagnation specific enthalpy at each station (J/kg/K)
        mdot (np.ndarray): Mass flow rate at each station (kg/s)
        gamma (np.ndarray): Specific gas constant at each station 
        cp (np.ndarray): Isobaric specific heat capacity at each station (J/kg/K)
        V (np.ndarray): Gas velocity at each station (m/s)
    """
    def __init__(self, 
                 T0_in = 298.15, 
                 p0_in = 101325, 
                 mdot_in = 1.0, 
                 components = None,
                 model = "cantera"):

        if model != "cantera" and model != "perfect":
            raise ValueError("'model' argument must be 'perfect' or 'cantera'")

        self.T0_in = T0_in              # Inlet stagation temperature (K)
        self.p0_in = p0_in              # Inlet stagnation pressure (Pa)
        self.mdot_in = mdot_in          # Inlet mass flow rate of air (kg/s)
        self.components = components    # List of gas turbine components, e.g. [Compressor, SimpleCombustor, Turbine]
        self.model = model              # Model for the fluid. Can be "perfect" or "cantera".
        
        # Keep track of if we've solved the system
        self.solved = False

    def __str__(self):
        string = "<gazania.Cycle object>"
        if self.solved:
            for i in range(len(self.components)):
                if type(self.components[i]) is ChangeProperties:
                    data = [['p', self.p[i], self.p[i+1]], 
                            ['p0', self.p0[i], self.p0[i+1]],
                            ['T', self.T[i], self.T[i+1]],
                            ['T0', self.T0[i], self.T0[i+1]],
                            ['h (incomparable)', self.h[i], self.h[i+1]],
                            ['h0 (incomparable)', self.h0[i], self.h0[i+1]],
                            ['gamma', self.gamma[i], self.gamma[i+1]],
                            ['cp', self.cp[i], self.cp[i+1]],
                            ['mdot', self.mdot[i], self.mdot[i+1]],
                            ['rho', self.rho[i], self.rho[i+1]],
                            ['V', self.V[i], self.V[i+1]]]  
                else:
                    data = [['p', self.p[i], self.p[i+1]], 
                            ['p0', self.p0[i], self.p0[i+1]],
                            ['T', self.T[i], self.T[i+1]],
                            ['T0', self.T0[i], self.T0[i+1]],
                            ['h', self.h[i], self.h[i+1]],
                            ['h0', self.h0[i], self.h0[i+1]],
                            ['gamma', self.gamma[i], self.gamma[i+1]],
                            ['cp', self.cp[i], self.cp[i+1]],
                            ['mdot', self.mdot[i], self.mdot[i+1]],
                            ['rho', self.rho[i], self.rho[i+1]],
                            ['V', self.V[i], self.V[i+1]]]

                string = string + "\n" + str(self.components[i]) + "\n"
                string = string + str(tabulate(data, headers=['Property', 'Inlet', 'Outlet'])) + "\n"

        else:
            string = "Cycle unsolved. Contains the following components:"
            for i in range(len(self.components)):
                string = string + str(self.components[i]) + "\n"

        return string

    def solve(self):
         # Nozzles can only be at the end
        if sum(isinstance(component, Nozzle) for component in self.components) > 1:
            raise ValueError("Can only have one nozzle per Cycle object, and it must be at the end.")

        for i in range(len(self.components)):
            if type(self.components[i]) is Nozzle and i != len(self.components) - 1:
                raise ValueError(f"Nozzle must be in the last component in the cycle, but it is at index {i} (out of {len(self.components) - 1})")     

        # Properties at each node
        self.p = np.zeros(len(self.components) + 1)         # Static pressure (Pa)
        self.p0 = np.zeros(len(self.components) + 1)        # Stagnation pressure (Pa)
        self.T = np.zeros(len(self.components) + 1)         # Static temperature (K)
        self.T0 = np.zeros(len(self.components) + 1)        # Stagnation temperature (K)
        self.h = np.zeros(len(self.components) + 1)         # Specific enthalpy (J/kg)
        self.h0 = np.zeros(len(self.components) + 1)        # Stagnation specific enthalpy (J/kg)
        self.gamma = np.zeros(len(self.components) + 1)     # Ratio of specific heats, cp/cv
        self.cp = np.zeros(len(self.components) + 1)        # Isobaric specific heat capacity (J/kg/K)
        self.mdot = np.zeros(len(self.components) + 1)      # Mass flow rate (kg/s)
        self.rho = np.zeros(len(self.components) + 1)       # Static density (kg/m3)
        self.V = np.zeros(len(self.components) + 1)         # Flow velocity (m/s)

        # Keep track of bleed air using a dictionary
        self.bleeds = {}

        # Initialise
        self.p[0] = self.p0_in
        self.p0[0] = self.p0_in              
        self.T[0] = self.T0_in         
        self.T0[0] = self.T0_in 
        self.mdot[0] = self.mdot_in   

        if self.model == "perfect":
            self.gamma[:] = GAMMA_AIR
            self.cp[:] = CP_AIR
            self.h[0] = self.cp[0] * self.T[0]
            self.h0[0] = self.h[0]
            self.rho[0] = self.p[0] / (R_AIR * self.T[0])    # p = rho * R * T (ideal gas)

        elif self.model == "cantera":
            self.gas = [None] * (len(self.components) + 1)          # Store cantera.Quantity objects to represent the gas flow at each stage

            gri30 = ct.Solution('gri30.yaml')         

            self.gas[0] = ct.Quantity(gri30, constant = "HP")     # Whenever it needs to mix or equilibrate, it will keep enthalpy and pressure constant
            self.gas[0].mass = self.mdot[0]
            self.gas[0].TPY = self.T0_in, self.p0_in, AIR_COMP

            self.h[0] = self.gas[0].h
            self.h0[0] = self.h[0]
            self.cp[0] = self.gas[0].cp
            self.rho[0] = self.gas[0].density
            self.gamma[0] = self.cp[0] / self.gas[0].cv

        # Run through components and solve
        for i in range(len(self.components)):

            if type(self.components[i]) is Compressor:

                self.p[i+1] = self.p[i] * self.components[i].PR
                self.p0[i+1] = self.p[i+1]
                self.mdot[i+1] = self.mdot[i]

                if self.model == "perfect":

                    if hasattr(self.components[i], 'isen'):
                        T2s = (self.p[i+1] / self.p[i])**((self.gamma[i] - 1) / self.gamma[i]) * self.T[i]
                        self.T[i+1] = self.T[i] + (T2s - self.T[i]) / self.components[i].isen

                    elif hasattr(self.components[i], 'poly'):
                        self.T[i+1] = self.T[i] * (self.p[i+1] / self.p[i])**( (self.gamma[i] - 1) / (self.gamma[i] * self.components[i].poly) )

                    self.T0[i+1] = self.T[i+1]
                    self.h[i+1] = self.cp[i] * self.T[i+1] 
                    self.h0[i+1] = self.h[i+1]
                    R = self.cp[i+1] * (1 - 1/self.gamma[i+1])
                    self.rho[i+1] = self.p[i+1] / (R * self.T[i+1])    # p = rho * R * T (ideal gas)

                elif self.model == "cantera":
                    # Imperfect gas
                    self.gas[i+1] = ct.Quantity(gri30, constant = "HP", mass = self.gas[i].mass)
                    self.gas[i+1].TPY = self.gas[i].TPY     # Copy current gas across before editing
                    
                    if hasattr(self.components[i], 'isen'):
                        h1 = self.gas[i+1].h
                        s1 = self.gas[i+1].s
                        self.gas[i+1].SP = s1, self.p[i+1]
                        h2s = self.gas[i+1].h
                        h2 = h1 + (h2s - h1) / self.components[i].isen
                        self.gas[i+1].HP = h2, self.p[i+1]

                    elif hasattr(self.components[i], 'poly'):
                        # Need to do a numerical integration
                        p_poly_integral = np.linspace(self.p[i], self.p[i+1], N_POLY_INTEGRATION)
                        for j in range(len(p_poly_integral) - 1):

                            h_i = self.gas[i+1].h
                            s_i = self.gas[i+1].s
                            p_ip1 = p_poly_integral[j+1]

                            # Small isentropic compression
                            self.gas[i+1].SP = s_i, p_ip1

                            dhs = self.gas[i+1].h - h_i
                            dh = dhs / self.components[i].poly

                            self.gas[i+1].HP = h_i + dh, p_ip1

                    assert abs(self.gas[i+1].P - self.p[i+1]) < self.p[i+1] * 1e-12, f"Incorrect exit pressure calculated (dp/p = {(self.gas[i+1].P - self.p[i+1])/self.p[i+1]} Pa)"        # Make sure I've ended up at the right pressure

                    self.T[i+1] = self.gas[i+1].T
                    self.T0[i+1] = self.T[i+1]
                    self.h[i+1] = self.gas[i+1].h
                    self.h0[i+1] = self.h[i+1]
                    self.cp[i+1] = self.gas[i+1].cp
                    self.gamma[i+1] = self.cp[i+1] / self.gas[i+1].cv
                    self.rho[i+1] = self.gas[i+1].density

            elif type(self.components[i]) is Mixer:
                if self.model == "perfect":
                    raise ValueError("'Mixer' component does not work for model = 'perfect'. Must use model = 'cantera'.")
                
                # Isenthalpically decompress the inlet fluid
                if self.components[i].fluid.P < self.gas[i].P and (not self.components[i].ignore_p):
                    print(f"Warning: Mixer at position {i} has an inlet fluid with pressure lower than the core cycle fluid ({self.components[i].fluid.P} Pa < {self.gas[i].P} Pa). Will still isenthalpically raise its pressure, but this may not be physically realistic.")

                input_gas = ct.Quantity(gri30, constant = "HP", mass = self.components[i].fluid.mass)
                input_gas.TPY = self.components[i].fluid.TPY

                if self.components[i].ignore_p:
                    input_gas.TP = input_gas.T,  self.gas[i].P

                if self.components[i].ignore_T:
                    input_gas.TP = self.gas[i].T,  input_gas.P

                input_gas.HP = input_gas.h, self.gas[i].P

                # Mix the two fluids
                self.gas[i+1] = self.gas[i] + input_gas

                # Keep track of properties
                self.p[i+1] = self.gas[i+1].P
                self.p0[i+1] = self.p[i+1]
                self.T[i+1] = self.gas[i+1].T
                self.T0[i+1] = self.T[i+1]     
                self.h[i+1] = self.gas[i+1].h
                self.h0[i+1] = self.h[i+1]
                self.cp[i+1] = self.gas[i+1].cp
                self.gamma[i+1] = self.cp[i+1] / self.gas[i+1].cv
                self.rho[i+1] = self.gas[i+1].density
                self.mdot[i+1] = self.gas[i+1].mass

            elif type(self.components[i]) is Equilibrium:    
                if self.model == "perfect":
                    raise ValueError("Mixer component does not work for model = 'perfect'. Must use model = 'cantera'.")

                gas_copy = ct.Quantity(gri30, constant = "HP", mass = self.gas[i].mass)
                gas_copy.TPY = self.gas[i].TPY
                gas_copy.equilibrate(self.components[i].constant)

                # Keep track of properties
                self.gas[i+1] = gas_copy

                self.p[i+1] = self.gas[i+1].P
                self.p0[i+1] = self.p[i+1]
                self.T[i+1] = self.gas[i+1].T
                self.T0[i+1] = self.T[i+1]     
                self.h[i+1] = self.gas[i+1].h
                self.h0[i+1] = self.h[i+1]
                self.cp[i+1] = self.gas[i+1].cp
                self.gamma[i+1] = self.cp[i+1] / self.gas[i+1].cv
                self.rho[i+1] = self.gas[i+1].density
                self.mdot[i+1] = self.mdot[i] 

            elif type(self.components[i]) is Turbine:
                self.p[i+1] = self.p[i] / self.components[i].PR
                self.p0[i+1] = self.p[i+1]
                self.mdot[i+1] = self.mdot[i]

                if self.model == "perfect":
                    if hasattr(self.components[i], 'isen'):
                        T2s = (self.p[i+1] / self.p[i])**((self.gamma[i] - 1) / self.gamma[i]) * self.T[i]
                        self.T[i+1] = self.T[i] + (T2s - self.T[i]) * self.components[i].isen
                    
                    elif hasattr(self.components[i], 'poly'):
                        self.T[i+1] = self.T[i] * (self.p[i+1] / self.p[i])**( (self.gamma[i] - 1) * self.components[i].poly / self.gamma[i] )

                    self.T0[i+1] = self.T[i+1]     
                    self.h[i+1] = self.cp[i] * self.T[i+1]
                    self.h0[i+1] = self.h[i+1]
                    R = self.cp[i+1] * (1 - 1/self.gamma[i+1])
                    self.rho[i+1] = self.p[i+1] / (R * self.T[i+1])    # p = rho * R * T (ideal gas)

                elif self.model == "cantera":
                    # Copy current gas across before editing
                    self.gas[i+1] = ct.Quantity(gri30, constant = "HP", mass = self.gas[i].mass)
                    self.gas[i+1].TPY = self.gas[i].TPY

                    if hasattr(self.components[i], 'isen'):
                        h1 = self.gas[i+1].h
                        s1 = self.gas[i+1].s
                        self.gas[i+1].SP = s1, self.p[i+1]
                        h2s = self.gas[i+1].h
                        h2 = h1 + (h2s - h1) * self.components[i].isen
                        self.gas[i+1].HP = h2, self.p[i+1]

                    elif hasattr(self.components[i], 'poly'):
                        
                        # Need to do a numerical integration
                        p_poly_integral = np.linspace(self.p[i], self.p[i+1], N_POLY_INTEGRATION)
                        for j in range(len(p_poly_integral) - 1):
                    
                            h_i = self.gas[i+1].h
                            s_i = self.gas[i+1].s
                            p_ip1 = p_poly_integral[j+1]

                            # Small isentropic expansion
                            self.gas[i+1].SP = s_i, p_ip1

                            dhs = self.gas[i+1].h - h_i
                            dh = dhs *  self.components[i].poly

                            self.gas[i+1].HP = h_i + dh, p_ip1

                    assert abs(self.gas[i+1].P - self.p[i+1]) < self.p[i+1]*1e-12, f"Incorrect exit pressure calculated (dp/p = {(self.gas[i+1].P - self.p[i+1])/self.p[i+1]} Pa)"        # Make sure I've ended up at the right pressure
                    
                    self.T[i+1] = self.gas[i+1].T
                    self.T0[i+1] = self.T[i+1]     
                    self.h[i+1] = self.gas[i+1].h
                    self.h0[i+1] = self.h[i+1]
                    self.cp[i+1] = self.gas[i+1].cp
                    self.gamma[i+1] = self.cp[i+1] / self.gas[i+1].cv
                    self.rho[i+1] = self.gas[i+1].density

            elif type(self.components[i]) is Nozzle:
                # Cannot expand to a pressure above the stagnation pressure
                if self.components[i].type == "CD" and self.components[i].p_e > self.p0[i]:
                    raise ValueError("Nozzle exit pressure (type = 'CD') is above available stagnation pressure.")

                # Isentropic expansion 
                self.p0[i+1] = self.p0[i]
                self.T0[i+1] = self.T0[i]
                self.h0[i+1] = self.h0[i]
                self.mdot[i+1] = self.mdot[i]

                if self.model == "perfect":
                    if self.components[i].type == "CD":
                        # Expansion to ambient pressure
                        self.p[i+1] = self.components[i].p_e
                        M = np.sqrt( 2 / (self.gamma[i+1] - 1) * ((self.p[i+1] / self.p0[i+1]) ** -((self.gamma[i+1] - 1) / self.gamma[i+1]) - 1) )
                        self.T[i+1] = self.T0[i+1] * (1 + (self.gamma[i+1] - 1) / 2 * M**2)**(-1)
                        self.h[i+1] = self.cp[i+1] * self.T[i+1]
                        R = self.cp[i+1] * (1 - 1/self.gamma[i+1])                  # Using R = cp - cv and gamma = cv/cv
                        self.V[i+1] = M * (self.gamma[i+1] * R * self.T[i+1])**0.5  
                        self.rho[i+1] = self.p[i+1] / (R * self.T[i+1])             # p = rho * R * T (ideal gas)

                    elif self.components[i].type == "C":
                        # First expand to exit pressure
                        self.p[i+1] = self.components[i].p_e
                        M = np.sqrt( 2 / (self.gamma[i+1] - 1) * ((self.p[i+1] / self.p0[i+1]) ** -((self.gamma[i+1] - 1) / self.gamma[i+1]) - 1) )

                        # Check if flow is subsonic - if so then use this subsonic solution
                        if M < 1:
                            self.T[i+1] = self.T0[i+1] * (1 + (self.gamma[i+1] - 1) / 2 * M**2)**(-1)
                            self.h[i+1] = self.cp[i+1] * self.T[i+1]
                            R = self.cp[i+1] * (1 - 1/self.gamma[i+1])
                            self.V[i+1] = M * (self.gamma[i+1] * R * self.T[i+1])**0.5  
                            self.rho[i+1] = self.p[i+1] / (R * self.T[i+1])             # p = rho * R * T (ideal gas)

                        # If flow is supersonic, limit it to being sonic at exit
                        else:
                            # Expansion to M = 1
                            self.p[i+1] = self.p0[i+1] * (1 + (self.gamma[i] - 1) / 2) ** (-self.gamma[i] / (self.gamma[i] - 1))
                            self.T[i+1] = self.T0[i+1] * (1 + (self.gamma[i] - 1) / 2) ** (-1)
                            self.h[i+1] = self.cp[i+1] * self.T[i+1]
                            R = self.cp[i+1] * (1 - 1/self.gamma[i+1])
                            self.V[i+1] = (self.gamma[i+1] * R * self.T[i+1])**0.5
                            self.rho[i+1] = self.p[i+1] / (R * self.T[i+1])             # p = rho * R * T (ideal gas)

                    elif self.components[i].type == "subsonic":
                        m_bar_desired =  self.mdot[i+1] * np.sqrt(self.cp[i+1] * self.T0[i+1])/(self.components[i].A_e * self.p0[i+1])
                        M_e = M_sub(m_bar_desired = m_bar_desired, gamma = self.gamma[i+1])
                        self.p[i+1] = compressible_p(p0 = self.p0[i+1], M = M_e, gamma = self.gamma[i+1])
                        self.T[i+1] = compressible_T(T0 = self.T0[i+1], M = M_e, gamma = self.gamma[i+1])
                        self.h[i+1] = self.cp[i+1] * self.T[i+1]

                        R = self.cp[i+1] * (1 - 1/self.gamma[i+1])
                        self.V[i+1] = M_e * (self.gamma[i+1] * R * self.T[i+1]) ** 0.5
                        self.rho[i+1] = self.p[i+1] / (R * self.T[i+1])             # p = rho * R * T (ideal gas)

                elif self.model == "cantera":
                    self.gas[i+1] = ct.Quantity(gri30, constant = "HP", mass = self.gas[i].mass)
                    self.gas[i+1].TPY = self.gas[i].TPY
                    s0 = self.gas[i+1].s

                    if self.components[i].type == "CD":
                        # Expansion to ambient pressure
                        self.p[i+1] = self.components[i].p_e
                        self.gas[i+1].SP = s0, self.components[i].p_e       # Isentropic expansion
                        self.T[i+1] = self.gas[i+1].T
                        self.h[i+1] = self.gas[i+1].h
                        self.cp[i+1] = self.gas[i+1].cp
                        self.gamma[i+1] = self.cp[i+1] / self.gas[i+1].cv
                        self.V[i+1] = np.sqrt(2 * (self.h0[i+1] - self.h[i+1]) )
                        self.rho[i+1] = self.gas[i+1].density

                    elif self.components[i].type == "C":
                        # First check if the flow would be subsonic
                        self.gas[i+1].SP = s0, self.components[i].p_e       # Isentropic expansion

                        self.p[i+1] = self.gas[i+1].P                       # Update all variables
                        self.T[i+1] = self.gas[i+1].T
                        self.h[i+1] = self.gas[i+1].h
                        self.cp[i+1] = self.gas[i+1].cp
                        self.gamma[i+1] = self.cp[i+1] / self.gas[i+1].cv
                        R = self.cp[i+1] * (1 - 1/self.gamma[i+1])
                        self.V[i+1] = np.sqrt(2 * (self.h0[i+1] - self.h[i+1]) )
                        self.rho[i+1] = self.gas[i+1].density

                        # Check if subsonic
                        if self.V[i+1] < np.sqrt(self.gamma[i+1] * R * self.T[i+1]):
                            # Use the subsonic solution (already updated variables earlier)
                            pass    
                        
                        # Otherwise limit the flow to M = 1.
                        else:
                            # Make initial guess assuming perfect gas assumption
                            p_guess = self.p0[i+1] * (1 + (self.gamma[i] - 1) / 2) ** (-self.gamma[i] / (self.gamma[i] - 1))
                        
                            # Guess pressure and solve until velocity is sonic
                            def velocity_error(p):
                                # Update all variables with each guess 
                                # I believe this is frozen flow (not equilibrium), because I don't run gas.equilibriate() 
                                self.p[i+1] = p
                                self.gas[i+1].SP = s0, p                          # Isentropic expansion
                                self.T[i+1] = self.gas[i+1].T
                                self.h[i+1] = self.gas[i+1].h
                                self.cp[i+1] = self.gas[i+1].cp
                                self.gamma[i+1] = self.cp[i+1] / self.gas[i+1].cv
                                R = self.cp[i+1] * (1 - 1/self.gamma[i+1])
                                self.V[i+1] = np.sqrt(2 * (self.h0[i+1] - self.h[i+1]) )
                                self.rho[i+1] = self.gas[i+1].density

                                # Difference between V and sonic velocity - ideally should be zero
                                error = self.V[i+1] - np.sqrt(self.gamma[i+1] * R * self.T[i+1])
                                return error       

                            # Solve until exit velocity is sonic - variables will automatically be updated by the function
                            scipy.optimize.root_scalar(velocity_error, bracket = [p_guess*0.90, 0.90*p_guess + 0.1*self.p0[i]])

                    elif self.components[i].type == "subsonic":
                        # First find the pressure at which the gas is sonic
                        p_guess = self.p0[i+1] * (1 + (self.gamma[i] - 1) / 2) ** (-self.gamma[i] / (self.gamma[i] - 1))    # Iinitial guess assuming perfect gas assumption
                    
                        # Guess pressure and solve until velocity is sonic - same as for nozzle type = 'C'
                        def velocity_error_sonic(p):
                            self.gas[i+1].SP = s0, p                          # Isentropic expansion
 
                            gamma = self.gas[i+1].cp / self.gas[i+1].cv
                            R = self.gas[i+1].cp * (1 - 1 / gamma)
                            V = np.sqrt(2 * (self.h0[i+1] - self.gas[i+1].h) )
                            rho = self.gas[i+1].density

                            # Difference between V and sonic velocity - ideally should be zero
                            error = V - np.sqrt(gamma * R * self.gas[i+1].T)
                            return error       

                        # Solve until exit velocity is sonic - variables will automatically be updated by the function
                        sonic_sol = scipy.optimize.root_scalar(velocity_error_sonic, bracket = [p_guess*0.90, 0.90*p_guess + 0.1*self.p0[i]])
                        p_sonic = sonic_sol.root

                        V_sonic = np.sqrt(2 * (self.h0[i+1] - self.gas[i+1].h) )
                        A_sonic = self.mdot[i+1] / (self.gas[i+1].density * V_sonic)

                        if A_sonic > self.components[i].A_e:
                            raise ValueError("Nozzle exit area is smaller than the area that would choke the flow. You must make the exit area larger, or reduce the mass flow rate.")

                        # Now we know the exit pressure must be above this, for the gas to be subsonic
                        # Initial guess for actual velocity using perfect gas solution
                        m_bar_guess =  self.mdot[i] * np.sqrt(self.cp[i] * self.T0[i])/(self.components[i].A_e * self.p0[i])
                        try:
                            M_guess = M_sub(m_bar_desired = m_bar_guess, gamma = self.gamma[i])
                            p_guess = compressible_p(p0 = self.p0[i+1], M = M_guess, gamma = self.gamma[i])
                        
                        # Can sometimes fail when we are close to M = 1.
                        except ValueError as e:
                            if "M_sub(): No solution for Mach number. If using a Nozzle, the area is likely too small for the given mass flow rate." in str(e):
                                # Estimate static pressure assuming incompressible flow
                                V_guess = self.mdot[i+1] / (self.rho[i] * self.components[i].A_e)   # Continuinty using stagnation density
                                p_guess = self.p0[i+1] - 0.5 * self.rho[i] * V_guess**2             # Bernoulli

                            else:
                                raise ValueError(str(e))

                        # Refine using semi-perfect gas model, flow area matches that desired
                        def area_error(p):
                            self.gas[i+1].SP = s0, p                          # Isentropic expansion

                            V = np.sqrt(2 * (self.h0[i+1] - self.gas[i+1].h) )
                            A = self.mdot[i+1] / (self.gas[i+1].density * V)
                            error = A - self.components[i].A_e

                            return error
                        
                        subsonic_sol = scipy.optimize.root_scalar(area_error, bracket = [p_sonic, (1.0 - 1e-12) * self.p0[i+1]])
                        p_subsonic = subsonic_sol.root

                        self.gas[i+1].SP = s0, p_subsonic

                        self.p[i+1] = self.gas[i+1].P
                        self.T[i+1] = self.gas[i+1].T
                        self.h[i+1] = self.gas[i+1].h
                        self.cp[i+1] = self.gas[i+1].cp
                        self.gamma[i+1] = self.cp[i+1] / self.gas[i+1].cv
                        self.rho[i+1] = self.gas[i+1].density
                        self.V[i+1] = self.mdot[i+1] / (self.components[i].A_e * self.gas[i+1].density)

                        # Check that we converged on the subsonic and not supersonic solution
                        R = self.cp[i+1] * (1 - 1/self.gamma[i+1])
                        if self.V[i+1] / np.sqrt(self.gamma[i+1] * R * self.T[i+1]) > 1.0:
                            raise ValueError("Converged on supersonic solution in nozzle, instead of subsonic")

            elif type(self.components[i]) is BleedOut:
                if self.model == "perfect":
                    raise ValueError("BleedOut not currently implemented for model = 'perfect'. Must use model = 'cantera'.")
                
                if self.components[i].id in self.bleeds.keys():
                    raise ValueError(f"A BleedOut object already exists with the ID '{self.components[i].id}' - two BleedOuts cannot have same ID.")

                # Keep track of how much mass flow rate was taken out, and its condition
                gas_copy = ct.Quantity(gri30, constant = "HP", mass = self.components[i].mdot_out)
                gas_copy.TPY = self.gas[i].TPY 
                self.bleeds[self.components[i].id] = {"mdot_out" : self.components[i].mdot_out, "mdot_in" : [], "gas" : gas_copy}

                # Remove mass flow rate from cycle and update properties
                self.mdot[i+1] = self.mdot[i] - self.components[i].mdot_out
                self.gas[i+1] = ct.Quantity(gri30, constant = "HP")
                self.gas[i+1].TPY = self.gas[i].TPY
                self.gas[i+1].mass = self.mdot[i+1]

                self.p[i+1] = self.gas[i+1].P
                self.p0[i+1] = self.p[i+1]
                self.T[i+1] = self.gas[i+1].T
                self.T0[i+1] = self.T[i+1]     
                self.h[i+1] = self.gas[i+1].h
                self.h0[i+1] = self.h[i+1]
                self.cp[i+1] = self.gas[i+1].cp
                self.gamma[i+1] = self.cp[i+1] / self.gas[i+1].cv
                self.rho[i+1] = self.gas[i+1].density

            elif type(self.components[i]) is BleedIn:

                if not self.components[i].id in self.bleeds.keys():
                    raise ValueError(f"BleedOut at index {i} has ID '{self.components[i].id}', but there is no BleedOut object with the same ID.")

                # Keep track of how much mass flow rate we put back in
                self.bleeds[self.components[i].id]["mdot_in"].append(self.components[i].mdot_in)

                # Check to make sure we're not putting more air back in than was available
                mdot_out = self.bleeds[self.components[i].id]["mdot_out"]
                mdot_in_tot = sum(self.bleeds[self.components[i].id]["mdot_in"])

                if mdot_in_tot > mdot_out:
                    raise ValueError(f"Bleed with ID '{self.components[i].id}' has more returning to the cycle than leaving (mdot_out = {mdot_out} kg/s, mdot_in = {mdot_in_tot} kg/s")

                # Use cantera to mix the gases
                inlet_gas = ct.Quantity(gri30, constant = "HP", mass = self.components[i].mdot_in)
                inlet_gas.TPY = self.bleeds[self.components[i].id]["gas"].TPY
                inlet_gas.HP = inlet_gas.h, self.gas[i].P                       # Isenthalpic decompression through a valve

                main_gas = ct.Quantity(gri30, constant = "HP", mass = self.gas[i].mass)
                main_gas.TPY = self.gas[i].TPY 

                self.gas[i+1] = main_gas + inlet_gas

                # Add mass flow rate back into cycle and update properties
                self.mdot[i+1] = self.mdot[i] + self.components[i].mdot_in

                self.p[i+1] = self.gas[i+1].P
                self.p0[i+1] = self.p[i+1]
                self.T[i+1] = self.gas[i+1].T
                self.T0[i+1] = self.T[i+1]     
                self.h[i+1] = self.gas[i+1].h
                self.h0[i+1] = self.h[i+1]
                self.cp[i+1] = self.gas[i+1].cp
                self.gamma[i+1] = self.cp[i+1] / self.gas[i+1].cv
                self.rho[i+1] = self.gas[i+1].density

            elif type(self.components[i]) is HeatExchanger:

                # Account for pressure drop (HeatExchanger.dp or HeatExchanger.PR must be input as a function of mass flow rate)
                if hasattr(self.components[i], "dp"):
                    self.p[i+1] = self.p[i] - self.components[i].dp(self.mdot[i]) 
                elif hasattr(self.components[i], "PR"):
                    self.p[i+1] = self.p[i] * self.components[i].PR(self.mdot[i]) 
                else:
                    raise ValueError("Cycle.sovle(): Error with heat exchanger attributes")

                self.p0[i+1] = self.p[i+1]

                if self.model == "perfect":
                    # Increase temperature
                    if hasattr(self.components[i], "Qdot"):
                        self.T[i+1] = self.T[i] + self.components[i].Qdot/(self.mdot[i] * self.cp[i])
                    elif hasattr(self.components[i], "T_out"):
                        self.T[i+1] = self.components[i].T_out
                    else:
                        raise ValueError("Cycle.sovle(): Error with heat exchanger attributes")
                    self.T0[i+1] = self.T[i+1]

                    # Update other properties
                    R = self.cp[i+1] * (1 - 1/self.gamma[i+1])
                    self.rho[i+1] = self.p[i+1] / (R * self.T[i+1]) 
                    self.h[i+1] = self.cp[i+1] * self.T[i+1]
                    self.h0[i+1] = self.cp[i+1] * self.T0[i+1]
                    self.mdot[i+1] = self.mdot[i] 

                elif self.model == "cantera":
                    # Update properties
                    self.gas[i+1] = ct.Quantity(gri30, constant = "HP")
                    self.gas[i+1].TPY = self.gas[i].T, self.p[i+1], self.gas[i].Y   # Take into account pressure drop
                    self.gas[i+1].mass = self.mdot[i]

                    # If Qdot given - specific enthalpy rises by Qdot/mdot
                    if hasattr(self.components[i], "Qdot"):
                        self.gas[i+1].HP = (self.gas[i+1].h + self.components[i].Qdot/self.gas[i+1].mass), self.gas[i+1].P
                    # If T_out given - just set the outlet temperature
                    elif hasattr(self.components[i], "T_out"):
                        self.gas[i+1].TP = self.components[i].T_out, self.gas[i+1].P
                    else:
                        raise ValueError("Cycle.sovle(): Error with heat exchanger attributes")

                    # Update everything else
                    self.T[i+1] = self.gas[i+1].T
                    self.T0[i+1] = self.T[i+1]     
                    self.h[i+1] = self.gas[i+1].h
                    self.h0[i+1] = self.h[i+1]
                    self.cp[i+1] = self.gas[i+1].cp
                    self.gamma[i+1] = self.cp[i+1] / self.gas[i+1].cv
                    self.rho[i+1] = self.gas[i+1].density
                    self.mdot[i+1] = self.mdot[i] 

            elif type(self.components[i]) is ChangeProperties:
                if self.model != "perfect":
                    raise ValueError("ChangeProperties component only works with model = 'perfect'. ")
                
                # Change cp, gamma and mdot if requested
                if not (self.components[i].new_cp is None):
                    self.cp[i+1:] = self.components[i].new_cp

                if not (self.components[i].new_gamma is None):
                   self.gamma[i+1:] = self.components[i].new_gamma             

                if not (self.components[i].new_mdot is None):
                   self.mdot[i+1] = self.components[i].new_mdot  

                else:
                    self.mdot[i+1] = self.mdot[i] 

                # Leave all the other properties unchanged
                self.p[i+1] = self.p[i]
                self.p0[i+1] = self.p0[i]
                self.T[i+1] = self.T[i]
                self.T0[i+1] = self.T0[i]     
                self.h[i+1] = self.cp[i+1] * self.T[i+1]    # Enthalpies before and after are incomparable
                self.h0[i+1] = self.h[i+1]
                self.rho[i+1] = self.rho[i]
                

            else:
                raise ValueError(f"Component of type {type(self.components[i])} is not recognised as a Cycle component")

        # Keep track of if we've solved the system
        self.solved = True

    def plot(self, axes = "TS"):
        """Plot the cycle on a given set of axes. Note that you must run matplotlib.pyplot.show() after this function, to display the plot.

        Args:
            axes (str, optional): Axes to use, currently the only option is "TS". Defaults to "TS".
        """
        if axes == "TS":
            
            s = np.zeros(len(self.T))

            air = ct.Solution('gri30.yaml')        # Choose the starting entropy to be the same as what cantera would choose
            air.TPY = self.T[0], self.p[0], AIR_COMP
            s[0] = air.s

            if self.model == "perfect":
                for i in range(len(s) - 1):
                    R = self.cp[i] * (1 - 1/ self.gamma[i])
                    s[i+1] = s[i] + self.cp[i] * np.log(self.T[i+1] / self.T[i]) - R * np.log(self.p[i+1] / self.p[i])

            if self.model == "cantera":
                print(f"Warning: When model = 'cantera' is used, the T-S diagrams are plotted by assuming air throughout.")
                air = ct.Solution('gri30.yaml')         # Cantera Solution object that will be edited to calculate properties
                air.Y = AIR_COMP

                for i in range(len(self.T)):
                    air.TP = self.T[i], self.p[i]
                    s[i] = air.s

                T_bleed_out = {}
                s_bleed_out = {}
                T_bleed_in = {}
                s_bleed_in = {}

                for i in range(len(self.components)):
                    if type(self.components[i]) is BleedOut:
                        T_bleed_out[self.components[i].id] = self.T[i]
                        s_bleed_out[self.components[i].id] = s[i]
                    
                    elif type(self.components[i]) is BleedIn:
                        if self.components[i].id in T_bleed_in.keys():
                            T_bleed_in[self.components[i].id].append(self.T[i+1])
                            s_bleed_in[self.components[i].id].append(s[i+1]) 
                        else:
                            T_bleed_in[self.components[i].id] = [ self.T[i+1] ]
                            s_bleed_in[self.components[i].id] = [ s[i+1] ]  

                # Plot air bleed paths
                if len(T_bleed_out.keys()) != 0:
                    plt.plot(s[0], self.T[0], label = "Bleed paths", color = "C0", linestyle = "--")

                    for i in range(len(T_bleed_out.keys())):
                        key = list(T_bleed_out.keys())[i]

                        for j in range(len(T_bleed_in[key])):
                            T_to_plot = [ T_bleed_out[key], T_bleed_in[key][j] ]
                            s_to_plot = [ s_bleed_out[key], s_bleed_in[key][j] ]
                            plt.plot(s_to_plot, T_to_plot, color = "C0", linestyle = "--")

            # Plot main cycle
            plt.plot(s, self.T, label = "Main cycle")
            plt.scatter(s, self.T, s = 20, marker = "o")

            # Labels
            plt.xlabel("Specific entropy, s (J/kg/K)")
            plt.ylabel("Temperature, T (K)")
            plt.legend()
            plt.grid()
        
        else:
            raise ValueError("Only 'TS' option is currently allowed for 'axes' input.")


class Compressor:
    """Object for representing a compressor. Can choose to input either an isentropic or polytropic efficiency.

    Args:
        PR (float): Pressure ratio (exit pressure / inlet pressure).

    Keyword Args:
        isen (float): Isentropic efficiency of the compressor.
        poly (float): Polytropic efficiency of the compressor.
    """
    def __init__(self, PR, **kwargs):
        self.PR = PR        # Pressure ratio (P2/P1)

        if "isen" in kwargs and "poly" in kwargs:
            raise ValueError("Cannot input both polytropic and isentropic efficiencies - must choose one ('poly' or 'isen' input).")

        if "isen" in kwargs:
            self.isen = kwargs["isen"]      # Isentropic efficiency (turbine definition)

        elif "poly" in kwargs:
            self.poly = kwargs["poly"]      # Polytropic efficiency (turbine definition)

        else:
            raise ValueError("Must input a compressor effiency - either polytropic ('poly') or isentropic ('isen').")

    def __str__(self):
        return f"Compressor (isen = {self.isen}, PR = {self.PR})"



class Mixer:
    """
    Class for adding a fluid into a gas turbine cycle, which will then be mixed with the core fluid (at constant enthalpy and pressure).

    Note:
        Do not forget to set the 'mass' property for the cantera.Quantity, as this will be used for mixing. The object's temperature/pressure will also be used.

    Args:
        fluid (cantera.Quantity): Cantera.Quantity object to represent the fluid entering. Will be isenthalpically depressurised through a valve before mixing.
        ignore_T (bool): Whether or not to ignore the 'fluid' argument's temperature, and just overwrite it with the main cycle temperature.
        ignore_p (bool): Whether or not to ignore the 'fluid' argument's pressure, and just overwrite it with the main cycle temperature.
        
    """
    def __init__(self, fluid, ignore_T = False, ignore_p = False):

        if type(fluid) is not ct.Quantity:
            raise ValueError("'fluid' argument must be a cantera.Quantity object.")

        if fluid.report()[3:8] != "gri30":
            raise ValueError("Must use 'gri30.yaml' phase model for 'fluid' argument.")

        self.fluid = fluid
        self.ignore_T = ignore_T
        self.ignore_p = ignore_p


class Equilibrium:
    """Upon passing through this, a fluid will be forced to reach its equilibrium composition.

    Args:
        constant (str): What is to be held constant whilst reaching equilibrium, e.g. 'HP' for isenthalpic combustion, 'TP' to equilibrate at current temperature and pressure. See Cantera documentation for options. Defaults to 'HP'.
    """
    def __init__(self, constant = 'HP'):
        self.constant = constant


class Turbine:
    """Object for representing a turbine. Can choose to input either an isentropic or polytropic efficiency.

    Args:
        PR (float): Pressure ratio (inlet pressure / exit pressure).

    Keyword Args:
        isen (float): Isentropic efficiency of the turbine.
        poly (float): Polytropic efficiency of the turbine.
    """
    def __init__(self, PR, **kwargs):

        self.PR = PR        # Pressure ratio (P1/P2)

        if "isen" in kwargs and "poly" in kwargs:
            raise ValueError("Cannot input both polytropic and isentropic efficiencies - must choose one ('poly' or 'isen' input).")

        if "isen" in kwargs:
            self.isen = kwargs["isen"]      # Isentropic efficiency (turbine definition)

        elif "poly" in kwargs:
            self.poly = kwargs["poly"]      # Polytropic efficiency (turbine definition)

        else:
            raise ValueError("Must input a turbine effiency - either polytropic ('poly') or isentropic ('isen').")

    def __str__(self):
        return f"Turbine (isen = {self.isen}, PR = {self.PR})"


class Nozzle:
    """Class for representing a Nozzle object.

    Args:
        type (str, optional): Type of nozzle. 'C' for converging, 'CD' for con-di, 'subsonic' for a subsonic-unchoked nozzle. Defaults to "C".

    Keyword Args:
        p_e (float, optional): Exit plane pressure (Pa). If type = "CD", the flow is isentropically expanded to this pressure.
        p_amb (float, option): Ambient pressure (Pa). If type == 'C', we try to isentropically expand the flow to this pressure, but cap the flow to M = 1. 
        A_e (float, optional): Exit area (m2). If type = "subsonic", this area is used to calculate the exit velocity. There will be a subsonic and supersonic solution - the subsonic one is used.

    """
    def __init__(self, type = "C", **kwargs):
        self.type = type    

        if type == "C":
            if not ("p_amb" in kwargs):
                raise ValueError("Need to input ambient pressure 'p_amb' with nozzle type = 'C'. This is to check for a potential subsonic solution.")
            else:
                self.p_e = kwargs["p_amb"]  

        elif type == "CD":
            self.p_e = kwargs["p_e"]  

        elif type == "subsonic":
            self.A_e = kwargs["A_e"]

        else:
            raise ValueError("Nozzle 'type' must be 'C' (for converging), 'CD' (for converging-diverging) or 'subsonic' for a subsonic nozzle with a given area.")


    def __str__(self):
        if self.type == "CD":
            return f"Nozzle (type = {self.type}, p_e = {self.p_e})"

        elif self.type == "subsonic":
            return f"Nozzle (type = {self.type}, A_e = {self.A_e})"

        else:
            return f"Nozzle (type = {self.type})"

class BleedOut:
    def __init__(self, mdot_out, id = "example"):
        if not (type(id) is str):
            raise ValueError("'id' input for 'BleedOut must be a string")

        self.mdot_out = mdot_out         # Mass flow rate of fluid that is bled out (kg/s)
        self.id = id                    # Identifying string - can be used to with a BleedIn object to bring the flow back into the system.


class BleedIn:
    def __init__(self, mdot_in, id = "example"):
        if not (type(id) is str):
            raise ValueError("'id' input for 'BleedIn must be a string")

        self.mdot_in = mdot_in         # Mass flow rate of fluid that brought back in (kg/s) - must be less than or equal to the amount that was withdrawn.
        self.id = id                    # Identifying string corresponding to the BleedOut object where the fluid was extracted.


class HeatExchanger:
    def __init__(self, **kwargs):
        """Object for representing heat exchangers. Note that either PR or dp can be given, but not both. By default, there is no pressure drop across the heat exchanger.

        Keyword Args:
            Qdot (float): Heat transfer rate - positive for heat going into the cycle (W)
            T_out (float): Outlet temperature of the heat exchanger (for the cycle side) (K)
            PR (float, int or function): Pressure ratio (P_out / P_in) to account for pressure losses, can be given as a fixed value or a function of mass flow rate (in the form func(mdot)).  Defaults to 1.
            dp (float, int or function): Pressure drop across the heat exchanger defined by (dp = P_in - P_out) (Pa). Can be given as a fixed value or a function of mass flow rate (in the form func(mdot)). 

        """
        if "dp" in kwargs and "PR" in kwargs:
            raise ValueError("Cannot input both 'dp' and 'PR' for a HeatExchanger - must choose one.")

        elif "dp" in kwargs:
            if type(kwargs["dp"]) is float or type(kwargs["dp"]) is int:
                if kwargs["dp"] < 0:
                    print("Warning: Your HeatExchanger pressure drop (dp) is negative. This means there is a pressure rise across the heat exchanger (instead of a pressure drop).")
                self.dp = lambda mdot: kwargs["dp"]               # Convert fixed value to a function

            elif callable(kwargs["dp"]):
                self.dp = kwargs["dp"]  

            else:
                raise ValueError("'dp' input for heat exchanger must be a float, int or function of the mass flow rate (i.e. it will be caused using dp(mdot)). Leave blank to default to zero pressure drop.")

        elif "PR" in kwargs:
            if type(kwargs["PR"]) is float or type(kwargs["PR"]) is int:
                if kwargs["PR"] > 1:
                    print("Warning: Your HeatExchanger pressure ratio (PR) is greater than 1. This means there is a pressure rise across the heat exchanger (instead of a pressure drop).")
                self.PR = lambda mdot: kwargs["PR"]               # Convert fixed value to a function

            elif callable(kwargs["PR"]):
                self.PR = kwargs["PR"]  

            else:
                raise ValueError("'PR' input for heat exchanger must be a float, int or function of the mass flow rate (i.e. it will be caused using dp(mdot)). Leave blank to default to zero pressure drop.")

        else:
            raise ValueError("Must input either 'dp' and 'PR' for a HeatExchanger.")

              
        if "Qdot" in kwargs and "T_out" in kwargs:
            raise ValueError("Cannot input both 'T_out' and 'Qdot' for a HeatExchanger - must choose one.")
        elif "Qdot" in kwargs:
            self.Qdot = kwargs["Qdot"]            
        elif "T_out" in kwargs:
            self.T_out = kwargs["T_out"]            
        else:
            raise ValueError("Must input either 'T_out' and 'Qdot' for a HeatExchanger.")



class ChangeProperties:
    """Class for changing the proprties of a perfect gas in the Cycle object.

    Args:
        new_cp (float, optional): New isobaric heat capacity to use. Defaults to None (in which case it is unchanged).
        new_gamma (float, optional): New ratio of specific heats (cp/cv) to use. Defaults to None (in which case it is unchanged).
        new_mdot (float, optional): New mass flow rate through the cycle. Defaults to None (in which case it is unchanged).
    """
    def __init__(self, new_cp = None, new_gamma = None, new_mdot = None):
        self.new_cp = new_cp
        self.new_gamma = new_gamma
        self.new_mdot = new_mdot