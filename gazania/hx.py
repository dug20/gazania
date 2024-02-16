"""
References:
[1] - Forced Convection Heat Transfer, Simon Fraser University (https://www.sfu.ca/~mbahrami/ENSC%20388/Notes/Forced%20Convection.pdf)
[2] - Critical Reynolds Number, nuclear-power.com (https://www.nuclear-power.com/nuclear-engineering/fluid-dynamics/reynolds-number/critical-reynolds-number/)
[3] - Dittus-Boelter Equation, nuclear-power.com (https://www.nuclear-power.com/nuclear-engineering/heat-transfer/convection-convective-heat-transfer/dittus-boelter-equation/)
[4] - Darcy friction factor, Wikipedia (https://en.wikipedia.org/wiki/Darcy_friction_factor_formulae#Colebrook%E2%80%93White_equation)
[5] - Fundamentals of Heat and Mass Transfer, Incropera et al.

Notes:
    - (Darcy friction factor) = 4 * (Fanning friction factor)
    - Fanning friction factor = Cf (Technically, Cf is defined at the wall I think, whereas Fanning can technically be anywhere in the flow)

"""

import numpy as np
import scipy.optimize


def Cf(type = "modified-reynolds", **kwargs):
    """Get skin friction coefficient, Cf = tau_w / (0.5 * rho * V^2) (equal to the Fanning friction factor, or 0.25 times the Darcy friction factor).

    Args:
        type (str, optional): Correlation to use. Defaults to 'modified-reynolds'.

    Keyword Args:
        Nu (float): Nusselt number
        Re (float): Reynolds number
        Pr (float): Prandtl number

    Returns:
        float: Skin friction coefficient, Cf
    """

    if type == "modified-reynolds":
        return 2 * kwargs["Nu"] / ( kwargs["Re"] * kwargs["Pr"]**(1/3) )

    elif type == "colebrook-white":

        def darcy_func(f):
            return 1/(f**0.5) + 2 * np.log10( kwargs["roughness"] / (3.7 * kwargs["Dh"]) + 2.51 / (kwargs["ReDh"] * f**0.5) )

        f_darcy = scipy.optimize.fsolve(func = darcy_func, x0 = 0.005)

        return f_darcy / 4

    else:
        raise ValueError(f"type = {type} is not recognised. Check documentation for options.")

def Nu(type = "plate-Q", **kwargs):
    """Get the Nusselt number (Nu_L = h * L / k). The following inputs are needed for each 'type':
        - plate-lam-Q: Pr, Rex
        - plate-lam-T: Pr, Rex
        - plate-turb-Q: Pr, Rex
        - plate-turb-T: Pr, Rex
        - plate-T: Pr, Rex (Rex_transition optional)
        - plate-Q: Pr, Rex (Rex_transition optional)
        - pipe-lam-T: 
        - pipe-lam-Q: 
        - pipe-dittus-boelter: ReDh, Pr
        - pipe-sieder-tate: ReDh, Pr, mu_mus
        - pipe-gnielinski: ReDh, Pr, (Cf optional)

    Args:
        type (str, optional): Correlation to use. Defaults to "plate-Q". See code for options.

    Keyword Args:
        Pr (float): Prandtl number.
        Rex (float): Reynolds number based on distance 'x' from the leading edge of a flat plate.
        ReDh (float): Reynolds number based on hydraulic diameter of a pipe.
        mu_mus (float): Ratio of freestream viscosity to wall viscosity - i.e. mu(T = T_inf, p = p_inf) / mu(T = T_wall, p = p_wall ~= p_inf).
    
    Returns:
        float: Nusselt number.
    """
    if type == "plate-lam-Q":
        return 0.453 * kwargs["Pr"]**(1/3) * kwargs["Rex"]**(1/2)               # Constant heat flux, laminar, flat plate

    elif type == "plate-lam-T":   
        return 0.332 * kwargs["Pr"]**(1/3) * kwargs["Rex"]**(1/2)               # Constant wall temperature, laminar, flat plate

    elif type == "plate-turb-Q":
        return 0.0308 * kwargs["Pr"]**(2/3) * kwargs["Rex"]**(4/5)              # Constant heat flux, turbulent, flat plate

    elif type == "plate-turb-T":
        return 0.0296 * kwargs["Pr"]**(2/3) * kwargs["Rex"]**(4/5)              # Constant heat flux, turbulent, flat plate

    elif type == "plate-T":
        # Critical Reynolds number for transition to turbulence [2]
        if "Rex_transition" in kwargs:
            Rex_transition = kwargs["Rex_transition"]           

        else:
            Rex_transition = 500000                                             

        # Choose appropriate laminar or turbulent correlation
        if kwargs["Rex"] < Rex_transition:
            return Nu(type = "plate-lam-T", Rex = kwargs["Rex"], Pr = kwargs["Pr"]) 
        
        else:
            return Nu(type = "plate-turb-T", Rex = kwargs["Rex"], Pr = kwargs["Pr"]) 

    elif type == "plate-Q":
        # Critical Reynolds number for transition to turbulence [2]
        if "Rex_transition" in kwargs:
            Rex_transition = kwargs["Rex_transition"]           

        else:
            Rex_transition = 500000                                             

        # Choose appropriate laminar or turbulent correlation
        if kwargs["Rex"] < Rex_transition:
            return Nu(type = "plate-lam-Q", Rex = kwargs["Rex"], Pr = kwargs["Pr"]) 
        
        else:
            return Nu(type = "plate-turb-Q", Rex = kwargs["Rex"], Pr = kwargs["Pr"]) 


    elif type == "pipe-lam-T":
        return 3.66             # Constant temperature laminar pipe flow [5]

    elif type == "pipe-lam-Q":
        return 4.36             # Constant heat flux laminar pipe flow [5]

    elif type == "pipe-dittus-boelter":
        if kwargs["ReDh"] < 10000:
            print("Warning: Dittus-Boelter should be used with ReDh > 10000")
        
        if kwargs["Pr"] < 0.6 or kwargs["Pr"] > 160:
            print("Warning: Dittus-Boelter should be used with 0.6 < Pr < 160")

        return 0.023 * kwargs["ReDh"]**0.8 * kwargs["Pr"]**0.4                      # Reference [3]
    
    elif type == "pipe-sieder-tate":
        if kwargs["ReDh"] < 10000:
            print("Warning: Sieder-Tate should be used with ReDh > 10000")
        
        if kwargs["Pr"] < 0.7 or kwargs["Pr"] > 16700:
            print("Warning: Sieder-Tate should be used with 0.7 < Pr < 16700")

        return 0.027 * kwargs["ReDh"]**(4/5) * kwargs["Pr"]**(1/3) * kwargs["mu_mus"]**0.14                     # Reference [3]
    
    elif type == "pipe-gnielinski":
        if kwargs["ReDh"] < 3000 or kwargs["ReDh"] > 5e6:
            print("Warning: Gnielisnki should be used with 3000 < ReDh > 5e6")
        
        if kwargs["Pr"] < 0.5 or kwargs["Pr"] > 2000:
            print("Warning: Gnielisnki should be used with 0.5 < Pr < 2000")

        if "Cf" in kwargs:
            f_darcy = 4 * kwargs["Cf"]
        else:
            f_darcy = (0.790 * np.log(kwargs["ReDh"]) - 1.64)**(-2)     # Petukhov equation [5]

        return (f_darcy/8) * (kwargs["ReDh"] - 1000) * kwargs["Pr"] / ( 1 + 12.7 * (f_darcy/8)**(1/2) * (kwargs["Pr"]**(2/3) - 1) ) # Reference [5]

    else:
        raise ValueError(f"type = {type} is not recognised. Check documentation for options.")

def R_conv(h, A):
    """Convective thermal resistance

    Args:
        h (float): Convective heat transfer coefficient
        A (float): 'Wetted area'

    Returns:
        float: Convective thermal resistance
    """
    return 1 / (h * A)

def R_cond_rad(r1, r2, L, k):
    """Conductive thermal resistance, for radial conduction

    Args:
        r1 (float): Inner radius
        r2 (float): Outer radius
        L (float): Length of the cylinder
        k (float): Thermal conductivity of the material

    Returns:
        float: Thermal resistance
    """
    return np.log(r2/r1) / (2 * np.pi * k * L)

def R_cond(A, t, k):
    """Conductive thermal resistance for a flat plate

    Args:
        A (float): Area (perpindicular to heat flow) (m2)
        t (float): Thickness (parallel to heat flow) (m)
        k (float): Thermal conductivity (W/m/K)

    Returns:
        float: Thermal resistance
    """
    return t / (A * k)


class Circuit:
    def __init__(self, T1, T2, R = None):
        """Object for representing and solving thermal cicuits. 

        Args:
            T1 (float): Starting temperature (any unit)
            T2 (float): Final temperature (any unit)
            R (float, optional): List of thermal resistances. Defaults to None.

        Attributes:
            Q (float): Heat transfer rate, positive in the direction T1 --> T2. Equal to (T1 - T2) / sum(R). Must run Circuit.solve() to calculate.
            T (list): List of temperatures at each point. Must run Circuit.solve() to calculate.

        """

        self.T1 = T1
        self.T2 = T2

        if R == None:
            self.R = np.array([])
        else:
            self.R = np.array(R)

    def solve(self):
        self.Q = (self.T1 - self.T2) / sum(self.R)

        R_cumu = np.zeros(len(self.R) + 1)  
        R_cumu[1:] = np.cumsum(self.R)          # First resistance will be zero.

        self.T = self.T1 - self.Q * R_cumu
