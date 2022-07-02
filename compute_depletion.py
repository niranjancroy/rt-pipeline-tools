# Contains the depletions of individual elements from Jenkins, 2009, ApJ, 700, 1299
if __name__ == "__main__": 
    import matplotlib 
    matplotlib.use('Agg')
    matplotlib.rcParams['text.usetex'] = True 
    matplotlib.rcParams['xtick.direction'] = 'in' 
    matplotlib.rcParams['xtick.top'] = True 
    matplotlib.rcParams['xtick.bottom'] = True 
    matplotlib.rcParams['xtick.major.top'] = True 
    matplotlib.rcParams['xtick.major.bottom'] = True 
    matplotlib.rcParams['xtick.minor.top'] = True 
    matplotlib.rcParams['xtick.minor.bottom'] = True 
    matplotlib.rcParams['ytick.direction'] = 'in' 
    matplotlib.rcParams['ytick.left'] = True 
    matplotlib.rcParams['ytick.right'] = True 
    matplotlib.rcParams['ytick.major.left'] = True 
    matplotlib.rcParams['ytick.major.right'] = True 
    matplotlib.rcParams['ytick.minor.left'] = True 
    matplotlib.rcParams['ytick.minor.right'] = True

import numpy as np
import pylab 
from matplotlib.font_manager import FontProperties 
import sys
#import my_colour_maps as mcm

# Arrays of the fit coefficients, given in Table 4.
# In the order: Ax, Bx, zx 
# Elements: C, N, O, Mg, Si, P, Cl, Ti, Cr, Mn, Fe,
# Ni, Cu, Zn, Ge, Kr
pars_C = np.array([-0.101, -0.193, 0.803])
pars_N = np.array([0.0, -0.109, 0.55]) 
pars_O = np.array([-0.225, -0.145, 0.598]) 
pars_Mg = np.array([-0.997, -0.800, 0.531]) 
pars_Si = np.array([-1.136, -0.570, 0.305])
pars_P = np.array([-0.945, -0.166, 0.488]) 
pars_Cl = np.array([-1.242, -0.314, 0.609]) 
pars_Ti = np.array([-2.048, -1.957, 0.43]) 
pars_Cr = np.array([-1.447, -1.508, 0.47]) 
pars_Mn = np.array([-0.857, -1.354, 0.52]) 
pars_Fe = np.array([-1.285, -1.513, 0.437]) 
pars_Ni = np.array([-1.490, -1.829, 0.599]) 
pars_Cu = np.array([-0.710, -1.102, 0.711]) 
pars_Zn = np.array([-0.610, -0.279, 0.555]) 
pars_Ge = np.array([-0.615, -0.725, 0.69]) 
pars_Kr = np.array([-0.166, -0.332, 0.684])

# Arrays of the fit coefficents, A2 and B2, in Table 3 of
# De Cia et al., 2016, A&A, 596, 97
DC16_pars_O = np.array([-0.02, -0.15])
DC16_pars_Mg = np.array([-0.03, -0.61])
DC16_pars_Si = np.array([-0.03, -0.63])
DC16_pars_P = np.array([0.01, -0.10])
DC16_pars_S = np.array([-0.04, -0.28])
DC16_pars_Cr = np.array([0.15, -1.32])
DC16_pars_Mn = np.array([0.04, -0.95])
DC16_pars_Fe = np.array([-0.01, -1.26]) 
DC16_pars_Zn = np.array([0.0, -0.27])

# Atomic masses (in the same order as above)
atomic_mass = np.array([12.0, 14.0, 16.0, 24.0, 28.0, 31.0, 35.0, 48.0, 52.0, 55.0, 56.0, 59.0, 64.0, 65.0, 73.0, 84.0])
atomic_mass_S = 32.0 

# Solar abundances, as log_10(NX/NH) + 12
# From Lodders et al. (2003)
#print("WARNING: this dust-to-gas ratio is computed for the Lodders et al. (2003) solar abundances. To calibrate to CHIMES' default abundances, we need to re-scale this to the Wiersma et al. (2009) metallicity.") 
#solar_abundance = np.array([8.46, 7.90, 8.76, 7.62, 7.61, 5.54, 5.33, 5.00, 5.72, 5.58, 7.54, 6.29, 4.34, 4.70, 3.70, 3.36])

# Cloudy default abundances
solar_abundance = 12.0 + np.log10(np.array([2.45e-4, 8.51e-5, 4.90e-4, 3.47e-5, 3.47e-5, 3.20e-7, 1.91e-7, 1.05e-7, 4.68e-7, 2.88e-7, 2.82e-5, 1.78e-6, 1.62e-8, 3.98e-8, 5.01e-9, 2.29e-9]))
solar_abundance_S = 12.0 + np.log10(1.86e-5) 

# Hydrogen mass fraction
XH = 0.7065 

def compute_Fstar(nH):
    # Returns the parameter F_star, as a function
    # nH (in cgs units). Uses the best-fit relation
    # from Fig. 16. 
    Fstar = 0.772 + (np.log10(nH) * 0.461)
    if Fstar > 1.0:
        return 1.0
    else:
        return Fstar

def compute_Fstar_array(nH):
    # As above, but nH and 
    # Fstar are numpy arrays. 
    Fstar = 0.772 + (np.log10(nH) * 0.461)
    Fstar[(Fstar > 1.0)] = 1.0 
    return Fstar 

def element_linear_fit(Fstar, pars, extrapolate = 1):
    # Returns [X_gas / H]fit, as given by equation 10.
    # pars contains the fit coefficients, as given in
    # Table 4.
    Ax = pars[0]
    Bx = pars[1]
    zx = pars[2]

    if extrapolate == 0: 
        # Set all metals to be in the gas phase for Fstar < 0 
        if Fstar < 0.0:
            return 0.0
        else: 
            return Bx + (Ax * (Fstar - zx))
    else:
        # Smoothly extrapolate depltion factors at Fstar < 0
        # until they go to zero
        output = Bx + (Ax * (Fstar - zx))
        if output > 0.0:
            return 0.0
        else:
            return output 

def element_linear_fit_array(Fstar, pars):
    # As above, but Fstar and output 
    # are numpy arrays 
    Ax = pars[0]
    Bx = pars[1]
    zx = pars[2]

    # Smoothly extrapolate depltion factors at Fstar < 0
    # until they go to zero
    output = Bx + (Ax * (Fstar - zx))
    output[(output > 0.0)] = 0.0 
    return output 

def DC16_element_linear_fit(Fstar, pars):
    # Returns [X_gas / H]fit, as given by equation 5 
    # of DC16. 
    A2 = pars[0]
    B2 = pars[1]

    # Note: DC16 always allows Fstar < 0
    Zn_over_Fe = (Fstar + 1.5) / 1.48 
    output = A2 + (B2 * Zn_over_Fe) 
    
    if output > 0.0:
        return 0.0
    else:
        return output 

def DC16_element_linear_fit_array(Fstar, pars):
    # As above, but Fstar and output 
    # are numpy arrays 
    A2 = pars[0]
    B2 = pars[1]

    # Note: DC16 always allows Fstar < 0
    Zn_over_Fe = (Fstar + 1.5) / 1.48 
    output = A2 + (B2 * Zn_over_Fe) 
    output[(output > 0.0)] = 0.0 
    return output 
        
def compute_dust_to_gas_ratio(nH, reference_flag):
    # computes the ratio of dust mass (summing over all
    # of the above elements) to gas mass, for a given nH.

    Fstar = compute_Fstar(nH)
    dust_to_gas = 0.0 

    # Sum dust to gas mass ratios over all elements
    if reference_flag == 0:
        # J09 
        dust_to_gas += atomic_mass[2] * XH * (10.0 ** (solar_abundance[2] - 12.0)) * (1.0 - (10.0 ** element_linear_fit(Fstar, pars_O)))
        dust_to_gas += atomic_mass[3] * XH * (10.0 ** (solar_abundance[3] - 12.0)) * (1.0 - (10.0 ** element_linear_fit(Fstar, pars_Mg))) 
        dust_to_gas += atomic_mass[4] * XH * (10.0 ** (solar_abundance[4] - 12.0)) * (1.0 - (10.0 ** element_linear_fit(Fstar, pars_Si))) 
        dust_to_gas += atomic_mass[5] * XH * (10.0 ** (solar_abundance[5] - 12.0)) * (1.0 - (10.0 ** element_linear_fit(Fstar, pars_P))) 
        dust_to_gas += atomic_mass[8] * XH * (10.0 ** (solar_abundance[8] - 12.0)) * (1.0 - (10.0 ** element_linear_fit(Fstar, pars_Cr))) 
        dust_to_gas += atomic_mass[9] * XH * (10.0 ** (solar_abundance[9] - 12.0)) * (1.0 - (10.0 ** element_linear_fit(Fstar, pars_Mn))) 
        dust_to_gas += atomic_mass[10] * XH * (10.0 ** (solar_abundance[10] - 12.0)) * (1.0 - (10.0 ** element_linear_fit(Fstar, pars_Fe))) 
        dust_to_gas += atomic_mass[13] * XH * (10.0 ** (solar_abundance[13] - 12.0)) * (1.0 - (10.0 ** element_linear_fit(Fstar, pars_Zn)))
    elif reference_flag == 1:
        # DC16 
        # J09 
        dust_to_gas += atomic_mass[2] * XH * (10.0 ** (solar_abundance[2] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_O)))
        dust_to_gas += atomic_mass[3] * XH * (10.0 ** (solar_abundance[3] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_Mg))) 
        dust_to_gas += atomic_mass[4] * XH * (10.0 ** (solar_abundance[4] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_Si))) 
        dust_to_gas += atomic_mass[5] * XH * (10.0 ** (solar_abundance[5] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_P))) 
        dust_to_gas += atomic_mass[8] * XH * (10.0 ** (solar_abundance[8] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_Cr))) 
        dust_to_gas += atomic_mass[9] * XH * (10.0 ** (solar_abundance[9] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_Mn))) 
        dust_to_gas += atomic_mass[10] * XH * (10.0 ** (solar_abundance[10] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_Fe))) 
        dust_to_gas += atomic_mass[13] * XH * (10.0 ** (solar_abundance[13] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_Zn)))
        
        dust_to_gas += atomic_mass_S * XH * (10.0 ** (solar_abundance_S - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_S)))

    # The following are only in J09 
    dust_to_gas += atomic_mass[0] * XH * (10.0 ** (solar_abundance[0] - 12.0)) * (1.0 - (10.0 ** element_linear_fit(Fstar, pars_C))) 
    dust_to_gas += atomic_mass[1] * XH * (10.0 ** (solar_abundance[1] - 12.0)) * (1.0 - (10.0 ** element_linear_fit(Fstar, pars_N))) 
    dust_to_gas += atomic_mass[6] * XH * (10.0 ** (solar_abundance[6] - 12.0)) * (1.0 - (10.0 ** element_linear_fit(Fstar, pars_Cl))) 
    dust_to_gas += atomic_mass[7] * XH * (10.0 ** (solar_abundance[7] - 12.0)) * (1.0 - (10.0 ** element_linear_fit(Fstar, pars_Ti))) 
    dust_to_gas += atomic_mass[11] * XH * (10.0 ** (solar_abundance[11] - 12.0)) * (1.0 - (10.0 ** element_linear_fit(Fstar, pars_Ni))) 
    dust_to_gas += atomic_mass[12] * XH * (10.0 ** (solar_abundance[12] - 12.0)) * (1.0 - (10.0 ** element_linear_fit(Fstar, pars_Cu))) 
    dust_to_gas += atomic_mass[14] * XH * (10.0 ** (solar_abundance[14] - 12.0)) * (1.0 - (10.0 ** element_linear_fit(Fstar, pars_Ge))) 
    dust_to_gas += atomic_mass[15] * XH * (10.0 ** (solar_abundance[15] - 12.0)) * (1.0 - (10.0 ** element_linear_fit(Fstar, pars_Kr))) 
    
    return dust_to_gas
        
def compute_dust_to_gas_ratio_array(nH):
    # As above, but nH is an array. 
    # Only for reference_flag == 1 

    Fstar = compute_Fstar_array(nH)
    dust_to_gas = np.zeros(len(Fstar))  

    # Sum dust to gas mass ratios over all elements
    # DC16 
    dust_to_gas += atomic_mass[2] * XH * (10.0 ** (solar_abundance[2] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit_array(Fstar, DC16_pars_O)))
    dust_to_gas += atomic_mass[3] * XH * (10.0 ** (solar_abundance[3] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit_array(Fstar, DC16_pars_Mg))) 
    dust_to_gas += atomic_mass[4] * XH * (10.0 ** (solar_abundance[4] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit_array(Fstar, DC16_pars_Si))) 
    dust_to_gas += atomic_mass[5] * XH * (10.0 ** (solar_abundance[5] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit_array(Fstar, DC16_pars_P))) 
    dust_to_gas += atomic_mass[8] * XH * (10.0 ** (solar_abundance[8] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit_array(Fstar, DC16_pars_Cr))) 
    dust_to_gas += atomic_mass[9] * XH * (10.0 ** (solar_abundance[9] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit_array(Fstar, DC16_pars_Mn))) 
    dust_to_gas += atomic_mass[10] * XH * (10.0 ** (solar_abundance[10] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit_array(Fstar, DC16_pars_Fe))) 
    dust_to_gas += atomic_mass[13] * XH * (10.0 ** (solar_abundance[13] - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit_array(Fstar, DC16_pars_Zn)))
    
    dust_to_gas += atomic_mass_S * XH * (10.0 ** (solar_abundance_S - 12.0)) * (1.0 - (10.0 ** DC16_element_linear_fit_array(Fstar, DC16_pars_S)))

    # The following are only in J09 
    dust_to_gas += atomic_mass[0] * XH * (10.0 ** (solar_abundance[0] - 12.0)) * (1.0 - (10.0 ** element_linear_fit_array(Fstar, pars_C))) 
    dust_to_gas += atomic_mass[1] * XH * (10.0 ** (solar_abundance[1] - 12.0)) * (1.0 - (10.0 ** element_linear_fit_array(Fstar, pars_N))) 
    dust_to_gas += atomic_mass[6] * XH * (10.0 ** (solar_abundance[6] - 12.0)) * (1.0 - (10.0 ** element_linear_fit_array(Fstar, pars_Cl))) 
    dust_to_gas += atomic_mass[7] * XH * (10.0 ** (solar_abundance[7] - 12.0)) * (1.0 - (10.0 ** element_linear_fit_array(Fstar, pars_Ti))) 
    dust_to_gas += atomic_mass[11] * XH * (10.0 ** (solar_abundance[11] - 12.0)) * (1.0 - (10.0 ** element_linear_fit_array(Fstar, pars_Ni))) 
    dust_to_gas += atomic_mass[12] * XH * (10.0 ** (solar_abundance[12] - 12.0)) * (1.0 - (10.0 ** element_linear_fit_array(Fstar, pars_Cu))) 
    dust_to_gas += atomic_mass[14] * XH * (10.0 ** (solar_abundance[14] - 12.0)) * (1.0 - (10.0 ** element_linear_fit_array(Fstar, pars_Ge))) 
    dust_to_gas += atomic_mass[15] * XH * (10.0 ** (solar_abundance[15] - 12.0)) * (1.0 - (10.0 ** element_linear_fit_array(Fstar, pars_Kr))) 
    
    return dust_to_gas

def plot_dust_to_gas(outfile, reference_flag):
    nH_array = 10.0 ** np.arange(-5.0, 2.0, 0.01)
    dust_to_gas_array = []

    for i in nH_array:
        dust_to_gas_array.append(compute_dust_to_gas_ratio(i, reference_flag))

    print("D/G (saturated) = %.4e" % (dust_to_gas_array[-1], ))

    dust_to_gas_array = np.array(dust_to_gas_array)

    # Normalise to the saturated value 
    dust_to_gas_array /= dust_to_gas_array[-1] 

    pylab.plot(nH_array, dust_to_gas_array, 'k-', linewidth = 1.8, label = r"$\rm{Jenkins} \, (2009)$")
    pylab.xscale('log')
    pylab.xlim(1.0e-5, 1.0e2)
    pylab.xlabel(r"$n_{\rm{H}} \, (\rm{cm}^{-3})$", fontsize = 14)
    pylab.ylabel(r"$[D / G] / [D / G]_{\odot}$", fontsize = 14)
    pylab.savefig(outfile, dpi = 300)
    pylab.close()

    return

def plot_depletion_factors(outfile, reference_flag):

    nH_array = 10.0 ** np.arange(-6.0, 3.01, 0.02) 

    C_depl = []
    N_depl = []
    O_depl = []
    Mg_depl = []
    Si_depl = []
    Fe_depl = [] 

    for nH in nH_array: 
        Fstar = compute_Fstar(nH)

        if reference_flag == 0:
            # Use Jenkins (2009) for all 
            O_depl.append(10.0 ** element_linear_fit(Fstar, pars_O)) 
            Mg_depl.append(10.0 ** element_linear_fit(Fstar, pars_Mg)) 
            Si_depl.append(10.0 ** element_linear_fit(Fstar, pars_Si)) 
            Fe_depl.append(10.0 ** element_linear_fit(Fstar, pars_Fe))
        elif reference_flag == 1:
            # Use De Cia et al. (2016) where possible, otherwise
            # use Jenkins (2009) 
            O_depl.append(10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_O)) 
            Mg_depl.append(10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_Mg)) 
            Si_depl.append(10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_Si)) 
            Fe_depl.append(10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_Fe))
        else:
            print("ERROR: reference_flag %d not recognised. Aborting" % (reference_flag, ))
            return 

        # The following are only in Jenkins (2009) 
        C_depl.append(10.0 ** element_linear_fit(Fstar, pars_C)) 
        N_depl.append(10.0 ** element_linear_fit(Fstar, pars_N)) 

    cols = mcm.viridis_colmap()

    pylab.plot(nH_array, C_depl, '-', color = cols[15], linewidth = 1.8, label = "C") 
    pylab.plot(nH_array, N_depl, '-', color = cols[60], linewidth = 1.8, label = "N") 
    pylab.plot(nH_array, O_depl, '-', color = cols[105], linewidth = 1.8, label = "O") 
    pylab.plot(nH_array, Mg_depl, '-', color = cols[150], linewidth = 1.8, label = "Mg") 
    pylab.plot(nH_array, Si_depl, '-', color = cols[195], linewidth = 1.8, label = "Si") 
    pylab.plot(nH_array, Fe_depl, '-', color = cols[240], linewidth = 1.8, label = "Fe")
    leg1 = pylab.legend(loc='lower left', bbox_to_anchor = (0.0, 0.0), ncol = 1, frameon = False)
    pylab.xscale('log')
    pylab.yscale('log')
    pylab.ylim(4.0e-3, 1.5) 
    pylab.xlabel(r"$n_{\rm{H}} \, (\rm{cm}^{-3})$", fontsize = 14)
    pylab.ylabel(r"$\rm{Depletion} \, \rm{factor}$", fontsize = 14)
    pylab.savefig(outfile, dpi = 300)
    pylab.close()

    return

def plot_combined(outfile, reference_flag):
    nH_array = 10.0 ** np.arange(-6.0, 3.01, 0.02) 

    C_depl = []
    N_depl = []
    O_depl = []
    Mg_depl = []
    Si_depl = []
    Fe_depl = []
    S_depl = [] 

    for nH in nH_array: 
        Fstar = compute_Fstar(nH)

        if reference_flag == 0:
            # Use Jenkins (2009) for all 
            O_depl.append(10.0 ** element_linear_fit(Fstar, pars_O)) 
            Mg_depl.append(10.0 ** element_linear_fit(Fstar, pars_Mg)) 
            Si_depl.append(10.0 ** element_linear_fit(Fstar, pars_Si)) 
            Fe_depl.append(10.0 ** element_linear_fit(Fstar, pars_Fe))

            S_depl.append(1.0) 
        elif reference_flag == 1:
            # Use De Cia et al. (2016) where possible, otherwise
            # use Jenkins (2009) 
            O_depl.append(10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_O)) 
            Mg_depl.append(10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_Mg)) 
            Si_depl.append(10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_Si)) 
            Fe_depl.append(10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_Fe))

            S_depl.append(10.0 ** DC16_element_linear_fit(Fstar, DC16_pars_S)) 
        else:
            print("ERROR: reference_flag %d not recognised. Aborting" % (reference_flag, ))
            return 

        # The following are only in Jenkins (2009) 
        C_depl.append(10.0 ** element_linear_fit(Fstar, pars_C)) 
        N_depl.append(10.0 ** element_linear_fit(Fstar, pars_N))

    # Total dust to gas ratio    
    dust_to_gas_array = []

    for i in nH_array:
        dust_to_gas_array.append(compute_dust_to_gas_ratio(i, reference_flag))

    dust_to_gas_array = np.array(dust_to_gas_array)

    # Normalise to the saturated value 
    dust_to_gas_array /= dust_to_gas_array[-1] 

    cols = mcm.viridis_colmap()
    
    x_min = 1.0e-6
    x_max = 1000.0 
    x_tick = [1.0e-6, 1.0e-4, 0.01, 1.0, 100.0]
    x_tick_minor = [1.0e-5, 1.0e-3, 0.1, 10.0, 1000.0]
    x_tick_labels = [r"$-6$", r"$-4$", r"$-2$", r"$0$", r"$2$"]
    
    y1_min = 10.0 ** (-2.5)
    y1_max = 1.5 
    y1_tick = [0.01, 0.1, 1.0]
    y1_tick_minor = [10.0 ** (-2.5), 10.0 ** (-1.5), 10.0 ** (-0.5)]
    y1_tick_labels = [r"$-2$", r"$-1$", r"$0$"]
    
    y2_min = 0.0
    y2_max = 1.1 
    y2_tick = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    y2_tick_minor = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
    y2_tick_labels = [r"$0.0$", r"$0.2$", r"$0.4$", r"$0.6$", r"$0.8$", r"$1.0$"]
    
    fontP = FontProperties()
    fontP.set_size(14)
    
    fig = pylab.figure(figsize = (6.0, 7.0)) 
    ax1 = pylab.axes([0.12, 0.53, 0.85, 0.45]) 

    ax1.plot(nH_array, C_depl, '-', color = cols[10], linewidth = 1.8, label = r"$\rm{C}$") 
    ax1.plot(nH_array, N_depl, '-', color = cols[50], linewidth = 1.8, label = r"$\rm{N}$") 
    ax1.plot(nH_array, O_depl, '-', color = cols[90], linewidth = 1.8, label = r"$\rm{O}$") 
    ax1.plot(nH_array, Mg_depl, '-', color = cols[130], linewidth = 1.8, label = r"$\rm{Mg}$") 
    ax1.plot(nH_array, Si_depl, '-', color = cols[170], linewidth = 1.8, label = r"$\rm{Si}$") 
    ax1.plot(nH_array, S_depl, '-', color = cols[200], linewidth = 1.8, label = r"$\rm{S}$") 
    ax1.plot(nH_array, Fe_depl, '-', color = cols[240], linewidth = 1.8, label = r"$\rm{Fe}$")
    leg1 = pylab.legend(loc='lower left', bbox_to_anchor = (0.0, 0.0), ncol = 1, prop = fontP, frameon = False)
    pylab.gca().spines["bottom"].set_linewidth(1.8) 
    pylab.gca().spines["top"].set_linewidth(1.8) 
    pylab.gca().spines["left"].set_linewidth(1.8) 
    pylab.gca().spines["right"].set_linewidth(1.8) 
    ax1.xaxis.set_tick_params(width=1.6, length=4.0) 
    ax1.xaxis.set_tick_params(which='minor', width=1.4, length=2.3) 
    ax1.yaxis.set_tick_params(width=1.6, length=4.0) 
    ax1.yaxis.set_tick_params(which='minor', width=1.4, length=2.3)
    pylab.xscale('log')
    pylab.yscale('log')
    pylab.xlim(x_min, x_max)
    pylab.ylim(y1_min, y1_max) 
    ax1.xaxis.set_ticks(x_tick)
    ax1.xaxis.set_ticks(x_tick_minor, minor = True)
    ax1.xaxis.set_ticklabels([], minor = True, fontsize=14)
    pylab.setp(ax1.get_xticklabels(), visible = False) 
    ax1.yaxis.set_ticks(y1_tick)
    ax1.yaxis.set_ticks(y1_tick_minor, minor = True)
    ax1.yaxis.set_ticklabels(y1_tick_labels, fontsize=14)
    pylab.ylabel(r"$\log_{10} [ M_{X}^{\rm{gas}} / M_{X}^{\rm{tot}} ]$", fontsize = 14)

    ax2 = pylab.axes([0.12, 0.08, 0.85, 0.45]) 
    ax2.plot(nH_array, dust_to_gas_array, 'k-', linewidth = 1.8)                
    pylab.gca().spines["bottom"].set_linewidth(1.8) 
    pylab.gca().spines["top"].set_linewidth(1.8) 
    pylab.gca().spines["left"].set_linewidth(1.8) 
    pylab.gca().spines["right"].set_linewidth(1.8) 
    ax2.xaxis.set_tick_params(width=1.6, length=4.0) 
    ax2.xaxis.set_tick_params(which='minor', width=1.4, length=2.3) 
    ax2.yaxis.set_tick_params(width=1.6, length=4.0) 
    ax2.yaxis.set_tick_params(which='minor', width=1.4, length=2.3)
    pylab.xscale('log')
    pylab.xlim(x_min, x_max)
    pylab.ylim(y2_min, y2_max) 
    ax2.xaxis.set_ticks(x_tick)
    ax2.xaxis.set_ticks(x_tick_minor, minor = True)
    ax2.xaxis.set_ticklabels(x_tick_labels, fontsize=14)
    ax2.xaxis.set_ticklabels([], minor = True, fontsize=14)
    ax2.yaxis.set_ticks(y2_tick)
    ax2.yaxis.set_ticks(y2_tick_minor, minor = True)
    ax2.yaxis.set_ticklabels(y2_tick_labels, fontsize=14)
    
    pylab.xlabel(r"$\log_{10} [ n_{\rm{H}} \, (\rm{cm}^{-3}) ]$", fontsize = 14)
    pylab.ylabel(r"$DTM / DTM_{\rm{MW}}$", fontsize = 14)
    pylab.savefig(outfile, dpi = 300)
    pylab.close()

    return
    
def main():
    outfile = sys.argv[1]
    mode = int(sys.argv[2])
    reference_flag = int(sys.argv[3])  # 0 - J09
                                       # 1 - DC16 (where possible) 

    if mode == 0: 
        plot_dust_to_gas(outfile, reference_flag)
    elif mode == 1:
        plot_depletion_factors(outfile, reference_flag)
    elif mode ==2:
        plot_combined(outfile, reference_flag)
    else:
        print("ERROR: mode %d not recognised." % (mode, )) 

    return

if __name__ == "__main__": 
    main()
