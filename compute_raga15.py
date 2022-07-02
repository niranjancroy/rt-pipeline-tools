# Calculate Halpha and Hbeta emissivities from 
# recombination and collisional excitation, 
# based on Raga et al. 2015, RMxAA, 51, 231. 
import numpy as np
import matplotlib 
matplotlib.use('Agg') 
import pylab 
from matplotlib.font_manager import FontProperties 
import sys

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

def Einstein_coeff(u, l, case_A = True):
    # From Raga et al (2015) Table 1
    # Pad the edges so that the energy
    # indices start from 1, not 0. 
    A = np.zeros((6, 6), dtype = np.float64)

    # In s^-1 
    if case_A == True: 
        A[2, 1] = 4.695e8
        A[3, 1] = 5.567e7
        A[4, 1] = 1.279e7
        A[5, 1] = 4.128e6

    A[3, 2] = 4.41e7
    A[4, 2] = 8.419e6
    A[4, 3] = 8.986e6
    A[5, 2] = 2.53e6 
    A[5, 3] = 2.201e6 
    A[5, 4] = 2.699e6

    return A[u, l]

def branch_ratio(u, l, case_A = True):
    denominator = 0.0
    for idx in range(1, u):
        denominator += Einstein_coeff(u, idx, case_A = case_A)

    if denominator > 0.0: 
        return Einstein_coeff(u, l, case_A = case_A) / denominator
    else:
        return 0.0 

def compute_cascade_matrix():
    # Pad the edges so that we can index the energy
    # levels starting from 1 rather than 0 
    C = np.zeros((6, 6), dtype = np.float64)

    for idx in range(6):
        C[idx, idx] = 1.0

    C[2, 1] = (C[2, 2] * branch_ratio(2, 1)) + branch_ratio(2, 1)
    C[3, 2] = (C[3, 3] * branch_ratio(3, 2)) + branch_ratio(3, 2)
    C[3, 1] = (C[3, 2] * branch_ratio(2, 1)) + branch_ratio(3, 1)
    C[4, 3] = (C[4, 4] * branch_ratio(4, 3)) + branch_ratio(4, 3)
    C[4, 2] = (C[4, 3] * branch_ratio(3, 2)) + branch_ratio(4, 2)
    C[4, 1] = (C[4, 2] * branch_ratio(2, 1)) + branch_ratio(4, 1)
    C[5, 4] = (C[5, 5] * branch_ratio(5, 4)) + branch_ratio(5, 4)
    C[5, 3] = (C[5, 4] * branch_ratio(4, 3)) + branch_ratio(5, 3)
    C[5, 2] = (C[5, 3] * branch_ratio(3, 2)) + branch_ratio(5, 2)
    C[5, 1] = (C[5, 2] * branch_ratio(2, 1)) + branch_ratio(5, 1)

    return C

def energy(u, l):
    Ryd_const = 2.1798741e-11  # erg 

    return Ryd_const * ((1.0 / (l ** 2.0)) - (1.0 / (u ** 2.0)))

def q_coeff(k, T):
    # Again pad the energy level index
    # to start from 1, not 0 
    omega_coeff = np.zeros((6, 6), dtype = np.float64)

    omega_coeff[2, 0] = 0.7925 
    omega_coeff[2, 1] = 0.9385 
    omega_coeff[2, 2] = -1.5361 
    omega_coeff[2, 3] = 2.2035
    omega_coeff[2, 4] = -0.5345
    omega_coeff[2, 5] = 0.0409

    omega_coeff[3, 0] = 0.25 
    omega_coeff[3, 1] = 0.2461 
    omega_coeff[3, 2] = -0.3297 
    omega_coeff[3, 3] = 0.3892 
    omega_coeff[3, 4] = -0.0928 
    omega_coeff[3, 5] = 0.0071
    
    omega_coeff[4, 0] = 0.1125 
    omega_coeff[4, 1] = 0.1370 
    omega_coeff[4, 2] = -0.1152 
    omega_coeff[4, 3] = 0.1209 
    omega_coeff[4, 4] = -0.0276 
    omega_coeff[4, 5] = 0.0020 

    omega_coeff[5, 0] = 0.0773 
    omega_coeff[5, 1] = 0.0678 
    omega_coeff[5, 2] = -0.0945 
    omega_coeff[5, 3] = 0.0796 
    omega_coeff[5, 4] = -0.0177 
    omega_coeff[5, 5] = 0.0013 

    omega = 0.0
    log_T = np.log10(T / 1.0e4) 
    for idx in range(0, 6):
        omega += omega_coeff[k, idx] * (log_T ** idx)

    return (8.629e-6 / (2.0 * np.sqrt(T))) * omega * np.exp(- energy(k, 1) / (1.38064852e-16 * T)) 

def q_effective(k, T, case_A = True):
    try: 
        iter(T) 
        input_array = True 
    except TypeError: 
        input_array = False 

    C = compute_cascade_matrix()

    if input_array == True: 
        numerator = np.zeros(len(T), dtype = np.float64) 
    else: 
        numerator = 0.0

    for m in range(k, 6):
        numerator += C[m, k] * q_coeff(m, T)

    denominator = 0.0
    for m in range(1, k):
        denominator += Einstein_coeff(k, m, case_A = case_A)

    if denominator > 0.0:
        return numerator / denominator
    else:
        return 0.0

def alpha_coeff(k, T):
    b_coeff = np.zeros((6, 5), dtype = np.float64)

    b_coeff[1, 0] = -12.8049 
    b_coeff[1, 1] = -0.5323 
    b_coeff[1, 2] = -0.0344 
    b_coeff[1, 3] = -0.0305 
    b_coeff[1, 4] = -0.0017
    
    b_coeff[2, 0] = -13.1119 
    b_coeff[2, 1] = -0.6294 
    b_coeff[2, 2] = -0.0998 
    b_coeff[2, 3] = -0.0327 
    b_coeff[2, 4] = 0.0001
    
    b_coeff[3, 0] = -13.3377 
    b_coeff[3, 1] = -0.7161 
    b_coeff[3, 2] = -0.1435 
    b_coeff[3, 3] = -0.0386 
    b_coeff[3, 4] = 0.0077
    
    b_coeff[4, 0] = -13.5225 
    b_coeff[4, 1] = -0.7928 
    b_coeff[4, 2] = -0.1749 
    b_coeff[4, 3] = -0.0412 
    b_coeff[4, 4] = 0.0154 

    b_coeff[5, 0] = -13.682 
    b_coeff[5, 1] = -0.8629 
    b_coeff[5, 2] = -0.1957 
    b_coeff[5, 3] = -0.0375 
    b_coeff[5, 4] = 0.0199

    log_alpha = 0.0
    log_T = np.log10(T / 1.0e4)
    for idx in range(0, 5):
        log_alpha += b_coeff[k, idx] * (log_T ** idx)

    return 10.0 ** log_alpha

def alpha_effective(k, T, case_A = True):
    try: 
        iter(T) 
        input_array = True 
    except TypeError: 
        input_array = False 

    C = compute_cascade_matrix()

    if input_array == True: 
        numerator = np.zeros(len(T), dtype = np.float64) 
    else: 
        numerator = 0.0

    for m in range(k, 6):
        numerator += C[m, k] * alpha_coeff(m, T)

    denominator = 0.0
    for m in range(1, k):
        denominator += Einstein_coeff(k, m, case_A = case_A)

    if denominator > 0.0:
        return numerator / denominator
    else:
        return 0.0

def raga15_Halpha_col_caseA(T):
    return q_effective(3, T) * Einstein_coeff(3, 2) * energy(3, 2)

def raga15_Hbeta_col_caseA(T):
    return q_effective(4, T) * Einstein_coeff(4, 2) * energy(4, 2)

def raga15_Halpha_rec_caseA(T):
    return alpha_effective(3, T) * Einstein_coeff(3, 2) * energy(3, 2)

def raga15_Hbeta_rec_caseA(T):
    return alpha_effective(4, T) * Einstein_coeff(4, 2) * energy(4, 2)

def raga15_Halpha_col_caseB(T):
    return q_effective(3, T, case_A = False) * Einstein_coeff(3, 2, case_A = False) * energy(3, 2)

def raga15_Hbeta_col_caseB(T):
    return q_effective(4, T, case_A = False) * Einstein_coeff(4, 2, case_A = False) * energy(4, 2)

def raga15_Halpha_rec_caseB(T):
    return alpha_effective(3, T, case_A = False) * Einstein_coeff(3, 2, case_A = False) * energy(3, 2)

def raga15_Hbeta_rec_caseB(T):
    return alpha_effective(4, T, case_A = False) * Einstein_coeff(4, 2, case_A = False) * energy(4, 2)

def main():
    outfile = sys.argv[1]
    
    T = 10.0 ** np.arange(3.0, 6.05, 0.1)

    epsilon_Halpha_col = np.zeros(len(T), dtype = np.float64)
    epsilon_Hbeta_col = np.zeros(len(T), dtype = np.float64)
    epsilon_Halpha_rec = np.zeros(len(T), dtype = np.float64)
    epsilon_Hbeta_rec = np.zeros(len(T), dtype = np.float64)

    for i in range(len(T)):
        epsilon_Halpha_col[i] = raga15_Halpha_col_caseA(T[i]) 
        epsilon_Hbeta_col[i] = raga15_Hbeta_col_caseA(T[i])
        epsilon_Halpha_rec[i] = raga15_Halpha_rec_caseA(T[i]) 
        epsilon_Hbeta_rec[i] = raga15_Hbeta_rec_caseA(T[i])

    fontP = FontProperties()
    fontP.set_size(14)

    fig = pylab.figure(figsize = (5.5, 4.5)) 

    ax = pylab.axes([0.13, 0.1, 0.84, 0.87]) 
    ax.plot(T, epsilon_Halpha_col, 'k-', linewidth = 1.8, label = r"$\rm{H}\alpha, \, \rm{col}$")
    ax.plot(T, epsilon_Hbeta_col, 'r-', linewidth = 1.8, label = r"$\rm{H}\beta, \, \rm{col}$")
    ax.plot(T, epsilon_Halpha_rec, 'k--', linewidth = 1.8, label = r"$\rm{H}\alpha, \, \rm{rec}$")
    ax.plot(T, epsilon_Hbeta_rec, 'r--', linewidth = 1.8, label = r"$\rm{H}\beta, \, \rm{rec}$")
    leg = ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0), handletextpad = 0.05, columnspacing = 0.1, ncol=1, prop = fontP, frameon = False)
    pylab.xscale('log')
    pylab.yscale('log')
    pylab.xlim(1.0e3, 1.0e6) 
    pylab.ylim(1.0e-28, 3.0e-20)
    pylab.gca().spines["bottom"].set_linewidth(1.8) 
    pylab.gca().spines["top"].set_linewidth(1.8) 
    pylab.gca().spines["left"].set_linewidth(1.8) 
    pylab.gca().spines["right"].set_linewidth(1.8) 
    ax.xaxis.set_tick_params(width=1.6, length=4.0) 
    ax.xaxis.set_tick_params(which='minor', width=1.4, length=2.3) 
    ax.yaxis.set_tick_params(width=1.6, length=4.0) 
    ax.yaxis.set_tick_params(which='minor', width=1.4, length=2.3)
    pylab.xlabel(r"$T \, (\rm{K})$", fontsize = 16)
    pylab.ylabel(r"$\epsilon \, (\rm{erg} \, \rm{cm}^{3} \, \rm{s}^{-1})$", fontsize = 16)

    pylab.savefig(outfile, dpi = 300)

    pylab.close()

    return

if __name__ == "__main__":
    main()
    
