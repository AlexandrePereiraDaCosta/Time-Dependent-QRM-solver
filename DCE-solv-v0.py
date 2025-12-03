"""
Created on 27 Nov of 2025
@author: Alexandre Costa

In this code we integrate the coupled diferential equations (DE), derived from the time dependent Rabi Hamiltonian of the    j.physleta.2025.131163. 
The DE are separeted in real and imaginary parts:

    dta_n^(r) = [omega*n - Omega(t)/2]a_n^(i) + g[(n+1)^(1/2)b_(n+1)^(i) + n^(1/2) b_(n-1)^(i)]
    dta_n^(i) = -[omega*n - Omega(t)/2]a_n^(r) - g[(n+1)^(1/2)b_(n+1)^(r) + n^(1/2) b_(n-1)^(r)]
    dtb_n^(r) = [Omega(t)/2 + omega*n]b_n^(i) + g[(n+1)^(1/2)a_(n+1)^(i) + n^(1/2) a_(n-1)^(i)]
    dtb_n^(i) = -[Omega(t)/2 + omega*n]b_n^(r) - g[(n+1)^(1/2)a_(n+1)^(r) + n^(1/2) a_(n-1)^(r)]

From that we compute a couple of observables and save the results in .txt files
"""

import numpy as np
from scipy.integrate import solve_ivp
import math

# -------- parameters ----------
N = 30                     # Fock State
nu = 1.0                   # field frequency
g_1 = 0.05*nu              # coupling strength
Omega_0 = nu - 10.0 * g_1  # base atom frequency
epsilon_0 = 0.08*Omega_0   # modulation amplitude
eta0 = 2.00655*nu          # initial modulation frequency
alpha = -2.0e-8*nu         # modulation frequency drift
tmax = 2.0e4/nu            # max time
TOL = 1e-13                # error tolerance of the solver  
STATE_SIZE = 4 * (N + 1)   # State vector size: 4*(N+1) elements
N1 = N + 1                 # Pre-compute N+1 

# Pre-compute array offsets for direct indexing
OFFSET_A_R = 0
OFFSET_A_I = N1
OFFSET_B_R = 2 * N1
OFFSET_B_I = 3 * N1

# Pre-compute square roots for all n values
sqrt_n = np.sqrt(np.arange(N1, dtype=np.float64))           # sqrt_n[i] = sqrt(i) for i in [0, N]
sqrt_np1 = np.sqrt(np.arange(1, N1 + 1, dtype=np.float64))  # sqrt_np1[i] = sqrt(i+1) for i in [0, N]

# Pre-compute n and n^2 arrays for observables
n_array = np.arange(N1, dtype=np.float64)
n2_array = n_array ** 2

# Initial condition:
Y0 = np.zeros(STATE_SIZE, dtype=np.float64)
Y0[0] = 1.0  # a_0^(r) = 1.0

ic = 1j  # imaginary unit

def fcn(t, Y):
    """
    Compute derivatives of the coupled DE 
    Uses direct array indexing and pre-computed square roots
    """
    YP = np.zeros_like(Y)
    
    etat = eta0 + alpha * t
    omega_t = Omega_0 + epsilon_0 * math.sin(etat * t)
    omega_t_half = omega_t * 0.5 
    
    # Extract state vector slices for direct access
    a_r = Y[OFFSET_A_R:OFFSET_A_R + N1]
    a_i = Y[OFFSET_A_I:OFFSET_A_I + N1]
    b_r = Y[OFFSET_B_R:OFFSET_B_R + N1]
    b_i = Y[OFFSET_B_I:OFFSET_B_I + N1]
    
    # Pre-compute omega terms
    omega_term_a = nu * n_array - omega_t_half  # [omega*n - Omega(t)/2]
    omega_term_b = omega_t_half + nu * n_array   # [Omega(t)/2 + omegan]
    
    # Compute derivatives 
    for n in range(N1):
        # Coupling coeff
        coupling_a_i = 0.0
        coupling_a_r = 0.0
        coupling_b_i = 0.0
        coupling_b_r = 0.0
        
        if n + 1 < N1:
            sqrt_np1_val = sqrt_np1[n]
            coupling_a_i += g_1 * sqrt_np1_val * b_i[n + 1]
            coupling_a_r += g_1 * sqrt_np1_val * b_r[n + 1]
            coupling_b_i += g_1 * sqrt_np1_val * a_i[n + 1]
            coupling_b_r += g_1 * sqrt_np1_val * a_r[n + 1]
        
        if n > 0:
            sqrt_n_val = sqrt_n[n]
            coupling_a_i += g_1 * sqrt_n_val * b_i[n - 1]
            coupling_a_r += g_1 * sqrt_n_val * b_r[n - 1]
            coupling_b_i += g_1 * sqrt_n_val * a_i[n - 1]
            coupling_b_r += g_1 * sqrt_n_val * a_r[n - 1]
        
        YP[OFFSET_A_R + n] = omega_term_a[n] * a_i[n] + coupling_a_i
        YP[OFFSET_A_I + n] = -omega_term_a[n] * a_r[n] - coupling_a_r
        YP[OFFSET_B_R + n] = omega_term_b[n] * b_i[n] + coupling_b_i
        YP[OFFSET_B_I + n] = -omega_term_b[n] * b_r[n] - coupling_b_r
    
    return YP

def run_simulation(max_time_stop=tmax):
    """
    Run the simulation and compute observables
    """
    # Open output files with UTF-8 encoding
    files = {
        '0': open('0.txt', 'w', encoding='utf-8'),
        'n': open('n.txt', 'w', encoding='utf-8'),
        'n2': open('n2.txt', 'w', encoding='utf-8'),
        'E': open('E.txt', 'w', encoding='utf-8'),
        'Mandel': open('Mandel.txt', 'w', encoding='utf-8'),
        'g2': open('g2.txt', 'w', encoding='utf-8'),
        'entropy': open('entropy.txt', 'w', encoding='utf-8'),
        'LEatom': open('LE-atom.txt', 'w', encoding='utf-8'),
        'LEfield': open('LE-field.txt', 'w', encoding='utf-8')
    }
    f0, fn, fn2, fE, fMandel, fg2, fent, fLEatom, fLEfield = (
        files['0'], files['n'], files['n2'], files['E'], files['Mandel'],
        files['g2'], files['entropy'], files['LEatom'], files['LEfield']
    )
    
    # loop variables
    time = 0.0
    tend = 1e-6
    Y = Y0.copy()
    count = 40
    icount = 0
    dt = 0.005
    
    # Pre-allocate arrays for observables computation
    a_r_all = np.empty(N1, dtype=np.float64)
    a_i_all = np.empty(N1, dtype=np.float64)
    b_r_all = np.empty(N1, dtype=np.float64)
    b_i_all = np.empty(N1, dtype=np.float64)
    abs_a_sq = np.empty(N1, dtype=np.float64)  # |a_n|^2
    abs_b_sq = np.empty(N1, dtype=np.float64)  # |b_n|^2
    
    while time < tmax:
        # Integrate
        sol = solve_ivp(fcn, (time, tend), Y, rtol=TOL, atol=TOL, method='RK45')
        if not sol.success:
            print("Integrator failed at t=", time, " message:", sol.message)
            break
        Y = sol.y[:, -1]
        time = tend
        tend += dt
        
        icount += 1
        if icount == count or tend < 0.01:
            icount = 0
            
            # Extract all state values
            a_r_all[:] = Y[OFFSET_A_R:OFFSET_A_R + N1]
            a_i_all[:] = Y[OFFSET_A_I:OFFSET_A_I + N1]
            b_r_all[:] = Y[OFFSET_B_R:OFFSET_B_R + N1]
            b_i_all[:] = Y[OFFSET_B_I:OFFSET_B_I + N1]
            
            # Pre-compute |a_n|^2 and |b_n|^2
            abs_a_sq[:] = a_r_all**2 + a_i_all**2
            abs_b_sq[:] = b_r_all**2 + b_i_all**2
            
            # Compute single-index observables
            pe = np.sum(abs_b_sq)  # excited state probability
            pg = np.sum(abs_a_sq)  # ground state probability
            fotons = np.sum(n_array * (abs_a_sq + abs_b_sq))  # mean photon number
            var = np.sum(n2_array * (abs_a_sq + abs_b_sq))  # variance
            
            # Compute c2 (correlation term)
            # Pre-compute complex values for observables
            a_complex_all = a_r_all + ic * a_i_all
            a_complex_conj_all = a_r_all - ic * a_i_all
            b_complex_all = b_r_all + ic * b_i_all
            b_complex_conj_all = b_r_all - ic * b_i_all
            
            c2 = np.sum(a_complex_all * b_complex_conj_all)
            c2abs = abs(c2)**2
            
            # Compute field purity (purf)
            purf = 0.0
            
            for i in range(N1):
                abs_a_i_sq = abs_a_sq[i]
                abs_b_i_sq = abs_b_sq[i]
                a_i_c = a_complex_all[i]
                b_i_cc = b_complex_conj_all[i]
                
                for j in range(N1):
                    # hel expression
                    hel = a_i_c * a_complex_conj_all[j] * b_i_cc * b_complex_all[j]
                    
                    purf += abs_a_i_sq * abs_a_sq[j]
                    purf += abs_b_i_sq * abs_b_sq[j]
                    purf += 2.0 * hel.real
            
            # Avoid negative rounding issues under sqrt
            sqrt_arg = max(0.0, (pe - pg)**2 + 4.0 * c2abs)
            sqrt_val = math.sqrt(sqrt_arg)
            lap = 0.5 * (pe + pg + sqrt_val)
            lam = 0.5 * (pe + pg - sqrt_val)
            
            # Write outputs
            f0.write(f"{time} {1.0 - pe - pg}\n")
            fn.write(f"{time} {fotons}\n")
            fn2.write(f"{time} {var}\n")
            fE.write(f"{time} {pe}\n")
            
            # Mandel and g2 may be invalid if fotons==0 -> guard
            if fotons != 0.0:
                fotons_sq = fotons * fotons
                fMandel.write(f"{time} {(var - fotons_sq - fotons) / fotons}\n")
                fg2.write(f"{time} {(var - fotons) / fotons_sq}\n")
            else:
                fMandel.write(f"{time} 0.0\n")
                fg2.write(f"{time} 0.0\n")
            
            # Entropy (avoid log of zero)
            if abs(lam) > 1e-16 and abs(lap) > 1e-16:
                fent.write(f"{time} {-lam * math.log(lam) - lap * math.log(lap)}\n")
            elif abs(lam) > 1e-16:
                fent.write(f"{time} {-lam * math.log(lam)}\n")
            elif abs(lap) > 1e-16:
                fent.write(f"{time} {-lap * math.log(lap)}\n")
            else:
                fent.write(f"{time} 0.0\n")
            
            fLEatom.write(f"{time} {1.0 - (pg**2 + pe**2 + 2.0 * c2abs)}\n")
            fLEfield.write(f"{time} {1.0 - purf}\n")
        
        # Safety stop
        if time > max_time_stop:
            break
    
    # Close files
    for fh in files.values():
        fh.close()
    
    print("Simulation finished (stopped at t = {:.6g}). Files written in UTF-8.".format(time))

if __name__ == "__main__":
    run_simulation(max_time_stop=tmax)  #max_time_stop to run to tmax
