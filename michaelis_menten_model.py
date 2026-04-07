
'''
# --- 1. DEFINE ALL RAW DATA ---
time = [0, 3, 8, 13, 18, 23, 28, 33, 38,
        43, 48, 53, 58, 63, 68, 75, 85,
        95, 110, 130, 150, 175, 205, 235, 265]

aorta = [0, 124, 886, 5406, 29009, 84960, 90750, 78030, 57405,
         42104, 34094, 27709, 23290, 22109, 21168, 18760, 16693,
         15754, 14843, 14245, 13780, 13292, 12567, 12291, 11974]

brain = [0, 131, 120, 116, 271, 1984, 4982, 8035, 10641,
         12466, 13561, 14294, 14700, 15073, 15312, 15330, 15338,
         15308, 15131, 14855, 14643, 14123, 13874, 13473, 13141]
'''

def run(brain, V_brain, aorta, V_aorta, time):
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    from scipy import interpolate
    from scipy import optimize
    from scipy.signal import convolve
        
    # --- 2. DEFINE VOLUMES AND INTERPOLATED TIME ---
    times_interpolated = np.arange(0, 266, 1)
    dt_sec = times_interpolated[1] - times_interpolated[0] # Should be 1.0
    dt_min = dt_sec / 60.0 # Time step in minutes

    # --- 3. PREPARE INTERPOLATED DATA (CONCENTRATIONS) ---
    C_brain_raw = brain  # Use the raw brain data directly as concentration
    C_aorta_raw = aorta  # Use the raw aorta data directly as concentration

    # Create interpolation functions from the raw, normalized data
    interp_func_aorta = interpolate.interp1d(time, C_aorta_raw, kind='linear', fill_value="extrapolate")
    interp_func_brain = interpolate.interp1d(time, C_brain_raw, kind='linear', fill_value="extrapolate")

    # Create the continuous data curves for every second
    Cp_raw_interpolated = interp_func_aorta(times_interpolated) # RAW Plasma input function
    CT_measured = interp_func_brain(times_interpolated)     # Measured tissue curve (Concentration)


    # --- 4. DEFINE ALL HELPER & MODEL FUNCTIONS ---

    def get_dispersed_input(params_dispersion, Cp_raw):
        """
        Applies delay and dispersion to the raw input function.
        """
        delay_s = params_dispersion[0]
        tau_s = abs(params_dispersion[1]) # Dispersion time constant in seconds
        
        # 1. Create the dispersion kernel (an exponential decay)
        if tau_s < 1e-3: # If dispersion is tiny, just return the delayed curve
            return np.roll(Cp_raw, int(round(delay_s / dt_sec)))
            
        kernel_len = int(tau_s * 10) # Make kernel 10x the time constant
        kernel_time = np.arange(0, kernel_len, dt_sec)
        dispersion_kernel = (1.0 / tau_s) * np.exp(-kernel_time / tau_s)
        dispersion_kernel = dispersion_kernel / np.sum(dispersion_kernel) # Normalize
        
        # 2. Convolve the raw input with the kernel to "smear" it
        Cp_smeared = convolve(Cp_raw, dispersion_kernel, mode='full')[:len(Cp_raw)]
        
        # 3. Apply the time delay (shift the whole curve)
        # np.roll is a simple way to shift (delay) an array
        Cp_final = np.roll(Cp_smeared, int(round(delay_s / dt_sec)))
        
        return Cp_final
        

    def cT_from_NONLINEAR_1TCM(params_model, times, input_Cp_dispersed):
        """
        Simulates tissue CONCENTRATION (cT) using the *dispersed* input function.
        """
        # params_model = [Vmax, Km, k2]
        Vmax = abs(params_model[0]) # Units: (Bq/mL) / min
        Km = abs(params_model[1])   # Units: Bq/mL (same as input_Cp)
        k2 = abs(params_model[2])   # Units: 1/min
        
        cT = [0.0]  # Start with zero concentration in the tissue
        
        for i in range(1, len(times)):
            
            # 1. Get plasma concentration (already delayed and smeared)
            input_value = input_Cp_dispersed[i-1]
            
            # 2. Calculate the dynamic flow_in using Michaelis-Menten
            if (Km + input_value) == 0:
                flow_in_term = 0.0
            else:
                flow_in_term = (Vmax * input_value) / (Km + input_value)
                
            # 3. Calculate linear flow_out
            flow_out_term = k2 * cT[i-1]
            
            # 4. Calculate new concentration using Euler's method
            cT_new = cT[i-1] + dt_min * (flow_in_term - flow_out_term)
            
            cT.append(max(0, cT_new))  # Ensure concentration doesn't go negative
            
        return np.array(cT)

    def error_function_nonlinear(param):
        """
        Calculates the error between the simulated model and the measured data.
        """
        # *** ALL PARAMS IN ONE VECTOR ***
        # param = [Vmax, Km, k2, delay, tau, V_a_logit]
        params_model = param[0:3]
        params_dispersion = param[3:5]
        V_a_logit = param[5]

        # 1. Create the final, dispersed input function
        Cp_dispersed = get_dispersed_input(params_dispersion, Cp_raw_interpolated)
        
        # 2. Simulate the tissue CONCENTRATION (C_T)
        cT_simulated = cT_from_NONLINEAR_1TCM(params_model, times_interpolated, Cp_dispersed)
        
        # 3. Get the blood volume fraction (V_a)
        V_a = 1 / (1 + np.exp(-V_a_logit))
        
        # 4. Calculate the total simulated PET signal (CONCENTRATION)
        #    cPET = (1 - V_a) * c_Tissue + V_a * c_Plasma (dispersed)
        cPET_simulated = (1 - V_a) * cT_simulated + V_a * Cp_dispersed

        # 5. Calculate error
        sum_of_squares = np.sum((CT_measured - cPET_simulated)**2)
        
        if np.isnan(sum_of_squares) or np.isinf(sum_of_squares):
            return 1e30 # Return a very large number
            
        return sum_of_squares

    # --- 5. RUN THE OPTIMIZATION ---

    #print("Starting non-linear optimization with dispersion...")

    # New params: [Vmax, Km, k2, delay, tau, V_a_logit]
    initial_values = [10000.0, 50000.0, 0.5, 2.0, 5.0, -2.19] # V_a=0.1, tau=5s

    # [Vmax,         Km,           k2,    delay,  tau,   V_a_logit]
    bnds = [(0, 200000),  # Vmax: 0 to 200,000 (Bq/mL / min)
            (1, 200000),  # Km: > 0 to 200,000 (Bq/mL)
            (0, 10),      # k2: 0 to 10 (1/min)
            (-10, 20),    # delay: -10s to +20s
            (0, 30),      # tau: 0s to 30s dispersion
            (-4.6, -1.1)] # V_a_logit: -> V_a approx 1% to 25%

    res = optimize.minimize(error_function_nonlinear, 
                             x0=initial_values, 
                             method='L-BFGS-B', 
                             bounds=bnds)

    #print("Optimization finished.")
    #print('Sum of squares:', res.fun)

    # --- 6. DISPLAY THE RESULTS ---

    # Extract the best-fit parameters
    best_params = res.x
    vmax_fit = abs(best_params[0]) #k1
    km_fit = abs(best_params[1])   #k2
    k2_fit = abs(best_params[2])
    delay_fit = best_params[3]
    tau_fit = best_params[4]
    va_fit = 1 / (1 + np.exp(-best_params[5]))
    #print(res.fun, vmax_fit, km_fit,'haaaa')
    return res.fun, vmax_fit, km_fit
    '''
    print(f"Best-fit Vmax: {vmax_fit:.4f}")
    print(f"Best-fit Km: {km_fit:.4f}")
    print(f"Best-fit k2: {k2_fit:.4f}")
    print(f"Best-fit Delay: {delay_fit:.4f} s")
    print(f"Best-fit Tau (dispersion): {tau_fit:.4f} s")
    print(f"Best-fit V_a: {va_fit:.4f}")
    '''
    

    # Simulate the final, best-fit curves
    Cp_final_dispersed = get_dispersed_input(best_params[3:5], Cp_raw_interpolated)
    cT_final = cT_from_NONLINEAR_1TCM(best_params[0:3], times_interpolated, Cp_final_dispersed)
    cPET_final = (1 - va_fit) * cT_final + va_fit * Cp_final_dispersed

    # Plot the results
    plt.figure(figsize=(12, 8))

    # Plot 1: The main fit
    ax1 = plt.subplot(2, 1, 1)
    plt.plot(times_interpolated, CT_measured, 'b-', label='Measured Brain (C_T)', linewidth=2)
    plt.plot(times_interpolated, cPET_final, 'r--', label='Fitted Non-Linear Model (C_PET)', linewidth=2)
    plt.legend()
    plt.title('Non-Linear 1TCM Fit (with Dispersion)')
    plt.ylabel('Concentration (Bq/mL)')
    plt.grid(True)

    # Plot 2: The input functions
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    plt.plot(times_interpolated, Cp_raw_interpolated, 'k:', label='Raw Aorta Input (Cp_raw)')
    plt.plot(times_interpolated, Cp_final_dispersed, 'g-', label='Dispersed Aorta Input (Cp_dispersed)')
    plt.legend()
    plt.xlabel('Time (s)')
    plt.ylabel('Concentration (Bq/mL)')
    plt.grid(True)

    plt.tight_layout()
    #plt.show()


