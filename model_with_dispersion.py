"""

#
time=[0,3,8,13,18,23,28,33,38,
43,48,53,58,63,68,75,85,
95,110,130,150,175,205,235,265]

#Aortta data
aorta=[0,124,886,5406,29009,84960,90750,78030,57405,
42104,34094,27709,23290,22109,21168,18760,16693,
15754,14843,14245,13780,13292,12567,12291,11974]

#aivo data
brain=[0,131,120,116,271,1984,4982,8035,10641,
12466,13561,14294,14700,15073,15312,15330,15338,
15308,15131,14855,14643,14123,13874,13473,13141]
"""

def run(brain, V_brain, aorta, V_aorta, time):

    import numpy as np
    import matplotlib.pyplot as plt
    import math
    from scipy.interpolate import CubicSpline
    from scipy import interpolate
    from scipy import optimize
    from scipy.signal import convolve # <-- ADDED
    
    #all the seconds of the scan period. Sovitetaan mitattu aika halutulle välille 0-265, sekunnin välein
    times_interpolated = np.arange(0,266,1)
    dt_sec = times_interpolated[1] - times_interpolated[0] # <-- ADDED
    dt_min = dt_sec / 60.0 # <-- ADDED

    # --- We only need the raw concentration data ---
    #konsentraatio sovitukset
    cA_interpolate_input = interpolate.interp1d(time, aorta, fill_value="extrapolate")
    cA_raw_interpolated = cA_interpolate_input(times_interpolated) # This is our "spiky" input
    cT_interpolate_TAC = interpolate.interp1d(time, brain, fill_value="extrapolate")
    cT_measured_TAC = cT_interpolate_TAC(times_interpolated) # This is our "ground truth" target

    #yksikon määritys
    unit_of_times="s"
    unit_of_delay="s"

    #Varmistetan että käytetty aika kuuluu halutulle välille 0-265
    def interpolate_extended(t,x,y):
        # This function is used by aT_from_1TCM_with_delay
        # We create a temporary interpolation function for the array
        # This is not efficient, but it's how the base code works
        interpolate_1 = interpolate.interp1d(np.arange(len(y)), y, fill_value="extrapolate")
        
        # We need to find the index 't' corresponds to
        # This is tricky because 't' is a time and 'x' is an array of times
        # A simpler way is to assume x is times_interpolated
        if t < times_interpolated[0]:
            return y[0]
        if t > times_interpolated[-1]:
            return y[-1]
        
        # Find the closest index for time t
        # This is a bit of a hack to make the old function work with the new data
        # A better way would be to pass the interpolation function itself
        return np.interp(t, times_interpolated, y)


    def get_dispersed_input(params_dispersion, Cp_raw):
        """
        Applies delay and dispersion to the raw input function.
        """
        delay_s = params_dispersion[0]
        tau_s = abs(params_dispersion[1]) # Dispersion time constant in seconds
        
        if tau_s < 1e-3: # If dispersion is tiny, just return the delayed curve
            Cp_final = np.roll(Cp_raw, int(round(delay_s / dt_sec)))
            return Cp_final
            
        kernel_len = max(int(tau_s * 10), 1) # Make kernel 10x the time constant
        kernel_time = np.arange(0, kernel_len, dt_sec)
        if tau_s == 0: # Avoid division by zero if tau is exactly 0
            return np.roll(Cp_raw, int(round(delay_s / dt_sec)))
            
        dispersion_kernel = (1.0 / tau_s) * np.exp(-kernel_time / tau_s)
        dispersion_kernel = dispersion_kernel / np.sum(dispersion_kernel) # Normalize
        
        Cp_smeared = convolve(Cp_raw, dispersion_kernel, mode='full')[:len(Cp_raw)]
        Cp_final = np.roll(Cp_smeared, int(round(delay_s / dt_sec)))
        
        return Cp_final

    def aT_from_1TCM_with_delay(k1,k2,times,input,delay,unit_of_times,unit_of_delay):
        # This is the original LINEAR model from your base code
        aT=[0]
        for i in range(1,len(times)):
            # The input here is an array, not a function
            # We need to find the value at time[i-1]-delay
            input_value = np.interp(times[i-1] - delay, times, input)

            if unit_of_times=='s':
                aT.append(k1*(times[i]-times[i-1])/60*input_value+(1-k2*(times[i]-times[i-1])/60)*aT[i-1])
            if unit_of_times=='min':
                aT.append(k1*(times[i]-times[i-1])*input_value+(1-k2*(times[i]-times[i-1]))*aT[i-1])
        return np.array(aT)


    #virhe funktio: Sum of squares
    def error_function_linear_dispersion(param, input_aorta, target_brain):
        # param = [K1, k2, V_a_logit, delay, tau]
        K1 = abs(param[0])
        k2 = abs(param[1])
        V_a_logit = param[2]
        params_dispersion = [param[3], param[4]] # [delay, tau]

        # 1. Create the dispersed input function
        Cp_dispersed = get_dispersed_input(params_dispersion, input_aorta)
        
        # 2. Simulate tissue curve using LINEAR model
        #    We set delay=0 because dispersion function already handled it.
        aT = aT_from_1TCM_with_delay(K1, k2, times_interpolated, Cp_dispersed, 0, unit_of_times, unit_of_delay)
        
        # 3. Get V_a and calculate final PET signal
        V_a = 1 / (1 + math.exp(-V_a_logit))
        aPET = (1 - V_a) * aT + V_a * Cp_dispersed

        # 4. Calculate error
        sum_of_squares = np.sum((target_brain - aPET)**2)
        
        if np.isnan(sum_of_squares) or np.isinf(sum_of_squares):
            return 1e30
            
        return sum_of_squares
     
    #asetetaan alku arvot
    initial_values_disp = [0.5, 0.5, -2.0, 2.0, 10.0]

    # Määritellään rajat
    # [K1,    k2,    V_a_logit,  delay,  tau]
    bnds_disp = [(0, 10), (0, 10), (-3.5, -1.7), (-10, 20), (0, 30)]

    #print("Starting Fit: Linear Model with Dispersion (Raw Data)...")

    # Etsii parhaat arvot
    res = optimize.minimize(error_function_linear_dispersion,
                             x0=initial_values_disp,
                             args=(cA_raw_interpolated, cT_measured_TAC), # Use RAW data
                             method='L-BFGS-B',
                             bounds=bnds_disp)

    #print("Fit finished.")

    # Calculate n and MSE
    n_points = len(times_interpolated)
    mse = res.fun / n_points

    # Tulostetaan tulokset
    k1_fit = abs(res.x[0])
    k2_fit = abs(res.x[1])
    V_a_fit = 1 / (1 + math.exp(-res.x[2]))
    delay_fit = res.x[3]
    tau_fit = res.x[4]

    return res.fun, k1_fit, k2_fit
'''
    print(f'\n--- Fit Results: Linear + Dispersion ---')
    print(f'Sum of squares: {res.fun:.2f}')
    print(f'MSE (SS/n): {mse:.2f}')
    print(f'Vakiot: K1={k1_fit:.4f}, k2={k2_fit:.4f}')
    print(f'Fit V_a: {V_a_fit:.6f}')
    print(f'Fit Delay: {delay_fit:.4f} s')
    print(f'Fit Tau: {tau_fit:.4f} s')
'''
    # --- Simuloidaan lopullinen käyrä ---
    Cp_dispersed_final = get_dispersed_input([delay_fit, tau_fit], cA_raw_interpolated)
    aT_final = aT_from_1TCM_with_delay(k1_fit, k2_fit, times_interpolated, Cp_dispersed_final, 0, 's', 's')
    aPET_final = (1 - V_a_fit) * aT_final + V_a_fit * Cp_dispersed_final

    # --- Piirretään kuvaajat ---
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Kuvaaja 1: Mallin sovitus
    axs[0].plot(times_interpolated, cT_measured_TAC, '-b', label='Measured Brain (Raw)')
    axs[0].plot(times_interpolated, aPET_final, '--r', label='Fitted (Linear + Disp)')
    axs[0].set_title(f'Fit 3: Linear + Dispersion\nSS={res.fun:.2e}, V_a={V_a_fit:.4f}')
    axs[0].set_ylabel('Concentration (Bq/mL)')
    axs[0].legend()
    axs[0].grid(True)

    # Kuvaaja 2: Input-funktiot
    axs[1].plot(times_interpolated, cA_raw_interpolated, 'k:', label='Raw Aorta Input')
    axs[1].plot(times_interpolated, Cp_dispersed_final, 'g-', label='Dispersed Aorta Input')
    axs[1].set_title('Input Function Transformation')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Concentration (Bq/mL)')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    #plt.show()



