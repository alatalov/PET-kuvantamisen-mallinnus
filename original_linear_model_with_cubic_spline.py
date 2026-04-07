'''
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
'''
def run(brain, V_brain, aorta, V_aorta, time):
        
    import numpy as np
    import matplotlib.pyplot as plt
    import math
    from scipy.interpolate import CubicSpline
    from scipy import interpolate
    from scipy import optimize
    


    #Tietoja kudolsista:
    """
    aivojen tilavuus (mm^3): 1293938.4841918945
    lonkankoukistajan tilavuus (mm^3): 359071.16889953613

    henkilön paino (kg): 73.0
    injektoitu aktiivisuus (MBq): 107

    Verentilvuus 5.1 L, > AI
    """


    #yksikon määritys
    unit_of_times="s"
    unit_of_delay="s"

    #Laskee pisteiden välisen suoran tai sovittaa pisteiden väliin suoran, y=ax+b (suoranyhtälö)
    #Eli antaa ohjeet että voidaan sovittaa muutkin arvot samoilla ohjeilla
    interpolate_input = interpolate.interp1d(time, aorta)
    CuSp_interpolate_input = CubicSpline(time, aorta)

    #all the seconds of the scan period. Sovitetaan mitattu aika halutulle välille 0-265, sekunnin välein
    times_interpolated = np.arange(0,266,1)

    #produce cA with linear interpolation by ‘interp1d‘.
    #ELI: sovitetaan aorttadata uudelle aikavälille > sekunnin välein
    cA = interpolate_input(times_interpolated)
    CuSp_cA = CuSp_interpolate_input(times_interpolated)

    #Interpoloidaan aivo data ja time
    interpolate_TAC = interpolate.interp1d(time, brain)
    CuSp_interpolate_TAC = CubicSpline(time, brain)

    #tehdään sovitus aivodatasta haluttuun aikaväliin 0-265s
    measured_TAC = interpolate_TAC(times_interpolated)
    CuSp_measured_TAC = CuSp_interpolate_TAC(times_interpolated)

    #Varmistetan että käytetty aika kuuluu halutulle välille 0-265
    def interpolate_extended(t,x,y):
    #if the point is on the interval, interpolate normally
        if x[0]<t<x[-1]:
            interpolate_1=interpolate.interp1d(x,y)
            return(interpolate_1(t))
        #if the point is before the interval, return the value at the first point of the interval
        elif t<=x[0]:
            return(y[0])
        #if the point after the interval, return the value at the last point of the interval
        else:
            return(y[-1])

    def cT_from_1TCM_with_delay(k1,k2,times,input,delay,unit_of_times,unit_of_delay):
    #here k1,k2, and delay are float numbers, times and input should be vectors of the same length, and unit_of_times and unit_of_delayare ’s’ or ’min’
    #the units of k1 and k2 are assumed to be mL/min/mL and /min,respectively
    #the unit of times and delay are specified by unit_of_times and unit_of_delay
    #the first value of cT is set as zero
        cT=[0]
    #then the rest of the cT values are obtained in an iterative loop
        for i in range(1,len(times)):
        #let us define the value of input at time[i-1]-delay
            #varmistetaan oikeat yksiköt
            if unit_of_times==unit_of_delay:
                input_value=interpolate_extended(times[i-1]-delay,times,input)
            if unit_of_times=='min' and unit_of_delay=='s':
                input_value=interpolate_extended(times[i-1]-(1/60)*delay,times,input)
            if unit_of_times=='s' and unit_of_delay=='min':
                input_value=interpolate_extended(times[i-1]-60*delay,times,input)
                #let us append the current value of cT
            if unit_of_times=='s':
                cT.append(k1*(times[i]-times[i-1])/60*input_value+(1-k2*(times[i]-times[i-1])/60)*cT[i-1])
            if unit_of_times=='min':
                cT.append(k1*(times[i]-times[i-1])*input_value+(1-k2*(times[i]-times[i-1]))*cT[i-1])
        return np.array(cT)

    #virhe funktio: Sum of squares
    def error_function(param):
        #param is a list of four numeric values
        #simulate cT and then cPET from the parameters in the param vector
        #use absolute value for param[0] and param[1] since k1 and k2 should be non-negative
        cT=cT_from_1TCM_with_delay(abs(param[0]),abs(param[1]),times_interpolated, cA,param[3],unit_of_times,unit_of_delay)
        
        #use inverse-logit for param[2] as V_a is between 0 and 1
        V_a=1/(1+math.exp(-param[2]))
        cPET=(1-V_a)*cT+V_a*cA

        #compute and return the sum of squares between the measured TAC and cPET
        sum_of_squares=np.sum((measured_TAC-cPET)**2)
        return sum_of_squares

    def CuSp_error_function(param):
        CuSp_cT = cT_from_1TCM_with_delay(abs(param[0]),abs(param[1]),times_interpolated, CuSp_cA,param[3],unit_of_times,unit_of_delay)
        V_a=1/(1+math.exp(-param[2]))
        CuSp_cPET=(1-V_a)*CuSp_cT+V_a*CuSp_cA
        sum_of_squares=np.sum((CuSp_measured_TAC-CuSp_cPET)**2)
        return sum_of_squares

    """
    #Määritellää käytetty funktio
    def cT_from_1TCM(k1,k2,times,input,unit_of_times):
    #here k1 and k2 are float numbers, times and input should be vectorsof the same length, and unit_of_times is either ’s’ or ’min’
    #the units of k1 and k2 are assumed to be mL/min/mL and /min,respectively
    #the unit of the vector ’times’ is specified by unit_of_times
    #the tissue curve cT will be in the same unit as the input
    #we first set the first value of cT as zero
        cT=[0]

        for i in range(1,len(times)):
            if unit_of_times=="s":
            #we need the denominator 60 to since times are in seconds butK_1 and k_2 use /min
                cT.append(k1*(times[i]-times[i-1])/60*input[i-1]+
                (1-k2*(times[i]-times[i-1])/60)*cT[i-1])
            if unit_of_times=="min":
            #no denominator 60
                cT.append(k1*(times[i]-times[i-1])*input[i-1]+
                (1-k2*(times[i]-times[i-1]))*cT[i-1])
        return np.array(cT)
     """                            
    #choose initial values (here, param[2] is chosen so that V_a=0.1 after the inverse-logit transformation)
    #asetetaan alku arvot jos ruvetaan etsimään parempia > res
    initial_values=[1,1,math.log(9),0]

    #minimise the output of the error function
    #Etsii parhaat arvot
    res=optimize.minimize(error_function,x0=initial_values)
    CuSp_res=optimize.minimize(CuSp_error_function,x0=initial_values)
    '''
    #print the found minimum sum of squares
    print('Sum of squares:',res.fun)
    print('Sum of squares/N:',res.fun/len(times_interpolated))
    print('CuSp_Sum of squares:',CuSp_res.fun)
    print('CuSp_Sum of squares/N:',CuSp_res.fun/len(times_interpolated))
    print('CuSp_sovitus-lineaarien_sovitus',CuSp_res.fun-res.fun)
    print('CuSp_sovitus-lineaarien_sovitus, pistettä kohden',(CuSp_res.fun-res.fun)/len(times_interpolated))
    '''
    #fix k1,k2,V_a and delay to the values giving this minimum
    k1=abs(res.x[0])
    k2=abs(res.x[1])
    V_a=1/(1+math.exp(-res.x[2]))
    delay=res.x[3]

    #print('Vakiot k1, k2',k1,k2)

    CuSp_k1=abs(CuSp_res.x[0])
    CuSp_k2=abs(CuSp_res.x[1])
    V_a=1/(1+math.exp(-CuSp_res.x[2]))
    CuSp_delay=CuSp_res.x[3]

    #return 'lin virhe',res.fun, 'kuution virhe',CuSp_res.fun, 'lin_k1',k1, 'lin_k2',k2, 'kuuti_k1',CuSp_k1, 'kuuti_k2',CuSp_k2
    return res.fun, CuSp_res.fun, k1, k2, CuSp_k1, CuSp_k2
                                                                                                                   
    #print('Vakiot CuSp_sovituksessa, k1, k2',CuSp_k1,CuSp_k2)

    #simulate the cT and cPET from the found solution
    cT = cT_from_1TCM_with_delay(k1,k2,times_interpolated,cA,delay,'s','s')
    cPET = (1-V_a)*cT+V_a*cA

    #simulate the CuSp_cT and CuSp_cPET from the found solution
    CuSp_cT = cT_from_1TCM_with_delay(CuSp_k1,CuSp_k2,times_interpolated,CuSp_cA,CuSp_delay,'s','s')
    CuSp_cPET = (1-V_a)*CuSp_cT+V_a*CuSp_cA

    #plotti vierekkäine
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # Ensimmäinen kuvaaja
    ax1.plot(times_interpolated,measured_TAC,'-',color='blue')
    ax1.plot(times_interpolated,cPET,'--',color='black')
    ax1.legend(['Measured brain TAC', 'Fitted model curve cPET'])
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Consentration (Bq/mL)')

    # Toinen kuvaaja
    ax2.plot(times_interpolated,CuSp_measured_TAC,'-',color='red')
    ax2.plot(times_interpolated,CuSp_cPET,'--',color='orange')
    plt.legend(['CuSp_Measured brain TAC','CuSp_Fitted model curve cPET'])
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Consentration (Bq/mL)')

    # Säädetään väli ja näytetään kuva
    plt.tight_layout()
    #plt.show()




'''
#plot the fitted model curve together with the measured TAC
plt.plot(times_interpolated,measured_TAC,'-b',times_interpolated,cPET,'--k')
plt.legend(['Measured brain TAC','Fitted model curve cPET'])
plt.xlabel('Time (s)')
plt.ylabel('Consentration (Bq/mL)')

plt.show()

plt.plot(times_interpolated,CuSp_measured_TAC,'-',color='red')
plt.plot(times_interpolated,CuSp_cPET,'--',color='orange')
plt.legend(['CuSp_Measured brain TAC','CuSp_Fitted model curve cPET'])
#plt.xlabel('Time (s)')
#plt.ylabel('Consentration (Bq/mL)')

plt.show()
'''
"""
#Luodaan aortan consentraatio kBq/ml
a=[]
for i in cA:
    a.append(i/1000)
    
#simulate cT from any parameters (here K_1=0.9 and k_2=1)
cT=cT_from_1TCM(0.9,1,times_interpolated,cA,unit_of_times) #here, K_1=0.9 mL/min/mL and k_2=1/min

#aortan aktiivisuuden plot
plt.plot(time, a, marker = "o", color = "blue")

#plot the tissue curve cT (converted here into kBq/mL by dividing it by 1000)
plt.plot(times_interpolated, cT/1000, "-b")
plt.xlabel("Time (s)")
plt.ylabel("Kidoksen konsentraatio (kBq/mL)")
plt.show()
"""
