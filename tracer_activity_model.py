"""
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.interpolate import CubicSpline
from scipy import interpolate
from scipy import optimize

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

    #all the seconds of the scan period. Sovitetaan mitattu aika halutulle välille 0-265, sekunnin välein
    times_interpolated = np.arange(0,266,1)

    #Aktiivisuudet, tilanuudet litroissa
    A_brain=[i*(V_brain/1000000) for i in brain]
    A_aorta=[i*(V_aorta/1000000) for i in aorta]

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
    aA_interpolate_input = interpolate.interp1d(time, A_aorta)

    #produce cA with linear interpolation by ‘interp1d‘.
    #ELI: sovitetaan aorttadata uudelle aikavälille > sekunnin välein
    aA = aA_interpolate_input(times_interpolated)

    #Interpoloidaan aivo data ja time
    aT_interpolate_TAC = interpolate.interp1d(time, A_brain)

    #tehdään sovitus aivodatasta haluttuun aikaväliin 0-265s
    aT_measured_TAC = aT_interpolate_TAC(times_interpolated)

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

    def aT_from_1TCM_with_delay(k1,k2,times,input,delay,unit_of_times,unit_of_delay):
    #here k1,k2, and delay are float numbers, times and input should be vectors of the same length, and unit_of_times and unit_of_delayare ’s’ or ’min’
    #the units of k1 and k2 are assumed to be mL/min/mL and /min,respectively
    #the unit of times and delay are specified by unit_of_times and unit_of_delay
    #the first value of cT is set as zero
        aT=[0]
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
                aT.append(k1*(times[i]-times[i-1])/60*input_value+(1-k2*(times[i]-times[i-1])/60)*aT[i-1])
            if unit_of_times=='min':
                aT.append(k1*(times[i]-times[i-1])*input_value+(1-k2*(times[i]-times[i-1]))*aT[i-1])
        return np.array(aT)


    #virhe funktio: Sum of squares
    def error_function(param):
        #param is a list of four numeric values
        #simulate cT and then cPET from the parameters in the param vector
        #use absolute value for param[0] and param[1] since k1 and k2 should be non-negative
        aT=aT_from_1TCM_with_delay(abs(param[0]),abs(param[1]),times_interpolated, aA,param[3],unit_of_times,unit_of_delay)

        #use inverse-logit for param[2] as V_a is between 0 and 1
        V_a=1/(1+math.exp(-param[2]))
        aPET=(1-V_a)*aT+V_a*aA

        #compute and return the sum of squares between the measured TAC and cPET
        sum_of_squares=np.sum((aT_measured_TAC-aPET)**2)
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

    #fix k1,k2,V_a and delay to the values giving this minimum
    k1=abs(res.x[0])
    k2=abs(res.x[1])
    V_a=1/(1+math.exp(-res.x[2]))
    delay=res.x[3]

    #parametrit konsentraatioista

    '''
    #print the found minimum sum of squares
    print('Sum of squares:',res.fun/(1293.9384841918945)**2)
    print('Sum of squares/N:',(res.fun/len(times_interpolated))/(1293.9384841918945)**2)
    '''
    
    #print('Vakiot lineaarinen sovitus',k1,k2)

    return res.fun/(1293.9384841918945)**2, k1, k2

    #simulate the aT and aPET from the found solution
    aT=aT_from_1TCM_with_delay(k1,k2,times_interpolated,aA,delay,'s','s')
    aPET=(1-V_a)*aT+V_a*aA

    #plot the fitted model curve together with the measured TAC
    plt.plot(times_interpolated,aT_measured_TAC,'-b',times_interpolated,aPET,'--k')
    plt.legend(['Measured brain TAC','Fitted model curve aPET'])
    plt.xlabel('Time (s)')
    plt.ylabel('Activity (Bq)')

    #CS


    '''
    plt.plot(times_interpolated,CuSp_measured_TAC,'-',color='red')
    plt.plot(times_interpolated,CuSp_aPET,'--',color='orange')

    plt.plot(CuSp_measured_TAC,'-r',CuSp_aPET,'--g')
    plt.legend(['CuSp_Measured brain TAC','CuSp_Fitted model curve aPET'])
    '''
    #plt.show()
    


