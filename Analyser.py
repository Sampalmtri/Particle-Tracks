import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate as integrate
import scipy.special as scis

global m_sigma
global m_lambda
global L
global perp_mom
global L_lambda
global L_zs
global m_pion_o
global sigma
global pixel_width
global dx_error
global z_error

sigma = 1
pixel_width = 0.02036/2 #cm
dx_error = 4*pixel_width
z_error = dx_error*31.6/300

m_kaon = 493.677
m_pion = 139.57039
m_proton = 938.27208943
m_pion_o = 134.976

m_sigma = lambda p : np.sqrt((m_kaon + m_proton - np.sqrt(m_pion**2 + p**2))**2 - p**2)

m_lambda = lambda a : np.sqrt((m_proton**2 - m_pion**2)/a)

L = lambda L_perp, p_perp, p : L_perp*p/p_perp

perp_mom = lambda R : 3*1.625*R

L_lambda = lambda L_perp, L_z : np.sqrt(L_perp**2 + L_z**2)

L_zs = lambda L_plus, L_minus : np.abs(L_plus-L_minus)

def get_data(file, channel):
    #1 - index (type of decay),4-radius pixels, 5 - radius_cm,6-length pixels, 7 - decaylength_cm, 8 - mag_a, 9 - mag_b, 12 - phi_proton, 13 - phi_pion,
    #14 - origin_depth, 15 - decay_depth

    fopen = open(file, "r")

    lines = fopen.read()

    lines_split = lines.split("\n")

    list_of_data = []

    for i in range(1,len(lines_split)):
        list_of_data.append(lines_split[i].split(","))

    fopen.close()

    if(list_of_data[-1] == ['']):
        list_of_data.pop()
    else:
        pass

    if(channel == 'all'):
        data = np.array(list_of_data)[:,np.array([1,5,7,8,9,12,13,14,15])]
    else:
        data = np.array(list_of_data)[:,channel]

    error_data = np.array(list_of_data)[:,np.array([4,6,8,9,14,15])]

    return (data.astype('float64'),error_data.astype('float64'))

def get_decays(data, charge, error_data):

    plus_decays = []
    minus_decays = []
    plus_errors = []
    minus_errors = []

    charges = {'plus' : [plus_decays, 1, 2,plus_errors], 'minus' : [minus_decays, 3, 3,minus_errors]}

    for i,decay in enumerate(data):
        if(int(decay[0]) == charges[charge][1] or int(decay[0]) == charges[charge][2]):
            charges[charge][0].append(decay)
            charges[charge][3].append(error_data[i])
        else:
            pass

    return charges

def calc_sigma_mass(data, charge, error_data, R_gauss_error):

    charges, error_data = get_decays(data, charge, error_data)[charge][0],get_decays(data, charge, error_data)[charge][3]

    charges = np.array(charges)

    error_data = np.array(error_data)

    perp_moms = perp_mom(charges[:,1])

    average_perp_mom = np.average(perp_moms) 

    momentum = average_perp_mom/(np.pi/4)

    iso = isotropy(momentum,perp_moms)

    iso_error = (np.pi/4 - iso)/(2*np.pi/4) #divide by 2pi to get over the whole sphere

    mass = m_sigma(momentum)

    dx,mag_a,mag_b,o_depth = (error_data[:,0],error_data[:,2],error_data[:,3],error_data[:,4])

    rerror = R_error(mag_a,mag_b,o_depth,dx,R_gauss_error[1],charges[:,1],R_gauss_error[2])

    R_true_error = np.sqrt(rerror**2 + (R_gauss_error[0]/len(rerror))**2)

    p_error = np.average(P_perp_error(R_true_error))/np.sqrt(len(R_true_error))

    p_error = np.sqrt(p_error**2 + iso_error**2)

    mass_error = charged_m_error(momentum,mass,p_error)

    return mass, momentum, perp_moms, mass_error, P_perp_error(R_true_error), p_error

def calc_branch_ratio(data, branch):

    pion_num = 0
    proton_num = 0
    neutral_num = 0
    v_num = 0

    branches = {"sigma": [pion_num,2,proton_num,1], "lambda": [v_num,4,neutral_num,5]}


    for decay in data:
        if(int(decay[0]) == branches[branch][1]):
            branches[branch][0] += 1
        elif(int(decay[0]) == branches[branch][3]):
            branches[branch][2] += 1
        else:
            pass

    total = branches[branch][0] + branches[branch][2]

    return 100*branches[branch][2]/total, 100*branches[branch][0]/total, (branches[branch][2]+116)*total/((total+116)*branches[branch][2]), (branches[branch][0]+116)*total/((total+116)*branches[branch][0])

def calc_lambda_mass(data,proton_angle_error,pion_angle_error):

    v_decays = []
    for decay in data:
        if(int(decay[0]) == 4):
            v_decays.append(decay)
        else:
            pass

    v_decays = np.array(v_decays)

    sin_tot = np.sin(v_decays[:,6] + v_decays[:,5])

    alpha = np.sin(v_decays[:,6] - v_decays[:,5])/sin_tot

    alpha_error = error_alpha(v_decays[:,6], v_decays[:,5],pion_angle_error,proton_angle_error)

    alpha_avged_error = np.average(alpha_error)/np.sqrt(len(alpha_error))

    alpha_corrected = [] #just used to see anomalies
    for i in range(0,len(alpha)):
        if(np.abs(alpha[i]) < 5):
            alpha_corrected.append(alpha[i])
        else:
            print(v_decays[i])
            print(alpha[i])

    alpha = np.array(alpha)

    average_alpha = np.average(alpha)

    mass = m_lambda(average_alpha)

    mass_error = lambda_mass_error(average_alpha,alpha_avged_error,mass)

    return mass, mass_error

def func(x, a, b):
    return a*np.exp(-x/b)

def histogram(dist, width_mul, function, scan_width):
    if(function == func):
        width = (scan_width*width_mul)
        max_val = (np.max(dist)//width)*width + width

        bin_w = np.arange(0,max_val,width)
    elif(function == non_normal_gaussian or function == poisson):
        width = (scan_width*width_mul) #0.00005
        max_val = (np.max(dist)//width)*width + width
        min_val = (np.min(dist)//width)*width 

        bin_w = np.arange(min_val,max_val,width)

    counts, bins = np.histogram(dist,bins = bin_w)
    #print(len(bins))

    if(function == func):

        peak = np.max(counts)

        start = -1
        for i, count in enumerate(counts):
            if(count == peak):
                start = i
            else:
                pass
            
        counts = counts[start:]
        bins = bins[start:]
    elif(function == non_normal_gaussian):

        for i, val in enumerate(counts):
            if(val==np.max(counts)):
                central = i
                break
            else:
                pass

        bins = bins - bins[central]
        

    try:
        popt, pcov = curve_fit(function, bins[:len(bins)-1], counts)

        if(function == func):

            #print(pcov)

            return popt, np.diag(pcov)[1], bins, counts #np.linalg.cond(pcov)

        elif(function == non_normal_gaussian or function == poisson):

            return popt, pcov[0][0], bins, counts
    except:
        
        return [1], 1e99, bins, counts 

def iterate_best_width(dist, function, scan_width):
                       
    best_width = []

    for w in range(1,21):

        popt, pcov, bins, counts = histogram(dist, w, function, scan_width)
        best_width.append([pcov,w])

    best_width = np.array(best_width)

    #print(best_width)


    for i,val in enumerate(best_width):
        if(val[0] == np.min(best_width[:,0])):
            popt, pcov, bins, counts = histogram(dist, i+1, function, scan_width)
            #print(np.min(best_width[:,0]))
            #print(i+1)
            #print(bins)
            #print(counts)
            break
        else:
            pass

    return popt, pcov, bins, counts

def calc_sigma_lifetime(data, charge, momentum, perp_moms, mass, error_L, error_data, pperp_error, p_error, mass_error):

    charges, error_data = get_decays(data, charge, error_data)[charge][0],get_decays(data, charge, error_data)[charge][3]

    charges = np.array(charges)

    error_data = np.array(error_data)

    dist = L(charges[:,2]/100, perp_moms, momentum)

    scan_width = 0.005/100

    popt, pcov, bins, counts = iterate_best_width(dist, func, scan_width)

    dx,mag_a,mag_b,o_depth = (error_data[:,1],error_data[:,2],error_data[:,3],error_data[:,4])

    measured_L_error = d_or_l_error(mag_a,mag_b,o_depth,dx)

    error_L_combined = np.sqrt(measured_L_error**2 + error_L**2)

    error_L_true = L_error(dist,charges[:,2]/100,error_L_combined,perp_moms,pperp_error,momentum, p_error)

    lambda_error = calc_lambda_error(*popt, bins, counts, error_L_true/100) #error_L/100 because error_L is in cm and meters is used for the lifetime calc 
    
    tau_error = error_tau(mass,momentum,popt[1],lambda_error,mass_error,p_error)

    tau = popt[1]/((3e8)*(momentum/mass))

    E = np.sqrt(mass**2 + momentum**2)

    tau_error_corrected = corrected_tau_error(mass,momentum,E,tau,mass_error,p_error,tau_error)

    tau += 0.5*(tau**2)*(3e8)*0.248/(mass*((momentum/E)**3))

    text = {'plus' : 'Σ⁺', 'minus' : 'Σ⁻'}

    print("Mean lifetime for " + text[charge] + " is: " + str(tau) + " +/- " + str(tau_error_corrected) + " s")

    plt.stairs(counts, bins)

    plt.xlabel('L (m)')
    plt.ylabel('N')

    plt.plot(bins, func(bins, *popt),label=("Tau: " + str(tau)))
    
    plt.legend()
    plt.title(text[charge] + " Lifetime")

    plt.show()

def calc_lambda_lifetime(data, mass, error_data, error_L,mass_error):

    v_decays = []
    error_count = []
    for i,decay in enumerate(data):
        if(int(decay[0]) == 4):
            v_decays.append(decay)
            error_count.append(error_data[i])
        else:
            pass

    v_decays = np.array(v_decays)

    error_data = np.array(error_count)

    L_depth = L_zs(v_decays[:,7],v_decays[:,8])

    L = L_lambda(v_decays[:,2],L_depth)/100

    error_L_z = 2*z_error

    dx,mag_a,mag_b,o_depth,d_depth = (error_data[:,1],error_data[:,2],error_data[:,3],error_data[:,4],error_data[:,5])

    measured_L_error = d_or_l_error(mag_a,mag_b,(o_depth+d_depth)/2,dx)

    error_L_combined = np.sqrt(measured_L_error**2 + error_L**2)

    L_error = L_error_neutral(L,v_decays[:,2],L_depth,error_L_combined,error_L_z)

    scan_width = 0.005/100

    popt, pcov, bins, counts = iterate_best_width(L, func, scan_width)

    lambda_error = calc_lambda_error(*popt, bins, counts, L_error/100)

    E = m_kaon + m_proton

    momentum = np.sqrt(mass**4 - 2*(mass**2)*(m_pion_o**2) - 2*(E**2)*(mass**2) + m_pion_o**4 - 2*(E**2)*(m_pion_o**2) + E**4)/(2*E)

    tau = popt[1]/((3e8)*(momentum/mass))

    p_error = p_neutral_error(mass,momentum,mass_error)

    tau_error = error_tau(mass,momentum,popt[1],lambda_error,mass_error,p_error)

    print("Mean lifetime for Λ⁰ is: " + str(tau) + " +/- " + str(tau_error) + " s")

    plt.stairs(counts, bins)

    plt.xlabel('L (m)')
    plt.ylabel('N')

    plt.plot(bins, func(bins, *popt),label=("Tau: " + str(tau)))
    
    plt.legend()
    plt.title("Λ⁰ Lifetime")

    plt.show()

def non_normal_gaussian(x, c):
    return np.exp(-(x)**2/(2*(c**2)))

def gaussian(x, c):
    return np.exp(-(x)**2/(2*(c**2)))/(c*np.sqrt(2*np.pi))

def linear(x, a):
    return a*x

def length_error(data, gaussian_data,plot_show):

    scan_width = (np.max(gaussian_data)-np.min(gaussian_data))/(80)

    popt_gauss, pcov_gauss, bins_gauss, counts_gauss = fit_gaussian(gaussian_data, scan_width)

    width = np.max(bins_gauss) - np.min(bins_gauss)

    gauss_base = np.linspace(-width,width,100)

    gauss_dist = gaussian(gauss_base, popt_gauss[0])

    if(plot_show == True):

        plt.title('Length Measurements Distribution (ΔL)')

        plt.ylim((0,1.1*np.max(gauss_dist)))

        plt.ylabel('Relative Frequency')
        plt.xlabel('ΔL (cm)')

        plt.stairs(np.max(gauss_dist)*counts_gauss/np.max(counts_gauss),bins_gauss)

        plt.plot(gauss_base,gauss_dist,label=('Standard Deviation: ' + str(np.round(popt_gauss[0],3))))

        plt.legend()

        plt.show()
        

    data = data[np.array([15,14,13,12,11,0,16,17,18,19,20])]-data[0] #y data has been deemed insignificant

    length_error_base = np.linspace(-pixel_width,pixel_width,11)

    popt= fit_linear(length_error_base, data)

    base = np.linspace(-pixel_width,pixel_width,100)

    dist = linear(base, popt[0])

    if(plot_show == True):

        plt.title('Length Error Relation')

        plt.ylabel('ΔL (cm)')
        plt.xlabel('Δx (cm)')

        plt.plot(length_error_base,data)

        plt.plot(base,dist,label=("Gradient: " + str(np.round(popt[0],2)))) #should be approx 1

        plt.legend()

        plt.show()

    popt_gauss[0] /= popt[0] #conversion from delta L to delta x

    base_fit = np.linspace(-5*popt_gauss[0],5*popt_gauss[0],1000) #make 5 fitted sigma wide

    gauss_dist_fit = gaussian(base_fit, popt_gauss[0])

    if(plot_show == True):

        plt.title('Length Measurements Distribution Fitted (Δx)')

        plt.ylabel('Relative Frequency')
        plt.xlabel('Δx (cm)')

        plt.ylim((0,1.1*np.max(gauss_dist_fit)))

        plt.plot(base_fit,gauss_dist_fit,label=('Standard Deviation: ' + str(np.round(popt_gauss[0],3))))

        plt.legend()

        plt.show()

    dist_fit = linear(base_fit, popt[0])

    expected_dist = dist_fit*gauss_dist_fit

    E = integrate.quad(lambda x: gaussian(x, popt_gauss[0])*linear(x, popt[0]),-np.inf,np.inf)

    if(plot_show == True):

        plt.title('Expected Function')

        plt.ylabel('P(Δx)*ΔL(Δx)')
        plt.xlabel('Δx (cm)')

        plt.plot(base_fit,expected_dist,label=("Expected Value: " + str(E[0]) + " with integration error: " + str(E[1])))

        plt.legend()

        plt.show()

    variance_dist = (dist_fit**2)*gauss_dist_fit

    V = integrate.quad(lambda x: gaussian(x, popt_gauss[0])*(linear(x, popt[0])**2),-np.inf,np.inf)

    V += (-E[0]**2,2*E[1])

    S = (np.sqrt(V[0]),0.5*V[1])

    if(plot_show == True):

        plt.title('Variance Function')

        plt.ylabel('P(Δx)*(ΔL(Δx))^2')
        plt.xlabel('Δx (cm)')

        plt.plot(base_fit,variance_dist,label=("Standard Deviation: " + str(np.round(S[0],3)) + " with integration error: " + str(np.round(S[1],3))))

        plt.legend(loc='upper left')

        plt.show()

    
    length_error = 2*S[0] #maybe instead of 2* add the errors from 1st and 2nd decay in quadrature

    return length_error

def derivative_decay_plus_1(x,lmbda,N_0):
    return np.exp(-x/lmbda)*(x**2)/((lmbda**2)*(np.exp(-x/lmbda)+1/N_0)*np.log(np.exp(-x/lmbda)+1/N_0))

def calc_lambda_error(N_0, lmbda, bins, counts,error_L):

    bins = bins[:len(bins)-1]

    lambdas = []
    #bins_removed = []

    for i,val in enumerate(counts):

        lambdas.append(bins[i]/(-np.log((val+1)/N_0)))
        #bins_removed.append(bins[i])

        '''if(val != 0):
            lambdas.append(bins[i]/(-np.log(val/N_0)))
            bins_removed.append(bins[i])
        else:
            lambdas.append(bins[i]/(-np.log(0.01*np.exp(-bins[i]/lmbda)/N_0)))
            bins_removed.append(bins[i])'''

    lambdas = np.array(lambdas)
    #area = integrate.quad(lambda x: decay_plus_1(x,lmbda,N_0),bins[0],bins[-1])
    #width = bins[-1] - bins[0]
    #print(area[0]/width)
    #print(N_0)
    
    #bins_removed = np.array(bins_removed)
    #bins = bins_removed

    diff = len(bins) - len(error_L)

    error_L = error_L[:diff]

    errors_squared = (lambdas*error_L/(bins*derivative_decay_plus_1(bins,lmbda,N_0)))**2

    sigma_lambda = np.sqrt(np.average(errors_squared)/len(bins)) #we can do the average since the integral under the curve (avg) is lambda

    #need to finish, need M sigma and P sigma

    return sigma_lambda

def poisson(x, a):

    y = np.zeros((len(x)))

    for i,val in enumerate(x):
        fact = scis.factorial(val)
        if(fact == np.inf or fact == -np.inf):
            y[i] = 0
        else:
            y[i] = np.exp(val*np.log(a)-a)/fact

    
    return y

def poisson_single(x, a):
    fact = scis.factorial(x)

    if(fact == np.inf or fact == -np.inf):
        return 0
    else:
        return np.exp(x*np.log(a)-a)/scis.factorial(x)


def radius_error(data, poisson_data,plot_show):

    new_data = np.sort(poisson_data)

    poisson_data = new_data[:-1] #truncate largest value for better fit
             
    scan_width = (np.max(poisson_data)-np.min(poisson_data))/(200)

    popt_poiss, pcov_poiss, bins_poiss, counts_poiss = fit_poisson(poisson_data, scan_width)

    width = np.max(bins_poiss) - np.min(bins_poiss)

    poiss_base = np.linspace(0,width,100)

    poiss_dist = poisson(poiss_base, popt_poiss[0])

    if(plot_show == True):

        plt.title('Radius Measurements Distribution (ΔR)')

        plt.ylim((0,1.1*np.max(poiss_dist)))

        plt.ylabel('Relative Frequency')
        plt.xlabel('ΔR (cm)')

        plt.stairs(np.max(poiss_dist)*counts_poiss/np.max(counts_poiss),bins_poiss-popt_poiss[0])

        plt.plot(poiss_base-popt_poiss[0],poiss_dist,label=('Mean(of R)/Variance: ' + str(np.round(popt_poiss[0],3))))

        plt.legend()

        plt.show()

    data = data[np.array([6,5,4,3,2,1,0,7,8,9,10,11,12])]-data[0]

    for i,val in enumerate(data):
        if(val == data.max()):
            boosted = i
            
        else:
            pass

    radius_base = np.linspace(-2*pixel_width,2*pixel_width,13)

    x_shift = (data[boosted]*radius_base[boosted]+data[boosted+1]*radius_base[boosted+1])/(data[boosted]+data[boosted+1])

    global inverse_proportion

    def inverse_proportion(x, a):
        b=x_shift
        return a*(1/np.abs(x-b) - 1/np.abs(b))

    popt= fit_inverse(radius_base, data)

    base = np.linspace(-2*pixel_width,2*pixel_width,100)

    dist = inverse_proportion(base, *popt)

    if(plot_show == True):

        plt.title('Radius Error Relation')

        plt.ylabel('ΔR (cm)')
        plt.xlabel('Δx (cm)')

        plt.ylim((np.min(data)*1.05,np.max(data)*1.1))

        plt.plot(radius_base,data)

        plt.plot(base,dist,label=("Proportionality: " + str(np.round(popt[0],2)) + " => L = " + str(np.round(np.sqrt(8*popt[0]),2)) + "cm"))

        plt.legend()

        plt.show()
        

    base_fit = np.linspace(0,5*np.sqrt(popt_poiss[0]) + popt_poiss[0],1000) 

    poiss_dist_fit = poisson(base_fit, popt_poiss[0]) 

    dist_fit = linear(base_fit-popt_poiss[0], 1)

    expected_dist = dist_fit*poiss_dist_fit

    E = integrate.quad(lambda x: poisson_single(x, popt_poiss[0])*linear(x-popt_poiss[0], 1),0,np.inf)

    if(plot_show == True):

        plt.title('Expected Function')

        plt.ylabel('P(ΔR)*ΔR')
        plt.xlabel('ΔR (cm)')

        plt.plot(base_fit-popt_poiss[0],expected_dist,label=("E[ΔR]: " + str(np.round(E[0],6)) + " with integration error: " + str(np.round(E[1],6))))

        plt.legend(loc='upper left')

        plt.show()

    variance_dist = (dist_fit**2)*poiss_dist_fit

    V = integrate.quad(lambda x: poisson_single(x, popt_poiss[0])*(linear(x-popt_poiss[0], 1)**2),0,np.inf)

    V += (-E[0]**2,2*E[1])

    S = (np.sqrt(V[0]),0.5*V[1])

    if(plot_show == True):

        plt.title('Variance Function')

        plt.ylabel('P(ΔR)*ΔR^2')
        plt.xlabel('ΔR (cm)')

        plt.plot(base_fit,variance_dist,label=("SD[ΔR]: " + str(np.round(S[0],2)) + " with integration error: " + str(np.round(S[1],2))))

        plt.legend(loc='upper left')

        plt.show()

    
    radius_error = (E[0] + S[0])/np.sqrt(3) #maybe instead of 2* add the errors from 1st and 2nd decay in quadrature

    return (radius_error,np.sqrt(8*popt[0]),popt[0]/popt_poiss[0])

def angle_error(data,gaussian_data,plot_show):
    #5 - proton, 6 - piom

    scan_width = (np.max(gaussian_data[:,5])-np.min(gaussian_data[:,5]))/(40)

    popt_gauss_proton, pcov_gauss_proton, bins_gauss_proton, counts_gauss_proton = fit_gaussian(gaussian_data[:,5], scan_width)

    width_proton = np.max(bins_gauss_proton) - np.min(bins_gauss_proton)

    gauss_base_proton = np.linspace(-width_proton,width_proton,100)

    gauss_dist_proton = gaussian(gauss_base_proton, popt_gauss_proton[0])

    if(plot_show == True):

        plt.title('Angle Measurements Distribution Proton (Δϕ)')

        plt.ylim((0,1.1*np.max(gauss_dist_proton)))

        plt.ylabel('Relative Frequency')
        plt.xlabel('Δϕ (rad)')

        plt.stairs(np.max(gauss_dist_proton)*counts_gauss_proton/np.max(counts_gauss_proton),bins_gauss_proton)

        plt.plot(gauss_base_proton,gauss_dist_proton,label=('Standard Deviation: ' + str(np.round(popt_gauss_proton[0],3))))

        plt.legend()

        plt.show()

    scan_width = (np.max(gaussian_data[:,6])-np.min(gaussian_data[:,6]))/(50)

    popt_gauss_pion, pcov_gauss_pion, bins_gauss_pion, counts_gauss_pion = fit_gaussian(gaussian_data[:,6], scan_width)

    width_pion = np.max(bins_gauss_pion) - np.min(bins_gauss_pion)

    gauss_base_pion = np.linspace(-width_pion,width_pion,100)

    gauss_dist_pion = gaussian(gauss_base_pion, popt_gauss_pion[0])

    if(plot_show == True):

        plt.title('Angle Measurements Distribution Pion (Δϕ)')

        plt.ylim((0,1.1*np.max(gauss_dist_pion)))

        plt.ylabel('Relative Frequency')
        plt.xlabel('Δϕ (rad)')

        plt.stairs(np.max(gauss_dist_pion)*counts_gauss_pion/np.max(counts_gauss_pion),bins_gauss_pion)

        plt.plot(gauss_base_pion,gauss_dist_pion,label=('Standard Deviation: ' + str(np.round(popt_gauss_pion[0],3))))

        plt.legend()

        plt.show()
        
    data = data[:,np.array([5,6])]
    

    data_proton = data[:,0]

    data_proton = data_proton[np.array([27,26,25,24,23,22,28,29,30,31,32])]-data_proton[22] 

    length_error_base = np.linspace(-2*pixel_width,2*pixel_width,11)

    popt_proton= fit_linear(length_error_base, data_proton)

    base = np.linspace(-2*pixel_width,2*pixel_width,100)

    dist_proton = linear(base, popt_proton[0])

    if(plot_show == True):

        plt.title('Angle Error Relation Proton')

        plt.ylabel('Δϕ (rad)')
        plt.xlabel('Δx (cm)')

        plt.plot(length_error_base,data_proton)

        plt.plot(base,dist_proton,label=("Gradient: " + str(np.round(popt_proton[0],2)))) 

        plt.legend()

        plt.show()

    data_pion = data[:,1]

    data_pion = data_pion[np.array([16,15,14,13,12,11,17,18,19,20,21])]-data_pion[11] 

    popt_pion= fit_linear(length_error_base, data_pion)

    dist_pion = linear(base, popt_pion[0])

    if(plot_show == True):

        plt.title('Angle Error Relation Pion')

        plt.ylabel('Δϕ (rad)')
        plt.xlabel('Δx (cm)')

        plt.plot(length_error_base,data_pion)

        plt.plot(base,dist_pion,label=("Gradient: " + str(np.round(popt_pion[0],2)))) 

        plt.legend()

        plt.show()
    
    angle_error_proton = 2*popt_gauss_proton[0] #times 2 since it depends on the lambda path as well
    angle_error_pion = 2*popt_gauss_pion[0]

    return angle_error_proton, angle_error_pion


def angle_error_2(data, gaussian_data): #redundent
    #proton - 5, pion - 6

    error_phi_proton = []
    error_phi_pion = []

    

    x = np.arange(-0.0005,0.0006,0.0001) #change for width of line

    x_base = np.linspace(-0.0004,0.0004,100)

    proton_vars = fit_gaussian(gaussian_data,12)
    pion_vars = fit_gaussian(gaussian_data,13)

    x_5 = np.linspace(np.min(data[:,5]),np.max(data[:,5]),11)
    x_6 = np.linspace(np.min(data[:,6]),np.max(data[:,6]),11)

    x_5_base = np.linspace(np.min(data[:,5]),np.max(data[:,5]),100)
    x_6_base = np.linspace(np.min(data[:,6]),np.max(data[:,6]),100)

    pion_phi_base = np.linspace(np.min(data[:,6]),np.max(data[:,6]),100)
    proton_phi_base = np.linspace(np.min(data[:,5]),np.max(data[:,5]),100)

    proton_gaussian_dist = gaussian(proton_phi_base,*proton_vars[0]) #change width based on line measurement
    pion_gaussian_dist = gaussian(pion_phi_base,*pion_vars[0]) #change width based on line measurement

    plt.title("Proton Gaussian Distribution")

    plt.plot(proton_phi_base,proton_gaussian_dist)

    plt.show()

    plt.title("Pion Gaussian Distribution")

    plt.plot(pion_phi_base, pion_gaussian_dist)

    plt.show()

    for t in range(0,3):    

        dist_5 = data[t*11:(t+1)*11,5]

        dist_6 = data[t*11:(t+1)*11,6]
        
        add_5 = np.flip(dist_5[:6])

        add_6 = np.flip(dist_6[:6])

        dist_5 = np.concatenate((add_5, dist_5[6:]))

        dist_5 = np.abs(dist_5 - dist_5[5])

        dist_6 = np.concatenate((add_6, dist_6[6:]))

        dist_6 = np.abs(dist_6 - dist_6[5])

        print(np.max(dist_5))

        popt5, pcov5 = curve_fit(v_linear, x_5, dist_5)

        error_dist5 = v_linear(x_5_base, *popt5)# proton_gaussian_dist * 

        popt6, pcov6 = curve_fit(v_linear, x_6, dist_6)

        error_dist6 = pion_gaussian_dist * v_linear(x_base, *popt6)

        error_phi_proton.append(error_dist5)
        error_phi_pion.append(error_dist6)

    error_dist5 = np.sqrt(error_phi_proton[0]**2 + error_phi_proton[1]**2 + error_phi_proton[2]**2)
    error_dist6 = np.sqrt(error_phi_pion[0]**2 + error_phi_pion[1]**2 + error_phi_pion[2]**2)

    plt.title("Error Phi Proton Distribution")

    plt.plot(x_5_base, error_dist5)

    plt.plot(x_5_base, (np.zeros([100])+np.max(error_dist5)), '--', label=str(np.max(error_dist5)))

    plt.legend()

    plt.show()

    plt.title("Error Phi Pion Distribution")

    plt.plot(x_base, error_dist6)

    plt.plot(x_base, (np.zeros([100])+np.max(error_dist6)), '--', label=str(np.max(error_dist6)))

    plt.legend()

    plt.show()

    return np.max(error_dist5), np.max(error_dist6)



def p_neutral_error(mass,P,mass_error):

    E = m_kaon + m_proton

    error =(2*mass/P)*((E**2 + m_pion_o**2 - mass**2)/(2*E)**2)*mass_error

    return error

def lambda_mass_error(alpha,alpha_error,mass):

    error = (m_proton**2 + m_pion**2)*alpha_error/(2*mass*alpha**2)

    return error

def error_alpha(pion_angle,proton_angle,pion_angle_error,proton_angle_error):

    term1 = np.cos(pion_angle - proton_angle)/np.sin(pion_angle + proton_angle)

    term2 = np.sin(pion_angle - proton_angle)*np.cos(pion_angle+proton_angle)/(np.sin(pion_angle + proton_angle))**2

    error = np.sqrt((term1-term2)**2 * pion_angle_error**2 + (term1+term2)**2 * proton_angle_error**2)

    return error

def error_E(mass,P,E,mass_error,p_error):

    error = np.sqrt((mass/E)**2 * mass_error**2 + (P/E)**2 * p_error**2)

    return error

def corrected_tau_error(mass,P,E,tau,mass_error,p_error,tau_error):

    E_error = error_E(mass,P,E,mass_error,p_error)

    term = (E**3)*3e8*(0.248)/(mass*P**3)

    error = np.sqrt((1+tau*term)**2 * tau_error**2 + (3*term*tau**2/(2*P**4))**2 * p_error**2 + (term*tau**2/(2*mass))**2 * mass_error**2 + (3*term*tau**2/(2*E))**2 * E_error**2)

    return error

def error_tau(m,P,lmbda,lambda_error,mass_error,p_error):

    error = np.sqrt((m/(P*3e8))**2 * lambda_error**2 + (lmbda/(P*3e8))**2 * mass_error**2 + ((m*lmbda)/(3e8*P**2))**2 * p_error**2)

    return error

def L_error_neutral(L,L_perp,Lz,error_L_perp,error_L_z):

    error = np.sqrt((L_perp/L)**2 * error_L_perp**2 + (Lz/L)**2 * error_L_z**2)

    return error

def L_error(L_val,Lperp,Lperp_error,pperp,pperp_error,mom, perror):

    error = np.sqrt((L_val/Lperp)**2 * Lperp_error**2 + (L_val/pperp)**2 * pperp_error**2 + (L_val/mom)**2 * perror**2)

    return error

def charged_m_error(momentum,mass,p_error):

    error = ((1/mass)*((m_proton + m_kaon)-np.sqrt((m_pion**2 + momentum**2)))*(momentum/(m_pion**2 + momentum**2)**0.5)+momentum)*p_error

    return error

def P_perp_error(rerror):

    error = 3*1.625*rerror

    return error

def R_error(mag_a,mag_b,depth,dx,L,R,d):

    d_error = d_or_l_error(mag_a,mag_b,depth,dx*d/R)

    l_error = d_or_l_error(mag_a,mag_b,depth,dx*L/R)
    
    error = np.sqrt((2*R/L)**2 * l_error**2 + (L**2/(8*d**2))**2 * d_error**2)

    return error

def d_or_l_error(mag_a,mag_b,depth,dx):

    error = np.sqrt((mag_a + mag_b*depth)**2 * dx_error**2 + (mag_b * dx)**2 * z_error**2)

    return error   

def fit_gaussian(data,scan_width):
    
    popt, pcov, bins, counts = iterate_best_width(data, non_normal_gaussian, scan_width)

    return np.abs(popt), pcov, bins, counts

def fit_linear(x, y):

    popt, pcov = curve_fit(linear, x, y)

    return popt

def fit_inverse(x, y):

    popt, pcov = curve_fit(inverse_proportion, x, y)

    return popt

def fit_poisson(data, scan_width):

    popt, pcov, bins, counts = iterate_best_width(data, poisson, scan_width)

    return popt, pcov, bins, counts

def isotropy(momentum,perp_moms):

    perp_moms = perp_moms/momentum

    values = []
    for val in perp_moms:
        if(val < 1):
            values.append(val)
        
    values = np.array(values)

    bin_w = 20

    counts, bins = np.histogram(values,bins = bin_w)

    avg = bins*0 + np.average(counts)

    plt.stairs(counts,bins)

    plt.plot(bins,avg)

    plt.title('Distribution of Sin θ Values')

    plt.ylabel('Frequency')
    plt.xlabel('|Sin θ|')

    plt.show()

    return np.average(values)


    
    

data,error_data = get_data('/Users/sampalmer/Library/CloudStorage/OneDrive-UniversityofCambridge/E2b/Particle Tracks  - SP/event 0-115.csv', 'all')

angle_gaussian_data, discard = get_data('/Users/sampalmer/Library/CloudStorage/OneDrive-UniversityofCambridge/E2b/Particle Tracks  - SP/Angle Gaussian Measurements.csv','all')
angle_errors, discard = get_data('/Users/sampalmer/Library/CloudStorage/OneDrive-UniversityofCambridge/E2b/Particle Tracks  - SP/Angle Measurement Error 2.csv', 'all')

length_gaussian_data, discard = get_data('/Users/sampalmer/Library/CloudStorage/OneDrive-UniversityofCambridge/E2b/Particle Tracks  - SP/Length Gaussian Measurements.csv',7)
length_error_data, discard = get_data('/Users/sampalmer/Library/CloudStorage/OneDrive-UniversityofCambridge/E2b/Particle Tracks  - SP/Length Error measurements.csv',7)

radius_gaussian_data, discard = get_data('/Users/sampalmer/Library/CloudStorage/OneDrive-UniversityofCambridge/E2b/Particle Tracks  - SP/Radius Gaussian Measurements.csv',5)
radius_error_data, discard = get_data('/Users/sampalmer/Library/CloudStorage/OneDrive-UniversityofCambridge/E2b/Particle Tracks  - SP/Radius Error Measurements.csv',5)

error_of_length = length_error(length_error_data, length_gaussian_data, False)

error_of_radius = radius_error(radius_error_data, radius_gaussian_data, False)

proton_angle_error, pion_angle_error = angle_error(angle_errors, angle_gaussian_data, False)



plus_mass, plus_mom, plus_perp_mom, plus_mass_error, plus_p_perp_error, plus_p_error = calc_sigma_mass(data, 'plus',error_data,error_of_radius)                               
minus_mass, minus_mom, minus_perp_mom, minus_mass_error, minus_p_perp_error, minus_p_error = calc_sigma_mass(data, 'minus',error_data,error_of_radius)
neutral_mass, neutral_mass_error = calc_lambda_mass(data,proton_angle_error,pion_angle_error)
print("Average Σ⁻ Mass: " + str(np.round(minus_mass,0)) + " +/- " + str(np.round(minus_mass_error,0)) + " Mev/c^2")
print("Average Σ⁺ Mass: " + str(np.round(plus_mass,0)) + " +/- " + str(np.round(plus_mass_error,0)) +  " Mev/c^2")
print("Average Λ⁰ Mass: " + str(neutral_mass) + " +/- " + str(np.round(neutral_mass_error,0)) + " Mev/c^2")
print("")
proton_ratio, pion_ratio, pr_ratio_error, pi_ratio_error = calc_branch_ratio(data, "sigma")
print(str(proton_ratio) + "% of Σ⁺ decays were Σ⁺ ⇨ p + π⁰ decays, error: " + str(pr_ratio_error))
print(str(pion_ratio) + "% of Σ⁺ decays were Σ⁺ ⇨ n + π⁺ decays,error: "+ str(pi_ratio_error))
neutral_ratio, v_ratio, neutral_ratio_error, v_ratio_error = calc_branch_ratio(data, "lambda")
print(str(neutral_ratio) + "% of Λ⁰ decays were Λ⁰ ⇨ n + π⁰ decays, error: " + str(neutral_ratio_error))
print(str(v_ratio) + "% of Λ⁰ decays were Λ⁰ ⇨ p + π⁻ decays,error: "+ str(v_ratio_error))
print("")
calc_sigma_lifetime(data, "plus", plus_mom, plus_perp_mom, plus_mass, error_of_length,error_data, plus_p_perp_error, plus_p_error, plus_mass_error)
calc_sigma_lifetime(data, 'minus', minus_mom, minus_perp_mom, minus_mass, error_of_length,error_data, minus_p_perp_error, minus_p_error, minus_mass_error)
calc_lambda_lifetime(data, neutral_mass, error_data, error_of_length, neutral_mass_error)


#add error calcs, error in isotropic assumption, error between normal and Yuval's method, develop formula to produce best histogram bins
#systematic errors like charge capture, error from 'large' sample averaging being met
#error cannot be in the order of 9kev since an assumption says this is negligible


    
