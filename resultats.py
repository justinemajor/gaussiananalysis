import numpy as np
import uncertainties
import matplotlib.pyplot as plt
import scipy.optimize as opt
from calculs_exp import proprietes_diff, dia_cible, err_dia_cible, Z
from calculs_theoriques import section_eff as section_eff_th

# AJUSTER CODE POUR AVOIR DEJA NOMBRE DE COMPTES NET PAR SECONDE

### général ###

def erreur_mult_div(res, x, delta_x, y, delta_y):
    return res * np.sqrt((delta_x/x)**2 + (delta_y/y)**2)

def erreur_comptes(comptes):
    return np.sqrt(comptes)


### fraction des photons s’échappe de chaque diffuseur selon l’énergie des photons détectés (incluant θ = 0)

def fraction_photons_diffuses(nbre_comptes_ang, nbre_comptes_ref):
    return nbre_comptes_ang / nbre_comptes_ref

def erreur_fraction_photons_diffuses(comptes_ang, comptes_ref):
    err_comptes_ang = erreur_comptes(comptes_ang)
    err_comptes_ref = erreur_comptes(comptes_ref)
    frac = fraction_photons_diffuses(comptes_ang, comptes_ref)
    return erreur_mult_div(res=frac, x=comptes_ang, delta_x=err_comptes_ang, y=comptes_ref, delta_y=err_comptes_ref)


### Le nombre d’électron vu par la source pour chaque diffuseur

""" voir fichier calculs_exp.py """


### « net area/sec » corrigé pour chaque diffuseur/angle + section efficace

def nombre_coups_brutes_par_seconde(coups, t_coups):
    return coups/t_coups

def nombre_coups_par_seconde(coups_brutes_s, bruit_s):
    return coups_brutes_s - bruit_s

def erreur_nombre_coups_brutes_par_seconde(coups, t_coups):
    return erreur_comptes(comptes=coups) / t_coups

def erreur_nombre_coups_par_seconde(err_coups_s, err_bruit_s):
    return np.sqrt(err_coups_s**2 + err_bruit_s**2)


def attenuation_coefficient(coups_s, ref_coups_s, d_cible):
    return np.log(ref_coups_s/coups_s) / d_cible

def erreur_attenuation_coefficient(coups_s, err_coups_s, ref_coups_s, err_ref_coups_s, d_cible, err_d_cible):
    d_dI0 = 1 / (ref_coups_s*d_cible)
    d_dI = -1 / (coups_s*d_cible)
    d_dd = -np.log(ref_coups_s/coups_s) / d_cible**2
    return np.sqrt((d_dI0*err_ref_coups_s)**2 + (d_dI*err_coups_s)**2 + (d_dd*err_d_cible)**2)


def correction_attenuation(coups_sec, mu, d_cible):
    return coups_sec / np.exp(-mu*d_cible/2)

def erreur_correction_attenuation(coups_sec, err_coups_sec, mu, err_mu, d_cible, err_d_cible):
    d_dI = np.exp(mu*d_cible/2)
    d_dmu = coups_sec*d_cible/2*np.exp(mu*d_cible/2)
    d_dd = coups_sec*mu/2*np.exp(mu*d_cible/2)
    return np.sqrt((d_dI*err_coups_sec)**2 + (d_dmu*err_mu)**2 + (d_dd*err_d_cible)**2)

def correction_efficacite_detecteur(coups_sec, eff):
    return coups_sec/eff

def erreur_correction_efficacite_detecteur(corr_eff, coups_sec, err_coups_sec, eff, err_eff):
    return erreur_mult_div(res=corr_eff, x=coups_sec, delta_x=err_coups_sec, y=eff, delta_y=err_eff)

def section_efficace_diff_exp(coups_sec, flux_inc, n, angle_solide):
    return coups_sec / (flux_inc*n*angle_solide)

def erreur_section_efficace_diff_exp(section_eff, coups_sec, err_coups_sec, flux_inc, err_flux_inc, n, err_n, angle_solide, err_angle_solide):
    return section_eff * np.sqrt((err_coups_sec/coups_sec)**2 + (err_flux_inc/flux_inc)**2 + (err_n/n)**2 + (err_angle_solide/angle_solide)**2)


efficacites_gamma = {'0':(0.16, 0.02),
                     '45':(0.37, 0.02),
                     '60':(0.44, 0.02),
                     '75':(0.52, 0.02),
                     '90':(0.65, 0.05), 
                     '110':(0.75, 0.05)} # (efficacite, incertitude)

efficacites_elec = {'45':(0.9, 0.1),
                     '60':(0.65, 0.05),
                     '75':(0.54, 0.02),
                     '90':(0.47, 0.02), 
                     '110':(0.42, 0.02)} # (efficacite, incertitude)

"""
coups = {'plexi_0': 25.523, # VALEURS FICTIVES
        'al_0' : 25.535,
        'fe_0' : 25.541,
        'al_45' : 25.535,
        'plexi_60': 25.523,
        'al_60' : 25.535,
        'fe_60' : 25.541,
        'al_moyen_60' : 15.898,
        'al_petit_60' : 15.271,
        'al_mini_60' : 5.630,
        'al_75' : 5.535,
        'al_90' : 5.535,
        'al_110' : 5.535}

t_coups = {'plexi_0': 5,  # REVOIR ANGLES 0
        'al_0' : 5,
        'fe_0' : 5,
        'al_45' : 2.535,
        'plexi_60': 2.523,
        'al_60' : 2.535,
        'fe_60' : 5,
        'al_moyen_60' : 1.898,
        'al_petit_60' : 1.271,
        'al_mini_60' : 621,
        'al_75' : 2.535,
        'al_90' : 2.535,
        'al_110' : 2.535}

bruit = {'0': 5,  # REVOIR ANGLES 0
        '45' : 0.535,
        '60': 0.523,
        '75' : 0.535,
        '90' : 0.535,
        '110' : 0.535}

t_bruit = {'0': 2.553,  # VALEURS FICTIVES
        '45' : 2.535,
        '60' : 2.541,
        '75' : 2.535,
        '90' : 2.535,
        '110' : 2.535}
"""

counts_sec = {'plexi_0': 4539.41,
        'al_0' : 3646.8083333333334,
        'fe_0' : 1642.625641025641,
        'al_45' : 27.958123953098827,
        'plexi_60': 13.054878048780488,
        'al_60' : 21.259818731117825,
        'fe_60' : 26.520900321543408,
        'al_moyen_60' : 14.119444444444444,
        'al_petit_60' : 6.93631669535284,
        'al_mini_60' : 1.8325281803542672,
        'al_75' : 17.54590984974958,
        'al_90' : 14.945773524720893,
        'al_110' : 14.276018099547512}

err_counts_sec = {'plexi_0': 52.131614378463325,
        'al_0' : 35.90278832394886,
        'fe_0' : 11.326086002220272,
        'al_45' : 0.26323580745054953,
        'plexi_60': 0.2393044836729025,
        'al_60' : 0.3176635868767995,
        'fe_60' : 0.37729710816554735,
        'al_moyen_60' : 0.23726279325157937,
        'al_petit_60' : 0.12120240565880956,
        'al_mini_60' : 0.057273403811948884,
        'al_75' : 0.20044106995102506,
        'al_90' : 0.17822918511157765,
        'al_110' : 0.16827186977251365}

attenuation_coef = {}

attenuation_coef_th_inc = {'plexi': 0.08372 *1.19,
                    'al' : 0.07504 *2.70,
                    'fe' : 0.07392 *7.86}

attenuation_coef_th_diff = {'plexi_0': 0.08372 *1.19,
                    'al_0' : 0.07504 *2.70,
                    'fe_0' : 0.07392 *7.86,
                    'al_45' : 0.08612 *2.70,
                    'plexi_60': 0.1029 *1.19,
                    'al_60' : 0.09261 *2.70,
                    'fe_60' : 0.09383 *7.86,
                    'al_moyen_60' : 0.09261 *2.70,
                    'al_petit_60' : 0.09261 *2.70,
                    'al_mini_60' : 0.09261 *2.70,
                    'al_75' : 0.09988 *2.70,
                    'al_90' : 0.1063 *2.70,
                    'al_110' : 0.1147 *2.70}

nbre_coups_net_par_sec = {}
section_eff_diff = {}

#ref_comptes_0deg = 200
#t_ref_0deg = 5
#ref_comptes_sec = nombre_coups_brutes_par_seconde(coups=ref_comptes_0deg, t_coups=t_ref_0deg)
#err_ref_comptes_sec = erreur_nombre_coups_brutes_par_seconde(coups=ref_comptes_0deg, t_coups=t_ref_0deg)

#d_tot = 4.5+11.5+22.5+5
#angle_solide_ref = np.pi * (2.54/2)**2 /(d_tot)**2
#ref_comptes_sec = 5691.44578313253 / angle_solide_ref
#err_ref_comptes_sec = ref_comptes_sec * np.sqrt((76.8524351746035/5691.44578313253)**2 + 2*(0.01/2.54)**2 + 2*(0.2/d_tot)**2) 

ref_comptes_sec = 5691.44578313253 
err_ref_comptes_sec = 76.8524351746035

loop = ['plexi_0', 'al_0', 'fe_0', 'al_45', 'plexi_60', 'al_60', 'al_moyen_60', 'al_petit_60', 'al_mini_60', 'fe_60', 'al_75', 'al_90', 'al_110']

for acquisition in loop:
    print(acquisition)
    description = acquisition.split('_')
    if len(description)==3:
        cible=f'{description[0]}_{description[1]}'
    else:
        cible=description[0]
    angle_diff=description[-1]
    I_inc, err_I_inc = proprietes_diff[cible]['flux']
    nbre_elec, err_nbre_elec = proprietes_diff[cible]['n']
    angle_solide, err_angle_solide = proprietes_diff[cible]['angle solide']
    densite_elec = proprietes_diff[cible]['densite elec'][0]
    densite_surface_elec = proprietes_diff[cible]['densite surface elec']
    D_cible = dia_cible[cible]

    #nbre_coups_brutes_sec = nombre_coups_brutes_par_seconde(coups=coups[acquisition], t_coups=t_coups[acquisition])
    #err_nbre_coups_brutes_sec = erreur_nombre_coups_brutes_par_seconde(coups=coups[acquisition], t_coups=t_coups[acquisition])
    #nbre_bruit_sec = nombre_coups_brutes_par_seconde(coups=bruit[angle_diff], t_coups=t_bruit[angle_diff])
    #err_nbre_bruit_sec = erreur_nombre_coups_brutes_par_seconde(coups=bruit[angle_diff], t_coups=t_bruit[angle_diff])

    #nbre_coups_sec_nocorr = nombre_coups_par_seconde(coups_brutes_s=nbre_coups_brutes_sec, bruit_s=nbre_bruit_sec)
    #err_nbre_coups_sec_nocorr = erreur_nombre_coups_par_seconde(err_coups_s=err_nbre_coups_brutes_sec, err_bruit_s=err_nbre_bruit_sec)

    nbre_coups_sec_nocorr = counts_sec[acquisition]
    err_nbre_coups_sec_nocorr = err_counts_sec[acquisition]

    if angle_diff == '0':
        mu = attenuation_coefficient(coups_s=nbre_coups_sec_nocorr, ref_coups_s=ref_comptes_sec, d_cible=D_cible)
        err_mu = erreur_attenuation_coefficient(coups_s=nbre_coups_sec_nocorr, err_coups_s=err_nbre_coups_sec_nocorr, ref_coups_s=ref_comptes_sec, err_ref_coups_s=err_ref_comptes_sec, d_cible=D_cible, err_d_cible=err_dia_cible)
        attenuation_coef[description[0]] = (mu, err_mu)
        mu_pm = uncertainties.ufloat(mu, err_mu)
        print(f"Coefficient d'atténuation : {mu_pm:.1u}")

    nbre_coups_sec_mu_corr = correction_attenuation(coups_sec=nbre_coups_sec_nocorr, mu=attenuation_coef_th_diff[acquisition], d_cible=D_cible)
    err_nbre_coups_sec_mu_corr = erreur_correction_attenuation(coups_sec=nbre_coups_sec_nocorr, err_coups_sec=err_nbre_coups_sec_nocorr, mu=attenuation_coef_th_diff[acquisition], err_mu=0, d_cible=D_cible, err_d_cible=err_dia_cible)

    nbre_coups_sec_corr = correction_efficacite_detecteur(coups_sec=nbre_coups_sec_mu_corr, eff=efficacites_gamma[angle_diff][0])
    err_nbre_coups_sec_corr = erreur_correction_efficacite_detecteur(corr_eff=nbre_coups_sec_corr, coups_sec=nbre_coups_sec_mu_corr, err_coups_sec=err_nbre_coups_sec_mu_corr, eff=efficacites_gamma[angle_diff][0], err_eff=efficacites_gamma[angle_diff][1])
    nbre_coups_sec_corr_pm = uncertainties.ufloat(nbre_coups_sec_corr, err_nbre_coups_sec_corr)
    print(f'Nombre de coups net par seconde : {nbre_coups_sec_corr_pm:.1u}')

    nbre_coups_net_par_sec[acquisition] = (nbre_coups_sec_corr, err_nbre_coups_sec_corr)

    section_eff = section_efficace_diff_exp(coups_sec=nbre_coups_sec_corr, flux_inc=I_inc, n=nbre_elec, angle_solide=angle_solide)
    err_section_eff = erreur_section_efficace_diff_exp(section_eff=section_eff, coups_sec=nbre_coups_sec_corr, err_coups_sec=err_nbre_coups_sec_corr, flux_inc=I_inc, err_flux_inc=err_I_inc, n=nbre_elec, err_n=err_nbre_elec, angle_solide=angle_solide, err_angle_solide=err_angle_solide)
    section_eff_pm = uncertainties.ufloat(section_eff, err_section_eff)
    print(f'Section efficace : {section_eff_pm:.1u}')
    
    section_eff_diff[acquisition] = (section_eff, err_section_eff)

    frac_diff = fraction_photons_diffuses(nbre_comptes_ang=nbre_coups_sec_corr, nbre_comptes_ref=ref_comptes_sec)
    err_frac_diff = erreur_fraction_photons_diffuses(comptes_ang=nbre_coups_sec_corr, comptes_ref=ref_comptes_sec)
    frac_diff_pm = uncertainties.ufloat(frac_diff, err_frac_diff)
    print(f'Fraction des électrons diffusés : {frac_diff_pm:.1u}')



### graph nombre de coups net (Net area)/sec en fonction de Z pour un angle fixe pour les diffuseurs de même taille physique

def graph_nbre_coups_fct_Z(nbre_coups_sec:dict, Z:dict):
    coups = [nbre_coups_sec['plexi_60'][0], nbre_coups_sec['al_60'][0], nbre_coups_sec['fe_60'][0]]
    err_coups = [nbre_coups_sec['plexi_60'][1], nbre_coups_sec['al_60'][1], nbre_coups_sec['fe_60'][1]]
    z = [Z['plexi'], Z['al'], Z['fe']]
    plt.plot(z, coups, 'k.')
    plt.errorbar(z, coups, yerr=err_coups, capsize=5, c='k', fmt='.')
    plt.xlabel('Z')
    plt.ylabel('Nombre de coups net / sec')
    plt.show()

graph_nbre_coups_fct_Z(nbre_coups_sec=nbre_coups_net_par_sec, Z=Z)


### graph nombre de coups net(Net area)/sec en fonction de la taille ( é/cm2 ) pour un angle fixe et Z fixe
    
def graph_nbre_coups_fct_densite_elec(nbre_coups_sec:dict, proprietes:dict):
    coups = [nbre_coups_sec['al_60'][0], nbre_coups_sec['al_moyen_60'][0], nbre_coups_sec['al_petit_60'][0], nbre_coups_sec['al_mini_60'][0]]
    err_coups = [nbre_coups_sec['al_60'][1], nbre_coups_sec['al_moyen_60'][1], nbre_coups_sec['al_petit_60'][1], nbre_coups_sec['al_mini_60'][1]]
    #densite = [proprietes['al']['densite elec'][0], proprietes['al_moyen']['densite elec'][0], proprietes['al_petit']['densite elec'][0], proprietes['al_mini']['densite elec'][0]]
    densite_surf = [proprietes['al']['densite surface elec'][0], proprietes['al_moyen']['densite surface elec'][0], proprietes['al_petit']['densite surface elec'][0], proprietes['al_mini']['densite surface elec'][0]]
    err_densite_surf = [proprietes['al']['densite surface elec'][1], proprietes['al_moyen']['densite surface elec'][1], proprietes['al_petit']['densite surface elec'][1], proprietes['al_mini']['densite surface elec'][1]]
    plt.plot(densite_surf, coups, 'k.')
    plt.errorbar(densite_surf, coups, xerr=err_densite_surf, yerr=err_coups, capsize=5, c='k', fmt='.')
    #plt.xlabel('Densité électronique [é/cm$^3$]')
    plt.xlabel('Taille du diffuseur [é/cm$^2$]')
    plt.ylabel('Nombre de coups net / sec')
    plt.show()


graph_nbre_coups_fct_densite_elec(nbre_coups_sec=nbre_coups_net_par_sec, proprietes=proprietes_diff)


### graph sections efficaces différentielles théoriques et expérimentales en fonction de l’angle    
    
def graph_section_eff_fct_angle(section_eff_exp:dict, section_eff_th:dict):
    angles_th = []
    section_eff_th_plot = []
    angles_exp = []
    section_eff_exp_plot = []
    err_section_eff_exp_plot = []

    for key in section_eff_exp.keys():
        if '_0' in key:
            continue
        angles_exp.append(float(key.split('_')[-1]))
        section_eff_exp_plot.append(section_eff_exp[key][0])
        err_section_eff_exp_plot.append(section_eff_exp[key][1])

    for key in section_eff_th.keys():
        if key == '0':
            continue
        angles_th.append(float(key))
        section_eff_th_plot.append(section_eff_th[key])
    
    plt.plot(angles_th, section_eff_th_plot, 'kx', label="Théorique")
    plt.plot(angles_exp, section_eff_exp_plot, 'k.', label="Expérimental")
    plt.errorbar(angles_exp, section_eff_exp_plot, yerr=err_section_eff_exp_plot, xerr=[1 for i in range(len(angles_exp))], capsize=5, c='k', fmt='.')
    plt.xlabel('Angle de diffusion [°]')
    plt.ylabel('Section efficace différentielle [cm$^2$]')
    plt.legend(loc='best')
    plt.show()
    
graph_section_eff_fct_angle(section_eff_exp=section_eff_diff, section_eff_th=section_eff_th)


E_gamma_diff_th = np.array([479.90, 401.76, 337.72, 288.39, 241.73])

E_gamma_diff = np.array([485.8797788475164, 410.904721021309, 345.5373989648364, 297.1882484528253, 249.33693954179228])
err_E_gamma_diff = np.array([5.179719733616313, 5.011308943380486, 4.617934156014403, 4.373734944475478, 4.1489599791758565])

E_gamma_inc = 661.8816073270356
err_E_gamma_inc = 5.4307997625784346


T_elec_th = np.array([182.10, 260.24, 324.28, 373.61, 420.27])

T_elec_exp = np.array([188.35927695155502, 263.845770665512, 329.0085861986625, 381.12805365559063, 427.9927595377146])
err_T_elec_exp = np.array([5.652612262337268, 5.918388327917103, 5.9572444601002825, 6.012876570490097, 6.010448968062331])


### graph énergie hv’ du gamma diffusé vs angle de diffusion avec barres d’erreur (±sigma ou encore ±FWHM/2.35)

def graph_energie_diffuse_fct_angle(E_diff_exp, err_E_diff, E_diff_th):
    angles = [45, 60, 75, 90, 110] # mettre 0 ou non??
    plt.plot(angles, E_diff_th, 'kx', label="Théorique")
    plt.plot(angles, E_diff_exp, 'k.', label='Expérimental')
    plt.errorbar(angles, E_diff_exp, yerr=err_E_diff, xerr=[1 for i in range(len(angles))], capsize=5, c='k', fmt='.')
    plt.xlabel('Angle de diffusion [°]')
    plt.ylabel('Énergie du gamma diffusé [keV]')
    plt.legend(loc='best')
    plt.show()

graph_energie_diffuse_fct_angle(E_diff_exp=E_gamma_diff, err_E_diff=err_E_gamma_diff, E_diff_th=E_gamma_diff_th)


### graph linéarité de la relation entre le rapport ho / hv’ et (1 - cos theta)

def func_lineaire(x, a, b):
    return a*x + b

def erreur_rapport_energies(E_num, err_E_num, E_denum, err_E_denum):
    return erreur_mult_div(res=E_num/E_denum, x=E_num, delta_x=err_E_num, y=E_denum, delta_y=err_E_denum)

def erreur_1_costheta(theta, err_theta):
    return abs(np.sin(theta)*err_theta)

#def erreur_1_costheta(res, err_theta_deg):
#    return abs(res*err_theta_deg/360)

def graph_linearite_ratioEgamma_costheta(E_inc, err_E_inc, E_diff, err_E_diff):
    rapport = E_inc/E_diff
    err_rapport = erreur_rapport_energies(E_inc, err_E_inc, E_diff, err_E_diff)
    angles = np.array([45, 60, 75, 90, 110]) # mettre 0 ou non??
    err_angles = 1/360*2*np.pi # ou transformer en rad ??? à voir selon erreur graph
    x=1-np.cos(np.deg2rad(angles))
    err_x = erreur_1_costheta(angles, err_angles)
    plt.plot(x, rapport, 'k.', label='Données')
    #plt.errorbar(x, rapport, xerr=err_x, yerr=err_rapport, capsize=5, c='k', fmt='.')
    plt.errorbar(x, rapport, yerr=err_rapport, capsize=5, c='k', fmt='.')

    x_axis = np.linspace(0, 2)
    popt, pcov = opt.curve_fit(func_lineaire, x, rapport)
    std_popt = np.sqrt(np.diag(pcov))
    plt.plot(x_axis,func_lineaire(x_axis, *popt), 'k-', label=f'Régression ({popt[0]:.3f} +- {std_popt[0]:.3f}) x + ({popt[1]:.3f} +- {std_popt[1]:.3f})')
    print(f'Régression linéaire ({popt[0]} +- {std_popt[0]}) x + ({popt[1]} +- {std_popt[1]})')

    plt.xlabel('$1 - \cos(θ)$ [-]')
    plt.ylabel("$hν_0 / hν'$ [-]")
    plt.legend(loc='best')
    plt.show()

graph_linearite_ratioEgamma_costheta(E_inc=E_gamma_inc, err_E_inc=err_E_gamma_inc, E_diff=E_gamma_diff, err_E_diff=err_E_gamma_diff)


### validation conservation énergie en fonction de theta

def energie_finale(E_diff, T):
    return E_diff + T

def erreur_energie_finale(err_E_diff, err_T):
    return np.sqrt(err_E_diff**2 + err_T**2)

for idx, ang in enumerate([45, 60, 75, 90, 110]):
    E_tot = energie_finale(E_gamma_diff[idx], T_elec_exp[idx])
    err_E_tot = erreur_energie_finale(err_E_gamma_diff[idx], err_T_elec_exp[idx])
    E_tot_pm = uncertainties.ufloat(E_tot, err_E_tot)
    print(f'Énergie totale finale a {ang} : {E_tot_pm:.1u} keV')



### graph énergie cinétique T de l'électron de recul vs angle de diffusion avec barres d’erreur (±sigma ou encore ±FWHM/2.35)

def graph_energie_electron_fct_angle(T_exp, err_T, T_th):
    angles = [45, 60, 75, 90, 110]
    plt.plot(angles, T_exp, 'k.', label='Expérimental')
    plt.plot(angles, T_th, 'kx', label='Théorique')
    plt.errorbar(angles, T_exp, yerr=err_T, xerr=[1 for i in range(len(angles))], capsize=5, c='k', fmt='.')
    plt.xlabel('Angle de diffusion [°]')
    plt.ylabel("Énergie cinétique de l'électron de recul [keV]")
    plt.legend(loc='best')
    plt.show()

graph_energie_electron_fct_angle(T_elec_exp, err_T_elec_exp, T_elec_th)


### graph linéarité de la relation entre le rapport ho / T et (1 - cos theta)^-1
def erreur_inv_1_costheta(theta, err_theta):
    deriv = np.sin(theta) / (1-np.cos(theta))**2
    return abs(deriv*err_theta)

def graph_linearite_ratioEelectron_invcostheta(E_inc, err_E_inc, T, err_T):
    rapport = E_inc/T
    err_rapport = erreur_rapport_energies(E_inc, err_E_inc, T, err_T)
    angles = np.array([45, 60, 75, 90, 110])
    err_angles = 1 # ou transformer en rad ??? à voir selon erreur graph
    x=(1-np.cos(np.deg2rad(angles)))**-1
    err_x = erreur_inv_1_costheta(angles, err_angles)
    plt.plot(x, rapport, 'k.', label='Données')
    #plt.errorbar(x, rapport, xerr=err_x, yerr=err_rapport, capsize=5, c='k', fmt='.')
    plt.errorbar(x, rapport, yerr=err_rapport, capsize=5, c='k', fmt='.')

    x_axis = np.linspace(0, 4)
    popt, pcov = opt.curve_fit(func_lineaire, x, rapport)
    std_popt = np.sqrt(np.diag(pcov))
    plt.plot(x_axis,func_lineaire(x_axis, *popt), 'k-', label=f'Régression ({popt[0]:.3f} +- {std_popt[0]:.3f}) x + ({popt[1]:.2f} +- {std_popt[1]:.2f})')
    print(f'Régression linéaire ({popt[0]} +- {std_popt[0]}) x + ({popt[1]} +- {std_popt[1]})')

    plt.xlabel('$1 - \cos(θ)$ [-]')
    plt.ylabel("$hν_0 / T$ [-]")
    plt.legend(loc='best')
    plt.show()

graph_linearite_ratioEelectron_invcostheta(E_gamma_diff, err_E_gamma_diff, T_elec_exp, err_T_elec_exp)


def plot_attenuation_coef(coef_exp, coef_th):
    line = np.linspace(0, 1)
    plt.plot(line, line, ':k')
    for key in coef_exp.keys():
        plt.errorbar(coef_th[key], coef_exp[key][0], yerr=coef_exp[key][1], capsize=5, fmt='.', label=f'{key}')
    plt.legend(loc='best')
    plt.xlabel('Théorique')
    plt.ylabel("Expérimental")
    plt.show()

plot_attenuation_coef(coef_exp=attenuation_coef, coef_th=attenuation_coef_th_inc)