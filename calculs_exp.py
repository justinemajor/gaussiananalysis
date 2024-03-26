import numpy as np
import uncertainties
from calculs_theoriques import A_adj

D_faisceau_cible = 3.3 # cm valeur obtenue avec triangles semblables
err_D_faisceau_cible = 0.1
D_detecteur = 2.54 # cm
err_D_detecteur = 0.01
d_detecteur_cible = (4.5+11.5) # cm
d_source_cible = (5+22.5) # cm
err_d_detecteur_cible = 0.2
err_d_source_cible = 0.2

dia_cible = {'plexi': 2.523,
        'al' : 2.535,
        'fe' : 2.541,
        'al_moyen' : 1.898,
        'al_petit' : 1.271,
        'al_mini' : 0.630} # cm
err_dia_cible = 0.001

density = {'plexi': 1.19,
        'al' : 2.70,
        'fe' : 7.86} # g/cm3
masse_molaire = {'plexi': 13.03,
        'al' : 27,
        'fe' : 55.85} # g/mol
Z = {'plexi': 7.03,
        'al' : 13,
        'fe' : 26} # nbre elec / atom

NA = 6.02214076 *10**23  # atoms/mol

def erreur_mult_div(res, x, delta_x, y, delta_y):
    return res * np.sqrt((delta_x/x)**2 + (delta_y/y)**2)

def flux_gamma_inc(A, aire, angle_solide_source):
    return 0.851 * A / d_source_cible**2
    #return 0.851 * A / aire * angle_solide_source

def erreur_flux_gamma_inc(I, A, err_A, aire, err_aire):
    return  I * np.sqrt((err_A/A)**2 + 2*(err_d_source_cible/d_source_cible)**2)
    #return erreur_mult_div(res=I, x=A, delta_x=err_A, y=aire, delta_y=err_aire) * 0.851

def angle_solide_source(l, h, d_source):
    return angle_solide(aire_eff=aire_faisceau_cible(l, h), d=d_source)

def aire_faisceau_cible(l, h):
    return l*h

def erreur_aire_faisceau_cible(aire, l, err_l, h, err_h):
    return erreur_mult_div(res=aire, x=l, delta_x=err_l, y=h, delta_y=err_h)

def largeur_moyenne_faisceau_cible(d_cible):
    aire_disque = np.pi*(d_cible/2)**2
    return aire_disque / d_cible

def erreur_largeur_moyenne_faisceau_cible(l_moy, d_cible, err_d_cible):
    return err_d_cible / d_cible * l_moy


def volume_cible(d_cible, h):
    return np.pi * (d_cible/2)**2 * h

def nombre_electrons_cible(V, rho, M, Z, NA) :
    return  (V * rho * NA * Z) / M

def erreur_volume_cible(d_cible, err_d_cible, h, err_h):
    dV_dd = np.pi * d_cible / 2 * h
    dV_dh = np.pi * d_cible**2 / 4
    return np.sqrt((dV_dd * err_d_cible)**2 + (dV_dh * err_h)**2)

def erreur_nombre_electrons_cible(nbre_electrons, volume, err_vol):
    return err_vol/volume * nbre_electrons

def nombre_electrons_cm3(rho, M, Z, NA) :
    return  (rho * NA * Z) / M


def aire_efficace_detecteur(d):
    return np.pi * (d/2)**2

def erreur_aire_efficace_detecteur(aire_eff, d, err_d):
    return np.pi / 4 * erreur_mult_div(res=aire_eff, x=d, delta_x=err_d, y=d, delta_y=err_d)

def angle_solide(aire_eff, d):
    return aire_eff / d**2

def erreur_angle_solide(A, err_A, d, err_d):
    d_dA = 1 / d**2
    d_dd = -2 * A / d**3
    return np.sqrt((d_dA * err_A)**2 + (d_dd * err_d)**2)


def nombre_coups_par_seconde(coups, t_coups, bruit, t_bruit):
    return coups/t_coups - bruit/t_bruit

def correction_attenuation(coups_sec, mu, d_cible):
    return coups_sec / np.exp(-mu*d_cible/2)

def correction_efficacite_detecteur(coups_sec, eff):
    return coups_sec/eff

def section_efficace_diff_exp(coups_sec, flux_inc, n, angle_solide):
    return coups_sec / (flux_inc*n*angle_solide)

efficacites = []


loop = ['plexi', 'al', 'fe', 'al_moyen', 'al_petit', 'al_mini']
proprietes_diff = {}

for cible in loop:
    print(cible)
    D_cible = dia_cible[cible]

    material = cible.split('_')[0]
    rho = density[material]
    M = masse_molaire[material]
    nbre_elec_atom = Z[material]
        
    l_moy_faisceau = largeur_moyenne_faisceau_cible(d_cible=D_cible)
    err_l_moy = erreur_largeur_moyenne_faisceau_cible(l_moy=l_moy_faisceau, d_cible=D_cible, err_d_cible=err_dia_cible)
    A_faisceau_cible = aire_faisceau_cible(l=l_moy_faisceau, h=D_faisceau_cible)
    err_A_faisceau = erreur_aire_faisceau_cible(aire=A_faisceau_cible, l=l_moy_faisceau, err_l=err_l_moy, h=D_faisceau_cible, err_h=err_D_faisceau_cible)
    angle_solide_inc = angle_solide_source(l=l_moy_faisceau, h=D_faisceau_cible, d_source=d_source_cible)
    I_inc = flux_gamma_inc(A=A_adj*10**6, aire=A_faisceau_cible, angle_solide_source=angle_solide_inc)
    err_I_inc = erreur_flux_gamma_inc(I=I_inc, A=A_adj*10**6, err_A=0.1*10**6, aire=A_faisceau_cible, err_aire=err_A_faisceau)

    I_inc_pm = uncertainties.ufloat(I_inc, err_I_inc)
    print(f'Flux gamma incident sur cible : {I_inc_pm:.1u} gamma / s cm^2')

    V_irr = volume_cible(d_cible=D_cible, h=D_faisceau_cible)
    err_V_irr = erreur_volume_cible(d_cible=D_cible, err_d_cible=err_dia_cible, h=D_faisceau_cible, err_h=err_D_faisceau_cible)

    nbre_elec = nombre_electrons_cible(V=V_irr, rho=rho, M=M, Z=nbre_elec_atom, NA=NA)
    err_nbre_elec = erreur_nombre_electrons_cible(nbre_electrons=nbre_elec, volume=V_irr, err_vol=err_V_irr)
    
    nbre_elec_pm = uncertainties.ufloat(nbre_elec, err_nbre_elec)
    print(f'Nombre electrons cible : {nbre_elec_pm:.1u}')

    nbre_elec_cm3 = nombre_electrons_cm3(rho=rho, M=M, Z=nbre_elec_atom, NA=NA)
    print(f'Densite electrons cible : {nbre_elec_cm3}')

    nbre_elec_cm2 = nbre_elec_cm3 * D_cible
    err_nbre_elec_cm2 = err_dia_cible / D_cible * nbre_elec_cm2
    print(f'Densite surface electrons cible : {nbre_elec_cm2} +- {err_nbre_elec_cm2}')
        

    A_eff_det = aire_efficace_detecteur(d=D_detecteur)
    err_A_eff_det = erreur_aire_efficace_detecteur(aire_eff=A_eff_det, d=D_detecteur, err_d=err_D_detecteur)
    angle_solide_det = angle_solide(aire_eff=A_eff_det, d=d_detecteur_cible)
    err_angle_solide_det = erreur_angle_solide(A=A_eff_det, err_A=err_A_eff_det, d=d_detecteur_cible, err_d=err_d_detecteur_cible)

    angle_solide_pm = uncertainties.ufloat(angle_solide_det, err_angle_solide_det)
    print(f'angle solide : {angle_solide_pm:.1u} steradian')

    proprietes_diff[cible] = {'flux' : (I_inc, err_I_inc), 
                              'n' : (nbre_elec, err_nbre_elec),
                              'angle solide' : (angle_solide_det, err_angle_solide_det), 
                              'densite elec': (nbre_elec_cm3, 0),
                              'densite surface elec': (nbre_elec_cm2, err_nbre_elec_cm2)}
    