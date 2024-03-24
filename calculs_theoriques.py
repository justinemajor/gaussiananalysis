import numpy as np

EnergieInc = 662 # keV
ALPHA = 1.2955
r0 = 2.8179*10**-13 # cm

def energie_photon_diffuse(theta):
    return EnergieInc / (1 + ALPHA*(1-np.cos(theta)))

def energie_cinetique_electron_recul(theta):
    return EnergieInc * ALPHA * (1 - np.cos(theta)) / (1 + ALPHA * (1 - np.cos(theta)))

def angle_electron_recul(theta):
    tan_phi = 1 / ((1+ALPHA)*np.tan(theta/2))
    angle_rad = np.arctan(tan_phi)
    return np.rad2deg(angle_rad)

def section_efficace_par_electron(theta):
    dsigma_domega = r0**2/2
    dsigma_domega *= ((1+(np.cos(theta))**2)/(1 + ALPHA*(1-np.cos(theta)))**2)
    dsigma_domega *= (1+(ALPHA**2*(1-np.cos(theta))**2)/((1+(np.cos(theta))**2)*(1+ALPHA*(1-np.cos(theta)))))
    return dsigma_domega

def section_efficace_totale(theta, nbre_electrons):
    dsigma_domega = section_efficace_par_electron(theta)
    return dsigma_domega*nbre_electrons


theta = np.deg2rad(np.array([0, 45, 60, 75, 90, 110]))

print('Energie photon diffuse : ', energie_photon_diffuse(theta))
print('Energie cinetique electron : ', energie_cinetique_electron_recul(theta))
print('Angle electrons recul : ', angle_electron_recul(theta))
print('Section differentielle par electron : ', section_efficace_par_electron(theta))

section_eff = {}
for angle in [0, 45, 60, 75, 90, 110]:
    section_eff[str(angle)] = section_efficace_par_electron(np.deg2rad(angle))


def activite(A0, dt, demivie):
    lambda_ = np.log(2) / demivie
    return A0 * np.exp(-lambda_ * dt)

demivie_137Cs = 30.17 * 365.25 # jours
dt = 557 # jours entre 1er sep 2022 et 11 mars 2024

A_adj = activite(A0=99.9, dt=dt, demivie=demivie_137Cs)
print(f'Activité 137Cs en MBq : {A_adj:.1f}')