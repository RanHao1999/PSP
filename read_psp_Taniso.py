"""
@author = RanHao, (contact me at: hao.ran.24@ucl.ac.uk)

This file is for reading Parker Solar Probe data at a large scale (e.g., over a timespan of over 10 days).

This code reads critical parameters (B, V, T, etc) from the .cdf files, and save them in an .h5 file for future easy extraction.
"""




import os, sys
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
import scipy
import scipy.interpolate
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import plasmapy
import math

from sunpy.io.cdf import read_cdf
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames

import scipy.constants as const
import astropy.units as u

os.chdir(sys.path[0])

def resample_to_target_time(dataframe, target_tstart, target_tend, target_dt, keys, units=None):
    # The function for resampling the data to the target time resolution.
    # Turn the index of the dataframe to numbers according to their gap to the first point (time index).
    time = dataframe.index
    time_index_in_num = [(time[i] - time[0]).total_seconds() for i in range(len(time))]

    # Derive the time index numbers of the target time range.
    target_time_gap = (target_tend - target_tstart).total_seconds()
    point_num = int(target_time_gap / target_dt)
    target_start_index = (target_tstart - time[0]).total_seconds()
    target_end_index = (target_tend - time[0]).total_seconds()

    target_timeindex_in_num = np.linspace(target_start_index, target_end_index, point_num + 1)


    res_dict = {}
    for i in range(len(keys)):
        values = dataframe[keys[i]].values
        f = scipy.interpolate.interp1d(time_index_in_num, values)
        y_new = f(target_timeindex_in_num)
        res_dict[keys[i]] = y_new

    target_timestamps = [time[0] + timedelta(seconds=x) for x in target_timeindex_in_num]
    res_df = pd.DataFrame(res_dict, index=target_timestamps)

    return res_df


def read_psp(path_fld, path_swp, target_tstart, target_tend, target_dt):
    """
    :param path_fld: Path for the folder containing psp fields data.
    :param path_swp: Path for the folder containing psp swp data, should be further divided into /spc and /spi folders.
    :param target_tstart: format: pd.timestamp.
    :param target_tend: format: pd.timestamp.
    :param target_dt: int, unit in seconds.
    :return: a dataframe that contains all the needed data columns in psp, in the target time resolution.
    """
    
    # Read the fields data.
    file_names = sorted([x for x in os.listdir(path_fld) if 'fld_l2_mag' in x])
    fld_dataframe = pd.concat([read_cdf(path_fld + '/' + x)[0].to_dataframe() for x in file_names])[::100]
    fld_magnitude = np.sqrt(fld_dataframe['psp_fld_l2_mag_RTN_0'] ** 2 +
                            fld_dataframe['psp_fld_l2_mag_RTN_1'] ** 2 +
                            fld_dataframe['psp_fld_l2_mag_RTN_2'] ** 2)

    fld_dataframe['psp_fld_l2_mag_RTN_magnitude'] = fld_magnitude
    fld_dataframe = fld_dataframe.dropna()

    # Read the SWP data.
    filenames_spi_alpha = sorted([x for x in os.listdir(path_swp + '/spi')
                                if 'sf0a' in x])
    spi_alpha_dataframe = pd.concat([read_cdf(path_swp + '/spi/' + x)[0].to_dataframe()
                                    for x in filenames_spi_alpha]).dropna()

    filenames_spi_proton = sorted([x for x in os.listdir(path_swp + '/spi')
                                if 'sf00' in x])
    spi_proton_dataframe = pd.concat([read_cdf(path_swp + '/spi/' + x)[0].to_dataframe()
                                    for x in filenames_spi_proton]).dropna()
    
    filenames_spc = sorted([x for x in os.listdir(path_swp + '/spc')
                            if'spc' in x])
    #spc_dataframe = pd.concat([read_cdf(path_swp + '/spc/' + x)[0].to_dataframe() for x in filenames_spc])[::50].dropna() 

    # Read the qtn data.
    filenames_qtn = sorted([x for x in os.listdir(path_fld) if 'fld_l3_sqtn' in x])
    qtn_dataframe = pd.concat([read_cdf(path_fld + '/' + x)[0].to_dataframe() for x in filenames_qtn])
    qtn_dataframe = qtn_dataframe[qtn_dataframe["electron_density"] != -1.0e31].dropna()

    # Read coordinates data. (HCI and carrington)
    # PSP_Orbit.txt is downloaded from https://psp-gateway.jhuapl.edu/
    with open('data/PSP_Orbit.txt') as f:
        lines = f.readlines()[1:]
    year = [line.split()[0] for line in lines]
    month = [line.split()[1] for line in lines]
    day = [line.split()[2] for line in lines]
    hour = [line.split()[3] for line in lines]
    minute = [line.split()[4] for line in lines]
    second = [line.split()[5] for line in lines]
    date = [pd.Timestamp(year[i] + '-' + month[i] + '-' + day[i] + ' ' + hour[i] + ':' + minute[i] + ':' + second[i]) for i in range(len(year))]
    HCI_X = np.array([float(line.split()[6]) for line in lines])
    HCI_Y = np.array([float(line.split()[7]) for line in lines])
    HCI_Z = np.array([float(line.split()[8]) for line in lines])
    Radius = np.sqrt(HCI_X ** 2 + HCI_Y ** 2 + HCI_Z ** 2)

    coord_HCI_dataframe = pd.DataFrame({'sc_pos_HCI_0': HCI_X,
                                    'sc_pos_HCI_1': HCI_Y,
                                    'sc_pos_HCI_2': HCI_Z,
                                    'Radius': Radius}, index=date)

    coords_HCI = [SkyCoord(HCI_X[i] * u.km, HCI_Y[i] * u.km, HCI_Z[i] * u.km,
                           frame=frames.HeliocentricInertial, obstime=date[i],
                           representation_type='cartesian') for i in range(len(HCI_X))]
    coords_carrington = [coord.transform_to(frame=frames.HeliographicCarrington(observer='Sun',
                                                                                obstime=coord.obstime))
                         for coord in coords_HCI]
    lon_carr = [coord.lon.value for coord in coords_carrington]
    lat_carr = [coord.lat.value for coord in coords_carrington]

    coord_carr_dataframe = pd.DataFrame({'carr_longitude': lon_carr,
                                         'carr_latitude': lat_carr}, index=date)
    
    # key words that I want to study
    keys_spi = ['VEL_RTN_SUN_0', 'VEL_RTN_SUN_1', 'VEL_RTN_SUN_2', 'DENS', 'TEMP', 
                'T_TENSOR_INST_0', 'T_TENSOR_INST_1', 'T_TENSOR_INST_2', 'T_TENSOR_INST_3', 'T_TENSOR_INST_4', 'T_TENSOR_INST_5', 
                'MAGF_INST_0', 'MAGF_INST_1', 'MAGF_INST_2']
    units_spi = [u.km / u.s, u.km / u.s, u.km / u.s, u.km / u.s, u.cm ** -3, u.eV, u.eV, u.eV, u.eV, u.eV, u.eV, u.eV, u.nT, u.nT, u.nT]

    keys_fld = ['psp_fld_l2_mag_RTN_0', 'psp_fld_l2_mag_RTN_1', 'psp_fld_l2_mag_RTN_2', 'psp_fld_l2_mag_RTN_magnitude']
    units_fld = [u.nT, u.nT, u.nT, u.nT, u.nT]

    # Resample the data (from the selected starting time to the end time) to the target time resolution.
    spi_alpha_dataframe = resample_to_target_time(spi_alpha_dataframe, target_tstart, target_tend, target_dt, keys_spi, units_spi)
    spi_proton_dataframe = resample_to_target_time(spi_proton_dataframe, target_tstart, target_tend, target_dt, keys_spi, units_spi)
    #spc_dataframe = resample_to_target_time(spc_dataframe, target_tstart, target_tend, target_dt, keys_spc, units_spc)
    fld_dataframe = resample_to_target_time(fld_dataframe, target_tstart, target_tend, target_dt, keys_fld ,units_fld)
    qtn_dataframe = resample_to_target_time(qtn_dataframe, target_tstart, target_tend, target_dt, ['electron_density'], [u.cm ** -3])
    coord_HCI_dataframe = resample_to_target_time(coord_HCI_dataframe, target_tstart, target_tend, 
                                                  target_dt, ['sc_pos_HCI_0', 'sc_pos_HCI_1', 'sc_pos_HCI_2', 'Radius'])
    coord_carr_dataframe = resample_to_target_time(coord_carr_dataframe, target_tstart, target_tend,
                                                    target_dt, ['carr_longitude', 'carr_latitude'])

    # Rename some keywords so wen can keep all the data in one dataframe.
    spi_alpha_dataframe.rename(columns={"VEL_RTN_SUN_0": "VEL_RTN_SUN_0_alpha",
                                        "VEL_RTN_SUN_1": "VEL_RTN_SUN_1_alpha",
                                        "VEL_RTN_SUN_2": "VEL_RTN_SUN_2_alpha",
                                        "VEL_RTN_SUN_0_buttered": "VEL_RTN_SUN_0_buttered_alpha",
                                        "DENS": "DENS_alpha",
                                        "TEMP": 'TEMP_alpha', 
                                        "T_TENSOR_INST_0": "T_TENSOR_INST_0_alpha", 
                                        "T_TENSOR_INST_1": "T_TENSOR_INST_1_alpha", 
                                        "T_TENSOR_INST_2": "T_TENSOR_INST_2_alpha",
                                        "T_TENSOR_INST_3": "T_TENSOR_INST_3_alpha",
                                        "T_TENSOR_INST_4": "T_TENSOR_INST_4_alpha",
                                        "T_TENSOR_INST_5": "T_TENSOR_INST_5_alpha",
                                        "MAGF_INST_0": "MAGF_INST_0_alpha",
                                        "MAGF_INST_1": "MAGF_INST_1_alpha",
                                        "MAGF_INST_2": "MAGF_INST_2_alpha"}, inplace=True)
    spi_proton_dataframe.rename(columns={"VEL_RTN_SUN_0": "VEL_RTN_SUN_0_proton",
                                         "VEL_RTN_SUN_1": "VEL_RTN_SUN_1_proton",
                                         "VEL_RTN_SUN_2": "VEL_RTN_SUN_2_proton",
                                         "VEL_RTN_SUN_0_buttered": "VEL_RTN_SUN_0_buttered_proton",
                                         "DENS": "DENS_proton", 
                                         "TEMP": "TEMP_proton",
                                         "T_TENSOR_INST_0": "T_TENSOR_INST_0_proton",
                                         "T_TENSOR_INST_1": "T_TENSOR_INST_1_proton",
                                         "T_TENSOR_INST_2": "T_TENSOR_INST_2_proton",
                                         "T_TENSOR_INST_3": "T_TENSOR_INST_3_proton",
                                         "T_TENSOR_INST_4": "T_TENSOR_INST_4_proton",
                                         "T_TENSOR_INST_5": "T_TENSOR_INST_5_proton",
                                         "MAGF_INST_0": "MAGF_INST_0_proton",
                                         "MAGF_INST_1": "MAGF_INST_1_proton",
                                         "MAGF_INST_2": "MAGF_INST_2_proton"}, inplace=True)
    qtn_dataframe.rename(columns={"electron_density": "DENS_electron"}, inplace=True)

    # Merge all the dataframes into one.
    psp_dataframe = pd.concat([spi_alpha_dataframe, spi_proton_dataframe, fld_dataframe, qtn_dataframe, coord_HCI_dataframe, coord_carr_dataframe], axis=1)

    #===============================================
    # Key words that are not in the original file, you calculate them here.
    #===============================================
    
    # Calculate some important key elements.

    # Butter some key parameters.
    def butter_lowpass_filter(data, cutoff, fs, order=5):
        # The butter low pass filter.
        def butter_lowpass(cutoff, fs, order=5):
            return butter(order, cutoff, fs=fs, btype='low')

        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    np_buttered = butter_lowpass_filter(psp_dataframe['DENS_proton'], 2e-4, 1.0/30.0, order=1)
    na_buttered = butter_lowpass_filter(psp_dataframe['DENS_alpha'], 2e-4, 1.0/30.0, order=1)

    psp_dataframe['DENS_proton_buttered'] = np_buttered
    psp_dataframe['DENS_alpha_buttered'] = na_buttered


    # Alfven speed.
    proton = plasmapy.particles.Particle('proton')

    density = psp_dataframe["DENS_electron"].values * (u.cm**-3)
    B = psp_dataframe['psp_fld_l2_mag_RTN_magnitude'].values * u.nT

    VA = plasmapy.formulary.Alfven_speed(B, density, ion='p+', z_mean=None)
    VA = VA.to(u.km / u.s).value.tolist()
    psp_dataframe['VEL_Alfven'] = VA


    # The cosine values between B_vector and V_diff_vector
    B_vectors = [np.array((psp_dataframe['psp_fld_l2_mag_RTN_0'].values[i],
                  psp_dataframe['psp_fld_l2_mag_RTN_1'].values[i],
                  psp_dataframe['psp_fld_l2_mag_RTN_2'].values[i])) for i in range(len(psp_dataframe))]

    # v_alpha - v_p
    v_diff_vectors = [((psp_dataframe["VEL_RTN_SUN_0_alpha"] - psp_dataframe["VEL_RTN_SUN_0_proton"]).values[i],
                       (psp_dataframe["VEL_RTN_SUN_1_alpha"] - psp_dataframe["VEL_RTN_SUN_1_proton"]).values[i],
                       (psp_dataframe["VEL_RTN_SUN_2_alpha"] - psp_dataframe["VEL_RTN_SUN_2_proton"]).values[i])
                      for i in range(len(psp_dataframe))]

    def magnitude(vector):
        # calculate the magnitude of a vector
        return math.sqrt(sum(pow(element, 2) for element in vector))

    cosines = [np.dot(B_vectors[i], v_diff_vectors[i]) / (magnitude(B_vectors[i]) * magnitude(v_diff_vectors[i]))
               for i in range(len(B_vectors))]

    psp_dataframe["COSINES"] = cosines


    # V_p magnitude
    vp_magnitude = [magnitude((psp_dataframe["VEL_RTN_SUN_0_proton"].values[i],
                               psp_dataframe["VEL_RTN_SUN_1_proton"].values[i],
                               psp_dataframe["VEL_RTN_SUN_2_proton"].values[i]))
                    for i in range(len(psp_dataframe))]
    psp_dataframe['VEL_RTN_SUN_MAGNITUDE_proton'] = vp_magnitude

    # V_alpha magnitude
    va_magnitude = [magnitude((psp_dataframe["VEL_RTN_SUN_0_alpha"].values[i],
                               psp_dataframe["VEL_RTN_SUN_1_alpha"].values[i],
                               psp_dataframe["VEL_RTN_SUN_2_alpha"].values[i]))
                    for i in range(len(psp_dataframe))]
    psp_dataframe['VEL_RTN_SUN_MAGNITUDE_alpha'] = va_magnitude

    # v_{\alpha, p} = sign(|V_{\alpha}| - |V_{p}|) / |V_{\alpha} - V_{p}|
    v_ap = [np.sign(va_magnitude[i] - vp_magnitude[i]) * magnitude(v_diff_vectors[i]) for i in range(len(v_diff_vectors))]
    psp_dataframe["v_ap"] = v_ap

    # Solar Distance
    Radius = np.sqrt(psp_dataframe['sc_pos_HCI_0'] ** 2 + psp_dataframe['sc_pos_HCI_1'] ** 2 + psp_dataframe['sc_pos_HCI_2'] ** 2)
    psp_dataframe['Radius'] = Radius

    # Collision Age for proton and alpha particles.
    def get_collision_age(n, T, q, m, D, Vsw):
        """
        Calculate the collision age of a particle.
        n: number density, unit: cm^-3
        T: temperature, unit: K
        q: charge, unit: C
        m: mass, unit: kg
        D: Solar distance, unit: km
        Vsw: Solar wind speed, unit: km/s
        """
        # Constants
        pi = const.pi
        epsilon_0 = const.epsilon_0
        kB = const.k
        qp = const.e
        mp = const.m_p

        # Calculate the collision age
        Lambda = ((12*pi) / (qp * q**2)) * ((epsilon_0 ** 3 * kB ** 3 * T ** 3) / (n)) ** (0.5)
        tau = 11.4 * (1.0 / np.log(Lambda)) * (m / mp) ** (0.5) * (q / qp) ** (-4) *(n) ** (-1) * (T) ** (3.0 / 2.0)

        Ac  = D / (Vsw * tau)

        return Ac

    # Proton.
    density_p = psp_dataframe['DENS_electron'] # ne represent np, cm^-3
    Tp = psp_dataframe['TEMP_proton'] * 11606.0 # Tp, K
    qp = const.e # proton charge, C
    mp = const.m_p # proton mass, kg
    D = psp_dataframe['Radius'] - 695550.0 # D, km
    Vsw = psp_dataframe['VEL_RTN_SUN_MAGNITUDE_proton'] # Vsw, km/s

    collosion_age_proton = get_collision_age(density_p, Tp, qp, mp, D, Vsw)
    psp_dataframe['Collision_Age_proton'] = collosion_age_proton
    
    # Alpha
    na = psp_dataframe['DENS_alpha_buttered'] # na, cm^-3
    Ta = psp_dataframe['TEMP_alpha'] * 11606.0 # Tp, K
    qa = 2 * const.e # alpha charge, C
    ma = 4 * const.m_p # alpha mass, kg

    collosion_age_alpha = get_collision_age(na, Ta, qa, ma, D, Vsw)
    psp_dataframe['Collision_Age_alpha'] = collosion_age_alpha

    
    # Temperature anisotropy

    # Proton
    B_Matrix_proton = [np.matrix(np.array([x, y, z])) / np.sqrt(x ** 2 + y ** 2 + z ** 2) for x, y, z in zip(psp_dataframe['MAGF_INST_0_proton'], psp_dataframe['MAGF_INST_1_proton'], psp_dataframe['MAGF_INST_2_proton'])]
    T_Tensor_Matrix_proton = [np.matrix(np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])) for xx, yy, zz, xy, xz, yz in zip(psp_dataframe['T_TENSOR_INST_0_proton'], psp_dataframe['T_TENSOR_INST_1_proton'], psp_dataframe['T_TENSOR_INST_2_proton'], psp_dataframe['T_TENSOR_INST_3_proton'], psp_dataframe['T_TENSOR_INST_4_proton'], psp_dataframe['T_TENSOR_INST_5_proton'])]

    Temp_parallel_proton = [(x * y * x.T)[0, 0] for x, y in zip(B_Matrix_proton, T_Tensor_Matrix_proton)]
    psp_dataframe['TEMP_parallel_proton'] = Temp_parallel_proton

    Temp_perpendicular_proton = 0.5 * (psp_dataframe['T_TENSOR_INST_0_proton'] + psp_dataframe['T_TENSOR_INST_1_proton'] + psp_dataframe['T_TENSOR_INST_2_proton'] - Temp_parallel_proton)
    psp_dataframe['TEMP_perpendicular_proton'] = Temp_perpendicular_proton

    # Alpha
    B_Matrix_alpha = [np.matrix(np.array([x, y, z])) / np.sqrt(x ** 2 + y ** 2 + z ** 2) for x, y, z in zip(psp_dataframe['MAGF_INST_0_alpha'], psp_dataframe['MAGF_INST_1_alpha'], psp_dataframe['MAGF_INST_2_alpha'])]
    T_Tensor_Matrix_alpha = [np.matrix(np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])) for xx, yy, zz, xy, xz, yz in zip(psp_dataframe['T_TENSOR_INST_0_alpha'], psp_dataframe['T_TENSOR_INST_1_alpha'], psp_dataframe['T_TENSOR_INST_2_alpha'], psp_dataframe['T_TENSOR_INST_3_alpha'], psp_dataframe['T_TENSOR_INST_4_alpha'], psp_dataframe['T_TENSOR_INST_5_alpha'])]

    Temp_parallel_alpha = [(x * y * x.T)[0, 0] for x, y in zip(B_Matrix_alpha, T_Tensor_Matrix_alpha)]
    psp_dataframe['TEMP_parallel_alpha'] = Temp_parallel_alpha

    Temp_perpendicular_alpha = 0.5 * (psp_dataframe['T_TENSOR_INST_0_alpha'] + psp_dataframe['T_TENSOR_INST_1_alpha'] + psp_dataframe['T_TENSOR_INST_2_alpha'] - Temp_parallel_alpha)
    psp_dataframe['TEMP_perpendicular_alpha'] = Temp_perpendicular_alpha

    # Total thermal temperature
    Temp_thermal_proton = (psp_dataframe['TEMP_parallel_proton'] + 2 * psp_dataframe['TEMP_perpendicular_proton']) / 3.0
    psp_dataframe['TEMP_thermal_proton'] = Temp_thermal_proton

    Temp_thermal_alpha = (psp_dataframe['TEMP_parallel_alpha'] + 2 * psp_dataframe['TEMP_perpendicular_alpha']) / 3.0
    psp_dataframe['TEMP_thermal_alpha'] = Temp_thermal_alpha


    # Proton plasma beta (parallel to the local B)
    mu0 = const.mu_0
    density_p = psp_dataframe['DENS_electron'] * 1e6 # ne represent np, m^-3
    k = const.k
    Tp_para = psp_dataframe['TEMP_parallel_proton'] * 11606.0 # Tp, K
    B = psp_dataframe['psp_fld_l2_mag_RTN_magnitude'] * 1e-9 # B, T

    plasma_beta_proton_para = (2 * mu0 * density_p * k * Tp_para) / (B ** 2)
    psp_dataframe['Plasma_beta_proton_parallel'] = plasma_beta_proton_para


    # Alpha plasma beta (parallel to the local B)
    density_a = psp_dataframe['DENS_alpha'] * 1e6 # na, m^-3
    Ta_para = psp_dataframe['TEMP_parallel_alpha'] * 11606.0 # Ta, K

    plasma_beta_alpha_para = (2 * mu0 * density_a * k * Ta_para) / (B ** 2)
    psp_dataframe['Plasma_beta_alpha_parallel'] = plasma_beta_alpha_para

    # Save the file.
    if os.path.exists('data/psp/psp_dataframe_Taniso.h5') == True:
        os.remove('data/psp/psp_dataframe_Taniso.h5')
    store_psp = pd.HDFStore('data/psp/psp_dataframe_Taniso.h5')
    store_psp['psp_dataframe_Taniso'] = psp_dataframe
    store_psp.close()

    return 0


def read_fov(path_spi, target_tstart, target_tend, target_dt):
    file_names_proton = sorted([x for x in os.listdir(path_spi) if 'sf00' in x])
    file_names_alpha = sorted([x for x in os.listdir(path_spi) if 'sf0a' in x])

    proton_dataframe = pd.concat([read_cdf(path_spi + '/' + x)[0].to_dataframe() for x in file_names_proton]).dropna()
    alpha_dataframe = pd.concat([read_cdf(path_spi + '/' + x)[0].to_dataframe() for x in file_names_alpha]).dropna()

    key_all = proton_dataframe.keys()
    key_energy = [key for key in key_all if 'EFLUX_VS_ENERGY' in key]
    energy_vals = [key for key in key_all if 'ENERGY_VALS' in key]
    key_phi = [key for key in key_all if 'EFLUX_VS_PHI' in key]
    phi_vals = [key for key in key_all if 'PHI_VALS' in key]
    key_theta = [key for key in key_all if 'EFLUX_VS_THETA' in key]
    theta_vals = [key for key in key_all if 'THETA_VALS' in key]

    proton_dataframe = resample_to_target_time(proton_dataframe, target_tstart, target_tend, target_dt, 
                                               key_energy + energy_vals + key_phi + phi_vals + key_theta + theta_vals)
    alpha_dataframe = resample_to_target_time(alpha_dataframe, target_tstart, target_tend, target_dt, 
                                              key_energy + energy_vals + key_phi + phi_vals + key_theta + theta_vals)


    proton_dataframe.rename(columns={key: key+'_proton' for key in 
                                     key_energy + energy_vals + key_phi + phi_vals + key_theta + theta_vals}, 
                                     inplace=True)
    alpha_dataframe.rename(columns={key: key+'_alpha' for key in 
                                    key_energy + energy_vals + key_phi + phi_vals + key_theta + theta_vals}, 
                                    inplace=True)
    
    fov_dataframe = pd.concat([proton_dataframe, alpha_dataframe], axis=1)

    # Save the file.
    if os.path.exists('data/psp/fov_dataframe') == True:
        os.remove('data/psp/fov_dataframe.h5')
    store_psp = pd.HDFStore('data/psp/fov_dataframe.h5')
    store_psp['fov_dataframe'] = fov_dataframe
    store_psp.close() 

    return 0

def main():
    # Read PSP data.
    path_fld = 'data/psp/FIELDS'
    path_swp = 'data/psp/SWEAP'
    target_tstart = pd.Timestamp("2021-11-15 00:10:00")
    target_tend = pd.Timestamp("2021-11-24 23:50:00")
    target_dt = 2

    print("Reading Process is on...")
    read_psp(path_fld, path_swp, target_tstart, target_tend, target_dt)
    print("Reading Process is done!")

    #print("Reading fov process in on...")
    #read_fov(path_swp + '/spi', target_tstart, target_tend, target_dt)
    #print("Reading fov process is done!")

    return 0



if __name__ == "__main__":
    main()
