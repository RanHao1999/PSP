"""
@author = RanHao

This code is for tracing the source region of solar wind observed by PSP.

Parameters to be set:  PFSS parameters: |R_ss (Height of the Source Surface)
                                        |nrho
                       PSP time range:  |Format( for both start and end time): "YYYY-mm-ddTHH:MM:SS"
                                        The timerange is set for the code to download the corresponding
                                        PSP SWEAP data, namely "psp_swp_spc_l3i_YYYYMMDD_vVV".

Principle:
1. Ballistic Parker Spiral extrapolation from PSP position to source surface.
    The input solar wind velocity is measured by PSP.

2. PFSS extrapolation from source surface to the solar surface.


Process:
    Input:  (i). PFSS parameters: nrho, rss

            (ii). Time range of the psp trajectory one wants to trace.
                  Format( for both start and end time): "YYYY-mm-ddTHH:MM:SS"


    Schematic Illustration:
        Spatial and Temporal coordinates of the selected spacecraft
           -- (Parker Spiral Extrapolation) -->
        The foot point (Point A) of the interplanetary field line at 2.5 R_sun
           -- (PFSS Extrapolation) -->
        The foot point (Point B) of Point A at solar surface

    Output:
        Maps with coordinates of the SC on the source surface and the field lines connecting the SC to the photosphere.
    Last update: 2022.11.15
    "It's valentines day (2023.02.14), I want to confess love to my girlfriend zsy! Love you 3000~"
"""

# import packages

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
from matplotlib.patches import ConnectionPatch
import pandas as pd
import pylab as pl
from datetime import datetime, timedelta
import scipy
from scipy.signal import butter, filtfilt
import asdf
import sys

import pfsspy
import pfsspy.utils
from pfsspy import tracing
import pyvista as pv

import sunpy
import sunpy.map
from sunpy.io.cdf import read_cdf
from sunpy.net import Fido, attrs as a
from sunpy.coordinates import frames, get_body_heliographic_stonyhurst
from sunkit_pyvista import SunpyPlotter

from astropy.visualization import AsymmetricPercentileInterval, ImageNormalize, LogStretch
import astropy.constants as const
import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.io.fits as fits
from astropy.wcs import WCS
from reproject import reproject_interp
import concurrent.futures

from download_data import download_psp, download_gong_adapt, download_aia_synoptic, mkd
os.chdir(sys.path[0])

# This parameter is adjustible. It shifts the output aia-backgroud map, making its center longitude to the parameter.
reproject_central_lon = 180.0

def reproject_map(map, central_lon, shape_out):
    """
    central_lon: float, not quantity
    """
    header_out = map.fits_header
    header_out['CRVAL1'] = central_lon
    header_out['CRVAL2'] = 0.0
    wcs_out = WCS(header_out)
    array_out, footprint = reproject_interp((map.data, map.wcs), wcs_out, shape_out)
    out_map = sunpy.map.Map((array_out, header_out))

    return out_map

def temporal_resample(dataframe, target_dt, keys):
    # unit for target_dt must be second!!!
    time = dataframe.index
    time_index = [(time[i] - time[0]).total_seconds() for i in
                  range(len(time))]  # trasfer timestamps to numbers by there gap to the first timestamp (sec)

    ts = int(time_index[0])  # time index start, always 0.0
    te = (time_index[-1])  # time index end

    time_target_index = np.linspace(ts, te - te % target_dt, int((te - te % target_dt) / target_dt) + 1)

    res_dict = {}
    for key in keys:
        values = dataframe[key].values
        f = scipy.interpolate.interp1d(time_index, values)
        y_new = f(time_target_index)
        res_dict[key] = y_new

    time_target_stamps = [time[0] + timedelta(seconds=x) for x in time_target_index]
    res_df = pd.DataFrame(res_dict, index=time_target_stamps)

    return res_df

def read_psp(spi_path, spc_path, times_sub, res_path):
    """
    Function for reading en2 psp_swp_spc_l3i data.
    :param psp_path: the path of the folder that contains all the psp codes.
    :param res_path: the path of the folder that saves the solar wind plot.
    :return: sw_vel as a dataframe.
    """

    def butter_lowpass_filter(data, cutoff, fs, order=5):
        # The butter low pass filter
        def butter_lowpass(cutoff, fs, order=5):
            return butter(order, cutoff, fs=fs, btype='low')

        b, a = butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y

    # read psp datas and concat to a dataframe
    # for convenience, spi scale down to 10%.
    spi_files = [x for x in os.listdir(spi_path) if 'swp_spi' in x]
    spi_files.sort()
    spi_dataframe = pd.concat([read_cdf(spi_path + '/' + x)[0].to_dataframe() for x in spi_files])[::10]
    spi_dataframe = spi_dataframe.dropna()

    keys_spi = ['VEL_RTN_SUN_0']
    target_dt = 300
    new_df_spi = temporal_resample(spi_dataframe, target_dt, keys_spi)
    values_sw = new_df_spi['VEL_RTN_SUN_0'].values

    # for convenience, spc scale down to 1 %
    spc_files = [x for x in os.listdir(spc_path) if 'swp_spc' in x]
    spc_files.sort()
    spc_dataframe = pd.concat([read_cdf(spc_path + '/' + x)[0].to_dataframe() for x in spc_files])[::100]
    spc_dataframe = spc_dataframe.dropna()

    keys_spc = ['sc_pos_HCI_0', 'sc_pos_HCI_1', 'sc_pos_HCI_2', 'carr_longitude', 'carr_latitude']
    new_df_spc = temporal_resample(spc_dataframe, target_dt, keys_spc)

    # Cutting-off frequency in the butter-low-pass filter
    cutoff = 2e-4
    fs = 1.0 / target_dt

    new_y = butter_lowpass_filter(values_sw, cutoff, fs, order=1)

    # The to-be-returned new solar wind velocity dataframe.
    sw_buttered_df = pd.DataFrame({'VEL_RTN_SUN_0': new_y}, index=new_df_spi.index)

    # Get the coordinates of PSP.
    psp_HCI_x = new_df_spc['sc_pos_HCI_0'] * u.km
    psp_HCI_y = new_df_spc['sc_pos_HCI_1'] * u.km
    psp_HCI_z = new_df_spc['sc_pos_HCI_2'] * u.km

    psp_carr_lon = new_df_spc['carr_longitude'] * u.deg
    psp_carr_lat = new_df_spc['carr_latitude'] * u.deg

    # Position of PSP, HCI and HGS

    psp_HCI_radius = np.sqrt(psp_HCI_x ** 2 + psp_HCI_y ** 2 + psp_HCI_z ** 2)
    psp_HCI_lat = np.radians(np.arcsin(psp_HCI_z / psp_HCI_radius))
    psp_HCI_lon = np.radians(np.arctan2(psp_HCI_y, psp_HCI_x))

    psp_coordinates_HCI = [SkyCoord(psp_HCI_lon[i], psp_HCI_lat[i], psp_HCI_radius[i],
                                    frame=frames.HeliocentricInertial,
                                    obstime=psp_HCI_x.keys()[i]) for i in range(len(psp_HCI_x))]

    psp_coordinates_carr = [SkyCoord(psp_carr_lon[i], psp_carr_lat[i], psp_HCI_radius[i],
                                     frame=frames.HeliographicCarrington(observer='Sun', obstime=psp_HCI_x.keys()[i]))
                            for i in range(len(psp_carr_lat))]


    return psp_coordinates_HCI, psp_coordinates_carr, sw_buttered_df, psp_carr_lon, psp_carr_lat

def Parker_Spiral_extrapolation(SC_coordinates, SC_coordinates_carr, psp_carr_lon, psp_carr_lat, sw_vel, ss_height, times_sub, res_path):
    """
    The function to connect the spacecraft to 2.5 R_sun via Parker Spiral extrapolation.
    Parker Spiral Equation:

    | $ B_R (R, \theta, \Phi) = B_R (R_0, \theta, \Phi_0) \frac{R_0 ^2}{R^2} $
    | $ B_{\Phi} (R, \theta, \Phi) = - B_R (R_0, \theta, \Phi_0) \frac{\Omega R_0^2 sin(\theta)}{V_R R} $
    | $ B_{\theta} (R, \theta, \Phi) = 0 $

    :param SC_coordinates: coordinates of the selected spacecraft.
    :param vsw: velocity of the solar wind.
    :param ss_height: the height of the source surface (unit: R_sun)
    :return: coordinates of the foot point at 2.5 R_sun.
    """
    def Omega_sun(phi):
        A = 14.713 * (u.deg / u.day)
        B = -2.396 * (u.deg / u.day)
        C = -1.787 * (u.deg / u.day)
        return A + B * (np.sin(phi)).value ** 2 + C * (np.sin(phi)).value ** 4


    N = 400
    t = np.linspace(0, 2 * np.pi, N) * (u.rad)

    r_ss = ss_height * const.R_sun.to(u.Rsun)

    r_psp = [x.distance.to(u.Rsun) for x in SC_coordinates]
    lat_psp = [x.lat.to(u.rad) for x in SC_coordinates]
    lon_psp = [x.lon.to(u.rad) for x in SC_coordinates]


    t_start_str = datetime.strftime(times_sub[0][0], '%Y-%m-%dT%H:%M:%S')  # start date in string
    t_end_str = datetime.strftime(times_sub[-1][-1], '%Y-%m-%dT%H:%M:%S')  # end date in string

    Earth1 = get_body_heliographic_stonyhurst('earth', t_start_str.split('T')[0]).transform_to(frames.HeliocentricInertial)

    select_indexes = [(len(sw_vel[sw_vel.index < x[0]]), len(sw_vel[sw_vel.index < x[1]])) for x in times_sub]


    fig, ax = plt.subplots(figsize = (10, 5), subplot_kw = {'projection' : 'polar'})
    ax.plot([x.value for x in lon_psp], [x.value for x in r_psp], color = 'blue', label='PSP trajectory')
    Source_Sur = pl.Circle((0.0, 0.0), ss_height, transform=ax.transData._b, color="yellow", alpha=0.4, label='Source Surface')
    ax.add_artist(Source_Sur)

    lon_ss = []
    lon_ss_carr = []
    lat_ss = lat_psp

    lat_carr = psp_carr_lat
    lon_carr = psp_carr_lon

    for i in range(len(SC_coordinates)):
        solarwind_velocity = sw_vel['VEL_RTN_SUN_0'][i] * u.km / u.s
        N2 = 500
        PS_line = np.linspace(r_ss.value, r_psp[i].value, N2) * (u.Rsun)
        phi_sc = lon_psp[i].to(u.rad)
        lons = phi_sc + Omega_sun(lat_psp[i]) / solarwind_velocity * (r_psp[i] - PS_line)
        lons = lons.to(u.rad)
        lon_ss.append(lons[0])

        phi_sc_carr = lon_carr[i].to(u.rad)
        lons_carr = phi_sc_carr + Omega_sun(lat_psp[i]) / solarwind_velocity * (r_psp[i] - PS_line)
        lons_carr = lons_carr.to(u.rad)
        lon_ss_carr.append(lons_carr[0])

        for select_index in select_indexes:
            if i == select_index[0]:
                ax.plot(lons, PS_line, 'green', label='Parker Spiral')
                ax.scatter(lons[-1], PS_line[-1], marker='o', color='brown', label=t_start_str)
            if i == select_index[1]:
                ax.plot(lons, PS_line, 'green')
                ax.scatter(lons[-1], PS_line[-1], marker='o', color='orange', label=t_end_str)

    ax.plot(Earth1.lon.to(u.rad), 80 * u.Rsun, marker='o', color='cornflowerblue', label='Earth ' + t_start_str.split('T')[0])
    ax.set_title('Parker Spiral Line, r_unit = R_sun')
    plt.legend(loc=3, bbox_to_anchor=(1.05, 0))
#    plt.show()
    plt.savefig(res_path + '/Parker_Spiral_extrapolation.jpg', format='jpg', dpi=400)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(lon_ss_carr[0].to(u.deg).value, lat_carr[0].to(u.deg).value, marker='o', label='start', color='green')
    ax.scatter(lon_ss_carr[-1].to(u.deg).value, lat_carr[-1].to(u.deg).value, marker='o', label='end', color='red')
    ax.plot([x.to(u.deg).value for x in lon_ss_carr], [x.to(u.deg).value for x in lat_carr])
    for select_index in select_indexes:
        select_index_start = select_index[0]
        select_index_end = select_index[1]
        ax.plot([x.to(u.deg).value for x in lon_ss_carr[select_index_start:select_index_end]],
                [x.to(u.deg).value for x in lat_carr[select_index_start:select_index_end]],
                color='orange', label='Selected Part')
    ax.set_title('PSP trajectory on Source Surface')
    ax.set_xlabel('Carrington Longitude (deg)')
    ax.set_ylabel('Carrington Latitude (deg)')
    plt.legend()
#    plt.show()
    plt.savefig(res_path + '/psp_trajectory_ss.jpg', format='jpg', dpi=400)

    coordinates_ss_carr = [SkyCoord(lon_ss_carr[i], lat_carr[i], r_ss.to(u.Rsun),
                               frame=SC_coordinates_carr[i].frame)
                      for i in range(len(SC_coordinates_carr))]

    return coordinates_ss_carr, select_indexes

def pfss_trace(coord_carr, rss, time_gong, res_path, select_indexes):
    """
    Function for pfss trace from source surface to photosphere
    :param hmi_map: hmi synoptic map
    :param ss_coordinate: the coordinate of the observer traced back to the source surface via PS extrapolation
    :return: the coordinates on the photosphere
    """

    def read_hmi_synoptic():
        hmi_map = sunpy.map.Map('data/hmi.Synoptic_Mr.2243.fits')
        # Replace NaN with the average value of the closest ten values
        nan_posi = np.where(np.isnan(hmi_map.data))
        for i in range(len(nan_posi[0])):
            position = (nan_posi[0][i], nan_posi[1][i])
            hmi_map.data[position] = np.average(hmi_map.data[position[0]-10:position[0], position[1]])
        return hmi_map
#    mag_map = read_hmi_synoptic()
#    mag_map = mag_map.resample([720, 360] * u.pix)

    def read_adapt_gong():
        adapt_fname = [x for x in os.listdir('data/gong/adapt/' + time_gong.split('T')[0].replace('-', '_') + '/' +
                                             time_gong.split('T')[1].replace(':', '')) if 'adapt403' in x]
        adapt_fname = 'data/gong/adapt/' + time_gong.split('T')[0].replace('-', '_') + '/' + \
                      time_gong.split('T')[1].replace(':', '') + '/' + adapt_fname[0]
        adapt_fits = fits.open(adapt_fname)
        sum_array = np.zeros(adapt_fits[0].data[0].shape)
        for i in range(len(adapt_fits[0].data)):
            sum_array += adapt_fits[0].data[i]
        average_array = sum_array / len(adapt_fits[0].data)
        adapt_average_map = sunpy.map.Map(average_array, adapt_fits[0].header)
        adapt_map_cea = pfsspy.utils.car_to_cea(adapt_average_map)
        return adapt_map_cea

    mag_map_original = read_adapt_gong()
    mag_map = reproject_map(mag_map_original, reproject_central_lon, (180, 360))

    # Set the model parameters
    nrho = 35

    # Construct the inputs, and calculate the outputs
    pfss_in = pfsspy.Input(mag_map, nrho, rss)
    print('calculating pfss...')
    pfss_out = pfsspy.pfss(pfss_in)
    print('Finished calculating pfss')

    # Trace the magnetic field lines in 3D.
    # Set up the tracing seeds
    lon = [c.lon for c in coord_carr]
    lat = [c.lat for c in coord_carr]
    r = [c.radius.to(u.m) for c in coord_carr]

    lons_select = [[c.lon for c in coord_carr[select_index[0]: select_index[1]+1]]
                   for select_index in select_indexes]

    lats_select = [[c.lat for c in coord_carr[select_index[0]: select_index[1]+1]]
                   for select_index in select_indexes]

    rs_select = [[c.radius.to(u.m) for c in coord_carr[select_index[0]: select_index[1]+1]]
                 for select_index in select_indexes]

    seeds = [SkyCoord(lons_select[i], lats_select[i], rs_select[i], frame=pfss_out.coordinate_frame)
             for i in range(len(lons_select))]
    seeds_all = SkyCoord(lon, lat, r, frame=pfss_out.coordinate_frame)

    # Trace the field lines
    print('Tracing field lines...')
    tracer = tracing.FortranTracer()
    field_lines = [tracer.trace(seed, pfss_out) for seed in seeds]
    print('Finished tracing field lines')


    fig = plt.figure()
    m = pfss_in.map
    ax = plt.subplot(projection=m)
    m.plot(axes=ax)
    for seed in seeds:
        for c in seed:
            ax.plot_coord(c, color='orange', marker='o', linewidth=1, markersize=2, label='psp trajectory')
    ax.plot_coord(pfss_out.source_surface_pils[0])
    plt.legend(['PSP trajectory'])
#    plt.show()
    plt.savefig(res_path + '/overplot_gong.jpg', format='jpg', dpi=400)

    return field_lines, seeds, seeds_all, coord_carr, pfss_in, pfss_out, mag_map, mag_map_original

def Plots(field_lines, seeds, seeds_all, pfss_in, pfss_out, mag_map, mag_map_original, res_path, select_indexes, bottom_left_coords, top_right_coords):
    m = pfss_in.map
    m = reproject_map(m, reproject_central_lon, (180, 360))
    # PLot field line over GONG magnetogram and open/closed field lines map.
    # ===============================================================================================================
    # Set up the full map grids
    # Number of steps in cos(latitude)
    r = const.R_sun
    # Spatial resolution of the global tracing result
    nsteps = 45
    lon_1d = np.linspace(0, 2 * np.pi, 360)
    lat_1d = np.arcsin(np.linspace(-1, 1, 180))
    lon, lat = np.meshgrid(lon_1d, lat_1d, indexing='ij')
    lon, lat = lon * u.rad, lat * u.rad
    seeds_global = SkyCoord(lon.ravel(), lat.ravel(), r, frame=pfss_out.coordinate_frame)
    tracer = tracing.FortranTracer(max_steps=2000)
    print('Global Tracing ...')
    field_lines_all = tracer.trace(seeds_global, pfss_out)
    print('Finished global tracing')

    # Get the coronal hole map
    cmap = colors.ListedColormap(['tab:red', 'black', 'tab:blue'])
    norm = colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], ncolors=3)
    pols = field_lines_all.polarities.reshape(360, 180).T
    CH_map = sunpy.map.Map(pols, mag_map_original.fits_header)
    CH_map = reproject_map(CH_map, reproject_central_lon, (180, 360))
    comp_map = sunpy.map.Map(mag_map, CH_map, composite=True, plot_settings=mag_map.plot_settings)

    levels = [-1.0, 0, 1.0]
    comp_map.set_levels(index=1, levels=levels)

    bottom_left = SkyCoord(bottom_left_coords[0] * u.deg, bottom_left_coords[1] * u.deg, frame=m.coordinate_frame)
    top_right = SkyCoord(top_right_coords[0] * u.deg, top_right_coords[1] * u.deg, frame=m.coordinate_frame)

    mag_small = mag_map.submap(bottom_left, top_right=top_right)
    CH_small = CH_map.submap(bottom_left, top_right=top_right)
    comp_small = sunpy.map.Map(mag_small, CH_small, composite=True, plot_settings=mag_small.plot_settings)
    levels = [-1.0, 0, 1.0]
    comp_small.set_levels(index=1, levels=levels)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection=mag_small)
    comp_small.plot(axes=ax)
    for c in seeds_all:
        ax.plot_coord(c, color='blue', marker='o', linewidth=1, markersize=1)

    for i in range(len(select_indexes)):
        seed = seeds[i]
        field_line = field_lines[i]
        for c in seed:
                ax.plot_coord(c, color='green', marker='o', linewidth=1, markersize=1)
        for fline in field_line[::2]:
            if len(fline.coords) > 0:
                ax.plot_coord(fline.coords, alpha=0.7, linewidth=0.5, color='green')

    flg1 = pfss_out.source_surface_pils[0].lon > bottom_left.lon
    flg1 = np.where(flg1 == True)[0][0]

    flg2 = pfss_out.source_surface_pils[0].lon > top_right.lon
    flg2 = np.where(flg2 == False)[0][-1]

    subpil1 = pfss_out.source_surface_pils[0][flg1: flg2 + 1]
    flg3 = subpil1.lat > bottom_left.lat
    flg3 = np.where(flg3 == True)[0]

    flg4 = subpil1.lat < top_right.lat
    flg4 = np.where(flg4 == True)[0]

    flg = [a for a in flg3 if a in flg4]
    subpil = subpil1[flg]

    ax.plot_coord(subpil, color='black')
    plt.savefig(res_path + '/trace_gong_CH.jpg', format='jpg', dpi=400)
#    plt.show()

    return CH_map

def make_aia_syn_plot(aia_path, pfss_out, CH_map, field_lines, coord_carr, res_path, select_indexes, obstime, aia_CR_number, bottom_left_coords, top_right_coords):
    """
    Create a AIA synoptic map and overplot trajectory/field lines over the aia map.
    """

    # Step 1: Create the synoptic AIA map for CR2243.
    # ================================================================================================
    # The original header is not standard, errors happen when one wants to map directly.
    if os.path.exists(aia_path) == False:
        download_aia_synoptic(aia_CR_number)

    data = sunpy.map.Map(aia_path).data

    # crete a new header
    plot_settings = sunpy.map.Map('data/aia/aia_synoptic_20210501.fits').plot_settings
    shape_out = (360, 720)
    header_new = sunpy.map.make_fitswcs_header(shape_out,
                                               SkyCoord(-180.0, 0.0, unit=u.deg,
                                                        frame=frames.HeliographicCarrington,
                                                        observer='earth',
                                                        obstime=obstime),
                                               scale=[360 / shape_out[1],
                                                      180 / shape_out[0]] * u.deg / u.pix,
                                               instrument='aia193',
                                               observatory='sdo',
                                               wavelength=193 * u.AA,  projection_code='CAR')

    out_map = sunpy.map.Map(data, header_new)
    out_map = reproject_map(out_map, reproject_central_lon, (360, 720))
    out_map.plot_settings = plot_settings
    out_map = out_map.reproject_to(CH_map.wcs)

    # get the seeds for pfss.
    lon = [c.lon for c in coord_carr]
    lat = [c.lat for c in coord_carr]
    r = [c.radius.to(u.m) for c in coord_carr]

    lons_select = [[c.lon for c in coord_carr[select_index[0]: select_index[1] + 1]]
                   for select_index in select_indexes]

    lats_select = [[c.lat for c in coord_carr[select_index[0]: select_index[1] + 1]]
                   for select_index in select_indexes]

    rs_select = [[c.radius.to(u.m) for c in coord_carr[select_index[0]: select_index[1] + 1]]
                 for select_index in select_indexes]

    seeds = [SkyCoord(lons_select[i], lats_select[i], rs_select[i], frame=out_map.coordinate_frame)
             for i in range(len(lons_select))]
    seeds_all = SkyCoord(lon, lat, r, frame=out_map.coordinate_frame)


    bottom_left = SkyCoord(bottom_left_coords[0] * u.deg, bottom_left_coords[1] * u.deg, frame=CH_map.coordinate_frame)
    top_right = SkyCoord(top_right_coords[0] * u.deg, top_right_coords[1] * u.deg, frame=CH_map.coordinate_frame)
    out_submap = out_map.submap(bottom_left, top_right=top_right)
    CH_submap = CH_map.submap(bottom_left, top_right=top_right)


    fig = plt.figure(figsize=(7.3, 7))
    ax = fig.add_subplot(111, projection=out_submap)
    out_submap.plot(axes=ax)
    bounds = ax.axis()
    CH_submap.draw_contours(levels=[-1.0, 0.0, 1.0], axes=ax, colors=['deeppink', 'deepskyblue'],
                            linestyles='solid', alpha=0.8)
    ax.axis(bounds)
    for c in seeds_all:
        ax.plot_coord(c, color='blue', marker='o', linewidth=1, markersize=1)
    for i in range(len(select_indexes)):
        seed = seeds[i]
        field_line = field_lines[i]
        for c in seed:
                ax.plot_coord(c, color='green', marker='o', linewidth=1, markersize=1)
        for fline in field_line[::2]:
            if len(fline.coords) > 0:
                ax.plot_coord(fline.coords, alpha=0.7, linewidth=0.5, color='green')

    flg1 = pfss_out.source_surface_pils[0].lon > bottom_left.lon
    flg1 = np.where(flg1 == True)[0][0]

    flg2 = pfss_out.source_surface_pils[0].lon > top_right.lon
    flg2 = np.where(flg2 == False)[0][-1]

    subpil1 = pfss_out.source_surface_pils[0][flg1: flg2 + 1]
    flg3 = subpil1.lat > bottom_left.lat
    flg3 = np.where(flg3 == True)[0]

    flg4 = subpil1.lat < top_right.lat
    flg4 = np.where(flg4 == True)[0]

    flg = [a for a in flg3 if a in flg4]
    subpil = subpil1[flg]

    ax.plot_coord(subpil, color='black')
    ax.set_title(obstime, fontsize=16)
    ax.minorticks_on()

    longitude = ax.coords[0]
    latitude = ax.coords[1]

    longitude.display_minor_ticks(True)
    latitude.display_minor_ticks(True)
    longitude.set_minor_frequency(2)
    latitude.set_minor_frequency(3)

    longitude.set_axislabel('Carrington Longitude', fontsize=16)
    latitude.set_axislabel('Carrington Latitude', fontsize=16)

    longitude.set_ticks(spacing=30 * u.deg)

    ax.tick_params(axis='both', labelsize=15)
    ax.tick_params(which='major', length=7.0)
    ax.tick_params(which='minor', length=4.5)
    plt.savefig(res_path + '/aia_overplot.jpg', format='jpg', dpi=400)
#    plt.show()

    # ============================================
    # save important variables for further use.
    out_submap.save(res_path + '/out_submap.fits', overwrite=True)
    CH_submap.save(res_path + '/CH_submap.fits', overwrite=True)

    tree = {'seeds_all': seeds_all, 'subpil': subpil}
    with asdf.AsdfFile(tree) as asdf_file:
        asdf_file.write_to(res_path + '/seeds_pil.asdf')

    df_select_indexes = pd.DataFrame({'select_indexes': select_indexes})
    df_select_indexes.to_csv(res_path + '/select_indexes.csv', index=False)

    return 0

def main():
    hh = '00'
    times_sub = [(pd.Timestamp("2021-08-05 00:10:00"), pd.Timestamp("2021-08-14 23:50:00"))]
    rss = 2.5
    date_gong ='2021_08_17'
    cr_aia_num = 2247
    obstime = date_gong.replace('_', '-') + ' ' + ':'.join([hh, '00', '00'])

    bottom_left_coords = [0.0, -90.0]
    top_right_coords = [180.0, 90.0]

    # Step0: Download the gong data
    download_gong_adapt(obstime.replace(' ', 'T'))

    # Step1: read psp data, return coordinates of the spacecraft and the solar wind velocity.
    if os.path.exists('res/source_region/' + date_gong) == False:
        mkd('res/source_region/' + date_gong)
    SC_coordinates, SC_coordinates_carr, sw_vel, psp_carr_lon, psp_carr_lat = read_psp('data/psp/SWEAP/spi', 'data/psp/SWEAP/spc', times_sub, 'res/source_region/' + date_gong)

    # Step2: PSP extrapolation.
    coordinates_ss_carr, select_indexes = Parker_Spiral_extrapolation(SC_coordinates, SC_coordinates_carr, psp_carr_lon, psp_carr_lat, sw_vel, rss, times_sub, 'res/source_region/' + date_gong)

    time_gong = date_gong.replace('_', '-') + 'T' + hh + ':00:00'
    res_path = 'res/source_region/' + time_gong.split('T')[0].replace('-', '_') + '/' + time_gong.split('T')[1].replace(
        ':', '')
    if os.path.exists(res_path) == False:
        mkd(res_path)
    field_lines, seeds, seeds_all, coord_carr, pfss_in, pfss_out, mag_map, mag_map_original = pfss_trace(coordinates_ss_carr, rss,
                                                                                       time_gong, res_path, select_indexes)

    CH_map = Plots(field_lines, seeds, seeds_all, pfss_in, pfss_out, mag_map, mag_map_original, res_path, select_indexes, bottom_left_coords, top_right_coords)
    make_aia_syn_plot('data/aia/aia193_synmap_cr'+str(cr_aia_num)+'.fits', pfss_out, CH_map,
                      field_lines, coord_carr, res_path, select_indexes, obstime, cr_aia_num, bottom_left_coords, top_right_coords)

    return 0


if __name__ == "__main__":
    main()

