"""
This code is for trace the source region of a specific solar wind observed by PSP.

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
"""

# import packages

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import gridspec
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import pandas as pd
import pylab as pl
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pyspedas

import pfsspy
import pfsspy.utils
from pfsspy import tracing
import pyvista as pv

import sunpy
import sunpy.map
from sunpy.io.cdf import read_cdf
from sunpy.net import Fido, attrs as a
from sunpy.coordinates import frames, get_body_heliographic_stonyhurst
#from sunkit_pyvista import SunpyPlotter

from astropy.visualization import AsymmetricPercentileInterval, ImageNormalize, LogStretch
import astropy.constants as const
import astropy.units as u
from astropy.coordinates import SkyCoord
import astropy.io.fits as fits
from astropy.wcs import WCS

from download_data import download_psp, download_gong_adapt

def read_psp_data(psp_path):
    """
    The function for reading hmi and psp files.
    :date: format: YYYY-MM-DD
    :return: a hmi map and a psp data.
    """
    # Deal with psp data, return psp location.
    # ======================================================================================
    list_psp = os.listdir(psp_path)
    psp_l3i = [x for x in list_psp if 'psp_swp_spc_l3i' in x]
    psp_l3i.sort()
    psp_dataframes_lst = [read_cdf(psp_path + '/' + x)[0].to_dataframe() for x in psp_l3i]
    psp_dataframe = pd.concat(psp_dataframes_lst)

    # Derive the solar wind velocity
    sw_vel = np.sqrt(psp_dataframe['vp1_fit_RTN_0'] ** 2 + psp_dataframe['vp1_fit_RTN_1'] ** 2 + psp_dataframe['vp1_fit_RTN_2'] ** 2)[::2000] * (u.km / u.s)

    psp_HCI_x = psp_dataframe['sc_pos_HCI_0'][::2000] * u.km # 1/2000 data selection rate
    psp_HCI_y = psp_dataframe['sc_pos_HCI_1'][::2000] * u.km
    psp_HCI_z = psp_dataframe['sc_pos_HCI_2'][::2000] * u.km

    psp_carr_lon = psp_dataframe['carr_longitude'][::2000] * u.deg
    psp_carr_lat = psp_dataframe['carr_latitude'][::2000] * u.deg
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


    # =======================================================================================

    return psp_coordinates_HCI, psp_coordinates_carr, sw_vel

def psp_trajectory_nd():
    psp_pos = read_cdf('data/psp/psp_helio1day_position_20180813_v01.cdf')
    psp_pos = psp_pos[0]
    psp_pos_dataframe = psp_pos.to_dataframe()

    lats = psp_pos_dataframe['HGI_LAT']['2021-04-15' : '2021-05-25']
    lons = psp_pos_dataframe['HGI_LON']['2021-04-15' : '2021-05-25']
    Radiis = psp_pos_dataframe['RAD_AU']['2021-04-15' : '2021-05-25']

    lats_rad = [(x * u.deg).to(u.rad) for x in lats]
    lons_rad = [(x * u.deg).to(u.rad) for x in lons]
    Radiis_Rs = [(x * u.AU).to(u.Rsun) for x in Radiis]

    fig, ax = plt.subplots(figsize=(5,5), subplot_kw={'projection' : 'polar'})
    ax.plot([x.value for x in lons_rad], [x.value for x in Radiis_Rs], label='PSP')
#    ax.scatter(lons_rad[6].value, Radiis_Rs[6].value, marker='*', color='red', label='0501')
#    ax.scatter(lons_rad[7].value, Radiis_Rs[7].value, marker='*', color='green', label='0502')
#    plt.legend(['psp', '0501', '0502'])
    plt.show()
    return 0

def Parker_Spiral_extrapolation(SC_coordinates, SC_coordinates_carr, sw_vel, ss_height, t_start, t_end):
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

    N = 400
    Omega_sun = 2.662e-6 * (u.rad / u.s)  # rad/s
    t = np.linspace(0, 2 * np.pi, N) * (u.rad)

    r_ss = ss_height * const.R_sun.to(u.Rsun)

    r_psp = [x.distance.to(u.Rsun) for x in SC_coordinates]
    lat_psp = [x.lat.to(u.rad) for x in SC_coordinates]
    lon_psp = [x.lon.to(u.rad) for x in SC_coordinates]

    # 0501 projection
    dstart = t_start.split('T')[0] # start date
    dend = t_end.split('T')[0]  # end date

    SC_coordinates_0501 = []
    SC_coordinates_0501_carr = []
    SW_vel_0501 = []
    for i in range(len(SC_coordinates)):
        if SC_coordinates[i].obstime.value < datetime.strptime(dend, '%Y-%m-%d') and \
                SC_coordinates[i].obstime.value > datetime.strptime(dstart, '%Y-%m-%d'):
            SC_coordinates_0501.append(SC_coordinates[i])
            SC_coordinates_0501_carr.append(SC_coordinates_carr[i])
            SW_vel_0501.append(sw_vel[i])

    r_0501 = [x.distance.to(u.Rsun) for x in SC_coordinates_0501]
    lat_0501 = [x.lat.to(u.rad) for x in SC_coordinates_0501]
    lon_0501 = [x.lon.to(u.rad) for x in SC_coordinates_0501]
    lat_0501_carr = [x.lat.to(u.rad) for x in SC_coordinates_0501_carr]
    lon_0501_carr = [x.lon.to(u.rad) for x in SC_coordinates_0501_carr]

    Earth1 = get_body_heliographic_stonyhurst('earth', dstart).transform_to(frames.HeliocentricInertial)
    Earth2 = get_body_heliographic_stonyhurst('earth', dend).transform_to(frames.HeliocentricInertial)

    fig, ax = plt.subplots(figsize = (8, 5), subplot_kw = {'projection' : 'polar'})
    ax.plot([x.value for x in lon_psp], [x.value for x in r_psp], color = 'blue', label='PSP trajectory')
    Source_Sur = pl.Circle((0.0, 0.0), ss_height, transform=ax.transData._b, color="yellow", alpha=0.4, label='Source Surface')
    ax.add_artist(Source_Sur)
    lon_ss = []
    lon_ss_carr = []
    lat_ss = lat_0501

    for i in range(len(SC_coordinates_0501)):
        N2 = 500
        PS_line = np.linspace(r_ss.value, r_0501[i].value, N2) * (u.Rsun)
        phi_sc = lon_0501[i].to(u.rad)
        lons = phi_sc + Omega_sun / sw_vel[i] * (r_0501[i] - PS_line)
        lons = lons.to(u.rad)
        lon_ss.append(lons[0])

        phi_sc_carr = lon_0501_carr[i].to(u.rad)
        lons_carr = phi_sc_carr + Omega_sun / sw_vel[i] * (r_0501[i] - PS_line)
        lons_carr = lons_carr.to(u.rad)
        lon_ss_carr.append(lons_carr[0])
        if i == 0:
            ax.plot(lons, PS_line, 'green', label='Parker Spiral')
            ax.scatter(lons[-1], PS_line[-1], marker='o', color='brown', label=dstart)
        if i == len(SC_coordinates_0501) - 1:
            ax.plot(lons, PS_line, 'green')
            ax.scatter(lons[-1], PS_line[-1], marker='o', color='orange', label=dend)
        if SC_coordinates_0501[i].obstime.value - datetime.strptime('2021-05-01T00:30', '%Y-%m-%dT%H:%M') < timedelta(minutes=10)\
                and SC_coordinates_0501[i].obstime.value - datetime.strptime('2021-05-01T00:30', '%Y-%m-%dT%H:%M') > timedelta(minutes=0):
            ax.plot(lons, PS_line, 'green')
            ax.scatter(lons[-1], PS_line[-1], marker='o', color='olive', label='2021-05-01')
    ax.plot(Earth1.lon.to(u.rad), 80 * u.Rsun, marker='o', color='cornflowerblue', label='Earth ' + dstart)
    ax.plot(Earth2.lon.to(u.rad), 80 * u.Rsun, marker='o', color='midnightblue', label='Earth ' + dend)
    ax.set_title('Parker Spiral Line, r_unit = R_sun')
    plt.legend(loc=3, bbox_to_anchor=(1.05, 0))
#    if os.path.exists('res/adapt/'+dstart.replace('-','_')) == False:
#        os.mkdir('res/adapt/'+dstart.replace('-'))
#    plt.show()
    plt.savefig('res/adapt/Parker_Spiral_extrapolation.jpg', format='jpg', dpi=400)

    coordinates_ss = [SkyCoord(lon_ss[i], lat_ss[i], r_ss.to(u.Rsun),
                               frame=SC_coordinates_0501[i].frame)
                      for i in range(len(SC_coordinates_0501))]

    coordinates_ss_carr = [SkyCoord(lon_ss_carr[i], lat_0501_carr[i], r_ss.to(u.Rsun),
                                    frame=SC_coordinates_0501_carr[i].frame)
                           for i in range(len(SC_coordinates_0501_carr))]

    return coordinates_ss, coordinates_ss_carr

def coordinate_ss_respect_to_Sun(coordinates_ss, carrington_frame):
    """
    Plot the relative (considering the solar rotation) coordinates on the source surface.
    """
    def solar_omega(phi):
        # phi should be a quantity object
        A = 14.713 * (u.deg / u.day)
        B = -2.396 * (u.deg / u.day)
        C = -1.787 * (u.deg / u.day)
        return A + B * (np.sin(phi) ** 2) + C * (np.sin(phi) ** 4)
    coordinates_relative = []
    coordinates_relative.append(coordinates_ss[0])

    for i in range(1, len(coordinates_ss)):
        time_delta = (coordinates_ss[i].obstime.value - coordinates_ss[i-1].obstime.value).seconds * u.s
        lon_delta = coordinates_ss[i].lon - coordinates_ss[i-1].lon
        solar_rotation = solar_omega(coordinates_ss[i].lat)
        solar_rad = solar_rotation * time_delta
        relative_rotation = lon_delta - solar_rad
        lon_relative_i = coordinates_relative[i-1].lon + relative_rotation
        cooord_relavtive_i = SkyCoord(lon_relative_i, coordinates_ss[i].lat, coordinates_ss[i].distance,
                                      frame = coordinates_ss[i].frame)
        coordinates_relative.append(cooord_relavtive_i)

    return coordinates_relative

def pfss_trace(coord_carr, rss, time_gong, index_gong):
    """
    Function for pfss trace from source surface to photosphere
    :param hmi_map: hmi synoptic map
    :param ss_coordinate: the coordinate of the observer traced back to the source surface via PS extrapolation
    :return: the coordinates on the photosphere
    """

    def read_hmi_synoptic():
        hmi_map = sunpy.map.Map('data/hmi.Synoptic_Mr.2243.fits')

        # Replace NaN with the average value of the
        nan_posi = np.where(np.isnan(hmi_map.data))

        for i in range(len(nan_posi[0])):
            position = (nan_posi[0][i], nan_posi[1][i])
            hmi_map.data[position] = np.average(hmi_map.data[position[0]-10:position[0], position[1]])


        return hmi_map
#    mag_map = read_hmi_synoptic()
#    mag_map = mag_map.resample([720, 360] * u.pix)


    def read_adapt_gong():
        adapt_fnames = os.listdir('data/gong/adapt/'+time_gong.split('T')[0].replace('-', '_') + '/' +
                                  time_gong.split('T')[1].replace(':', ''))
        adapt_fname = 'data/gong/adapt/'+time_gong.split('T')[0].replace('-', '_') + '/' + \
                      time_gong.split('T')[1].replace(':', '') + '/' + adapt_fnames[1]
        adapt_fits = fits.open(adapt_fname)
        data_header_pairs = [(map_slice, adapt_fits[0].header) for map_slice in adapt_fits[0].data]
        adapt_maps = sunpy.map.Map(data_header_pairs, sequence=True)
        adapt_map_cea = pfsspy.utils.car_to_cea(adapt_maps[index_gong])
        return adapt_map_cea

    mag_map = read_adapt_gong()

    # Set the model parameters
    nrho = 35

    # Construct the inputs, and calculate the outputs
    pfss_in = pfsspy.Input(mag_map, nrho, rss)
    print('calculating pfss...')
    pfss_out = pfsspy.pfss(pfss_in)

    # Trace the magnetic field lines in 3D.
    # Set up the tracing seeds
    lon = [c.lon for c in coord_carr]
    lat = [c.lat for c in coord_carr]
    r = [c.radius.to(u.m) for c in coord_carr]
    seeds = SkyCoord(lon, lat, r, frame=pfss_out.coordinate_frame)

    # Trace the field lines
    print('Tracing field lines...')
    tracer = tracing.FortranTracer()
    field_lines = tracer.trace(seeds, pfss_out)
    #    field_lines_sph = tracer.trace(seeds_sph, pfss_out)
    print('Finished tracing field lines')

    fig = plt.figure()
    m = pfss_in.map
    ax = plt.subplot(projection=m)
    m.plot(axes=ax)
    for i in range(len(coord_carr)):
        c = seeds[i]
        ax.plot_coord(c, color='orange', marker='o', linewidth=1, markersize=2)
        if (coord_carr[i].obstime.value - datetime.strptime('2021-05-01T00', '%Y-%m-%dT%H')) < timedelta(hours=1) \
                and (coord_carr[i].obstime.value - datetime.strptime('2021-05-01T00', '%Y-%m-%dT%H')) > timedelta(hours=0):
            ax.plot_coord(c, color='black', marker='D', linewidth=0, markersize=5, label='2021-05-01')
    ax.plot_coord(pfss_out.source_surface_pils[0])
    plt.legend()
#    plt.show()
    plt.savefig('res/adapt/overplot_gong.jpg', format='jpg', dpi=400)


    return field_lines, seeds, coord_carr, pfss_in, pfss_out, mag_map

def Plots(field_lines, seeds_ss, coord_carr, pfss_in, pfss_out, mag_map):
    """
    Plot field lines over various of maps.
    :param field_lines: The field lines object returned from pfss extrapolation.
    :return: 0
    """
    m = pfss_in.map
    # PLot field line over GONG magnetogram and open/closed field lines map.
    # ===============================================================================================================
    # Set up the full map grids
    # Number of steps in cos(latitude)
    r = const.R_sun
    nsteps = 45
    lon_1d = np.linspace(0, 2 * np.pi, nsteps * 2 + 1)
    lat_1d = np.arcsin(np.linspace(-1, 1, nsteps + 1))
    lon, lat = np.meshgrid(lon_1d, lat_1d, indexing='ij')
    lon, lat = lon * u.rad, lat * u.rad
    seeds = SkyCoord(lon.ravel(), lat.ravel(), r, frame=pfss_out.coordinate_frame)
    tracer = tracing.FortranTracer(max_steps=2000)
    field_lines_all = tracer.trace(seeds, pfss_out)

    fig = plt.figure(figsize=(9, 10))
    ax = fig.add_subplot(2, 1, 1, projection=m)
    m.plot()
    for i in range(len(coord_carr)):
        c = seeds_ss[i]
        ax.plot_coord(c, color='orange', marker='o', linewidth=1, markersize=2)
        if (coord_carr[i].obstime.value - datetime.strptime('2021-05-01T00', '%Y-%m-%dT%H')) < timedelta(hours=1) \
                and (coord_carr[i].obstime.value - datetime.strptime('2021-05-01T00', '%Y-%m-%dT%H')) > timedelta(hours=0):
            ax.plot_coord(c, color='black', marker='D', linewidth=0, markersize=5, label='2021-05-01')
    ax.plot_coord(pfss_out.source_surface_pils[0])
    for fline in field_lines[::2]:
        if len(fline.coords) > 0:
            ax.plot_coord(fline.coords, alpha=0.7, linewidth=0.5, color='green')
    ax.set_title('Input GONG Map')
    plt.legend()

    ax = fig.add_subplot(2, 1, 2)
    # Plot HCS
    cmap = colors.ListedColormap(['tab:red', 'black', 'tab:blue'])
    norm = colors.BoundaryNorm([-1.5, -0.5, 0.5, 1.5], ncolors=3)
    pols = field_lines_all.polarities.reshape(2 * nsteps + 1, nsteps + 1).T
    ax.contourf(np.rad2deg(lon_1d), np.sin(lat_1d), pols, norm=norm, cmap=cmap)
    lons_carr = [c.lon.value for c in coord_carr] # in deg
    lats_carr = [c.lat.to(u.rad).value for c in coord_carr] # indeg
    pils_lons = [c.lon.value for c in pfss_out.source_surface_pils[0]]
    pils_lats = [c.lat.to(u.rad).value for c in pfss_out.source_surface_pils[0]]

    ax.plot(lons_carr, np.sin(lats_carr), color='orange', linewidth=5)

    for fline in field_lines[::2]:
        lon_fline = fline.coords.lon.value # in deg
        lat_fline = fline.coords.lat.to(u.rad).value # in rad
        ax.plot(lon_fline, np.sin(lat_fline), color='green', linewidth=1)
        ax.plot(pils_lons, np.sin(pils_lats), color='blue', linewidth=0.6)
    ax.set_ylabel('sin(Latitude)')
    ax.set_title('Open (blue/red) closed (black) field')
    ax.set_aspect(0.5 * 360 / 2)
#    plt.show()
#    if os.path.exists('res/adapt/' + h) == False:
#        os.mkdir('res/adapt/' + h)
    plt.savefig('res/adapt/gong_CH_overplot.jpg', format='jpg', dpi=400)
    # ===============================================================================================================


    """
    # Create a plotter
    # ===============================================================================================================
    plotter = SunpyPlotter()

    # Plot a map
    plotter.plot_map(gong_map, clip_interval=[1, 99] * u.percent)
    # Add an arrow to show the solar rotation axis
    plotter.plot_solar_axis()

    def my_fline_color_func(field_line):
        norm = colors.LogNorm(vmin=1, vmax=1000)
        cmap = plt.get_cmap('viridis')
        return cmap(norm(np.abs(field_line.expansion_factor)))

    #    plotter.plot_field_lines(field_lines_sph)
    plotter.plot_field_lines(field_lines, color_func=my_fline_color_func)
    plotter.plotter.add_mesh(pv.Sphere(radius=1))
    plotter.show()
    # ===============================================================================================================
    """

    return 0

def make_aia_syn_plot(field_lines, seeds_gong, coord_carr, pfss_in, pfss_out, mag_map):
    """
    Create a AIA synoptic map and overplot trajectory/field lines over the aia map.
    """

    # Step 1: Create the synoptic AIA map for CR2243.
    # ================================================================================================
    aia_raw = fits.open('data/aia/AIA_synoptic_CR2243.fits')

    # The original header is not standard, errors happen when one wants to map directly.
    header = aia_raw[0].header
    data = aia_raw[0].data
    header['rsun_ref'] = sunpy.sun.constants.radius.to_value(u.m)

    # crete a new header
    shape_out = (1080, 3600)
    header_new = sunpy.map.make_fitswcs_header(shape_out,
                                               SkyCoord(-180.0, 0, unit=u.deg,
                                                        frame=frames.HeliographicCarrington,
                                                        obstime='2021-05-01 00:00:00'),
                                               scale=[360 / shape_out[1],
                                                      180 / shape_out[0]] * u.deg / u.pix,
                                               wavelength=193 * u.AA,  projection_code='CAR')
    out_wcs = WCS(header_new)
    out_map = sunpy.map.Map(data, header_new)

    plot_settings = sunpy.map.Map('data/aia/aia_synoptic_20210501.fits').plot_settings
    out_map.plot_settings = plot_settings

    lon = [c.lon for c in coord_carr]
    lat = [c.lat for c in coord_carr]
    r = [c.radius.to(u.m) for c in coord_carr]
    seeds = SkyCoord(lon, lat, r, frame=out_map.coordinate_frame)


    fig = plt.figure(figsize=(9, 4))
    ax = fig.add_subplot(2, 1, 1, projection=out_map)
    out_map.plot(axes=ax)
    lon, lat = ax.coords
    lon.set_coord_type('longitude')
    lon.coord_wrap = 180
    lon.set_format_unit(u.deg)
    lat.set_coord_type('latitude')
    lat.set_format_unit(u.deg)

    lon.set_axislabel('Carrington Longitude', minpad=0.8)
    lat.set_axislabel('Carrington Latitude', minpad=0.9)
    lon.set_ticks(spacing=90*u.deg, color='k')
    lat.set_ticks(spacing=30*u.deg, color='k')

    lon.set_ticks_visible(True)

    for i in range(len(coord_carr)):
        c = seeds[i]
        ax.plot_coord(c, color='orange', marker='o', linewidth=1, markersize=2)
        if (coord_carr[i].obstime.value - datetime.strptime('2021-05-01T00', '%Y-%m-%dT%H')) < timedelta(hours=1) \
                and (coord_carr[i].obstime.value - datetime.strptime('2021-05-01T00', '%Y-%m-%dT%H')) > timedelta(
            hours=0):
            ax.plot_coord(c, color='black', marker='D', linewidth=0, markersize=5, label='2021-05-01')
    ax.plot_coord(pfss_out.source_surface_pils[0])
    for fline in field_lines[::2]:
        if len(fline.coords) > 0:
            fline_aia = SkyCoord(fline.coords.lon, fline.coords.lat, fline.coords.radius,
                                 frame=out_map.coordinate_frame)
            ax.plot_coord(fline_aia, alpha=0.7, linewidth=0.5, color='green')

    m = pfss_in.map
    ax = fig.add_subplot(2, 1, 2, projection=m)
    mag_map.plot(axes=ax)
    for i in range(len(coord_carr)):
        c = seeds_gong[i]
        ax.plot_coord(c, color='orange', marker='o', linewidth=1, markersize=2)
        if (coord_carr[i].obstime.value - datetime.strptime('2021-05-01T00', '%Y-%m-%dT%H')) < timedelta(hours=1) \
                and (coord_carr[i].obstime.value - datetime.strptime('2021-05-01T00', '%Y-%m-%dT%H')) > timedelta(hours=0):
            ax.plot_coord(c, color='black', marker='D', linewidth=0, markersize=5, label='2021-05-01')
    ax.plot_coord(pfss_out.source_surface_pils[0])
    for fline in field_lines[::2]:
        if len(fline.coords) > 0:
            ax.plot_coord(fline.coords, alpha=0.7, linewidth=0.5, color='green')
#    plt.show()
    plt.savefig('res/adapt/overplot_aia.jpg', format='jpg', dpi=400)

    # ================================================================================================
    return 0




def main():

    # Step1: Download psp data and GONG data.
    # ===============================================================================================================================
    t_start = input('Please enter the start time of your psp trajectory: (Format: YYYY-mm-ddTHH:MM:SS)')
    t_end = input('Please enter the end time of your psp trajectory: (Format: YYYY-mm-ddTHH:MM:SS)')
    download_psp(t_start, t_end)

    # Read PSP data
    SC_coordinates, SC_coordinates_carr, sw_vel = read_psp_data('data/psp')

    time_gong = input('Please enter the time of the GONG ADAPT data you want to download: (Format: YYYY-mm-ddTHH:00:00, HH is even number)')
    index_gong = int(input("Please enter the index of the gong adapt map: (0 ~ 11)"))
    download_gong_adapt(time_gong)

    rss = float(input('Please enter the height of Source Surface (unit: R_sun, usually 1.5 ~ 3.0):'))
# Step2: Parker Spiral extrapolation, spacecraft position --> Source Surface       # m/s
    ss_coordinates, ss_coordinates_carr = Parker_Spiral_extrapolation(SC_coordinates,SC_coordinates_carr, sw_vel, rss, t_start, t_end)

# Step3: pfss extrapolation, Source Surface --> Solar Surface
    field_lines, seeds, coord_carr, pfss_in, pfss_out, mag_map = pfss_trace(ss_coordinates_carr, rss, time_gong, index_gong)
    Plots(field_lines, seeds, coord_carr, pfss_in, pfss_out, mag_map)
#    make_aia_syn_plot(field_lines, seeds, coord_carr, pfss_in, pfss_out, mag_map)

    return 0


if __name__ == "__main__":
    main()
























