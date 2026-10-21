#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:01:03 2026

@author: jimbradford
"""



import numpy as np
import pandas as pd
import os
import sys
import obspy
import matplotlib.pyplot as plt
from obspy.core import UTCDateTime as utc
from obspy.core import Stream, read
from obspy import read_inventory
from obspy.io.sac import SACTrace
from obspy.signal.trigger import recursive_sta_lta
import obspy.taup.taup_geo as taup
from obspy.taup import TauPyModel
from glob import glob
import warnings
import matplotlib.dates as dates
import pygmt
import subprocess
import math
import geopandas as gpd
from pyproj import CRS, Transformer

import shapely
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from math import radians, sin, cos, sqrt, atan2
import io
from shapely.ops import unary_union
from scipy.interpolate import interp1d



warnings.simplefilter('ignore', category = UserWarning)
warnings.simplefilter('ignore', category = FutureWarning)


os.chdir('.')

region = 'TANGO_South'



def closest_idx(lst, K):
     lst = np.asarray(lst)
     idx = (np.abs(lst - K)).argmin()
     return idx

def haversine(lat1, lon1, lat2, lon2):
    # Convert latitude and longitude from degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

    # Haversine formula
    d_lat = lat2 - lat1
    d_lon = lon2 - lon1
    a = sin(d_lat/2)**2 + cos(lat1) * cos(lat2) * sin(d_lon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    distance = 6371 * c  # Earth radius in kilometers (approx.)
    return distance

def Return_Line_Elements(joints):


    points_per_km = 10

    # distance between joint-points
    d = np.array([0])
    num_points = np.array([])
    for i in range(len(joints)-1):
        d = np.append(d, haversine(joints[i,1], joints[i,0],joints[i+1,1], joints[i+1,0]))
        num_points = np.append(num_points, np.round(d[-1] * points_per_km))

    joints_dist = np.cumsum(d)


    # lat-lon coords of line
    line = np.array([0,0])
    for i in range(len(joints)-1):
        l = np.linspace(joints[i], joints[i+1], int(num_points[i]))
        line = np.vstack((line, l))

    line = line[1:len(line)]


    # cummulative distance along line
    d = np.array([0])
    for i in range(len(line)-1):
        d = np.append(d, haversine(line[i,1], line[i,0],line[i+1,1], line[i+1,0]))

    line_dist = np.cumsum(d)

    return joints_dist, line, line_dist



stat_table = pd.read_csv('../DATA/SPREADSHEETS/{}_PassingStations.csv'.format(region))
stat_table['Code'] = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i]) for i in range(len(stat_table))]

contributing_stations = pd.read_csv('../DATA/MAPPING/Contributing_Stations_RE.csv')
stat_table = stat_table[stat_table.Code.isin(contributing_stations.Code)]


pierce_points = pd.read_csv('../DATA/MAPPING/PiercePoints.csv')
contributing_pierce = pd.read_csv('../DATA/MAPPING/Contributing_Pierce_RE.csv')
pierce_points = pierce_points.iloc[contributing_pierce['Index']].reset_index(drop = True)




gmt_region = [-73.5, -67.5, -37, -34]
west, east, south, north = gmt_region



model ='BlendedModel_CSEM2-Hicks2014_vp.nc'
vpvs_model = 'BlendedModel_Hk-Hicks2014_vpvs.nc'

vp_var_name = 'vp'
vpvs_var_name = 'vpvs'



fig_x = 15
fig = pygmt.Figure()    # initialize the main map


# for slice_depth in np.flip(np.arange(-120, 0, 10)):
fig = pygmt.Figure()    # initialize the main map
with pygmt.config(FONT = '14p', MAP_FRAME_TYPE = 'plain', FORMAT_GEO_MAP = 'ddd.xx'):

    prj = 'M{}c'.format(fig_x)            # set the map projection
    fig.basemap(region=gmt_region, projection=prj, frame=['xafg+lLongitude', 'yafg+lLatitude', 'wsne'])
   
    # overlay national borders, coastlines, and make the water blue
    fig.coast(borders = ["1/1.5p,black"], shorelines="1/0.5p", water = "skyblue", transparency = 10)

    tecto_plates = gpd.read_file('../DATA/MAPPING/PlateBoundaries_Nazca.shp')
    fig.plot(data=tecto_plates[tecto_plates.Type == 'Convergent'], region = gmt_region, pen = '1.5p,black', style = "f1c/0.15c+r+t", fill = 'black')

    slice_depth = -100


    grid_color = pygmt.makecpt(cmap = 'jet',   series = [0, 10], continuous = True)
    
    with xr.open_dataset('../DATA/MAPPING/nc_models/JackKnife_Comparison_ABCD__CCP_TANGO_South_G-5.0_FZB_CSEM2-Hicks2014-vp_Hick2014-Hk-vpvs.nc') as jackknife:
        jackknife_slice = jackknife.interp(depth = slice_depth, method = 'linear', kwargs={"fill_value": None})
        jackknife_slice.Squared_Error.values = np.clip(jackknife_slice.Squared_Error.values, 0.000001, 10)
        # fig.grdimage(grid=jackknife_slice.Squared_Error, cmap = True)
        fig.grdcontour(grid=jackknife_slice.Squared_Error, annotation = [0.001])


    # now plot all station data
    fig.plot(x = stat_table.stlo, y = stat_table.stla, style='i0.3c', fill = 'gold',  pen='0.4p,black')
    fig.plot(x = stat_table[stat_table.net == 'ZR'].stlo, y =  stat_table[stat_table.net == 'ZR'].stla, style='i0.3c', fill = 'lightred', pen='0.6p,black')
    fig.plot(x = stat_table[stat_table.net == '1X'].stlo, y =  stat_table[stat_table.net == '1X'].stla, style='i0.25c',fill = 'cyan', pen='black')
    fig.plot(x = stat_table[stat_table.net == 'XM'].stlo, y =  stat_table[stat_table.net == 'XM'].stla, style='i0.3c',fill = 'magenta', pen='0.6p,black')

    joints =  np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_Line.txt', delimiter = ',')
    fig.plot(x = joints[:,0], y = joints[:,1], pen = '1.2p,black,--')
    Numerals = "ABCDEF"
    for i in range(len(joints)):
        fig.text(x = joints[i,0], y = joints[i,1], text = Numerals[i],  font = '12p, Helvetica-Bold', fill = 'white', pen = '0.3p,black')

    fig.basemap(map_scale="jTR+o0.75c/0.25c+w100k+u")


    fig.shift_origin(xshift = f'{fig_x+0.5}c')







    fig.basemap(region=gmt_region, projection=prj, frame=['xafg+lLongitude', 'yafg+lLatitude', 'wsnE'])
    fig.coast(borders = ["1/1.5p,black"], shorelines="1/0.5p", water = "skyblue", transparency = 10)
    # overlay the grid image DEM over the basemap
    fig.plot(data='../DATA/MAPPING/Payenia_Volcanic_Field.shp', region = gmt_region, pen = '0.5p,black', fill = 'pink' )

    tecto_plates = gpd.read_file('../DATA/MAPPING/PlateBoundaries_Nazca.shp')
    fig.plot(data=tecto_plates[tecto_plates.Type == 'Convergent'], region = gmt_region, pen = '1.5p,black', style = "f1c/0.15c+r+t", fill = 'black')


    slice_depth = -50
    

    # plot peirce points with an approximation of the first fresnel zone radius at depth
    pierce_map = pierce_points[pierce_points['depth'] == -slice_depth].reset_index(drop = True)

    
    
    for i in range(len(pierce_map)):
        if pierce_map.station[i] in contributing_stations.Code.to_list():
            fig.plot(x = pierce_map.longitude[i], y = pierce_map.latitude[i], style = '+0.25c', pen = '0.5p,black')
            fig.plot(x = pierce_map.longitude[i], y = pierce_map.latitude[i], style = f'E{pierce_map.fresnel_radius[i]*2}k', pen = '0.05p,black')





    # now plot station data
    fig.plot(x = stat_table.stlo, y = stat_table.stla, style='i0.3c', fill = 'gold',  pen='0.4p,black')
    fig.plot(x = stat_table[stat_table.net == 'ZR'].stlo, y =  stat_table[stat_table.net == 'ZR'].stla, style='i0.3c', fill = 'lightred', pen='0.6p,black')
    fig.plot(x = stat_table[stat_table.net == '1X'].stlo, y =  stat_table[stat_table.net == '1X'].stla, style='i0.25c',fill = 'cyan', pen='black')
    fig.plot(x = stat_table[stat_table.net == 'XM'].stlo, y =  stat_table[stat_table.net == 'XM'].stla, style='i0.3c',fill = 'magenta', pen='0.6p,black')


    volcanoes = pd.read_csv('../DATA/MAPPING/GVP_Holocene_Volcanoes.csv')
    volcanoes_PE = pd.read_csv('../DATA/MAPPING/GVP_Pleistocene_Volcanoes.csv')
    fig.plot(x = volcanoes.Longitude, y = volcanoes.Latitude, style='kvolcano/0.4c', fill = 'red', pen='0.75p,black')
    fig.plot(x = volcanoes_PE.Longitude, y = volcanoes_PE.Latitude, style='kvolcano/0.4c', fill = 'orange', pen='0.75p,black')


    joints =  np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_Line.txt', delimiter = ',')
    fig.plot(x = joints[:,0], y = joints[:,1], pen = '1.2p,black,--')
    Numerals = "ABCDEF"
    for i in range(len(joints)):
        fig.text(x = joints[i,0], y = joints[i,1], text = Numerals[i],  font = '12p, Helvetica-Bold', fill = 'white', pen = '0.3p,black')


    fig.text(x = -72.3, y = -36.85, text = f'depth slice: {np.abs(slice_depth)} km (b.s.l)', font = '12p,Helvetica-Bold', pen = '1p,black', fill = 'white')
    fig.basemap(map_scale="jTR+o0.75c/0.25c+w100k+u")
    
    fig.shift_origin(xshift = f'-{fig_x+0.5}c', yshift = f'{-10}c')

    
    
    
    
    fig.basemap(region=gmt_region, projection=prj, frame=['xafg+lLongitude', 'yafg+lLatitude', 'wSne'])
    fig.coast(borders = ["1/1.5p,black"], shorelines="1/0.5p", water = "skyblue", transparency = 10)
    # overlay the grid image DEM over the basemap
    fig.plot(data='../DATA/MAPPING/Payenia_Volcanic_Field.shp', region = gmt_region, pen = '0.5p,black', fill = 'pink' )

    tecto_plates = gpd.read_file('../DATA/MAPPING/PlateBoundaries_Nazca.shp')
    fig.plot(data=tecto_plates[tecto_plates.Type == 'Convergent'], region = gmt_region, pen = '1.5p,black', style = "f1c/0.15c+r+t", fill = 'black')


    slice_depth = -100
    

    # plot peirce points with an approximation of the first fresnel zone radius at depth
    pierce_map = pierce_points[pierce_points['depth'] == -slice_depth].reset_index(drop = True)

    
    
    for i in range(len(pierce_map)):
        if pierce_map.station[i] in contributing_stations.Code.to_list():
            fig.plot(x = pierce_map.longitude[i], y = pierce_map.latitude[i], style = '+0.25c', pen = '0.5p,black')
            fig.plot(x = pierce_map.longitude[i], y = pierce_map.latitude[i], style = f'E{pierce_map.fresnel_radius[i]*2}k', pen = '0.05p,black')





    # now plot station data
    fig.plot(x = stat_table.stlo, y = stat_table.stla, style='i0.3c', fill = 'gold',  pen='0.4p,black')
    fig.plot(x = stat_table[stat_table.net == 'ZR'].stlo, y =  stat_table[stat_table.net == 'ZR'].stla, style='i0.3c', fill = 'lightred', pen='0.6p,black')
    fig.plot(x = stat_table[stat_table.net == '1X'].stlo, y =  stat_table[stat_table.net == '1X'].stla, style='i0.25c',fill = 'cyan', pen='black')
    fig.plot(x = stat_table[stat_table.net == 'XM'].stlo, y =  stat_table[stat_table.net == 'XM'].stla, style='i0.3c',fill = 'magenta', pen='0.6p,black')


    volcanoes = pd.read_csv('../DATA/MAPPING/GVP_Holocene_Volcanoes.csv')
    volcanoes_PE = pd.read_csv('../DATA/MAPPING/GVP_Pleistocene_Volcanoes.csv')
    fig.plot(x = volcanoes.Longitude, y = volcanoes.Latitude, style='kvolcano/0.4c', fill = 'red', pen='0.75p,black')
    fig.plot(x = volcanoes_PE.Longitude, y = volcanoes_PE.Latitude, style='kvolcano/0.4c', fill = 'orange', pen='0.75p,black')


    joints =  np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_Line.txt', delimiter = ',')
    fig.plot(x = joints[:,0], y = joints[:,1], pen = '1.2p,black,--')
    Numerals = "ABCDEF"
    for i in range(len(joints)):
        fig.text(x = joints[i,0], y = joints[i,1], text = Numerals[i],  font = '12p, Helvetica-Bold', fill = 'white', pen = '0.3p,black')


    fig.text(x = -72.3, y = -36.85, text = f'depth slice: {np.abs(slice_depth)} km (b.s.l)', font = '12p,Helvetica-Bold', pen = '1p,black', fill = 'white')
    fig.basemap(map_scale="jTR+o0.75c/0.25c+w100k+u")
    
    
    fig.shift_origin(xshift = f'{fig_x+0.5}c')

    
    
    
    
    fig.basemap(region=gmt_region, projection=prj, frame=['xafg+lLongitude', 'yafg+lLatitude', 'wSnE'])
    fig.coast(borders = ["1/1.5p,black"], shorelines="1/0.5p", water = "skyblue", transparency = 10)
    # overlay the grid image DEM over the basemap
    fig.plot(data='../DATA/MAPPING/Payenia_Volcanic_Field.shp', region = gmt_region, pen = '0.5p,black', fill = 'pink' )

    tecto_plates = gpd.read_file('../DATA/MAPPING/PlateBoundaries_Nazca.shp')
    fig.plot(data=tecto_plates[tecto_plates.Type == 'Convergent'], region = gmt_region, pen = '1.5p,black', style = "f1c/0.15c+r+t", fill = 'black')


    slice_depth = -150
    

    # plot peirce points with an approximation of the first fresnel zone radius at depth
    pierce_map = pierce_points[pierce_points['depth'] == -slice_depth].reset_index(drop = True)

    
    
    for i in range(len(pierce_map)):
        if pierce_map.station[i] in contributing_stations.Code.to_list():
            fig.plot(x = pierce_map.longitude[i], y = pierce_map.latitude[i], style = '+0.25c', pen = '0.5p,black')
            fig.plot(x = pierce_map.longitude[i], y = pierce_map.latitude[i], style = f'E{pierce_map.fresnel_radius[i]*2}k', pen = '0.05p,black')





    # now plot station data
    fig.plot(x = stat_table.stlo, y = stat_table.stla, style='i0.3c', fill = 'gold',  pen='0.4p,black')
    fig.plot(x = stat_table[stat_table.net == 'ZR'].stlo, y =  stat_table[stat_table.net == 'ZR'].stla, style='i0.3c', fill = 'lightred', pen='0.6p,black')
    fig.plot(x = stat_table[stat_table.net == '1X'].stlo, y =  stat_table[stat_table.net == '1X'].stla, style='i0.25c',fill = 'cyan', pen='black')
    fig.plot(x = stat_table[stat_table.net == 'XM'].stlo, y =  stat_table[stat_table.net == 'XM'].stla, style='i0.3c',fill = 'magenta', pen='0.6p,black')


    volcanoes = pd.read_csv('../DATA/MAPPING/GVP_Holocene_Volcanoes.csv')
    volcanoes_PE = pd.read_csv('../DATA/MAPPING/GVP_Pleistocene_Volcanoes.csv')
    fig.plot(x = volcanoes.Longitude, y = volcanoes.Latitude, style='kvolcano/0.4c', fill = 'red', pen='0.75p,black')
    fig.plot(x = volcanoes_PE.Longitude, y = volcanoes_PE.Latitude, style='kvolcano/0.4c', fill = 'orange', pen='0.75p,black')


    joints =  np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_Line.txt', delimiter = ',')
    fig.plot(x = joints[:,0], y = joints[:,1], pen = '1.2p,black,--')
    Numerals = "ABCDEF"
    for i in range(len(joints)):
        fig.text(x = joints[i,0], y = joints[i,1], text = Numerals[i],  font = '12p, Helvetica-Bold', fill = 'white', pen = '0.3p,black')


    fig.text(x = -72.3, y = -36.85, text = f'depth slice: {np.abs(slice_depth)} km (b.s.l)', font = '12p,Helvetica-Bold', pen = '1p,black', fill = 'white')
    fig.basemap(map_scale="jTR+o0.75c/0.25c+w100k+u")

spec = io.StringIO(
    """
N 5
S 0.20c i 0.3c cyan 0.4p,black 0.65c TANGO Node
S 0.20c i 0.3c magenta 0.6p,black 0.65c TANGO Broadband
S 0.20c i 0.3c lightred 0.6p,black 0.65c LaMa Broadband
S 0.20c i 0.3c gold 0.6p,black 0.65c Other Network

S 0.3c kvolcano 0.60c red 0.75p,black 0.65c Holocene Volcano
S 0.3c kvolcano 0.60c orange 0.75p,black 0.65c Pleistocene Volcano
S 0.20c f+l+t 0.45c/-1/0.15c black 1.5,black 0.65c Chile Trench
S 0.20c - 0.45c/-1/0.25c black 1.5,black 0.65c Country Border
S 0.20c + 0.3c black 0.6p,black 0.65c Pierce Point
S 0.20c c 0.3c white 0.6p,black 0.65c Fresnel Zone Radius


G 0.07c
G 0.07c

    """
    )


with pygmt.config(FONT_ANNOT_PRIMARY="12p"):
    fig.legend(spec = spec,  position=f'jBL+w24c+o-{fig_x+0.5}c/-2.5c')


fig.show()

fig.savefig('../FIGURES/SupplementaryFigure5.png', dpi = 600)


