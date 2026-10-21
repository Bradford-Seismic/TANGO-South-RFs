# -*- coding: utf-8 -*-
"""
Created on Wed Apr  1 20:07:41 2026

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
import shapely
import xarray as xr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from math import radians, sin, cos, sqrt, atan2
import io
from shapely.ops import unary_union
from scipy.spatial import KDTree
from obspy.geodetics import degrees2kilometers
from scipy.signal import hilbert

from scipy.interpolate import interp1d

from pyproj import CRS, Transformer


warnings.simplefilter('ignore', category = UserWarning)
warnings.simplefilter('ignore', category = FutureWarning)


os.chdir('.')



stations = pd.read_csv('../DATA/SPREADSHEETS/SA_Stations.csv')
moveouts = pd.read_csv('../DATA/SPREADSHEETS/Moveout_Summary.csv')
downloads = pd.read_csv('../DATA/SPREADSHEETS/Download_Summary.csv')
snr = pd.read_csv('../DATA/SPREADSHEETS/SNR_Summary.csv')
itd = pd.read_csv('../DATA/SPREADSHEETS/Iterdecon_Summary.csv', keep_default_na=False)

region = 'TANGO_South'

stat_table = pd.read_csv('../DATA/SPREADSHEETS/{}_PassingStations.csv'.format(region))
stat_table['Code'] = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i]) for i in range(len(stat_table))]

contributing_stations = pd.read_csv('../DATA/MAPPING/Contributing_Stations_RE.csv')

stat_table = stat_table[stat_table.Code.isin(contributing_stations.Code)].reset_index(drop = True)



joints =  np.loadtxt('../DATA/MAPPING/{}_CCP_Line.txt'.format(region), delimiter = ',')

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

    return joints_dist, np.unique(line, axis = 0), np.unique(line_dist)



gmt_region = [-73.5, -67.5, -37, -34]
west, east, south, north = gmt_region



grid = pygmt.datasets.load_earth_relief(resolution="15s", region=gmt_region)
shade = pygmt.grdgradient(grid=grid, azimuth="315/45", normalize="e1")


fig_x = 20


fig = pygmt.Figure()    # initialize the main map
with pygmt.config(FONT = '14p', MAP_FRAME_TYPE = 'plain', FORMAT_GEO_MAP = 'ddd.xx'):

    prj = 'M{}c'.format(fig_x)            # set the map projection
    fig.basemap(region=gmt_region, projection=prj, frame=['xafg+lLongitude', 'yafg+lLatitude', 'wSnE'])

    # overlay the grid image DEM over the basemap
    fig.plot(data='../DATA/MAPPING/Tromen_Volcanic_Plateau.shp', region = gmt_region, pen = '0.5p,black', fill = 'lightred' )
    fig.plot(data='../DATA/MAPPING/Payenia_Volcanic_Field.shp', region = gmt_region, pen = '0.5p,black', fill = 'pink' )
    fig.grdimage(grid=grid, shading = shade, cmap = '../DATA/MAPPING/natural_mod.cpt', projection = prj, transparency = 55)

    # overlay national borders, coastlines, and make the water blue
    fig.coast(borders = ["1/1.5p,black"], shorelines="1/0.5p", water = "skyblue", transparency = 10)

    # # Sankha Seismicitiy
    # mahanti = pd.read_csv('../DATA/MAPPING/catalog_southline_hypodd.csv')
    # mahanti['id'] = np.arange(0, len(mahanti), 1)
    # mahanti = mahanti[mahanti.lat_err < 10][mahanti.lon_err < 10][mahanti.dep_err < 10]
    # eq_data = mahanti[['longitude', 'latitude', 'depth', 'id']] 

    # eq_depth = pygmt.makecpt(cmap = 'thermal', series = [1, 100], continuous = True)
    # fig.plot(x = eq_data.longitude, y = eq_data.latitude, fill = eq_data.depth, cmap = True, pen = '0.25p,black', style = 'c0.2c')
    # fig.colorbar(cmap = True, position = 'JMR+w10c/0.25c+o2.5c/0c', frame = 'xa20f10+ldepth')



    tecto_plates = gpd.read_file('../DATA/MAPPING/PlateBoundaries_Nazca.shp')
    fig.plot(data=tecto_plates[tecto_plates.Type == 'Convergent'], region = gmt_region, pen = '1.5p,black', style = "f1c/0.15c+r+t", fill = 'black')


    fig.plot(data='../DATA/MAPPING/Central_Valley.shp', region = gmt_region, pen = '0.5p,black,--', )
    # fig.plot(data='../DATA/MAPPING/Principal_Cordillera.shp', region = gmt_region, pen = '0.5p,black,--', )
    fig.plot(data='../DATA/MAPPING/Calderas.shp', region = gmt_region, pen = '1.25p,black,--', style = "f.5c/0.2c+r")


    faults = gpd.read_file('../DATA/MAPPING/Foreland_Faults.shp')

    fig.plot(data=faults[faults.Type=='thrust'], region = gmt_region, pen = '1.25p,black,--', style = "f1c/0.2c+r+t", fill = 'black')
    fig.plot(data=faults[faults.name=='Las Loicas'][faults.dip == 'east'], region = gmt_region, pen = '1.25p,black,--', style = "f1c/0.4c+l", fill = 'black')
    fig.plot(data=faults[faults.name=='Las Loicas'][faults.dip == 'west'], region = gmt_region, pen = '1.25p,black,--', style = "f1c/0.4c+r", fill = 'black')



    # now plot station data
    fig.plot(x = stat_table.stlo, y = stat_table.stla, style='i0.3c', fill = 'gold',  pen='0.4p,black')
    fig.plot(x = stat_table[stat_table.net == 'ZR'].stlo, y =  stat_table[stat_table.net == 'ZR'].stla, style='i0.3c', fill = 'lightred', pen='0.6p,black')
    fig.plot(x = stat_table[stat_table.net == '1X'].stlo, y =  stat_table[stat_table.net == '1X'].stla, style='i0.25c',fill = 'cyan', pen='black')
    fig.plot(x = stat_table[stat_table.net == 'XM'].stlo, y =  stat_table[stat_table.net == 'XM'].stla, style='i0.3c',fill = 'magenta', pen='0.6p,black')


    volcanoes = pd.read_csv('../DATA/MAPPING/GVP_Holocene_Volcanoes.csv')
    volcanoes_PE = pd.read_csv('../DATA/MAPPING/GVP_Pleistocene_Volcanoes.csv')
    fig.plot(x = volcanoes.Longitude, y = volcanoes.Latitude, style='kvolcano/0.4c', fill = 'red', pen='0.75p,black')
    fig.plot(x = volcanoes_PE.Longitude, y = volcanoes_PE.Latitude, style='kvolcano/0.4c', fill = 'red', pen='0.75p,black')



    cities = pd.read_csv('../DATA/MAPPING/Cities.csv')
    drop_index = cities[(cities.City == 'San Bernardo' ) | (cities.City == 'La Florida') | (cities.City == 'Maipú') | (cities.City == 'Puente Alto')].index
    cities = cities.drop(drop_index)
    fig.plot(x = cities.Longitude, y = cities.Latitude, style = 'd0.4c', fill = 'darkorange', pen = '1p,black')



    joints =  np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_Line.txt', delimiter = ',')
    fig.plot(x = joints[:,0], y = joints[:,1], pen = '1.2p,black,--')
    Numerals = "ABCDEF"
    for i in range(len(joints)):
        fig.text(x = joints[i,0], y = joints[i,1], text = Numerals[i],  font = '12p, Helvetica-Bold', fill = 'white', pen = '0.3p,black')

    fig.basemap(map_scale="jTR+o0.75c/0.25c+w100k+u")

    

    

spec = io.StringIO(
    """
N 5
S 0.20c i 0.3c cyan 0.4p,black 0.65c TANGO Node
S 0.20c i 0.3c magenta 0.6p,black 0.65c TANGO Broadband
S 0.20c i 0.3c lightred 0.6p,black 0.65c LaMa Broadband
S 0.20c i 0.3c gold 0.6p,black 0.65c Other Network

S 0.20c d 0.4c darkorange 1p,black 0.65c Major City
S 0.3c kvolcano 0.60c red 0.75p,black 0.65c Quaternary Volcano
S 0.20c f+l+t 0.45c/-1/0.15c black 1.5,black 0.65c Chile Trench
S 0.20c f+l 0.45c/-1/0.25c black 1.5,black 0.65c Normal Fault
S 0.20c - 0.45c/-1/0.25c black 1.5,black 0.65c Country Border


G 0.07c
G 0.07c

    """
    )


with pygmt.config(FONT_ANNOT_PRIMARY="12p"):
    fig.legend(spec = spec,  position='jBL+w24c+o-1c/-2.5c')


with fig.inset(position="jBL+w8c/14c+o-4.5c/-1.6c", box = False):

    inset_region = [-78, -53, -56, -15]

    fig.coast(
       region= inset_region,
       projection="U19S/?",
       dcw="CL,AR,PE,BO+gwhite+p0.5p",
       area_thresh=1000,
       transparency = 0)

    fig.coast(
       region= inset_region,
       projection="U19S/?",
       dcw="CL,AR,PE,BO+gcoral+p0.5p",
       area_thresh=1000,
       transparency = 30)

    fig.plot(data=tecto_plates[tecto_plates.Type == 'Convergent'], region = inset_region, pen = '1.5p,black', style = "f1c/0.15c+r+t", fill = 'black')
    fig.plot(data='../DATA/MAPPING/Central_Andes.shp', region = inset_region, pen = '0.5p,black', fill = 'green', transparency = 50)
    fig.plot(data='../DATA/MAPPING/Flat_Slab_Region.shp', region = inset_region, pen = '0.5p,black', fill = 'yellow', transparency = 50)
    fig.plot(data='../DATA/MAPPING/Southern_Andes.shp', region = inset_region, pen = '0.5p,black', fill = 'blue', transparency = 50)
    fig.plot(data='../DATA/MAPPING/Andes_Thrust_Belt.shp', region = inset_region, pen = '1p,black,--', style = "f1c/0.15c+r+t", fill = 'black')

    fig.plot(x = volcanoes.Longitude, y = volcanoes.Latitude, style='kvolcano/0.2c', fill = 'red', pen='0.1p,black')
    fig.plot(x = joints[:,0], y = joints[:,1], pen = '1.2p,black,--')
    for i in range(len(joints)):
        fig.text(x = joints[i,0], y = joints[i,1], text = Numerals[i],  font = '8p,Helvetica-Bold', fill = 'white', pen = '0.3p,black')

    # plot the general region we are focused on
    fig.plot(x = [west, west, east, east, west], y = [north, south, south, north, north], pen = '1.25p,cyan')




fig.show()

fig.savefig('../FIGURES/Fig1_MapFig.png', dpi = 600)

