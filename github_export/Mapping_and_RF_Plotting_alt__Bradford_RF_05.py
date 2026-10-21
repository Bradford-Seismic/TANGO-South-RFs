# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 15:01:56 2025

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



warnings.simplefilter('ignore', category = UserWarning)
warnings.simplefilter('ignore', category = FutureWarning)


os.chdir('.')
model = TauPyModel(model="iasp91")



stations = pd.read_csv('../DATA/SPREADSHEETS/SA_Stations.csv')
moveouts = pd.read_csv('../DATA/SPREADSHEETS/Moveout_Summary.csv')
downloads = pd.read_csv('../DATA/SPREADSHEETS/Download_Summary.csv')
snr = pd.read_csv('../DATA/SPREADSHEETS/SNR_Summary.csv')
itd = pd.read_csv('../DATA/SPREADSHEETS/Iterdecon_Summary.csv', keep_default_na=False)

region = 'TANGO_South'

#%% Collect Passing data information, station location, event info, etc
### Then plot passing data stations and information


itd_pass = itd[itd.Region == region][(itd.Accept == 'Man QC Pass')]


unique_events = np.unique(itd_pass.Event.to_numpy())
unique_stations = np.unique(itd_pass.Code.to_numpy())



stlas = stlos = stels = codes = nets = stats = np.array([])
for n, i in enumerate(unique_stations):
    print("\r", end="")
    print("Reading Stations: {:.1%} ".format(n/(len(unique_stations))), end="")

    code = i.replace('_', '.')
    codes = np.append(codes, code)

    files = glob('../DATA/{}/DATA_CCP/{}.*.itr'.format(region, code))
    if len(files) == 0:
        continue
    st = read(files[0])

    nets = np.append(nets, st[0].stats.network)
    stats = np.append(stats, st[0].stats.station)

    stlas = np.append(stlas, st[0].stats.sac.stla)
    stlos = np.append(stlos, st[0].stats.sac.stlo)
    stels = np.append(stels, st[0].stats.sac.stel)


stat_table = pd.DataFrame({'net': nets, 'stat':stats, 'stla': stlas, 'stlo': stlos, 'stel': stels})

stat_table.to_csv('../DATA/SPREADSHEETS/{}_PassingStations.csv'.format(region))

evlas = evlos = evdps = mags = events = phases = gcarcs = bazs = sense = event_id = np.array([])

for n, i in enumerate(unique_events):
    print("\r", end="")
    print("Reading Events: {:.1%} ".format(n/(len(unique_events))), end="")

    flag = ''
    events = np.append(events, i)

    files = glob('../DATA/{}/DATA_CCP/*.{}.*.itr'.format(region, i))
    if len(files) == 0:
        continue

    st = read(files[0])

    evlas = np.append(evlas, st[0].stats.sac.evla)
    evlos = np.append(evlos, st[0].stats.sac.evlo)
    evdps = np.append(evdps, st[0].stats.sac.evdp)
    gcarcs = np.append(gcarcs, st[0].stats.sac.gcarc)
    mags = np.append(mags, st[0].stats.sac.mag)
    phases = np.append(phases, downloads.PhaseType[downloads.Event == i].to_numpy()[0])
    bazs = np.append(bazs, st[0].stats.sac.baz)

    event_id = np.append(event_id, i)


    #check if file list has XM stations
    if any('XM' in f for f in files):
        flag = flag + '/XM'

    if any('XN' in f for f in files) or any('1X' in f for f in files):
        flag = flag + '/XN'



    sense = np.append(sense, flag)

event_table = pd.DataFrame({'event': event_id, 'evla': evlas, 'evlo':evlos, 'gcarc': gcarcs, 'phase': phases, 'baz': bazs, 'sense': sense})
event_table.to_csv('../DATA/SPREADSHEETS/{}_PassingEvents.csv'.format(region))



#%% Plotting Functions

def closest(lst, K):

     lst = np.asarray(lst)
     idx = (np.abs(lst - K)).argmin()
     return lst[idx]

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

def orthogonal_projection(point, line_point1, line_point2):
    """
    Calculates the orthogonal projection of a point onto a line.

    Args:
        point (tuple): The point to project.
        line_point1 (tuple): A point on the line.
        line_point2 (tuple): Another point on the line.

    Returns:
        tuple: The orthogonal projection of the point onto the line.
    """

    p = np.array(point)
    a = np.array(line_point1)
    b = np.array(line_point2)

    # Vector representing the line
    v = b - a

    # Projection of p onto v
    projection = a + np.dot(p - a, v) / np.dot(v, v) * v

    distance = haversine(projection[1], projection[0], a[1], a[0])


    return distance

def calculate_bearing(lat1, lon1, lat2, lon2):
    # Convert decimal degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    dlon = lon2_rad - lon1_rad

    y = math.sin(dlon) * math.cos(lat2_rad)
    x = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon)

    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)

    # Normalize to [0, 360) degrees
    bearing_deg = (bearing_deg + 360) % 360

    return bearing_deg

def destination_point(lat1, lon1, distance, bearing, earth_radius=6371):
    """
    Calculate the destination point given a starting latitude and longitude,
    a distance, and a bearing.

    Parameters:
    - lat1 (float): Starting latitude in degrees.
    - lon1 (float): Starting longitude in degrees.
    - distance (float): Distance to travel in kilometers.
    - bearing (float): Bearing from the starting point in degrees.
    - earth_radius (float): Radius of the Earth in kilometers (default is 6371 km).

    Returns:
    - (float, float): Tuple containing the destination latitude and longitude in degrees.
    """
    # Convert latitude, longitude, and bearing from degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    bearing_rad = math.radians(bearing)

    # Calculate the new latitude
    lat2_rad = math.asin(math.sin(lat1_rad) * math.cos(distance / earth_radius) +
                         math.cos(lat1_rad) * math.sin(distance / earth_radius) * math.cos(bearing_rad))

    # Calculate the new longitude
    lon2_rad = lon1_rad + math.atan2(math.sin(bearing_rad) * math.sin(distance / earth_radius) * math.cos(lat1_rad),
                                     math.cos(distance / earth_radius) - math.sin(lat1_rad) * math.sin(lat2_rad))

    # Convert latitude and longitude from radians to degrees
    lat2 = math.degrees(lat2_rad)
    lon2 = math.degrees(lon2_rad)

    return lat2, lon2

def Plot_Window(figsize = [15,15]):
    mapsize = ('{}c'.format(figsize[0]), '{}c'.format(figsize[1]))
    toposize = ('{}c'.format(figsize[0]), '{}c'.format(figsize[1]/4))

    fig = pygmt.Figure()    # initialize the map
    with fig.subplot(nrows=1, ncols =1, figsize = mapsize):
        with fig.set_panel(panel = 0):
            with pygmt.config(FONT="11p"):
                fig.basemap(region=[0, 10, 0, 10], projection='X{}/-{}'.format(mapsize[0], mapsize[1]), frame=['WSne', 'xaf1', 'ya1f1'])

    fig.shift_origin(yshift = '+{}'.format(mapsize[1]))
    with fig.subplot(nrows = 1, ncols = 1, figsize=toposize):
        with fig.set_panel(panel = 0):
            with pygmt.config(FONT="11p"):
                fig.basemap(region=[0, 10, 0, 10], projection='X{}/{}'.format(toposize[0], toposize[1]), frame=['WNe', 'xaf1', 'ya1f1'])

    return fig

def select_cross(region, stlas, stlos, line_length = 200):


    volcanoes = pd.read_csv('../DATA/MAPPING/GVP_Holocene_Volcanoes.csv')

    fig, ax = plt.subplots(figsize = (14, 14))
    ax.scatter(stlos, stlas, s= 150, marker = 'v', color = 'cyan', edgecolor = 'k', linewidth = 1)
    ax.scatter(volcanoes.Longitude, volcanoes.Latitude, s= 200, marker = '^', color = 'red', edgecolor = 'k', linewidth = 1)
    ax.set_xlim([stlos.min()-0.1, stlos.max()+0.1])
    ax.set_ylim([stlas.min()-0.1, stlas.max()+0.1])
    joints = np.array(plt.ginput(99))
    plt.close('all')


    return joints

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

def shift_line_ns(coords: np.ndarray, km: float) -> np.ndarray:
    """
    Shift a polyline north or south by a set number of kilometers.

    Parameters
    ----------
    coords : np.ndarray
        Array of shape (N, 2) with columns [lon, lat].
    km : float
        Shift distance in kilometers. Positive = north, Negative = south.

    Returns
    -------
    np.ndarray
        Shifted coordinates (N, 2) in [lon, lat].
    """
    # Make sure coords is Nx2
    coords = np.asarray(coords)
    assert coords.shape[1] == 2, "Input must be Nx2 [lon, lat]"

    # Center point for choosing an appropriate local projection
    lon0, lat0 = np.mean(coords[:, 0]), np.mean(coords[:, 1])

    # Define a local azimuthal equidistant projection centered on the line
    crs_geodetic = CRS.from_epsg(4326)  # WGS84
    crs_local = CRS.from_proj4(f"+proj=aeqd +lat_0={lat0} +lon_0={lon0} +datum=WGS84")

    fwd = Transformer.from_crs(crs_geodetic, crs_local, always_xy=True)
    inv = Transformer.from_crs(crs_local, crs_geodetic, always_xy=True)

    # Project to local meters
    x, y = fwd.transform(coords[:, 0], coords[:, 1])

    # Shift in north/south = add to y
    y_shifted = y + km * 1000.0

    # Transform back
    lon_shift, lat_shift = inv.transform(x, y_shifted)
    return np.column_stack([lon_shift, lat_shift])

def Plot_Topo(fig, y_shift, fig_x, fig_y, joints, stat_table, xlim_range):

    joints_dist, line, line_dist = Return_Line_Elements(joints)



    fig.shift_origin(yshift = '+{}c'.format(y_shift))
    with pygmt.config(FONT = '12p', MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p"):
        fig.basemap(region = [0, joints_dist[-1], 0,5], projection = 'X{}c/{}c'.format(fig_x, fig_y), frame = ['WsNe', 'xa50f10', 'ya5f1+e+lElev. (km)'])


        topo_x = np.append(line_dist[0], line_dist)
        topo_x = np.append(topo_x, line_dist[-1])



        # on-line topography

        x = xr.DataArray(line[:,0], dims='Distance_Along_Trend')
        y = xr.DataArray(line[:,1], dims='Distance_Along_Trend')

        dem = grid.interp(lat = y, lon = x, method = 'linear')
        dem = dem.assign_coords({'Distance_Along_Trend': line_dist})

        topo = np.append(0, dem.values)
        topo = np.append(topo, 0)


        fig.plot(x = topo_x, y = topo/1000, fill = 'lightgray', pen = '0.5p,black')




        # project volcanoes, repeat for PE volcanoes
        try:
            volcanoes = pd.read_csv(
                '../DATA/MAPPING/GVP_Pleistocene_Volcanoes.csv')
            volc_data = pd.DataFrame({'longitude': volcanoes.Longitude.values, 'latitude': volcanoes.Latitude.values,
                                     'elevation': volcanoes['Elevation (m)'].values, 'volc_id': volcanoes['Volcano Number'].values})
            used_data = []
            for i in range(len(joints) - 1):
                track = pygmt.project(data=volc_data, center=joints[i], endpoint=joints[i+1], width=[
                                      -12, 12], length=[0, joints_dist[i+1] - joints_dist[i]], unit=True)
                track = track.rename(columns={
                                     0: 'longitude', 1: 'latitude', 2: 'elevation', 3: 'volc_id', 4: 'p', 5: 'q', 6: 'r', 7: 's'})

                if len(track) > 0:
                    try:
                        data_used_check = [track.volc_id[j]
                                           in used_data for j in range(len(track))]
                        data_used_check = [not (bool(data_used_check[j]))
                                           for j in range(len(data_used_check))]
                        track = track[data_used_check].reset_index(drop=True)
                        used_data = np.append(
                            used_data, np.unique(track.volc_id.values))

                        volc_elev_interp = grid.interp(lat=xr.DataArray(track.latitude, dims='Distance'), lon=xr.DataArray(
                            track.longitude, dims='Distance'), method='linear')

                        fig.plot(x=track.p + joints_dist[i], y=volc_elev_interp.values/1000,
                                 style='kvolcano/0.6c', fill='orange', pen='0.3,black')
                    except:
                        pass
        except:
            pass




        # project volcanoes
        try:
            volcanoes = pd.read_csv('../DATA/MAPPING/GVP_Holocene_Volcanoes.csv')
            volc_data = pd.DataFrame({'longitude': volcanoes.Longitude.values, 'latitude': volcanoes.Latitude.values,
                                     'elevation': volcanoes['Elevation (m)'].values, 'volc_id': volcanoes['Volcano Number'].values})
            used_data = []
            for i in range(len(joints) - 1):

                # note the little buffer of on the 'length' parameter, for some reason, some lines if the vertex is
                # right on the point, it won't project
                track = pygmt.project(data=volc_data, center=joints[i], endpoint=joints[i+1], width=[-12, 12], length=[
                                      0, joints_dist[i+1] - joints_dist[i]], unit=True)
                track = track.rename(columns={
                                     0: 'longitude', 1: 'latitude', 2: 'elevation', 3: 'volc_id', 4: 'p', 5: 'q', 6: 'r', 7: 's'})

                if len(track) > 0:
                    data_used_check = [track.volc_id[j]
                                       in used_data for j in range(len(track))]
                    data_used_check = [not (bool(data_used_check[j]))
                                       for j in range(len(data_used_check))]
                    track = track[data_used_check].reset_index(drop=True)
                    used_data = np.append(
                        used_data, np.unique(track.volc_id.values))

                    volc_elev_interp = grid.interp(lat=xr.DataArray(track.latitude, dims='Distance'), lon=xr.DataArray(
                        track.longitude, dims='Distance'), method='linear')

                    fig.plot(x=track.p + joints_dist[i], y=volc_elev_interp.values/1000,
                             style='kvolcano/0.6c', fill='red', pen='0.3,black')
        except:
            pass

        # project stations
        codes = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i])
                 for i in range(len(stat_table))]
        stat_table['code'] = codes

        stat_data = pd.DataFrame({'longitude': stat_table.stlo.values, 'latitude': stat_table.stla.values,
                                 'elevation': stat_table.stel.values, 'code': stat_table.code.values})
        used_data = []
        for i in range(len(joints) - 1):
            track = pygmt.project(data=stat_data, center=joints[i], endpoint=joints[i+1], width=[
                                  -12, 12], length=[0, joints_dist[i+1] - joints_dist[i]], unit=True)
            track = track.rename(columns={
                                 0: 'longitude', 1: 'latitude', 2: 'elevation', 3: 'p', 4: 'q', 5: 'r', 6: 's', 7: 'code'})

            if len(track) > 0:
                data_used_check = [track.code[j]
                                   in used_data for j in range(len(track))]
                data_used_check = [not (bool(data_used_check[j]))
                                   for j in range(len(data_used_check))]
                track = track[data_used_check].reset_index(drop=True)
                used_data = np.append(used_data, np.unique(track.code.values))

                stat_elev_interp = grid.interp(lat=xr.DataArray(track.latitude, dims='Distance'), lon=xr.DataArray(
                    track.longitude, dims='Distance'), method='linear')

                fig.plot(x=track.p + joints_dist[i], y=stat_elev_interp.values /
                         1000, style='i0.3c', fill='gold', pen='0.3,black')

                try:
                    fig.plot(x=track.p[track.code.str.contains('ZR')] + joints_dist[i], y=stat_elev_interp.values[track.code.str.contains(
                        'ZR')]/1000, style='i0.3c', fill='lightred', pen='0.3,black')
                except:
                    pass

                try:
                    fig.plot(x=track.p[track.code.str.contains('1X')] + joints_dist[i],
                             y=stat_elev_interp.values[track.code.str.contains('1X')]/1000, style='i0.3c', fill='cyan', pen='0.3,black')
                except:
                    pass
                try:
                    fig.plot(x=track.p[track.code.str.contains('XM')] + joints_dist[i], y=stat_elev_interp.values[track.code.str.contains(
                        'XM')]/1000, style='i0.3c', fill='magenta', pen='0.3,black')
                except:
                    pass



        # Plot joints
        Numerals = 'ABCDEFGHIJK'
        fig.text(x = joints_dist, y = np.ones(len(joints_dist)) * 4.9, text = list(Numerals[0:len(joints_dist)]), font = '14.p, Helvetica-Bold', fill = 'white', pen = '0.3p,black', no_clip = True)

    return fig



#%% Map Fig

save_figs = 0
stat_fig_name = 'F1_Stations+Tectonics_Region-{}_ABCD.png'.format(region)



###### Read station objects and show only stations contributing to line

stat_table = pd.read_csv('../DATA/SPREADSHEETS/{}_PassingStations.csv'.format(region))
codes = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i]) for i in range(len(stat_table))]
stat_table['Code'] = codes


event_table = pd.read_csv('../DATA/SPREADSHEETS/{}_PassingEvents.csv'.format(region))







################# Define Region
# gmt_region = [stat_table.stlo.min(), stat_table.stlo.max(), stat_table.stla.min(), stat_table.stla.max()]
# gmt_region = [-74, -67.5, -39, -32]
gmt_region = [-73.5, -67.5, -37, -34]

west, east, south, north = gmt_region



contributing_data = 1
# filter data using the list of the line-contributing rays
if contributing_data:

    # sub-lists for stations and event contributing rays to joint-line
    contributing_stations = pd.read_csv('../DATA/SPREADSHEETS/Contributing_Stations_ABCD.csv')
    contributing_events = pd.read_csv('../DATA/SPREADSHEETS/Contributing_Events_ABCD.csv')


    stat_table = stat_table[stat_table.Code.isin(contributing_stations.Code)]
    event_table = event_table[event_table.event.isin(contributing_events.Events)]

    stat_table.reset_index(drop = True, inplace = True)
    event_table.reset_index(drop = True, inplace = True)


############ Read other map objects

cities = pd.read_csv('../DATA/MAPPING/Cities.csv')
drop_index = cities[(cities.City == 'San Bernardo' ) | (cities.City == 'La Florida') | (cities.City == 'Maipú') | (cities.City == 'Puente Alto')].index
cities = cities.drop(drop_index)

grid = pygmt.datasets.load_earth_relief(resolution="15s", region=gmt_region)
shade = pygmt.grdgradient(grid=grid, azimuth="315/45", normalize="e1")


volcanoes = pd.read_csv('../DATA/MAPPING/GVP_Holocene_Volcanoes.csv')
volcanoes_PE = pd.read_csv('../DATA/MAPPING/GVP_Pleistocene_Volcanoes.csv')

tecto_plates = gpd.read_file('../DATA/MAPPING/PlateBoundaries_Nazca.shp')


########### Construct Station Map


fig_x = 20


fig = pygmt.Figure()    # initialize the main map

with pygmt.config(FONT = '14p', MAP_FRAME_TYPE = 'plain', FORMAT_GEO_MAP = 'ddd.xx'):

     prj = 'M{}c'.format(fig_x)            # set the map projection


     fig.basemap(region=gmt_region, projection=prj, frame=['xafg+lLongitude', 'yafg+lLatitude', 'wSnE'])

     # overlay the grid image DEM over the basemap
     fig.grdimage(grid=grid, shading = shade, cmap = '../DATA/MAPPING/natural_mod.cpt', projection = prj, transparency = 55)


     # overlay national borders, coastlines, and make the water blue
     fig.coast(borders = ["1/1.5p,black"], shorelines="1/0.5p", water = "skyblue", transparency = 10)

     fig.plot(data=tecto_plates[tecto_plates.Type == 'Convergent'], region = gmt_region, pen = '1.5p,black', style = "f1c/0.15c+r+t", fill = 'black')


     fig.plot(data='../DATA/MAPPING/Central_Valley.shp', region = gmt_region, pen = '0.5p,black,--', )
     fig.plot(data='../DATA/MAPPING/Principal_Cordillera.shp', region = gmt_region, pen = '0.5p,black,--', )
     fig.plot(data='../DATA/MAPPING/Payenia.shp', region = gmt_region, pen = '0.5p,black,--', )

    # now plot station data
     fig.plot(x = stat_table.stlo, y = stat_table.stla, style='i0.3c', fill = 'gold',  pen='0.4p,black')



     try:
         fig.plot(x = stat_table[stat_table.net == 'ZR'].stlo, y =  stat_table[stat_table.net == 'ZR'].stla, style='i0.3c', fill = 'lightred', pen='0.6p,black')
     except:
         pass

     try:
         fig.plot(x = stat_table[stat_table.net == '1X'].stlo, y =  stat_table[stat_table.net == '1X'].stla, style='i0.25c',fill = 'cyan', pen='black')
     except:
         pass

     try:
         fig.plot(x = stat_table[stat_table.net == 'XM'].stlo, y =  stat_table[stat_table.net == 'XM'].stla, style='i0.3c',fill = 'magenta', pen='0.6p,black')
     except:
         pass


     fig.plot(x = volcanoes.Longitude, y = volcanoes.Latitude, style='kvolcano/0.4c', fill = 'red', pen='0.75p,black')
     fig.plot(x = volcanoes_PE.Longitude, y = volcanoes_PE.Latitude, style='kvolcano/0.4c', fill = 'orange', pen='0.75p,black')



     fig.plot(x = cities.Longitude, y = cities.Latitude, style = 'd0.4c', fill = 'darkorange', pen = '1p,black')


     joints =  np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_Line.txt', delimiter = ',')
     fig.plot(x = joints[:,0], y = joints[:,1], pen = '1.2p,black,--')
     Numerals = "ABCDEF"
     for i in range(len(joints)):
         fig.text(x = joints[i,0], y = joints[i,1], text = Numerals[i],  font = '12p, Helvetica-Bold', fill = 'white', pen = '0.3p,black')

     fig.basemap(map_scale="jTR+o0.75c/0.25c+w100k+u")








spec = io.StringIO(
    """
N 4
S 0.20c i 0.3c cyan 0.4p,black 0.65c TANGO Node
S 0.20c i 0.3c magenta 0.6p,black 0.65c TANGO Broadband
S 0.20c i 0.3c lightred 0.6p,black 0.65c LaMa Broadband
S 0.20c i 0.3c gold 0.6p,black 0.65c Other Network

S 0.20c d 0.4c darkorange 1p,black 0.65c Major City
S 0.3c kvolcano 0.60c red 0.75p,black 0.65c Holocene Volcano
S 0.3c kvolcano 0.60c orange 0.75p,black 0.65c Pleistocene Volcano



G 0.07c
G 0.07c

    """
    )


with pygmt.config(FONT_ANNOT_PRIMARY="12p"):
    fig.legend(spec = spec,  position='jBL+w20c+o0c/-2.5c')


with fig.inset(position="jBL+w8c/12c+o-5c/-0.5c", box = False):

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


# Plot_Topo(fig, y_shift = 'h', fig_x = fig_x, fig_y = 1.5, joints = joints, stat_table = stat_table, xlim_range = [])


fig.show(verbose = 'i')



if save_figs:
    fig.savefig('../FIGURES/{}'.format(stat_fig_name), dpi = 300)






#%% EQ Fig

### EQ + Rose diagram Figure



save_figs = 1
EQ_fig_name = 'F2_EQ+Rose_Region-{}.png'.format(region)


# produce a gmt map of the accepted events with respect to the phase type
fig = pygmt.Figure()
prj = "A-71.2/-20/100/15c"


with pygmt.config(FONT_ANNOT_PRIMARY="12p"):

    fig.coast(projection=prj, region = 'g', frame = 'g60', land="gray", shorelines = '0.5p, black', water = 'white')

    fig.plot(x = event_table.evlo[event_table.phase == 'P'], y = event_table.evla[event_table.phase == 'P'], style='c0.23c',fill = 'lightred', pen='0.6p,black', label = 'P phase')


    # plot the general region we are focused on
    fig.plot(x = [west, west, east, east, west], y = [north, south, south, north, north], pen = '1p,blue')
    fig.plot(x = event_table.evlo[event_table.sense == '/XM'], y = event_table.evla[event_table.sense == '/XM'], style='c0.3c', pen='0.5p,black', fill = 'magenta',label = 'TANGO-BB')
    fig.plot(x = event_table.evlo[event_table.sense == '/XN'], y = event_table.evla[event_table.sense == '/XN'], style='c0.3c', pen='0.5p,black', fill = 'cyan',label = 'TANGO-Node')
    fig.plot(x = event_table.evlo[event_table.sense == '/XM/XN'], y = event_table.evla[event_table.sense == '/XM/XN'], style='c0.3c', pen='0.5p,black',fill="p9+fcyan+bmagenta",label = 'TANGO-BB & Node')



    fig.legend( position='jTR+w5.5c+o0c/-.5c',  box='+gwhite+p1p')



# Back-Azimuth Distribution
fig.shift_origin(xshift = '-5c', yshift = '10c')
with fig.subplot(nrows=1, ncols =1, figsize = ('6c', '6c')):
    with fig.set_panel(panel = 0):

        baz_hits = []
        baz_bin = np.arange(0, 370, 10)
        for i in range(len(baz_bin-1)):
            hit_count = 0
            for j in event_table.baz[event_table.sense.str.contains('/XM', na = False)].values:
                if (j > baz_bin[i]) and (j < baz_bin[i+1]):
                    hit_count += 1

            baz_hits.append(hit_count)

        fig.rose(
            # use columns of the sample dataset as input for the length and azimuth
            # parameters
            length = baz_hits,
            azimuth = baz_bin,
            # specify the "region" of interest in the (r,azimuth) space
            # [r0, r1, az0, az1], here, r0 is 0 and r1 is 1, for azimuth, az0 is 0 and
            # az1 is 360 which means we plot a full circle between 0 and 360 degrees
            region=[0, np.max(baz_hits), 0, 360],
            # set the diameter of the rose diagram to 7.5 cm
            diameter="4c",
            # define the sector width in degrees, we append +r here to draw a rose
            # diagram instead of a sector diagram
            sector="10",
            # normalize bin counts by the largest value so all bin counts range from
            # 0 to 1

            # use red3 as color fill for the sectors
            fill="magenta",
            # define the frame with ticks and gridlines every 0.2
            # length unit in radial direction and every 30 degrees
            # in azimuthal direction, set background color to
            # lightgray
            labels = "W,E,S,N",
            frame=["x1g1", "y30g30", '+gwhite'],
            # use a pen size of 1p to draw the outlines
            pen="1p",
            no_scale = True

        )

        fig.text(position = 'BL', text = 'n = {} Events'.format(np.sum(baz_hits)), offset = '0.1c/0.1c',font = '10p,Helvetica,black', fill = 'white', pen = '0.5p,black')

fig.shift_origin(yshift = '-5.5c')
with fig.subplot(nrows=1, ncols =1, figsize = ('4c', '4c')):
    with fig.set_panel(panel = 0):


        baz_hits = []
        baz_bin = np.arange(0, 370, 10)
        for i in range(len(baz_bin-1)):
            hit_count = 0
            for j in event_table.baz[event_table.sense.str.contains('/XN', na = False)].values:
                if (j > baz_bin[i]) and (j < baz_bin[i+1]):
                    hit_count += 1

            baz_hits.append(hit_count)

        fig.rose(
            # use columns of the sample dataset as input for the length and azimuth
            # parameters
            length = baz_hits,
            azimuth = baz_bin,
            # specify the "region" of interest in the (r,azimuth) space
            # [r0, r1, az0, az1], here, r0 is 0 and r1 is 1, for azimuth, az0 is 0 and
            # az1 is 360 which means we plot a full circle between 0 and 360 degrees
            region=[0, np.max(baz_hits), 0, 360],
            # set the diameter of the rose diagram to 7.5 cm
            diameter="4c",
            # define the sector width in degrees, we append +r here to draw a rose
            # diagram instead of a sector diagram
            sector="10",
            # normalize bin counts by the largest value so all bin counts range from
            # 0 to 1

            # use red3 as color fill for the sectors
            fill="cyan",
            # define the frame with ticks and gridlines every 0.2
            # length unit in radial direction and every 30 degrees
            # in azimuthal direction, set background color to
            # lightgray
            labels = "W,E,S,N",
            frame=["x1g1", "y30g30", '+gwhite'],
            # use a pen size of 1p to draw the outlines
            pen="1p",
            no_scale = True

        )

    fig.text(position = 'BL', text = 'n = {} Events'.format(np.sum(baz_hits)), offset = '0.1c/0.1c',font = '10p,Helvetica,black', fill = 'white', pen = '0.5p,black')



fig.show()


if save_figs:
    fig.savefig('../FIGURES/{}'.format(EQ_fig_name), dpi = 300)



# plt.hist(bazs, bins = 36)


#%% Cross Sections


save_figs = 0


from scipy.spatial import KDTree
from obspy.geodetics import degrees2kilometers



## Plotting Params
G = 5.0
radius = 12      # radius in km for line search
data_scale = 10     # use 15 for G-2.5, 10 for G-5.0
astack_radius = 4 # km


top = 10
base = -0.2

km_per_cm = 10

fig_x = 20
fig_y = 5


use_line = 1
if use_line:
    joints =  np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_Line.txt', delimiter = ',')
else:
    joints = select_cross(gmt_region, stat_table.stla.to_numpy(), stat_table.stlo.to_numpy())

joints_dist, line, line_dist = Return_Line_Elements(joints)

################# Define Region
gmt_region = [-74, -67.5, -39, -32]
west, east, south, north = gmt_region

grid = pygmt.datasets.load_earth_relief(resolution="15s", region=gmt_region)


## Pull QC passing data
itd_use = itd[itd.G == G][(itd.Accept == 'Man QC Pass')][itd.Region == region]

## collect station location
stlas = [stations.Latitude[stations.Code == code.replace('_', '-')].values[0] for code in itd_use.Code]
stlos = [stations.Longitude[stations.Code == code.replace('_', '-')].values[0] for code in itd_use.Code]

stat_coords = np.c_[stlos, stlas]

itd_use['stlo'] = stlos
itd_use['stla'] = stlas


## search for data in range of line
line_tree = KDTree(line)

data_in_range = line_tree.query_ball_point(x = stat_coords, r = radius / degrees2kilometers(1))
data_in_range =[bool(data_in_range[j]) for j in range(len(data_in_range))]


## Read file data and time-stack for average station RF
files = np.unique(np.array(itd_use.File)[data_in_range])

nets, stlas, stlos, stels, tr_id = [],[],[],[],[]    # resetting stlas and stlos list to queried data
st = Stream()
for f in files:
    st += read('../DATA/{}/DATA_CCP/{}'.format(region, f))

st.stack(group_by='{network}.{station}', npts_tol=2)    # T-stack by station for average station RF
for i, tr in enumerate(st):
    nets.append(tr.stats.network)
    stlas.append(stations.Latitude[stations.Code == '{}-{}'.format(tr.stats.network, tr.stats.station)].values[0])
    stlos.append(stations.Longitude[stations.Code == '{}-{}'.format(tr.stats.network, tr.stats.station)].values[0])
    stels.append(grid.sel(lat = stlas[-1], lon = stlos[-1], method = 'nearest').values.item())
    tr_id.append(i)



## Plot selected data

fig = pygmt.Figure()
with pygmt.config(FONT = '14p', MAP_FRAME_TYPE = 'plain', FORMAT_GEO_MAP = 'ddd.xx'):
     prj = 'M10c'

     fig.basemap(region = gmt_region, projection = prj, frame = 'agf')
     fig.plot(x = stlos, y = stlas, style = 'i0.3c', pen = '0.5p,black', fill = 'cyan')


     fig.plot(x = joints[:,0], y = joints[:,1], pen = '1p,black,--')
     fig.text(x = joints[:,0], y = joints[:,1], text = list(Numerals[0:len(joints)]))

fig.show()


select_data = pd.DataFrame({'longitude': np.squeeze(stlos), 'latitude': np.squeeze(stlas), 'stel': np.squeeze(stels), 'tr_id': np.array(tr_id)})





## Project station data onto line and plot with radius-stack
fig = pygmt.Figure()
used_data = np.array([])
with pygmt.config(FONT = '14p', MAP_FRAME_TYPE = 'plain', FORMAT_GEO_MAP = 'ddd.xx'):
     prj = 'X{}c/-{}c'.format(fig_x, fig_y)

     fig.basemap(region = [0, joints_dist[-1], base, top], projection = prj, frame = ['WSne', 'xa20f10g20+lProfile Distance (km)', 'ya2f1+lTime (s)'])

     for i in range(len(joints) - 1):
        track = pygmt.project(data = select_data,center=joints[i],endpoint=joints[i+1],width=[-radius-1, radius+1], length = [0, joints_dist[i+1]], unit=True)
        track = track.rename(columns={0:'stlo',1:'stla',2:'stel',3:'tr_id',4:'p',5:'q',6:'r',7:'s'})

        ## check for re-used data from projection, remove those data
        data_used_check = [track.tr_id[j] in used_data for j in range(len(track))]
        data_used_check = [not(bool(data_used_check[j])) for j in range(len(data_used_check))]
        track = track[data_used_check].reset_index(drop = True)
        used_data = np.append(used_data, np.unique(track.tr_id.values))


        for j in range(len(track)):

            station_dist = track.p[j] + joints_dist[i]

            # collect data within astack_ range of each center station
            center_tree = KDTree(np.c_[track.stlo[j], track.stla[j]])
            in_range = center_tree.query_ball_point(x = np.c_[select_data.longitude, select_data.latitude], r = astack_radius / degrees2kilometers(1))
            in_range =[bool(in_range[k]) for k in range(len(in_range))]
            in_range = select_data[in_range].tr_id.values

            st_stack = Stream()
            for trace in in_range:
                st_stack += st[trace].copy()

            st_stack.stack(npts_tol = 2)

            tr = st_stack[0]
            data = tr.data
            time = tr.times() - 10

            time_trim = time[(time >= base) & (time <= top)]
            data_trim = data[(time >= base) & (time <= top)] * data_scale + station_dist

            pos = data_trim.copy()
            neg = data_trim.copy()

            pos[0] = station_dist
            pos[pos < station_dist] = station_dist
            pos[-1] = station_dist

            neg[0] = station_dist
            neg[neg > station_dist] = station_dist
            neg[-1] = station_dist

            fig.plot(x = pos, y = time_trim,pen = '0.1p,black', no_clip = True, fill = 'red', transparency = 20)
            fig.plot(x = neg, y = time_trim,pen = '0.1p,black', no_clip = True, fill = 'blue', transparency = 20)


Plot_Topo(fig, y_shift = fig_y, fig_x = fig_x, fig_y = 1, joints = joints, stat_table = stat_table, xlim_range = [])

fig.show()



if save_figs:
    fig.savefig('../FIGURES/RFT_{}_G-{}_AStack.png'.format(region, G), dpi = 300)









