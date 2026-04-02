#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 19:34:39 2026

@author: bradford
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


# active_dir = 'C:/Users/7418888/Documents/TANGO/TANGO_South/SCRIPTING'
active_dir = '/tango/bradford/TANGO_South/SCRIPTING'



os.chdir(active_dir)
model = TauPyModel(model="iasp91")



stations = pd.read_csv('../DATA/SPREADSHEETS/SA_Stations.csv')
moveouts = pd.read_csv('../DATA/SPREADSHEETS/Moveout_Summary.csv')
downloads = pd.read_csv('../DATA/SPREADSHEETS/Download_Summary.csv')
snr = pd.read_csv('../DATA/SPREADSHEETS/SNR_Summary.csv')
itd = pd.read_csv('../DATA/SPREADSHEETS/Iterdecon_Summary.csv', keep_default_na=False)

region = 'TANGO_South'

stat_table = pd.read_csv('../DATA/SPREADSHEETS/{}_PassingStations.csv'.format(region))
stat_table['Code'] = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i]) for i in range(len(stat_table))]

contributing_stations = pd.read_csv('../DATA/SPREADSHEETS/Contributing_Stations.csv')

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

def moveout_correction(tr, vp_bulk = 6.4, p_ref = 0.06):

    p = tr.stats.sac.user4
    tr_time = tr.times() - 10
    tr_data = tr.data

    modifier = (np.sqrt(1 - (p*vp_bulk)**2)) / (np.sqrt(1 - (p_ref*vp_bulk)**2))
    t_corrected = tr_time * modifier


    tr_interp = interp1d(t_corrected, tr_data, kind = 'linear', bounds_error = False, fill_value = 0)
    tr.data = tr_interp(tr_time)

    return tr

def Phase_Weighted_Stack(st, v=2, sm=False, sl=15):

    # compute the signal coherency for Phase Weighted Stacking
    # return the PWS of a set of traces
    # https://geophydog.cool/post/signal_pws_stack/

    m = len(st)
    n = st[0].stats.npts
    dt = st[0].stats.delta
    t = np.arange(n) * dt
    c = np.zeros(n, dtype=complex)
    for i, tr in enumerate(st):
        h = hilbert(tr.data)
        c += h/abs(h)
    c = abs(c/m)
    if sm:
        operator = np.ones(sl) / sl
        c = np.convolve(c, operator, 'same')

    stc = st.copy()
    stc.stack(npts_tol = 2)
    tr = stc[0]
    tr.data = tr.data*c**v
    return stc

def tr_normalize(tr):
    ##  normalize to p-wave amplitude
     extrema_limit = tr.data.max() * 0.10
     # cycle through the data array to find all major extrema
     extrema=[]
     extrema_i = []

     # Note, maintain k as iterable variable
     k = 1
     for dat in tr.data[1:len(tr.data)-2]:
         left = tr.data[k - 1]
         right = tr.data[k + 1]
         if (dat > left) and (dat > right):
                 if dat > extrema_limit:
                     extrema.append(dat)
                     extrema_i.append(k)
         elif (dat < left) and (dat < right):
             if dat < -extrema_limit:
                 extrema.append(dat)
                 extrema_i.append(k)
         k += 1

     p_amp = extrema[0]
     dat = tr.data.copy() / p_amp
     tr.data = dat
     return tr

def plot_topo(fig):
    fig.basemap(region = [0, joints_dist[-1], 0,7], projection = 'X{}c/{}c'.format(fig_x, 1.5), frame = ['WsNe', 'xa100f10', 'ya5f1+e+lElev. (km)'])

    topo_x = np.append(line_dist[0], line_dist)
    topo_x = np.append(topo_x, line_dist[-1])


    # on-line topography

    dem = grid.interp(lat = y, lon = x, method = 'linear')
    dem = dem.assign_coords({'Distance_Along_Trend': line_dist})

    topo = np.append(0, dem.values)
    topo = np.append(topo, 0)


    fig.plot(x = topo_x, y = topo/1000, fill = 'lightgray', pen = '0.5p,black')





    # project tectonic zones
    try:
        int_dists = []
        fig.plot(x = [30, 30], y = [4, 5.5], pen = '1p,black')
        fig.plot(x = [joints_dist[-1], joints_dist[-1]], y = [4, 5.5], pen = '1p,black')

        used_zones = []
        for j in range(len(joints)-1):
            zone = gpd.read_file('../DATA/MAPPING/Central_Valley.shp')
            line_gpd = gpd.GeoSeries(data = shapely.geometry.LineString([joints[j], joints[j+1]]), crs = zone.crs)

            tecto_zones = ['../DATA/MAPPING/Central_Valley.shp',
                           '../DATA/MAPPING/Principal_Cordillera.shp',
                           ]

            for zone_name in tecto_zones:
                try:
                    if zone_name in used_zones:
                        continue

                    else:
                        zone = gpd.read_file(zone_name)

                        poly_int = zone.union_all().intersection(line_gpd)
                        poly_int = poly_int.get_coordinates().to_numpy()[0,:]

                        int_dist = haversine(joints[j,1], joints[j,0], poly_int[1], poly_int[0]) + joints_dist[j]
                        int_elev = dem.values[closest_idx(dem['Distance_Along_Trend'], int_dist)]
                        int_dists.append(int_dist)

                        fig.plot(x = [int_dist, int_dist], y = [4, 5.5], pen = '1p,black')
                        fig.plot(x = [int_dist, int_dist], y = [4, 5.5], pen = '1p,black')

                        used_zones.append(zone_name)


                except:
                      pass
        fig.plot(x = [30, joints_dist[-1]], y = [5.5, 5.5], pen = '1p,black')

    except:
        pass

    # Project Malargue FTB
    mftb = gpd.read_file('../DATA/MAPPING/Foreland_Faults.shp')
    mftb = mftb[mftb.name == 'Malargue'].union_all()
    for j in range(len(joints)-1):
        try:
            line_gpd = gpd.GeoSeries(data = shapely.geometry.LineString([joints[j], joints[j+1]]), crs = zone.crs)
            poly_int = mftb.intersection(line_gpd)
            poly_int = poly_int.get_coordinates().to_numpy()[0,:]
            int_dist = haversine(joints[j,1], joints[j,0], poly_int[1], poly_int[0]) + joints_dist[j]
            fig.plot(x = [int_dist, int_dist], y = [4, 5.5], pen = '1p,black')
            fig.plot(x = [int_dist, int_dist], y = [4, 5.5], pen = '1p,black')
        except:
            pass

    # Project San Rafael thrust
    mftb = gpd.read_file('../DATA/MAPPING/Foreland_Faults.shp')
    mftb = mftb[mftb.name == 'Las Malvinas'].union_all()
    for j in range(len(joints)-1):
        line_gpd = gpd.GeoSeries(data = shapely.geometry.LineString([joints[j], joints[j+1]]), crs = zone.crs)
        poly_int = mftb.intersection(line_gpd)
        try:
            poly_int = poly_int.get_coordinates().to_numpy()[0,:]
            int_dist = haversine(joints[j,1], joints[j,0], poly_int[1], poly_int[0]) + joints_dist[j]
            fig.plot(x = [int_dist, int_dist], y = [4, 5.5], pen = '1p,black')
            fig.plot(x = [int_dist, int_dist], y = [4, 5.5], pen = '1p,black')
        except:
            pass

    # project Payenia
    used_zones = []
    Payenia = gpd.read_file('../DATA/MAPPING/Payenia_Volcanic_Field.shp')

    poly_int = Payenia.union_all().intersection(line_gpd)
    poly_int = poly_int.get_coordinates().to_numpy()

    within_mask = shapely.within(shapely.points(line[:,0], line[:,1]), Payenia.union_all())
    Payenia_x = np.unique(line_dist[within_mask])
    Payenia_y = dem.interp(Distance_Along_Trend = Payenia_x, method = 'linear').copy().values/1000
    Payenia_x = np.append(Payenia_x[0], Payenia_x); Payenia_x = np.append(Payenia_x, Payenia_x[-1])
    Payenia_y = np.append(0, Payenia_y); Payenia_y = np.append(Payenia_y, 0)
    fig.plot(x = Payenia_x, y = Payenia_y,  fill = 'red', transparency = 70)





    # project stations
    codes = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i])
             for i in range(len(stat_table))]
    stat_table['code'] = codes

    stat_data = pd.DataFrame({'longitude': stat_table.stlo.values, 'latitude': stat_table.stla.values,
                             'elevation': stat_table.stel.values, 'code': stat_table.code.values})
    used_data = []
    for i in range(len(joints) - 1):
        track = pygmt.project(data=stat_data, center=joints[i], endpoint=joints[i+1], width=[
                              -18, 18], length=[0, joints_dist[i+1] - joints_dist[i]], unit=True)
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




            # stats_to_label = [ 'XM-ASB01', 'XM-ASB02', 'XM-ASB03', 'XM-ASB05', 'XM-ASB21', 'XM-ASB09',
            #                    'XM-CSB08', ]


            try:
                fig.plot(x=track.p[track.code.str.contains('1X')] + joints_dist[i],
                         y=stat_elev_interp.values[track.code.str.contains('1X')]/1000, style='i0.3c', fill='cyan', pen='0.3,black')
            except:
                pass
            try:
                fig.plot(x=track.p[track.code.str.contains('ZR')] + joints_dist[i],
                         y=stat_elev_interp.values[track.code.str.contains('ZR')]/1000, style='i0.3c', fill='lightred', pen='0.3,black')
            except:
                pass

            try:
                fig.plot(x=track.p[track.code.str.contains('XM')] + joints_dist[i], y=stat_elev_interp.values[track.code.str.contains(
                    'XM')]/1000, style='i0.3c', fill='magenta', pen='0.3,black')

            except:
                pass

            # if track.code.isin(stats_to_label).any():
            #     label_stats = track[track.code.isin(stats_to_label)].reset_index(drop = True)
            #     label_elevs = stat_elev_interp.values[track.code.isin(stats_to_label)]/1000
            #     for s in range(len(label_stats)):
            #         fig.text(x = label_stats.p[s] + joints_dist[i], y = label_elevs[s]*2.1, text = label_stats.code[s].split('-')[1], font = '7p,Helvetica')
            #         fig.plot(x = np.ones(2)*(label_stats.p[s] + joints_dist[i]), y = [label_elevs[s], label_elevs[s]*2.1 - .7], pen = '0.5p,black')


    # Plot joints
    Numerals = 'ABCDEFGHIJK'
    fig.text(x = joints_dist, y = np.ones(len(joints_dist)) * 7, text = list(Numerals[0:len(joints_dist)]), font = '14.p, Helvetica-Bold', fill = 'white', pen = '0.3p,black', no_clip = True)




    # project volcanoes
    try:
        volcanoes = pd.read_csv('../DATA/MAPPING/GVP_Holocene_Volcanoes.csv')
        volc_data = pd.DataFrame({'longitude': volcanoes.Longitude.values, 'latitude': volcanoes.Latitude.values,
                                 'elevation': volcanoes['Elevation (m)'].values, 'volc_id': volcanoes['Volcano Number'].values})
        used_data = []
        for i in range(len(joints) - 1):

            # note the little buffer of on the 'length' parameter, for some reason, some lines if the vertex is
            # right on the point, it won't project
            track = pygmt.project(data=volc_data, center=joints[i], endpoint=joints[i+1], width=[-18, 18], length=[
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
                         style='kvolcano/0.5c', fill='red', pen='0.5,black')
    except:
        pass


    # project volcanoes, repeat for PE volcanoes
    try:
        volcanoes = pd.read_csv(
            '../DATA/MAPPING/GVP_Pleistocene_Volcanoes.csv')
        volc_data = pd.DataFrame({'longitude': volcanoes.Longitude.values, 'latitude': volcanoes.Latitude.values,
                                 'elevation': volcanoes['Elevation (m)'].values, 'volc_id': volcanoes['Volcano Number'].values})
        used_data = []
        for i in range(len(joints) - 1):
            track = pygmt.project(data=volc_data, center=joints[i], endpoint=joints[i+1], width=[
                                  -18, 18], length=[0, joints_dist[i+1] - joints_dist[i]], unit=True)
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
                             style='kvolcano/0.5c', fill='orange', pen='0.5,black')
                except:
                    pass
    except:
        pass

    return fig




joints_dist, line, line_dist = Return_Line_Elements(joints)



gmt_region = [-74, -67.5, -39, -32]
west, east, south, north = gmt_region


grid = pygmt.datasets.load_earth_relief(resolution="15s", region=gmt_region)

fig_x = 20
depth = 100
t_depth = 14
aspect = 1

ratio = line_dist.max() / 20 # km per figure cm
fig_y = depth/ratio


x = xr.DataArray(line[:,0], dims='Distance_Along_Trend')
y = xr.DataArray(line[:,1], dims='Distance_Along_Trend')

fig = pygmt.Figure()
with pygmt.config(FONT = '12p', MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p"):

    fig = plot_topo(fig)

    fig.shift_origin(yshift = f'-{fig_y}c')
    fig.basemap(region = [0, line_dist.max(), -0.5, t_depth], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya5f1+lTime (s)'])


    ## Pull QC passing data
    itd_use = itd[itd.G == 1.0][ (itd.Accept == 'Man QC Pass')][itd.Region == region]

    ## collect station location
    stlas = [stations.Latitude[stations.Code == code.replace('_', '-')].values[0] for code in itd_use.Code]
    stlos = [stations.Longitude[stations.Code == code.replace('_', '-')].values[0] for code in itd_use.Code]

    stat_coords = np.c_[stlos, stlas]

    itd_use['stlo'] = stlos
    itd_use['stla'] = stlas




    ## search for data in range of A-B line
    line_tree = KDTree(line)

    data_in_range = line_tree.query_ball_point(x = stat_coords, r = 18 / degrees2kilometers(1))
    data_in_range =[bool(data_in_range[j]) for j in range(len(data_in_range))]

    ## Read file data and time-stack for average station RF
    files = np.unique(np.array(itd_use.File)[data_in_range])


    codes = [] # lists of collected network-stations
    st = Stream()
    for k, f in enumerate(files):
        try:
            print("\r", end="")
            print("Reading Stations: {:.1%} ".format(k/(len(files))), end="")

            st += read('../DATA/{}/DATA_CCP/{}'.format(region, f))
            st[-1] = tr_normalize(st[-1])
            st[-1] = moveout_correction(st[-1])     # optional/ standard moveout correction

            codes.append('{}.{}'.format(st[-1].stats.network, st[-1].stats.station))

        except:
            pass
    codes = np.unique(codes)

    st.stack(group_by='{network}.{station}', npts_tol=2)    # T-stack by station for average station RF



    nets, stlas, stlos, stels, tr_id = [],[],[],[],[]    # resetting stlas and stlos list to queried data
    for i, tr in enumerate(st):
        nets.append(tr.stats.network)
        stlas.append(stations.Latitude[stations.Code == '{}-{}'.format(tr.stats.network, tr.stats.station)].values[0])
        stlos.append(stations.Longitude[stations.Code == '{}-{}'.format(tr.stats.network, tr.stats.station)].values[0])
        stels.append(grid.sel(lat = stlas[-1], lon = stlos[-1], method = 'nearest').values.item())
        tr_id.append(i)

    select_data = pd.DataFrame({'longitude': np.squeeze(stlos), 'latitude': np.squeeze(stlas), 'stel': np.squeeze(stels), 'tr_id': np.array(tr_id)})
    select_data = select_data.sort_values('longitude')

    used_data = []
    for i in range(len(joints) - 1):
       track = pygmt.project(data = select_data,center=joints[i],endpoint=joints[i+1],width=[-18, 18], length = [0, joints_dist[i+1]], unit=True)
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
           in_range = center_tree.query_ball_point(x = np.c_[select_data.longitude, select_data.latitude], r = 6 / degrees2kilometers(1))
           in_range =[bool(in_range[k]) for k in range(len(in_range))]
           in_range = select_data[in_range].tr_id.values

           st_stack = Stream()
           for trace in in_range:
               st_stack += st[trace].copy()

           st_stack.stack(npts_tol = 2)



           tr = st_stack[0]
           data = tr.data
           time = tr.times() - 10

           time_trim = time[(time >= -0.5) & (time <= t_depth)]
           data_trim = data[(time >= -0.5) & (time <= t_depth)] * 15 + station_dist

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












    #######################################





    fig.shift_origin(yshift = f'-{fig_y+0.5}c')
    fig.basemap(region = [0, line_dist.max(), -0.5, t_depth], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya5f1+lTime (s)'])


    ## Pull QC passing data
    itd_use = itd[itd.G == 2.5][ (itd.Accept == 'Man QC Pass')][itd.Region == region]

    ## collect station location
    stlas = [stations.Latitude[stations.Code == code.replace('_', '-')].values[0] for code in itd_use.Code]
    stlos = [stations.Longitude[stations.Code == code.replace('_', '-')].values[0] for code in itd_use.Code]

    stat_coords = np.c_[stlos, stlas]

    itd_use['stlo'] = stlos
    itd_use['stla'] = stlas




    ## search for data in range of A-B line
    line_tree = KDTree(line)

    data_in_range = line_tree.query_ball_point(x = stat_coords, r = 18 / degrees2kilometers(1))
    data_in_range =[bool(data_in_range[j]) for j in range(len(data_in_range))]

    ## Read file data and time-stack for average station RF
    files = np.unique(np.array(itd_use.File)[data_in_range])


    codes = [] # lists of collected network-stations
    st = Stream()
    for k, f in enumerate(files):
        try:
            print("\r", end="")
            print("Reading Stations: {:.1%} ".format(k/(len(files))), end="")

            st += read('../DATA/{}/DATA_CCP/{}'.format(region, f))
            st[-1] = tr_normalize(st[-1])
            st[-1] = moveout_correction(st[-1])     # optional/ standard moveout correction

            codes.append('{}.{}'.format(st[-1].stats.network, st[-1].stats.station))

        except:
            pass
    codes = np.unique(codes)

    st.stack(group_by='{network}.{station}', npts_tol=2)    # T-stack by station for average station RF



    nets, stlas, stlos, stels, tr_id = [],[],[],[],[]    # resetting stlas and stlos list to queried data
    for i, tr in enumerate(st):
        nets.append(tr.stats.network)
        stlas.append(stations.Latitude[stations.Code == '{}-{}'.format(tr.stats.network, tr.stats.station)].values[0])
        stlos.append(stations.Longitude[stations.Code == '{}-{}'.format(tr.stats.network, tr.stats.station)].values[0])
        stels.append(grid.sel(lat = stlas[-1], lon = stlos[-1], method = 'nearest').values.item())
        tr_id.append(i)

    select_data = pd.DataFrame({'longitude': np.squeeze(stlos), 'latitude': np.squeeze(stlas), 'stel': np.squeeze(stels), 'tr_id': np.array(tr_id)})
    select_data = select_data.sort_values('longitude')

    used_data = []
    for i in range(len(joints) - 1):
       track = pygmt.project(data = select_data,center=joints[i],endpoint=joints[i+1],width=[-18, 18], length = [0, joints_dist[i+1]], unit=True)
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
           in_range = center_tree.query_ball_point(x = np.c_[select_data.longitude, select_data.latitude], r = 6 / degrees2kilometers(1))
           in_range =[bool(in_range[k]) for k in range(len(in_range))]
           in_range = select_data[in_range].tr_id.values

           st_stack = Stream()
           for trace in in_range:
               st_stack += st[trace].copy()

           st_stack.stack(npts_tol = 2)



           tr = st_stack[0]
           data = tr.data
           time = tr.times() - 10

           time_trim = time[(time >= -0.5) & (time <= t_depth)]
           data_trim = data[(time >= -0.5) & (time <= t_depth)] * 15 + station_dist

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








    #######################################





    fig.shift_origin(yshift = f'-{fig_y+0.5}c')
    fig.basemap(region = [0, line_dist.max(), -0.5, t_depth], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya5f1+lTime (s)'])


    ## Pull QC passing data
    itd_use = itd[itd.G == 5.0][ (itd.Accept == 'Man QC Pass')][itd.Region == region]

    ## collect station location
    stlas = [stations.Latitude[stations.Code == code.replace('_', '-')].values[0] for code in itd_use.Code]
    stlos = [stations.Longitude[stations.Code == code.replace('_', '-')].values[0] for code in itd_use.Code]

    stat_coords = np.c_[stlos, stlas]

    itd_use['stlo'] = stlos
    itd_use['stla'] = stlas




    ## search for data in range of A-B line
    line_tree = KDTree(line)

    data_in_range = line_tree.query_ball_point(x = stat_coords, r = 18 / degrees2kilometers(1))
    data_in_range =[bool(data_in_range[j]) for j in range(len(data_in_range))]

    ## Read file data and time-stack for average station RF
    files = np.unique(np.array(itd_use.File)[data_in_range])


    codes = [] # lists of collected network-stations
    st = Stream()
    for k, f in enumerate(files):
        try:
            print("\r", end="")
            print("Reading Stations: {:.1%} ".format(k/(len(files))), end="")

            st += read('../DATA/{}/DATA_CCP/{}'.format(region, f))
            st[-1] = tr_normalize(st[-1])
            st[-1] = moveout_correction(st[-1])     # optional/ standard moveout correction

            codes.append('{}.{}'.format(st[-1].stats.network, st[-1].stats.station))

        except:
            pass
    codes = np.unique(codes)

    st.stack(group_by='{network}.{station}', npts_tol=2)    # T-stack by station for average station RF



    nets, stlas, stlos, stels, tr_id = [],[],[],[],[]    # resetting stlas and stlos list to queried data
    for i, tr in enumerate(st):
        nets.append(tr.stats.network)
        stlas.append(stations.Latitude[stations.Code == '{}-{}'.format(tr.stats.network, tr.stats.station)].values[0])
        stlos.append(stations.Longitude[stations.Code == '{}-{}'.format(tr.stats.network, tr.stats.station)].values[0])
        stels.append(grid.sel(lat = stlas[-1], lon = stlos[-1], method = 'nearest').values.item())
        tr_id.append(i)

    select_data = pd.DataFrame({'longitude': np.squeeze(stlos), 'latitude': np.squeeze(stlas), 'stel': np.squeeze(stels), 'tr_id': np.array(tr_id)})
    select_data = select_data.sort_values('longitude')

    used_data = []
    for i in range(len(joints) - 1):
       track = pygmt.project(data = select_data,center=joints[i],endpoint=joints[i+1],width=[-18, 18], length = [0, joints_dist[i+1]], unit=True)
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
           in_range = center_tree.query_ball_point(x = np.c_[select_data.longitude, select_data.latitude], r = 6 / degrees2kilometers(1))
           in_range =[bool(in_range[k]) for k in range(len(in_range))]
           in_range = select_data[in_range].tr_id.values

           st_stack = Stream()
           for trace in in_range:
               st_stack += st[trace].copy()

           st_stack.stack(npts_tol = 2)



           tr = st_stack[0]
           data = tr.data
           time = tr.times() - 10

           time_trim = time[(time >= -0.5) & (time <= t_depth)]
           data_trim = data[(time >= -0.5) & (time <= t_depth)] * 15 + station_dist

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




    #######################################





    fig.shift_origin(yshift = f'-{fig_y+0.5}c')
    fig.basemap(region = [0, line_dist.max(), -0.5, t_depth], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya5f1+lTime (s)'])


    ## Pull QC passing data
    itd_use = itd[itd.G == 7.5][ (itd.Accept == 'Man QC Pass')][itd.Region == region]

    ## collect station location
    stlas = [stations.Latitude[stations.Code == code.replace('_', '-')].values[0] for code in itd_use.Code]
    stlos = [stations.Longitude[stations.Code == code.replace('_', '-')].values[0] for code in itd_use.Code]

    stat_coords = np.c_[stlos, stlas]

    itd_use['stlo'] = stlos
    itd_use['stla'] = stlas




    ## search for data in range of A-B line
    line_tree = KDTree(line)

    data_in_range = line_tree.query_ball_point(x = stat_coords, r = 18 / degrees2kilometers(1))
    data_in_range =[bool(data_in_range[j]) for j in range(len(data_in_range))]

    ## Read file data and time-stack for average station RF
    files = np.unique(np.array(itd_use.File)[data_in_range])


    codes = [] # lists of collected network-stations
    st = Stream()
    for k, f in enumerate(files):
        try:
            print("\r", end="")
            print("Reading Stations: {:.1%} ".format(k/(len(files))), end="")

            st += read('../DATA/{}/DATA_CCP/{}'.format(region, f))
            st[-1] = tr_normalize(st[-1])
            st[-1] = moveout_correction(st[-1])     # optional/ standard moveout correction

            codes.append('{}.{}'.format(st[-1].stats.network, st[-1].stats.station))

        except:
            pass
    codes = np.unique(codes)

    st.stack(group_by='{network}.{station}', npts_tol=2)    # T-stack by station for average station RF



    nets, stlas, stlos, stels, tr_id = [],[],[],[],[]    # resetting stlas and stlos list to queried data
    for i, tr in enumerate(st):
        nets.append(tr.stats.network)
        stlas.append(stations.Latitude[stations.Code == '{}-{}'.format(tr.stats.network, tr.stats.station)].values[0])
        stlos.append(stations.Longitude[stations.Code == '{}-{}'.format(tr.stats.network, tr.stats.station)].values[0])
        stels.append(grid.sel(lat = stlas[-1], lon = stlos[-1], method = 'nearest').values.item())
        tr_id.append(i)

    select_data = pd.DataFrame({'longitude': np.squeeze(stlos), 'latitude': np.squeeze(stlas), 'stel': np.squeeze(stels), 'tr_id': np.array(tr_id)})
    select_data = select_data.sort_values('longitude')

    used_data = []
    for i in range(len(joints) - 1):
       track = pygmt.project(data = select_data,center=joints[i],endpoint=joints[i+1],width=[-18, 18], length = [0, joints_dist[i+1]], unit=True)
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
           in_range = center_tree.query_ball_point(x = np.c_[select_data.longitude, select_data.latitude], r = 6 / degrees2kilometers(1))
           in_range =[bool(in_range[k]) for k in range(len(in_range))]
           in_range = select_data[in_range].tr_id.values

           st_stack = Stream()
           for trace in in_range:
               st_stack += st[trace].copy()

           st_stack.stack(npts_tol = 2)



           tr = st_stack[0]
           data = tr.data
           time = tr.times() - 10

           time_trim = time[(time >= -0.5) & (time <= t_depth)]
           data_trim = data[(time >= -0.5) & (time <= t_depth)] * 15 + station_dist

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







fig.show()
