#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:58:54 2026

@author: bradford
"""


import os
import sys
import io


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import pygmt
from math import radians, sin, cos, sqrt, atan2
import xarray as xr

import obspy
import obspy.taup.taup_geo as taup
from obspy.taup import TauPyModel
from obspy.core import UTCDateTime as utc
from obspy.core import Stream, read

from scipy.interpolate import interp1d

from scipy.spatial import KDTree
from scipy.interpolate import interpn
from obspy.geodetics import degrees2kilometers


from pyraysum import Model, Geometry, Control, run
import geopandas as gpd
import shapely





# active_dir = 'C:/Users/7418888/Documents/TANGO/Laguna_Del_Maule/SCRIPTING'
active_dir = '/tango/bradford/TANGO_South/SCRIPTING'

os.chdir(active_dir)




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

def plot_topo(fig, region):

    fig.basemap(region = region, projection = 'X{}c/{}c'.format(fig_x, 2), frame = ['WsNe', 'xa100f10', 'ya5f1+e+lElev. (km)'])

    topo_x = np.append(line_dist[0], line_dist)
    topo_x = np.append(topo_x, line_dist[-1])


    # on-line topography

    dem = grid.interp(lat = y, lon = x, method = 'linear')
    dem = dem.assign_coords({'Distance_Along_Trend': line_dist})

    topo = np.append(0, dem.values)
    topo = np.append(topo, 0)


    fig.plot(x = topo_x, y = topo/1000, fill = 'lightgray', pen = '0.5p,black')



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

def moveout_correction(tr, vp_bulk = 6.4, p_ref = 0.04):

    p = tr.stats.sac.user4
    tr_time = tr.times() - 10
    tr_data = tr.data

    modifier = (np.sqrt(1 - (p*vp_bulk)**2)) / (np.sqrt(1 - (p_ref*vp_bulk)**2))
    t_corrected = tr_time * modifier


    tr_interp = interp1d(t_corrected, tr_data, kind = 'linear', bounds_error = False, fill_value = 0)
    tr.data = tr_interp(tr_time)

    return tr


## Load mapping dataset
stations = pd.read_csv('../DATA/SPREADSHEETS/SA_Stations.csv')
moveouts = pd.read_csv('../DATA/SPREADSHEETS/Moveout_Summary.csv')
downloads = pd.read_csv('../DATA/SPREADSHEETS/Download_Summary.csv')
snr = pd.read_csv('../DATA/SPREADSHEETS/SNR_Summary.csv')
itd = pd.read_csv('../DATA/SPREADSHEETS/Iterdecon_Summary.csv', keep_default_na=False)

stat_table = pd.read_csv('../DATA/SPREADSHEETS/TANGO_South_PassingStations.csv')
stat_table['Code'] = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i]) for i in range(len(stat_table))]

contributing_stations = pd.read_csv('../DATA/SPREADSHEETS/Contributing_Stations.csv')


gmt_region = np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_GridArea-ABCD.txt')
north, south, east, west = gmt_region
grid = pygmt.datasets.load_earth_relief(resolution="15s", region=[west, east, south, north])

joints =  np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_Line.txt', delimiter = ',')
joints_dist, line, line_dist = Return_Line_Elements(joints)

x = xr.DataArray(line[:,0], dims='Distance_Along_Trend')
y = xr.DataArray(line[:,1], dims='Distance_Along_Trend')





#% Sedimentay basin + LVZ
z = [0, 5000, 12000, 35000]
thickn = [(z[i+1] - z[i])  for i in range(len(z) - 1)]
thickn.append(0)


vp = [4200, 6400, 5800, 7800]
vpvs = [1.80, 1.74, 1.85, 1.79]
rho = 2800


#Build a 1-layer-over-half-space subsurface model
model = Model(
    thickn = thickn,  # m; half-space thickness is irrelevant
    rho = rho,  # kg/m^3
    vs = list(np.array(vp) / np.array(vpvs)),  # m/s
    vp= vp,  # m/s
    # dip = dip,
    # strike = strike,
)
model.plot()


baz = 300
slow = np.arange(0.04, 0.08, 0.005)

# Define back-azimuth range and single horizontal slowness value
geometry = Geometry(baz=baz, slow=slow)
geometry.plot()

# Set sampling interval, number of points, alignment and ray-coordinate rotation
control = Control(dt=1./20., npts=3000, align="P", rot=1, mults = 2)

# Run the simulation
result = run(model, geometry, control, rf=True)



result.filter('rfs', 'lowpass', freq=1, zerophase=True, corners=2)




fig_x = 6
fig_y = 15
t_depth = 9

# define multples as flat interfaces, single depth value, and x-range
multiple_x = np.array([[340, 370]])
multiple_y = np.array([1.8])

vp_bulk = 5.2
vpvs_bulk = 1.8
p = 0.045

fig = pygmt.Figure()    # initialize the map
with pygmt.config(FONT="12p"):

    region = [330, 370, 0, 7]
    fig = plot_topo(fig, region)



    fig.shift_origin(yshift = f'-{fig_y}c')

    region = [330, 370, -0.5, t_depth]
    fig.basemap(region=region, projection=f'X{fig_x}c/-{fig_y}c', frame=['WSne', 'xaf+lProfile Distance (km)', 'ya1f.5g1+lPs-Delay Time (s)'])

    files = glob('../DATA/TANGO_South/DATA_CCP/1X.*.05.0.itr')


    xn_stat_range = np.arange(172, 190, 1).astype(int)
    st = Stream()
    for f in files:
        if int(f.split('/')[-1].split('.')[1]) in  xn_stat_range:
            st += read(f)

    ## Pull QC passing data
    itd_use = itd[itd.G == 5.0][ (itd.Accept == 'Man QC Pass')][itd.Region == 'TANGO_South']


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

            if int(f.split('/')[-1].split('.')[1]) in  xn_stat_range:

                st += read('../DATA/TANGO_South/DATA_CCP/{}'.format(f))
                st[-1] = tr_normalize(st[-1])

                st[-1] = moveout_correction(st[-1], p_ref = p)     # optional/ standard moveout correction
                # st[-1] = four_pin_moveout_correction(st[-1])    # optional/ four pin moveout correction of trace
                ### Through testing, four-pin correction provides only marginal difference in result

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
        try:
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
               data_trim = data[(time >= -0.5) & (time <= t_depth)] * 5 + station_dist

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

        except:
            pass

    for i in range(len(multiple_y)):

        fig.plot(x=  multiple_x[i,:], y = np.ones(2) * multiple_y[i], pen = '2.5p,cyan,--')

        qa = np.sqrt((1/(vp_bulk**2)) - p**2 )
        qb = np.sqrt((1/( (vp_bulk/vpvs_bulk)**2)) - p**2)

        t2 = (multiple_y[i] / (qb-qa)) * ( qb+qa )
        t3 = 2 * (multiple_y[i] / (qb-qa)) * qb


        fig.plot(x=  multiple_x[i,:], y = np.ones(2) * t2, pen = '2.5p,magenta,--')
        fig.plot(x=  multiple_x[i,:], y = np.ones(2) * t3, pen = '2.5p,green,--')




    fig.shift_origin(xshift = f'{fig_x}c')
    region = [0.03, 0.090, -0.5, t_depth]
    fig.basemap(region=region, projection=f'X{fig_x}c/-{fig_y}c', frame=['wSne', 'xaf+lRay param. (s/km)', 'ya1f.5g1+lPs-Delay Time (s)'])

    for tr in result.rfs:

        rf = tr[0]


        slow = rf.stats.slow
        data =( rf.data / 5 )  + slow

        rf_time = rf.times() + rf.stats.taxis[0]


        pos = data.copy()
        neg = data.copy()

        pos[pos < slow] = slow
        neg[neg > slow] = slow

        fig.plot(x = pos, y = rf_time,pen = '0.1p,black', fill = 'red', transparency = 20)
        fig.plot(x = neg, y = rf_time,pen = '0.1p,black', fill = 'blue', transparency = 20)


    for i in range(len(rf.stats.phase_names)):
        if (rf.stats.phase_names[i] == 'PpP') or (rf.stats.phase_names[i] == 'P'):
            continue
        if rf.stats.phase_names[i] == 'PpS':
            rf.stats.phase_names[i] = 'PpPs'
        elif  rf.stats.phase_names[i] == 'PsS':
            rf.stats.phase_names[i] = 'PpSs'

        elif  rf.stats.phase_names[i] == 'PS':
            rf.stats.phase_names[i] = 'Ps'


        fig.text(x = slow+0.003, y = rf.stats.phase_times[i], text= rf.stats.phase_names[i], font = '12p,Helvetica-Bold', fill = 'white', pen = '0.3p,black', clearance = '+tO')




    fig.shift_origin(xshift = f'{fig_x}c')
    with pygmt.config(FONT="11p"):
        fig.basemap(region=[1.8, 8.5, 0, 58], projection=f'X{fig_x}c/-{fig_y}c', frame=['SnE', 'xaf+lvelocity (km/s)', 'ya10f5+lDepth (km)'])


        vs_stair = np.repeat(model.vs, 2) / 1000
        vp_stair = np.repeat(model.vp, 2) / 1000

        z_stair = np.repeat(z, 2) / 1000
        z_stair = np.roll(z_stair, -1)

        z_stair[-1] = 60


        # z_div =np.append( np.array(z)/2000, 45)[1:None]
        z_div = [2.5, 8, 20, 45]


        fig.plot(x = vs_stair, y = z_stair, pen = '1p,blue')
        fig.plot(x = vp_stair, y = z_stair, pen = '1p,red')
        fig.text(x = np.ones(len(model.vs))*5, y = z_div, text = ['vp/vs = ' + i for i in (np.array(np.round(model.vpvs, 2), dtype = str))], font = '12p,Helvetica', fill = 'white', pen = '0.3p,black')
        fig.text(x = vs_stair[-1], y = 55, text = 'vs', font = '12p,Helvetica', fill = 'white', pen = '0.3p,black')
        fig.text(x = vp_stair[-1], y = 55, text = 'vp', font = '12p,Helvetica', fill = 'white', pen = '0.3p,black')

fig.show()



fig.savefig('../FIGURES/SupplementaryFigure6_SyntheticMultiples.png', dpi = 600)


