# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 14:25:13 2026

@author: jimbradford
"""

import pygmt
import obspy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import hilbert
import geopandas as gpd
import shapely

from obspy import Stream, read

import os
import sys
import io

import numpy as np
import pandas as pd
from glob import glob
import xarray as xr
from math import radians, sin, cos, sqrt, atan2



os.chdir('.')

plt.close('all')

# collect ensemble of receiver functions from select station


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

# average vp above the vp_contour for a station (d = 'net.stat') or an entire network (d = 'net'),
# averaged over the contributing station locations
def vp_solve(d):

    # locations (lon, lat) of contributing stations, same file search as Hk_solve
    files = glob(os.path.join(f'../DATA/TANGO_South/DATA_CCP/{d}.*.itr'))

    stla = []
    stlo = []
    for f in files:
        tr = read(f, headonly = True)[0]
        stlo.append(tr.stats.sac.stlo)
        stla.append(tr.stats.sac.stla)
    stat_coords = np.unique(np.c_[stlo, stla], axis = 0)

    # pull the vp-depth column of CSEM2 beneath each station
    stlo = xr.DataArray(stat_coords[:,0], dims='stat')
    stla = xr.DataArray(stat_coords[:,1], dims='stat')
    vp_tr = csem.VPV.interp(longitude = stlo, latitude = stla, method = 'linear', kwargs={"fill_value": None})

    # resample each column to an even 0.1 km depth spacing so the mean is weighted evenly with depth
    csem_z = vp_tr.depth.values
    order = np.argsort(csem_z)
    z = np.arange(csem_z.min(), csem_z.max(), 0.1)

    vp_avg = []
    z_contour = []
    for s in range(len(stat_coords)):
        vp_col = np.interp(z, csem_z[order], vp_tr.isel(stat = s).values[order])

        # column never reaches the vp_contour
        if not (vp_col >= vp_contour).any():
            continue

        # depth of the vp_contour, shallowest depth where vp reaches vp_contour km/s
        i = np.argmax(vp_col >= vp_contour)
        z_contour.append(z[i])

        # average vp from the top of the model to the vp_contour
        vp_avg.append(np.nanmean(vp_col[0:i]))

    print('station: {}, CSEM2 vp: {} km/s above {} km/s contour at {} km'.format(d, np.round(np.mean(vp_avg), 2), vp_contour, np.round(np.mean(z_contour), 1)))

    return np.mean(vp_avg), np.mean(z_contour)


def Hk_classic(st, vp = 6.4, w = (0.7, 0.2, 0.1), H = (30,55), k = (1.7, 2)):

    H_range = np.arange(*H, 0.1)
    k_range = np.arange(*k, 0.0005)

    H_grid, k_grid = np.meshgrid(H_range, k_range)


    Hk = np.zeros(H_grid.shape)
    H_tr_vals = []
    k_tr_vals = []

    for tr in st:

        p = tr.stats.sac.user4

        """

        qa = sqrt( (1/vp) **2 - p**2)
        qb = sqrt( (1/vp/k) **2 - p**2)

        t1 (tps) = H * (qb - qa)
        t2 (tppps) = H * (qb + qa)
        t3 (tppss) = 2H * qb


        """
        qb = np.sqrt( (1/(vp/k_grid)) **2 - p**2)
        qa = np.sqrt( (1/(np.ones(k_grid.shape)*vp)) **2 - p**2)

        t1  = H_grid * (qb - qa)
        t2  = H_grid * (qb + qa)
        t3  = 2*H_grid * qb


        rf_data = tr.data
        rf_time = tr.times()-10

        rf_interp = interp1d(rf_time, rf_data, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')

        rf_amp1 = rf_interp(t1)
        rf_amp2 = rf_interp(t2)
        rf_amp3 = rf_interp(t3)


        Hk_tr = w[0]*rf_amp1 + w[1]*rf_amp2 - w[2]*rf_amp3
        Hk = Hk + Hk_tr

        Hk_tr_xr = xr.DataArray(Hk_tr, dims = ['k', 'H'], coords = {'H': H_range, 'k': k_range})
        max_tr = Hk_tr_xr.where(Hk_tr_xr==Hk_tr_xr.max(), drop=True).squeeze()
        H_tr_vals.append(max_tr.H.values)
        k_tr_vals.append(max_tr.k.values)

    Hk_xr = xr.DataArray(Hk, dims = ['k', 'H'], coords = {'H': H_range, 'k': k_range})
    return Hk_xr, H_tr_vals, k_tr_vals

def Hk_solve(d, vp = None):
    
    print('station: {}'.format(d))

    files = glob(os.path.join(f'../DATA/TANGO_South/DATA_CCP/{d}.*.itr'))

    st = Stream()
    rayp = []
    for f in files:
        st += read(f)
        st[-1] = tr_normalize(st[-1])
        rayp.append(st[-1].stats.sac.user4)
        
    stla = st[-1].stats.sac.stla
    stlo = st[-1].stats.sac.stlo


    if vp == None:
        vp = vp_base

    num_traces = len(st)

    # stream contains traces from station RF set normalized
    # to p-wave amplitude


    fig = pygmt.Figure()
    fig.basemap(region = [-1, 30, -1.2, 1.2], projection = 'X10c/3c', frame = ['WSen', 'xaf', 'yaf'])

    data_stack = st[0].data
    data_time = st[0].times() - 10
    fig.plot(x = data_time, y = data_stack, pen = '0.5p,gray')

    for i in range(1,len(st)):
        data_stack = np.vstack((data_stack, st[i].data))
        fig.plot(x = data_time, y = data_stack[-1,:], pen = '0.5p,gray')
    rf_avg = np.mean(data_stack, axis = 0)
    rf_std = np.std(data_stack, axis = 0)


    fig.plot(x = data_time, y = rf_avg, pen = '1p,black')
    fig.plot(x = data_time, y = rf_avg + rf_std, pen = '.5p,black,--')
    fig.plot(x = data_time, y = rf_avg - rf_std, pen = '.5p,black,--')




    """

    s(H,k) = \Epsilon_j=1_N w1rj(t1) + w2rj(t2) - w3rj(t3)

    """


    # Hk result by classic method (e.g. Zhu and Kanamori, 2000)
    Hk_station, H_tr_vals, k_tr_vals = Hk_classic(st, vp = vp, w = weights, H = H, k = k)
    max_data = Hk_station.where(Hk_station==Hk_station.max(), drop=True).squeeze()
    H_std = np.std(H_tr_vals)
    k_std = np.std(k_tr_vals)



    return Hk_station, max_data, H_std, k_std, num_traces, st

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

def plot_topo(fig):
    
    fig.basemap(region = [125, joints_dist[-1], 0,7], projection = 'X{}c/{}c'.format(fig_x, 2), frame = ['WsNe', 'xa100f10', 'ya5f1+e+lElev. (km)'])

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
        fig.plot(x = [30, 30], y = [4, 5.5], pen = '1p,black,dotted')
        fig.plot(x = [joints_dist[-1], joints_dist[-1]], y = [4, 5.5], pen = '1p,black,dotted')

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

                        fig.plot(x = [int_dist, int_dist], y = [4, 5.5], pen = '1p,black,dotted')
                        fig.plot(x = [int_dist, int_dist], y = [4, 5.5], pen = '1p,black,dotted')

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




            stats_to_label = [ 'XM-ASB01', 'XM-ASB02', 'XM-ASB03', 'XM-ASB04', 'XM-ASB05', 'XM-ASB21', 'XM-ASB09',
                               'XM-CSB08', 'XM-CSB07']


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

            if track.code.isin(stats_to_label).any():
                label_stats = track[track.code.isin(stats_to_label)].reset_index(drop = True)
                label_elevs = stat_elev_interp.values[track.code.isin(stats_to_label)]/1000
                for s in range(len(label_stats)):                   
                    fig.text(x = label_stats.p[s] + joints_dist[i], y = label_elevs[s]*2.1, text = label_stats.code[s].split('-')[1], font = '7p,Helvetica')
                    fig.plot(x = np.ones(2)*(label_stats.p[s] + joints_dist[i]), y = [label_elevs[s], label_elevs[s]*2.1 - .7], pen = '0.5p,black')


    # Plot joints
    Numerals = 'ABCDEFGHIJK'
    fig.text(x = joints_dist[1:len(joints_dist)], y = np.ones(len(joints_dist[1:len(joints_dist)])) * 7, text = list('BCD'), font = '14.p, Helvetica-Bold', fill = 'white', pen = '0.3p,black', no_clip = True)


        
        
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
                             style='kvolcano/0.5c', fill='red', pen='0.5,black')
                except:
                    pass
    except:
        pass

    return fig





## Load mapping dataset


stations = pd.read_csv('../DATA/SPREADSHEETS/SA_Stations.csv')
moveouts = pd.read_csv('../DATA/SPREADSHEETS/Moveout_Summary.csv')
downloads = pd.read_csv('../DATA/SPREADSHEETS/Download_Summary.csv')
snr = pd.read_csv('../DATA/SPREADSHEETS/SNR_Summary.csv')
itd = pd.read_csv('../DATA/SPREADSHEETS/Iterdecon_Summary.csv', keep_default_na=False)

region = 'TANGO_South'

stat_table = pd.read_csv('../DATA/SPREADSHEETS/{}_PassingStations.csv'.format(region))
stat_table['Code'] = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i]) for i in range(len(stat_table))]

contributing_stations = pd.read_csv('../DATA/SPREADSHEETS/Contributing_Stations.csv')



gmt_region = np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_GridArea-ABCD.txt')
north, south, east, west = gmt_region
grid = pygmt.datasets.load_earth_relief(resolution="15s", region=[west, east, south, north])


joints =  np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_Line.txt', delimiter = ',')


joints_dist, line, line_dist = Return_Line_Elements(joints)

x = xr.DataArray(line[:,0], dims='Distance_Along_Trend')
y = xr.DataArray(line[:,1], dims='Distance_Along_Trend')

fig_x = 20


## Load CSEM2 vp model (as in Figure4_script.py), depth in km (positive-down), VPV in km/s
csem = xr.open_dataset('../DATA/MAPPING/nc_models/CSEM2-SouthAmerica.v2024.12.01.r0.0-n4.nc')
vp_contour = 7.5


### Data search and Hk criteria
data = [ 'XM.CSB07', 'XM.CSB08', 'ZR',
        'XM.ASB01', 'XM.ASB02', 'XM.ASB03', 'XM.ASB04', 'XM.ASB05', 'XM.ASB21']

fig = pygmt.Figure()
with pygmt.config(FONT = '10p'):

    
    fig = plot_topo(fig)


    vp_base = 6.4
    weights = (0.4, 0.3, 0.3)
    H = (35, 55)
    k = (1.7, 1.9)
    fig.shift_origin(yshift = '-2.75c')
    with fig.subplot(nrows=1, ncols=len(data), figsize=(f'{fig_x}c', "2c"), frame="lrtb"):
        
        cmap = pygmt.makecpt(cmap = 'jet', series = [0,1])

        for j in range(len(data)):
            with fig.set_panel(panel=j):
                if j == 0:
                    fig.basemap(region = [np.min(H), np.max(H), np.min(k), np.max(k)],
                                projection = 'X2c/2c', frame = ['WSen', 'xa10fg5+lThickness (H, km)', 'yafg.05+lvp/vs (k)'])
                else:
                    fig.basemap(region = [np.min(H), np.max(H), np.min(k), np.max(k)],
                                projection = 'X2c/2c', frame = ['wSen', 'xa10fg5', 'yafg.05'])
                
                
                
                vp_csem, z_csem = vp_solve(data[j])
                
                Hk_station, max_data, H_std, k_std, num_traces, st = Hk_solve(data[j], vp = vp_csem)
                Hk_max = Hk_station.max()
                Hk_min = Hk_station.min()
                Hk_station = (Hk_station / Hk_max)
                
                
                fig.grdimage(grid = Hk_station, cmap = cmap)
                fig.grdcontour(grid = Hk_station, levels = [Hk_station.values.max() * 0.90], pen = '0.5p,black')

                fig.plot(x = max_data.H, y = max_data.k, style = 'c.2c', fill = 'green', pen = '1p,black')


                fig.plot(x = [max_data.H - H_std, max_data.H + H_std], y = [max_data.k, max_data.k],  pen = '1p,black')
                fig.plot(x = [max_data.H , max_data.H ], y = [max_data.k - k_std, max_data.k + k_std], pen = '1p,black')

                fig.plot(x = [max_data.H, max_data.H], y = [Hk_station.k.values.min(), max_data.k],   pen = '.5p,white')
                fig.plot(x = [Hk_station.H.values.min(), max_data.H], y = [max_data.k, max_data.k],   pen = '.5p,white')

                fig.text(x = 47, y = 1.72, text = data[j], fill = 'white', font = '8p,Helvetica', pen = '0.5p,black')

                fig.text(x = 45, y = 1.93, text = f'H: {np.round(max_data.H.values, 1)}, k:{np.round(max_data.k.values, 2)}, vp:{vp_csem:.2f}', 
                         font = '6p,Helvetica', no_clip = True)
                
    
    
    spec = io.StringIO(
    """
N 4
S 0.20c i 0.3c cyan 0.4p,black 0.65c TANGO Node
S 0.20c i 0.3c magenta 0.6p,black 0.65c TANGO Broadband
S 0.20c i 0.3c lightred 0.6p,black 0.65c LaMa Broadband
S 0.20c i 0.3c gold 0.6p,black 0.65c Other Network

S 0.20c kvolcano 0.40c red 0.75p,black 0.65c Quaternary Volcano
S 0.20c r 0.40c red@70 0.75p,black 0.65c Payenia

# S 0.20c -- 0.40c black 0.75p,black,dashed 0.65c Slab 2.0 (Hayes, 2018)


G 0.07c
G 0.07c

    """
    )


    # fig.shift_origin(yshift = '-2c')


    fig.legend(spec = spec,  position='jBL+w{}c+o0c/-2.5c'.format(fig_x))

fig.show()


fig.savefig('../FIGURES/Fig3_Hk-v3.png', dpi = 1200)

