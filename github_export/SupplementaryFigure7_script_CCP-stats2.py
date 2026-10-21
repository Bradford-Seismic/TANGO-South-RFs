#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 18:04:15 2026

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

    fig.basemap(region = [0, joints_dist[-1], 0,7], projection = 'X{}c/{}c'.format(fig_x, 2), frame = ['WsNe', 'xa100f10', 'ya5f1+e+lElev. (km)'])

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
    stat_data = pd.DataFrame({'longitude': stat_table.stlo.values, 'latitude': stat_table.stla.values,
                             'elevation': stat_table.stel.values, 'code': stat_table.Code.values})
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
    fig.text(x = joints_dist, y = np.ones(len(joints_dist)) * 7, text = list('ABCD'), font = '14.p, Helvetica-Bold', fill = 'white', pen = '0.3p,black', no_clip = True)




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





stations = pd.read_csv('../DATA/SPREADSHEETS/SA_Stations.csv')
moveouts = pd.read_csv('../DATA/SPREADSHEETS/Moveout_Summary.csv')
downloads = pd.read_csv('../DATA/SPREADSHEETS/Download_Summary.csv')
snr = pd.read_csv('../DATA/SPREADSHEETS/SNR_Summary.csv')
itd = pd.read_csv('../DATA/SPREADSHEETS/Iterdecon_Summary.csv', keep_default_na=False)

region = 'TANGO_South'

stat_table = pd.read_csv('../DATA/SPREADSHEETS/{}_PassingStations.csv'.format(region))
stat_table['Code'] = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i]) for i in range(len(stat_table))]

contributing_stations = pd.read_csv('../DATA/MAPPING/Contributing_Stations_RE.csv')
stat_table = stat_table[stat_table.Code.isin(contributing_stations.Code)]




gmt_region = np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_GridArea-ABCD.txt')
north, south, east, west = gmt_region
grid = pygmt.datasets.load_earth_relief(resolution="15s", region=[west, east, south, north])


joints =  np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_Line.txt', delimiter = ',')


joints_dist, line, line_dist = Return_Line_Elements(joints)

x = xr.DataArray(line[:,0], dims='Distance_Along_Trend')
y = xr.DataArray(line[:,1], dims='Distance_Along_Trend')

fig_x = 20

aspect = 1
ratio = joints_dist[-1] / fig_x     # ratio of km to cm
z_lim = 120
fig_y = z_lim / (ratio * aspect)


# require a minimum number of bin hits to be present within the bin radius and bin std radius
# to be plotted
bin_hits_limit = 10
std_hits_limit = 1
std_limit = 0.15



fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig = plot_topo(fig)

    fig.shift_origin(yshift = f'-{fig_y}c')

    grid_region = [0, joints_dist[-1], 0, 120]
    prj = 'X{}c/-{}c'.format(fig_x, fig_y)
    fig.basemap(region=grid_region, projection=prj, frame=['Wsne', 'xa50f10g50+lProfile Distance (km)', 'ya20f10g50+lDepth (km)'])

    sca = [-0.5, 0.5]
    metric = 'Amplitude'

    with xr.open_dataset('../DATA/MAPPING/nc_models/CCP_TANGO_South_G-5.0_FZB_CSEM2-Hicks2014-vp_Hick2014-Hk-vp-csem2_vpvs.nc') as model:
        model_slice = model.interp(longitude = x, latitude = y, method = 'linear', kwargs={"fill_value": None})
        model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})
        model_slice['depth'] = -model_slice['depth']

        # truncate model_slice values to be within scale range
        model_slice[metric].values = np.clip(model_slice[metric].values, sca[0], sca[1])

        # nan grid space with bins less than bin hit limit
        model_slice[metric].values[np.where(model_slice['Bin_Hits'] < bin_hits_limit)] = np.nan
        model_slice[metric].values[np.where(model_slice['Bin_Hits_std'] < std_hits_limit)] = np.nan
        model_slice[metric].values[np.where(model_slice['Bootstrap_std'] > std_limit)] = np.nan


    grid_color = pygmt.makecpt(cmap = '../DATA/MAPPING/no_green.cpt',   series = [sca[0], sca[1]], continuous = True)
    fig.grdimage(grid = model_slice[metric])
    fig.basemap(region=grid_region, projection=prj, frame=['wsne', 'xg50)', 'yg50'])
    with pygmt.config(FONT_ANNOT_PRIMARY="20p", FONT_LABEL="20p"):
        fig.colorbar(cmap = True, position = 'JMR+w{}/0.25c+o0.25c/0c'.format(fig_y), frame = 'xa{}f{}+l{}'.format((sca[1] - sca[0]) / 2, (sca[1] - sca[0]) / 5, metric))


    fig.shift_origin(yshift = f'-{fig_y+0.5}c')
    sca = [0, 100]
    metric = 'Bin_Hits_std'

    fig.basemap(region=grid_region, projection=prj, frame=['Wsne', 'xa50f10g50+lProfile Distance (km)', 'ya20f10g50+lDepth (km)'])

    # truncate model_slice values to be within scale range
    model_slice[metric].values = np.clip(model_slice[metric].values, sca[0], sca[1])
    # nan grid space with bins less than bin hit limit
    model_slice[metric].values[np.where(model_slice['Bin_Hits'] < bin_hits_limit)] = np.nan
    model_slice[metric].values[np.where(model_slice['Bin_Hits_std'] < std_hits_limit)] = np.nan
    model_slice[metric].values[np.where(model_slice['Bootstrap_std'] > std_limit)] = np.nan

    model_slice[metric].values = np.clip(model_slice[metric].values, sca[0], sca[1])

    grid_color = pygmt.makecpt(cmap = '../DATA/MAPPING/no_green.cpt',   series = sca, continuous = True)
    fig.grdimage(grid = model_slice[metric])

    with pygmt.config(FONT_ANNOT_PRIMARY="20p", FONT_LABEL="20p"):
        fig.colorbar(cmap = True, position = 'JMR+w{}/0.25c+o0.25c/0c'.format(fig_y), frame = 'xa{}f{}+l{}'.format((sca[1] - sca[0]) / 2, (sca[1] - sca[0]) / 5, metric))


    fig.shift_origin(yshift = f'-{fig_y+0.5}c')
    sca = [0, 20]
    metric = 'Bin_std_Radius'

    fig.basemap(region=grid_region, projection=prj, frame=['Wsne', 'xa50f10g50+lProfile Distance (km)', 'ya20f10g50+lDepth (km)'])

    # truncate model_slice values to be within scale range
    model_slice[metric].values = np.clip(model_slice[metric].values, sca[0], sca[1])
    # nan grid space with bins less than bin hit limit
    model_slice[metric].values[np.where(model_slice['Bin_Hits'] < bin_hits_limit)] = np.nan
    model_slice[metric].values[np.where(model_slice['Bin_Hits_std'] < std_hits_limit)] = np.nan
    model_slice[metric].values[np.where(model_slice['Bootstrap_std'] > std_limit)] = np.nan

    model_slice[metric].values = np.clip(model_slice[metric].values, sca[0], sca[1])

    grid_color = pygmt.makecpt(cmap = '../DATA/MAPPING/no_green.cpt',   series = sca, continuous = True)
    fig.grdimage(grid = model_slice[metric])

    with pygmt.config(FONT_ANNOT_PRIMARY="20p", FONT_LABEL="20p"):
        fig.colorbar(cmap = True, position = 'JMR+w{}/0.25c+o0.25c/0c'.format(fig_y), frame = 'xa{}f{}+l{}'.format((sca[1] - sca[0]) / 2, (sca[1] - sca[0]) / 5, metric))


    fig.shift_origin(yshift = f'-{fig_y+0.5}c')
    sca = [0, 1]
    metric = 'BAZ_Concentration'

    fig.basemap(region=grid_region, projection=prj, frame=['WSne', 'xa50f10g50+lProfile Distance (km)', 'ya20f10g50+lDepth (km)'])

    # truncate model_slice values to be within scale range
    model_slice[metric].values = np.clip(model_slice[metric].values, sca[0], sca[1])
    # nan grid space with bins less than bin hit limit
    model_slice[metric].values[np.where(model_slice['Bin_Hits'] < bin_hits_limit)] = np.nan
    model_slice[metric].values[np.where(model_slice['Bin_Hits_std'] < std_hits_limit)] = np.nan
    model_slice[metric].values[np.where(model_slice['Bootstrap_std'] > std_limit)] = np.nan

    model_slice[metric].values = np.clip(model_slice[metric].values, sca[0], sca[1])

    grid_color = pygmt.makecpt(cmap = '../DATA/MAPPING/no_green.cpt',   series = sca, continuous = True)
    fig.grdimage(grid = model_slice[metric])

    with pygmt.config(FONT_ANNOT_PRIMARY="20p", FONT_LABEL="20p"):
        fig.colorbar(cmap = True, position = 'JMR+w{}/0.25c+o0.25c/0c'.format(fig_y), frame = 'xa{}f{}+l{}'.format((sca[1] - sca[0]) / 2, (sca[1] - sca[0]) / 5, metric))

    spec = io.StringIO(
    """
N 4
S 0.20c i 0.3c cyan 0.4p,black 0.65c TANGO Node
S 0.20c i 0.3c magenta 0.6p,black 0.65c TANGO Broadband
S 0.20c i 0.3c lightred 0.6p,black 0.65c LaMa Broadband
S 0.20c i 0.3c gold 0.6p,black 0.65c Other Network

S 0.20c kvolcano 0.40c red 0.75p,black 0.65c Quaternary Volcano
S 0.20c r 0.40c red@70 0.75p,black 0.65c Payenia



G 0.07c
G 0.07c

    """
    )


    fig.legend(spec = spec,  position='jBL+w{}c+o0c/-3c'.format(fig_x+1))



fig.show()

fig.savefig('../FIGURES/SupplementaryFigure7.png', dpi = 600)


