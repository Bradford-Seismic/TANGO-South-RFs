#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 11:05:37 2026

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

def moveout_correction(tr, vp_bulk = 6.4, p_ref = 0.06):

    p = tr.stats.sac.user4
    tr_time = tr.times() - 10
    tr_data = tr.data

    modifier = (np.sqrt(1 - (p*vp_bulk)**2)) / (np.sqrt(1 - (p_ref*vp_bulk)**2))
    t_corrected = tr_time * modifier


    tr_interp = interp1d(t_corrected, tr_data, kind = 'linear', bounds_error = False, fill_value = 0)
    tr.data = tr_interp(tr_time)

    return tr




#%% Load datasets

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



#%% Slab Design




z = [0, 40000, 50000]
thickn = [(z[i+1] - z[i])  for i in range(len(z) - 1)]
thickn.append(0)


### Realistic Model
vp = [6800, 5800, 8000]
vpvs = [1.75, 1.82, 1.79 ]
dip = [0, 30, 30]
strike = 16
rho = 2800

### Stable Model
# vp = [6400, 5800, 7800]
# vpvs = [1.75, 1.82, 1.79 ]
# dip = [0, 30, 30]
# strike = 0
# rho = 2800


#Build a 1-layer-over-half-space subsurface model
model = Model(
    thickn = thickn,  # m; half-space thickness is irrelevant
    rho = rho,  # kg/m^3
    vs = list(np.array(vp) / np.array(vpvs)),  # m/s
    vp= vp,  # m/s
    dip = dip,
    strike = strike,
)
model.plot()




### Ray Design
baz = np.arange(0., 360., 10.)
slow = 0.06
geometry = Geometry(baz=baz, slow=slow)
geometry.plot()



### Run Simulated Rays

# Set sampling interval, number of points, alignment and ray-coordinate rotation
control = Control(dt=1./20., npts=3000, align="P", rot='RTZ', mults = 0)

# Run the simulation
result = run(model, geometry, control, rf=True)

result.filter('rfs', 'lowpass', freq=1, zerophase=True, corners=2)



t_depth = 10
fig_x = 20
fig_y = 10

# Plot result
fig = pygmt.Figure()
with pygmt.config(FONT = '12p'):
    gmt_region = [0, 360, -0.5, t_depth]
    fig.basemap(region=gmt_region, projection=f'X{fig_x}c/-{fig_y}c', frame=['WSne', 'xafg+lBack Azimuth', 'yafg+lPs-Delay Time (s)'])

    for tr in result.rfs:

        try:
            rf = tr.select(channel = 'RFR')[0]
            baz = rf.stats.baz
            data =( rf.data * 300)  + baz

            rf_time = rf.times() + rf.stats.taxis[0]


            pos = data.copy()
            neg = data.copy()

            pos[pos < baz] = baz
            neg[neg > baz] = baz

            fig.plot(x = pos, y = rf_time,pen = '0.1p,black', fill = 'red', transparency = 20)
            fig.plot(x = neg, y = rf_time,pen = '0.1p,black', fill = 'blue', transparency = 20)

        except:
            pass


fig.show()


#%% Summarize current data coverage


## Pull QC passing data
itd_use = itd[itd.G == 5.0][ (itd.Accept == 'Man QC Pass')][itd.Region == region][itd.Net == '1X']


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
st_real = Stream()
bazs = []
evids = []
evlas = []
evlos = []
for k, f in enumerate(files):
    try:
        print("\r", end="")
        print("Reading stations: {:.1%} ".format(k/(len(files))), end="")

        st_real += read('../DATA/{}/DATA_CCP/{}'.format(region, f))
        st_real[-1] = tr_normalize(st_real[-1])

        st_real[-1] = moveout_correction(st_real[-1], vp_bulk = 6.8, p_ref = slow)     # optional/ standard moveout correction
        # st_real[-1] = four_pin_moveout_correction(st_real[-1])    # optional/ four pin moveout correction of trace
        ### Through testing, four-pin correction provides only marginal difference in result

        codes.append('{}.{}'.format(st_real[-1].stats.network, st_real[-1].stats.station))
        evid = f'{st_real[-1].stats.sac.evla}-{st_real[-1].stats.sac.evlo}'
        bazs.append(st_real[-1].stats.sac.baz)

        if evid not in evids:
            evids.append(evid)
            evlas.append(st_real[-1].stats.sac.evla)
            evlos.append(st_real[-1].stats.sac.evlo)

    except:
        pass

codes = np.unique(codes)




fig = pygmt.Figure()
prj = "A-71.2/-20/100/15c"


with pygmt.config(FONT="12p"):

    fig.coast(projection=prj, region = 'g', frame = 'g60', land="gray", shorelines = '0.5p, black', water = 'white')
    fig.plot(x = evlos, y = evlas, style='c0.23c',fill = 'lightred', pen='0.6p,black', label = 'P phase')

fig.show()



# Back-Azimuth Distribution
fig = pygmt.Figure()
with pygmt.config(FONT="12p"):

    baz_hits = []
    baz_bin = np.arange(0, 370, 10)
    for i in range(len(baz_bin-1)):
        hit_count = 0
        for j in bazs:
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
        diameter="10c",
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
        frame=["x1g100", "y10g10", '+gwhite'],
        # use a pen size of 1p to draw the outlines
        pen="1p",
        no_scale = True

    )

    fig.text(position = 'BL', text = 'n = {} Events'.format(len(evids)), offset = '0.1c/0.1c',font = '10p,Helvetica,black', fill = 'white', pen = '0.5p,black')

fig.show()

"""
1X network shows a lot of domination from events in the direction between
230-250 BAZ, and 340-350 BAZ


All other networks show mixed distribution, but a particular dominance between
130-140 BAZ, 220-240 BAZ, 320-350 BAZ

"""

#%% Set up simulated test area

gmt_region = [-75, -71, -36, -35]

tecto_plates = gpd.read_file('../DATA/MAPPING/PlateBoundaries_Nazca.shp')
grid = pygmt.datasets.load_earth_relief(resolution="03s", region=gmt_region)
shade = pygmt.grdgradient(grid=grid, azimuth="315/45", normalize="e1")

joints_sub = joints[0:2]

def segment_azimuths(lons, lats):
    lons = np.radians(lons)
    lats = np.radians(lats)

    dlon = np.diff(lons)

    x = np.sin(dlon) * np.cos(lats[1:])
    y = np.cos(lats[:-1]) * np.sin(lats[1:]) - \
        np.sin(lats[:-1]) * np.cos(lats[1:]) * np.cos(dlon)

    az = np.degrees(np.arctan2(x, y))
    return (az + 360) % 360

def calculate_slab_latlon(ds, lat_dim='latitude', lon_dim='longitude', z_var='z'):
    """
    Calculates strike and dip from an xarray dataset using Lat/Lon.
    Handles non-square grids by converting degrees to meters locally.
    """
    # 1. Physical constants
    R = 6371.0  # Earth radius in kilometers

    # 2. Get coordinate arrays and convert to radians for distance math
    lat_rad = np.radians(ds[lat_dim])

    # 3. Calculate local kilometers per degree
    # 1 deg lat is constant ~111km; 1 deg lon shrinks with cos(lat)
    km_per_deg_lat = (np.pi * R) / 180.0
    km_per_deg_lon = km_per_deg_lat * np.cos(lat_rad)

    # 4. Calculate gradients (dZ/dLat and dZ/dLon)
    # .differentiate() handles the non-square spacing of the lat/lon labels
    dz_dlat = ds[z_var].differentiate(lat_dim) / km_per_deg_lat
    dz_dlon = ds[z_var].differentiate(lon_dim) / km_per_deg_lon

    # 5. Calculate Dip (magnitude of gradient)
    # Gradient in meters/meter
    grad_mag = np.sqrt(dz_dlat**2 + dz_dlon**2)
    dip = np.degrees(np.arctan(grad_mag))

    # 6. Calculate Strike
    # North = +Y (lat), East = +X (lon)
    # dip_dir = arctan2(east_grad, north_grad)
    # Strike is dip_dir - 90 (Right Hand Rule)
    dip_direction = np.degrees(np.arctan2(dz_dlon, dz_dlat))
    strike = (dip_direction - 90) % 360

    # 7. Package results
    geometry = xr.Dataset(
        data_vars={
            'dip': (ds[z_var].dims, dip.data),
            'strike': (ds[z_var].dims, strike.data)
        },
        coords=ds.coords,
        attrs={'units': 'degrees', 'reference': 'WGS84'}
    )

    return geometry




slab = xr.open_dataset('../DATA/MAPPING/sam_slab2_dep_02.23.18.grd')
slab.coords['longitude'] = slab.x - 360; slab.coords['latitude'] = slab.y
slab = slab.swap_dims({'x': 'longitude', 'y': 'latitude'})
slab.z.values = -slab.z

slab_geometry = calculate_slab_latlon(slab)




fig = pygmt.Figure()
with pygmt.config(FONT = '12p'):

    prj = 'M20c'

    fig.basemap(region=gmt_region, projection=prj, frame=['xafg+lLongitude', 'yafg+lLatitude', 'WSne'])
    fig.grdimage(grid=grid, shading = shade, cmap = '../DATA/MAPPING/natural_mod.cpt', projection = prj, transparency = 55)
    fig.coast(borders = ["1/1.5p,black"], shorelines="1/0.5p", water = "skyblue", transparency = 40)
    fig.grdcontour(grid=slab.z, annotation= 20)

    fig.plot(data=tecto_plates[tecto_plates.Type == 'Convergent'], region = gmt_region, pen = '1.5p,black', style = "f1c/0.15c+r+t", fill = 'black')

    fig.plot(x = itd_use.stlo, y = itd_use.stla, style = 'i0.4c', pen = '0.5p,black', fill = 'white')


    joints =  np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_Line.txt', delimiter = ',')
    fig.plot(x = joints[:,0], y = joints[:,1], pen = '1.2p,black,--')
    Numerals = "ABCDEF"
    for i in range(len(joints)):
        fig.text(x = joints[i,0], y = joints[i,1], text = Numerals[i],  font = '12p, Helvetica', fill = 'white', pen = '0.3p,black')






    fig.shift_origin(yshift = '-8c')

    fig.basemap(region=gmt_region, projection=prj, frame=['xafg+lLongitude', 'yafg+lLatitude', 'WSne'])
    fig.coast(borders = ["1/1.5p,black"], shorelines="1/0.5p", water = "skyblue", transparency = 40)

    cmap = pygmt.makecpt(cmap = 'viridis', series = [0, 360])
    fig.grdimage(grid = slab_geometry.strike, cmap = True)
    fig.grdcontour(grid=slab.z, annotation= 20)

    fig.plot(data=tecto_plates[tecto_plates.Type == 'Convergent'], region = gmt_region, pen = '1.5p,black', style = "f1c/0.15c+r+t", fill = 'black')
    fig.plot(x = itd_use.stlo, y = itd_use.stla, style = 'i0.4c', pen = '0.5p,black', fill = 'white')

    fig.plot(x = joints[:,0], y = joints[:,1], pen = '1.2p,black,--')
    Numerals = "ABCDEF"
    for i in range(len(joints)):
        fig.text(x = joints[i,0], y = joints[i,1], text = Numerals[i],  font = '12p, Helvetica', fill = 'white', pen = '0.3p,black')


    fig.colorbar(cmap = True, position = 'JMR', frame = 'xa+lStrike')





    fig.shift_origin(yshift = '-8c')

    fig.basemap(region=gmt_region, projection=prj, frame=['xafg+lLongitude', 'yafg+lLatitude', 'WSne'])
    fig.coast(borders = ["1/1.5p,black"], shorelines="1/0.5p", water = "skyblue", transparency = 40)

    cmap = pygmt.makecpt(cmap = 'viridis', series = [0, 100])
    fig.grdimage(grid = slab_geometry.dip, cmap = True)
    fig.grdcontour(grid=slab.z, annotation= 20)

    fig.plot(data=tecto_plates[tecto_plates.Type == 'Convergent'], region = gmt_region, pen = '1.5p,black', style = "f1c/0.15c+r+t", fill = 'black')
    fig.plot(x = itd_use.stlo, y = itd_use.stla, style = 'i0.4c', pen = '0.5p,black', fill = 'white')

    fig.plot(x = joints[:,0], y = joints[:,1], pen = '1.2p,black,--')
    Numerals = "ABCDEF"
    for i in range(len(joints)):
        fig.text(x = joints[i,0], y = joints[i,1], text = Numerals[i],  font = '12p, Helvetica', fill = 'white', pen = '0.3p,black')


    fig.colorbar(cmap = True, position = 'JMR', frame = 'xa+lDip')


fig.show()


#%% Run Slab Synthetics


"""

At set points along the forearc region of the ABCD line, we want to:
    (1) identify the slab strike, dip, and depth

    (2) calculate the synthetic RF using the known baz distributions

    (3) migrate and sum those RFs in a representative estimate

    (4) plot the RFs along the line

    (5) compare synthetics to observed results

"""

x_dist = np.arange(30, 180, 10)
synth_rfs = Stream()
for k, x in enumerate(x_dist):

    print("\r", end="")
    print("Calculating Synthetics: {:.1%} ".format(k/(len(x_dist))), end="")

    line_i = closest_idx(x, line_dist)
    line_coord = line[line_i]

    slab_depth = slab.interp(longitude = line_coord[0], latitude = line_coord[1], method = 'linear').z.values
    slab_dip= slab_geometry.interp(longitude = line_coord[0], latitude = line_coord[1], method = 'linear').dip.values
    slab_strike = slab_geometry.interp(longitude = line_coord[0], latitude = line_coord[1], method = 'linear').strike.values

    slab_depth = np.round(slab_depth) * 1000
    slab_dip = np.round(slab_dip)

    z = [0, slab_depth, slab_depth + 10 ]
    thickn = [(z[i+1] - z[i])  for i in range(len(z) - 1)]
    thickn.append(0)


    ### Realistic Model
    vp = [6800, 5800, 8000]
    vpvs = [1.75, 1.82, 1.79 ]
    dip = [0, slab_dip, slab_dip]
    strike = np.round(slab_strike)
    rho = 2800

    ### Stable Model
    # vp = [6400, 5800, 7800]
    # vpvs = [1.75, 1.82, 1.79 ]
    # dip = [0, 30, 30]
    # strike = 0
    # rho = 2800


    #Build a 1-layer-over-half-space subsurface model
    model = Model(
        thickn = thickn,  # m; half-space thickness is irrelevant
        rho = rho,  # kg/m^3
        vs = list(np.array(vp) / np.array(vpvs)),  # m/s
        vp= vp,  # m/s
        dip = dip,
        strike = strike,
    )
    # model.plot()


    # ray design
    baz = np.append(np.arange(220, 270, 5), np.arange(335, 360, 5)) # consistent with data
    # baz= np.arange(0, 360, 10) # idealic full coverage
    slow = 0.06
    geometry = Geometry(baz=baz, slow=slow)
    # geometry.plot()

    control = Control(dt=1./20., npts=3000, align="P", rot='RTZ', mults = 0)
    result = run(model, geometry, control, rf=True)
    result.filter('rfs', 'lowpass', freq=1, zerophase=True, corners=2)



    # t_depth = 10
    # fig_x = 20
    # fig_y = 10

    # # Plot result
    # fig = pygmt.Figure()
    # with pygmt.config(FONT = '12p'):
    #     gmt_region = [0, 360, -0.5, t_depth]
    #     fig.basemap(region=gmt_region, projection=f'X{fig_x}c/-{fig_y}c', frame=['WSne', 'xafg+lBack Azimuth', 'yafg+lPs-Delay Time (s)'])


    #     for tr in result.rfs:
    #         try:
    #             rf = tr.select(channel = 'RFR')[0]
    #             baz = rf.stats.baz
    #             data =( rf.data * 300)  + baz

    #             rf_time = rf.times() + rf.stats.taxis[0]


    #             pos = data.copy()
    #             neg = data.copy()

    #             pos[pos < baz] = baz
    #             neg[neg > baz] = baz

    #             fig.plot(x = pos, y = rf_time,pen = '0.1p,black', fill = 'red', transparency = 20)
    #             fig.plot(x = neg, y = rf_time,pen = '0.1p,black', fill = 'blue', transparency = 20)

    #         except:
    #             pass


    # fig.show()


    sts = Stream()
    for tr in result.rfs:
        rf = tr.select(channel = 'RFR')[0]
        sts += rf




    synth_rfs += sts.stack()




# Plot result

t_depth = 10
fig_x = 20
fig_y = 10
print('\nPlotting Synthetic Results')

slab_moho_conversion = []
slab_top_conversion = []

fig = pygmt.Figure()
with pygmt.config(FONT = '12p'):
    gmt_region = [0, 200, -0.5, t_depth]
    fig.basemap(region=gmt_region, projection=f'X{fig_x}c/-{fig_y}c', frame=['WSne', 'xafg+lProfile Distance (km)', 'yafg+lPs-Delay Time (s)'])


    for i, rf in enumerate(synth_rfs):
        try:
            data =( rf.data *300)  + x_dist[i]

            rf_time = rf.times() + rf.stats.taxis[0]

            slab_top_conversion.append([x_dist[i], rf_time[np.where(data == data[rf_time>2].min())[0][0]]])
            slab_moho_conversion.append([x_dist[i], rf_time[np.where(data == data[rf_time>2].max())[0][0]]])

            pos = data.copy()
            neg = data.copy()

            pos[pos < x_dist[i]] = x_dist[i]
            neg[neg > x_dist[i]] = x_dist[i]

            fig.plot(x = pos, y = rf_time,pen = '0.1p,black', fill = 'red', transparency = 20)
            fig.plot(x = neg, y = rf_time,pen = '0.1p,black', fill = 'blue', transparency = 20)

        except:
            pass

    slab_top_conversion = np.array(slab_top_conversion)
    slab_moho_conversion = np.array(slab_moho_conversion)
    # fig.plot(x = slab_top_conversion[:,0], y = slab_top_conversion[:,1], pen = '1p,black,--')
    fig.plot(x = slab_moho_conversion[:,0], y = slab_moho_conversion[:,1], pen = '1p,black,--')

fig.show()



# sys.exit('here')


print('Plotting Observed results')

fig = pygmt.Figure()
with pygmt.config(FONT = '12p'):

    gmt_region = [0, 200, -0.5, t_depth]
    fig.basemap(region=gmt_region, projection=f'X{fig_x}c/-{fig_y}c', frame=['WSne', 'xafg+lProfile Distance (km)', 'yafg+lPs-Delay Time (s)'])

    nets, stlas, stlos, stels, tr_id = [],[],[],[],[]    # resetting stlas and stlos list to queried data
    for i, tr in enumerate(st_real):
        nets.append(tr.stats.network)
        stlas.append(stations.Latitude[stations.Code == '{}-{}'.format(tr.stats.network, tr.stats.station)].values[0])
        stlos.append(stations.Longitude[stations.Code == '{}-{}'.format(tr.stats.network, tr.stats.station)].values[0])
        stels.append(grid.sel(lat = stlas[-1], lon = stlos[-1], method = 'nearest').values.item())
        tr_id.append(i)

    select_data = pd.DataFrame({'longitude': np.squeeze(stlos), 'latitude': np.squeeze(stlas), 'stel': np.squeeze(stels), 'tr_id': np.array(tr_id)})
    select_data = select_data.sort_values('longitude')

    used_data = []
    for i in range(len(joints_sub) - 1):
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
               st_stack += st_real[trace].copy()

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

           fig.plot(x = pos, y = time_trim,pen = '0.1p,black', no_clip = False, fill = 'red', transparency = 20)
           fig.plot(x = neg, y = time_trim,pen = '0.1p,black', no_clip = False, fill = 'blue', transparency = 20)

    # fig.plot(x = slab_top_conversion[:,0], y = slab_top_conversion[:,1], pen = '1p,black,--')
    fig.plot(x = slab_moho_conversion[:,0], y = slab_moho_conversion[:,1], pen = '1p,black,--')


fig.show()

#%% Migrate Moho estimated arrival


"""

the array 'slab_moho_conversion' contains the distance-time vectors
of the estiamted slab Moho ps conversion from the synthetic results

in the previous cell we compared that to the observed data
( moveout corrected to the synthetic test parameters )

Here, we will 'pseudo migrate' those time arrivals to depth using
velocity model and methods in our CCP algorithm, then compare
that to the CCP stacks.

The overall goal is to prove that our "maybe" slab moho is reasonably
replicated in the synthetic tests

"""


model ='BlendedModel_CSEM2-Hicks2014_vp.nc'
vpvs_model = 'BlendedModel_Hk-Hicks2014_vpvs.nc'

vp_var_name = 'vp'
vpvs_var_name = 'vpvs'


## NOTE, we've already specified which station data we want to use


# velocity parameters
minv = 2.5  # lowest allowable shear wave velocity in model, any point lower will be raised to this
min_vpvs = 1.6  # lowest allowable vp/vs in model,
max_vpvs = 2.0  # highest allowable vp/vs in model,
ref_angle = 20  # reference angle for calculating amplitude scaling factor




# pull the line information using the sub-line in the forearc
joints =  np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_Line.txt', delimiter = ',')
joints_sub = joints[0:2]

joints_sub_dist, line_sub, line_sub_dist = Return_Line_Elements(joints_sub)

x = xr.DataArray(line_sub[:,0], dims='Distance_Along_Trend')
y = xr.DataArray(line_sub[:,1], dims='Distance_Along_Trend')

gmt_region = np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_GridArea-ABCD.txt')
north, south, east, west = gmt_region
grid = pygmt.datasets.load_earth_relief(resolution="15s", region=[west, east, south, north])


topo = grid.interp(lon = x, lat = y, method = 'linear', kwargs={"fill_value": None})
topo.values = topo.values / 1000





# synthetic RF data
times = synth_rfs[0].times() + synth_rfs[0].stats.taxis[0]
p = slow
dzi = 0.5


### Pull velocity data through xarray interpolation of coordinates
with xr.open_dataset('../DATA/MAPPING/nc_models/{}'.format(model)) as mod:

    # first interpolate model along line
    smod_slice = mod.interp(longitude = x, latitude = y, method = 'linear', kwargs={"fill_value": None})
    smod_slice = smod_slice.assign_coords({'Distance_Along_Trend': line_sub_dist})

    sz = mod.depth.values

### Pull vpvs through xarray interpolation of coordinates
with xr.open_dataset('../DATA/MAPPING/nc_models/{}'.format(vpvs_model)) as mod:

    kmod_slice = mod.interp(longitude = x, latitude = y, method = 'linear', kwargs={"fill_value": None})
    kmod_slice = kmod_slice.assign_coords({'Distance_Along_Trend': line_sub_dist})
    kz = mod.depth.values



slab_moho_depth = []
# cycle through points in x_dist
for i, xi in enumerate(x_dist):
    # interpoalte vp at point
    smod_point = smod_slice.interp(Distance_Along_Trend = xi, method = 'linear', kwargs={"fill_value": None})
    kmod_point = kmod_slice.interp(Distance_Along_Trend = xi, method = 'linear', kwargs={"fill_value": None})
    elevation = topo.interp(Distance_Along_Trend = xi, method = 'linear', kwargs={"fill_value": None}).values
    z_local = np.arange(np.round(elevation * 1/dzi) * dzi, 120 + dzi, dzi).astype(np.float16)


    vpu = smod_point.vp.values
    ksu = kmod_point.vpvs.values


    # Interpolate vs at mid-points between layers in z
    v_interp_func = interp1d(sz, vpu, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
    vp = v_interp_func(z_local + 0.5*dzi)

    k_interp_func = interp1d(kz, ksu, kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
    ku = k_interp_func(z_local + 0.5*dzi)

    ku = np.clip(ku, min_vpvs, max_vpvs)

    # multiply vp/k to obtain vs, then restrict values
    vs = vp/ku
    vs = np.clip(vs, minv, np.nanmax(vs))



    # Solve for qa and qb coefficients
    qa = np.sqrt((1/(vs)**2) - p**2)
    qb = np.sqrt((1/(vp)**2) - p**2)

    # solve for travel time within each depth interval
    dt = dzi * (qa - qb)

    tbot = np.cumsum(dt)    # total travel time at bottom of interval
    ttop = tbot - dt        # total travel time at top of interval
    tmid = np.mean([tbot, ttop], axis = 0) # estimated time at middle of interval

    ### tmid is index-tied to depth vector (Note, elevation already corrected at location)
    depth_interp_func = interp1d(tmid, z_local, kind = 'linear')
    phase_arrive = depth_interp_func(slab_moho_conversion[i, 1])


    slab_moho_depth.append(phase_arrive)

slab_moho_conversion_depth = np.c_[x_dist, slab_moho_depth]




## Now, plot the arrival against the CCP stacks



fig_x = 20

aspect = 1
ratio = joints_sub_dist[-1] / fig_x     # ratio of km to cm
z_lim = 120
fig_y = z_lim / (ratio * aspect)


# require a minimum number of bin hits to be present within the bin radius and bin std radius
# to be plotted
bin_hits_limit = 10
std_hits_limit = 1
std_limit = 0.15


fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):


    # fig = plot_topo(fig)
    # fig.shift_origin(yshift = f'-{fig_y}c')



    grid_region = [0, joints_sub_dist[-1], 0, 120]
    prj = 'X{}c/-{}c'.format(fig_x, fig_y)
    fig.basemap(region=grid_region, projection=prj, frame=['wsne'])

    with xr.open_dataset('../DATA/MAPPING/nc_models/CCP_TANGO_South_G-5.0_FZB_CSEM2-Hicks2014-vp_Hick2014-Hk-vpvs_BAZ-230-250.nc') as model:
        model_slice = model.interp(longitude = x, latitude = y, method = 'linear', kwargs={"fill_value": None})
        model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_sub_dist})
        model_slice['depth'] = -model_slice['depth']

        # truncate model_slice values to be within scale range
        model_slice['Amplitude'].values = np.clip(model_slice['Amplitude'].values, -0.5, 0.5)

        # nan grid space with bins less than bin hit limit
        model_slice['Amplitude'].values[np.where(model_slice['Bin_Hits'] < bin_hits_limit)] = np.nan
        model_slice['Amplitude'].values[np.where(model_slice['Bin_Hits_std'] < std_hits_limit)] = np.nan
        model_slice['Amplitude'].values[np.where(model_slice['Bootstrap_std'] > std_limit)] = np.nan

    grid_color = pygmt.makecpt(cmap = '../DATA/MAPPING/no_green.cpt',   series = [-0.5, 0.5], continuous = True)
    fig.grdimage(grid = model_slice['Amplitude'])

    fig.plot(x = slab_moho_conversion_depth[:,0], y = slab_moho_conversion_depth[:,1], pen = '1p,black,--')



    with pygmt.config(FONT_ANNOT_PRIMARY="20p", FONT_LABEL="20p"):
        fig.colorbar(cmap = True, position = 'JMR+w{}/0.25c+o0.25c/0c'.format(fig_y), frame = 'xa.5f.1+lAmplitude')


    fig.basemap(region=grid_region, projection=prj, frame=['WSne', 'xa100f10g50+lProfile Distance (km)', 'ya20f10g50+lDepth b.s.l. (km)'])


fig.show()





#%% Summary Figure



x_dist = np.arange(30, 180, 10)
synth_rfs = Stream()
for k, x in enumerate(x_dist):

    print("\r", end="")
    print("Calculating Synthetics: {:.1%} ".format(k/(len(x_dist))), end="")

    line_i = closest_idx(x, line_dist)
    line_coord = line[line_i]

    slab_depth = slab.interp(longitude = line_coord[0], latitude = line_coord[1], method = 'linear').z.values
    slab_dip= slab_geometry.interp(longitude = line_coord[0], latitude = line_coord[1], method = 'linear').dip.values
    slab_strike = slab_geometry.interp(longitude = line_coord[0], latitude = line_coord[1], method = 'linear').strike.values

    slab_depth = np.round(slab_depth) * 1000
    slab_dip = np.round(slab_dip)

    z = [0, slab_depth, slab_depth + 10 ]
    thickn = [(z[i+1] - z[i])  for i in range(len(z) - 1)]
    thickn.append(0)


    ### Realistic Model
    vp = [6800, 5800, 8000]
    vpvs = [1.75, 1.82, 1.79 ]
    dip = [0, slab_dip, slab_dip]
    strike = np.round(slab_strike)
    rho = 2800

    ### Stable Model
    # vp = [6400, 5800, 7800]
    # vpvs = [1.75, 1.82, 1.79 ]
    # dip = [0, 30, 30]
    # strike = 0
    # rho = 2800


    #Build a 1-layer-over-half-space subsurface model
    model = Model(
        thickn = thickn,  # m; half-space thickness is irrelevant
        rho = rho,  # kg/m^3
        vs = list(np.array(vp) / np.array(vpvs)),  # m/s
        vp= vp,  # m/s
        dip = dip,
        strike = strike,
    )
    # model.plot()


    # ray design
    baz = np.append(np.arange(220, 270, 5), np.arange(335, 360, 5)) # consistent with data
    # baz= np.arange(0, 360, 10) # idealic full coverage
    slow = 0.06
    geometry = Geometry(baz=baz, slow=slow)
    # geometry.plot()

    control = Control(dt=1./20., npts=3000, align="P", rot='RTZ', mults = 0)
    result = run(model, geometry, control, rf=True)
    result.filter('rfs', 'lowpass', freq=1, zerophase=True, corners=2)



    sts = Stream()
    for tr in result.rfs:
        rf = tr.select(channel = 'RFR')[0]
        sts += rf




    synth_rfs += sts.stack()



# Plot result

t_depth = 10
fig_x = 10
fig_y = 5
print('\nPlotting Synthetic Results')

slab_moho_conversion = []
slab_top_conversion = []

fig = pygmt.Figure()
with pygmt.config(FONT = '12p'):
    gmt_region = [0, 190, -0.5, t_depth]
    fig.basemap(region=gmt_region, projection=f'X{fig_x}c/-{fig_y}c', frame=['Wsne', 'xafg+lProfile Distance (km)', 'yafg+lPs-Delay Time (s)'])


    for i, rf in enumerate(synth_rfs):
        try:
            data =( rf.data *300)  + x_dist[i]

            rf_time = rf.times() + rf.stats.taxis[0]

            slab_top_conversion.append([x_dist[i], rf_time[np.where(data == data[rf_time>2].min())[0][0]]])
            slab_moho_conversion.append([x_dist[i], rf_time[np.where(data == data[rf_time>2].max())[0][0]]])

            pos = data.copy()
            neg = data.copy()

            pos[pos < x_dist[i]] = x_dist[i]
            neg[neg > x_dist[i]] = x_dist[i]

            fig.plot(x = pos, y = rf_time,pen = '0.1p,black', fill = 'red', transparency = 20)
            fig.plot(x = neg, y = rf_time,pen = '0.1p,black', fill = 'blue', transparency = 20)

        except:
            pass

    slab_top_conversion = np.array(slab_top_conversion)
    slab_moho_conversion = np.array(slab_moho_conversion)
    # fig.plot(x = slab_top_conversion[:,0], y = slab_top_conversion[:,1], pen = '1p,black,--')
    fig.plot(x = slab_moho_conversion[:,0], y = slab_moho_conversion[:,1], pen = '1p,black,--')






    #-------------------------------



    fig.shift_origin(yshift = f'-{fig_y+0.5}c')


    print('Plotting Observed results')

    fig.basemap(region=gmt_region, projection=f'X{fig_x}c/-{fig_y}c', frame=['WSne', 'xafg+lProfile Distance (km)', 'yafg+lPs-Delay Time (s)'])

    nets, stlas, stlos, stels, tr_id = [],[],[],[],[]    # resetting stlas and stlos list to queried data
    for i, tr in enumerate(st_real):
        nets.append(tr.stats.network)
        stlas.append(stations.Latitude[stations.Code == '{}-{}'.format(tr.stats.network, tr.stats.station)].values[0])
        stlos.append(stations.Longitude[stations.Code == '{}-{}'.format(tr.stats.network, tr.stats.station)].values[0])
        stels.append(grid.sel(lat = stlas[-1], lon = stlos[-1], method = 'nearest').values.item())
        tr_id.append(i)

    select_data = pd.DataFrame({'longitude': np.squeeze(stlos), 'latitude': np.squeeze(stlas), 'stel': np.squeeze(stels), 'tr_id': np.array(tr_id)})
    select_data = select_data.sort_values('longitude')

    used_data = []
    for i in range(len(joints_sub) - 1):
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
           in_range = center_tree.query_ball_point(x = np.c_[select_data.longitude, select_data.latitude], r = 9 / degrees2kilometers(1))
           in_range =[bool(in_range[k]) for k in range(len(in_range))]
           in_range = select_data[in_range].tr_id.values

           st_stack = Stream()
           for trace in in_range:
               st_stack += st_real[trace].copy()

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

           fig.plot(x = pos, y = time_trim,pen = '0.1p,black', no_clip = False, fill = 'red', transparency = 20)
           fig.plot(x = neg, y = time_trim,pen = '0.1p,black', no_clip = False, fill = 'blue', transparency = 20)

    # fig.plot(x = slab_top_conversion[:,0], y = slab_top_conversion[:,1], pen = '1p,black,--')
    fig.plot(x = slab_moho_conversion[:,0], y = slab_moho_conversion[:,1], pen = '1p,black,--')





    #-------------------------------------------------


    fig.shift_origin(xshift = f'{fig_x}+1c')


    joints =  np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_Line.txt', delimiter = ',')
    joints_sub = joints[0:2]

    joints_sub_dist, line_sub, line_sub_dist = Return_Line_Elements(joints_sub)

    x = xr.DataArray(line_sub[:,0], dims='Distance_Along_Trend')
    y = xr.DataArray(line_sub[:,1], dims='Distance_Along_Trend')

    fig_y2 = fig_y*2 + 0.5
    aspect = 1
    z_lim = 120
    x_lim = 190
    ratio = z_lim / fig_y2    # ratio of km to cm
    fig_x2 = x_lim / (ratio * aspect)

    # require a minimum number of bin hits to be present within the bin radius and bin std radius
    # to be plotted
    bin_hits_limit = 10
    std_hits_limit = 1
    std_limit = 0.15



    grid_region = [0, x_lim, 0, z_lim]
    prj = f'X{fig_x2}c/-{fig_y2}c'
    fig.basemap(region=grid_region, projection=prj, frame=['wsne'])

    with xr.open_dataset('../DATA/MAPPING/nc_models/CCP_TANGO_South_G-5.0_FZB_CSEM2-Hicks2014-vp_Hick2014-Hk-vpvs.nc') as model:
        model_slice = model.interp(longitude = x, latitude = y, method = 'linear', kwargs={"fill_value": None})
        model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_sub_dist})
        model_slice['depth'] = -model_slice['depth']

        # truncate model_slice values to be within scale range
        model_slice['Amplitude'].values = np.clip(model_slice['Amplitude'].values, -0.5, 0.5)

        # nan grid space with bins less than bin hit limit
        model_slice['Amplitude'].values[np.where(model_slice['Bin_Hits'] < bin_hits_limit)] = np.nan
        model_slice['Amplitude'].values[np.where(model_slice['Bin_Hits_std'] < std_hits_limit)] = np.nan
        model_slice['Amplitude'].values[np.where(model_slice['Bootstrap_std'] > std_limit)] = np.nan

    grid_color = pygmt.makecpt(cmap = '../DATA/MAPPING/no_green.cpt',   series = [-0.5, 0.5], continuous = True)
    fig.grdimage(grid = model_slice['Amplitude'],region=grid_region, projection=prj)

    fig.plot(x = slab_moho_conversion_depth[:,0], y = slab_moho_conversion_depth[:,1], region=grid_region, projection=prj, pen = '1p,black,--')

    with pygmt.config(FONT_ANNOT_PRIMARY="10p", FONT_LABEL="10p"):
        fig.colorbar(cmap = True, position = 'jBR+o1c/0.5c+w5/0.25c+h+m'.format(fig_y), frame = 'xa.5f.1+lAmplitude')



    fig.basemap(region=grid_region, projection=prj, frame=['wSnE', 'xa20f10g50+lProfile Distance (km)', 'ya20f10g50+lDepth b.s.l. (km)'])



fig.show()

fig.savefig('../FIGURES/Figure5.png', dpi = 600)





