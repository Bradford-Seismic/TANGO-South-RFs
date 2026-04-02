# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 08:27:09 2026

@author: 7418888
"""

import pygmt
import obspy
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import hilbert

from obspy import Stream, read

import os
import sys

import numpy as np
import pandas as pd
from glob import glob
import xarray as xr
from math import radians, sin, cos, sqrt, atan2



os.chdir('C:/Users/7418888/Documents/TANGO/TANGO_South/SCRIPTING')
# os.chdir('/tango/bradford/TANGO_South/SCRIPTING')

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


def Phase_Weighted_Stack(st, v=1, sm=False, sl=15):

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
    stc.stack()
    tr = stc[0]
    tr.data = tr.data*c**v
    return stc

def Coherency(st, v=1, sm=False, sl=15):

    # return just the coherency vector of a set of traces
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

    c = c ** v
    return c


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


def Hk_PWS(st, vp = 6.4, w = (0.7, 0.2, 0.1), H = (30,55), k = (1.7, 2)):


    # retrieve station coherency trace
    c = Coherency(st)


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

        t1 (tps) = H * sqrt(qb - qa)
        t2 (tppps) = H * sqrt(qb + qa)
        t3 (tppss) = 2H * sqrt(qb)


        """
        qb = np.sqrt( (1/(vp/k_grid)) **2 - p**2)
        qa = np.sqrt( (1/(np.ones(k_grid.shape)*vp)) **2 - p**2)

        t1  = H_grid * (qb - qa)
        t2  = H_grid * (qb + qa)
        t3  = 2*H_grid * (qb)


        rf_data = tr.data * c
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



### Data save material

H_results = []
H_std_results =[]
H_standard_mean_error = []
k_results = []
k_std_results = []
k_standard_mean_error = []
lon = []
lat = []
num_traces = []


### Data search and Hk criteria
data = [ 'XM.CSB10', 'XM.CSB14', 'XM.CSB07', 'XM.CSB08', 'ZR',]
# data = ['XM.ASB01', 'XM.ASB12', 'XM.ASB02', 'XM.ASB10', 'XM.ASB03', 'XM.ASB04', 'XM.ASB05', 'XM.ASB21', 'XM.ASB09']
# data = ['XM.ASB09']
G = '02.5'
vp_base = 6.4
weights = (0.4, 0.3, 0.3)
H = (35, 55)
k = (1.7, 1.9)

for d in data:

    print('station: {}'.format(d))

    files = glob(os.path.join(f'../DATA/TANGO_South/DATA_CCP/{d}.*.itr'))

    st = Stream()
    rayp = []
    stla = []
    stlo = []
    for f in files:
        st += read(f)
        st[-1] = tr_normalize(st[-1])
        rayp.append(st[-1].stats.sac.user4)
        stlo.append(st[-1].stats.sac.stlo)
        stla.append(st[-1].stats.sac.stla)


    lon.append(np.mean(stlo))
    lat.append(np.mean(stla))
    num_traces.append(len(st))

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
    Hk_station, H_tr_vals, k_tr_vals = Hk_classic(st, vp = vp_base, w = weights, H = H, k = k)
    max_data = Hk_station.where(Hk_station==Hk_station.max(), drop=True).squeeze()
    H_std = np.std(H_tr_vals)
    k_std = np.std(k_tr_vals)

    H_results.append(np.round(max_data.H.item(),3))
    k_results.append(np.round(max_data.k.item(),3))

    H_std_results.append(np.round(H_std,3))
    k_std_results.append(np.round(k_std,3))

    H_standard_mean_error.append(np.round(H_std / np.sqrt(len(st)), 3))
    k_standard_mean_error.append(np.round(k_std / np.sqrt(len(st)), 3))



    print('Classic Hk result:')
    print(max_data)

    qb = np.sqrt( (1/(vp_base/max_data.k.values)) **2 - np.average(rayp)**2)
    qa = np.sqrt( (1/vp_base) **2 - np.average(rayp)**2)

    t1  = max_data.H.values *( qb - qa)
    t2  = max_data.H.values * (qb + qa)
    t3  = 2*max_data.H.values * qb

    fig.plot(x = [t1, t1], y = [-0.5, 0.5], pen = '0.5p,red')
    fig.plot(x = [t2, t2], y = [-0.5, 0.5], pen = '0.5p,blue')
    fig.plot(x = [t3, t3], y = [-0.5, 0.5], pen = '0.5p,green')
    fig.text(x = 27, y = -1, text = d, fill = 'white', font = '10p,Helvetica', pen = '0.5p,black')
    fig.show()



    with pygmt.config(FONT = '14p'):
        fig2 = pygmt.Figure()
        fig2.basemap(region = [Hk_station.H.values.min(), Hk_station.H.values.max(), Hk_station.k.values.min() , Hk_station.k.values.max()],
                    projection = 'X10c/10c', frame = ['WSen', 'xafg', 'yafg'])
        cmap = pygmt.makecpt(cmap = 'jet', series = [0,np.ceil(Hk_station.values.max())])
        fig2.grdimage(grid = Hk_station, cmap = cmap)
        fig2.grdcontour(grid = Hk_station, levels = [Hk_station.values.max() * 0.90], pen = '0.5p,black')

        fig2.plot(x = max_data.H, y = max_data.k, style = 'c.2c', fill = 'green', pen = '1p,black')


        fig2.plot(x = [max_data.H - H_std, max_data.H + H_std], y = [max_data.k, max_data.k],  pen = '1p,black')
        fig2.plot(x = [max_data.H , max_data.H ], y = [max_data.k - k_std, max_data.k + k_std], pen = '1p,black')

        fig2.plot(x = [max_data.H, max_data.H], y = [Hk_station.k.values.min(), max_data.k],   pen = '.25p,white')
        fig2.plot(x = [Hk_station.H.values.min(), max_data.H], y = [max_data.k, max_data.k],   pen = '.25p,white')

        fig2.text(x = 52, y = 1.71, text = d, fill = 'white', font = '14p,Helvetica', pen = '0.5p,black')

        fig2.colorbar(cmap = cmap)


        fig2.show()

sys.exit('here')





#%% Spatial Kriging

gmt_region = np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_GridArea-ABCD.txt')
north, south, east, west = gmt_region
joints = np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_Line.txt', delimiter = ',')

krig_model = gstools.Exponential(dim=2, var=np.var(k_results), len_scale=1)
krig = gstools.krige.Simple(krig_model, mean=1.745, cond_pos=(lon, lat), cond_val=(k_results))

grid_x = np.arange(np.min(lon) - 0.2, np.max(lon) + 0.2, 0.1)
grid_y = np.arange(np.min(lat) - 0.2, np.max(lat) + 0.2, 0.1)

krig((grid_x, grid_y), mesh_type='structured')


k_krig_da = xr.DataArray(
        krig[0].T,
        coords={"longitude": grid_x, "latitude": grid_y},
        dims=["latitude", "longitude"],
        name="k"
    )


fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [west, east, south, north], projection = 'M15c', frame = ['WSne', 'afg', 'afg'])
    cmap = pygmt.makecpt(cmap = 'jet', series = [1.7, 2.0])
    fig.grdimage(grid = k_krig_da, cmap = True)
    fig.plot(x = lon, y = lat, style = 'i0.3c', fill = 'magenta', pen = '1p,black')
    fig.plot(x = joints[:,0], y = joints[:,1], pen = '1p,black')
    fig.text(x = joints[:,0], y = joints[:,1], text = list("ABCD"), font = '12p,Helvetica', fill = 'white', pen = '1p,black')
    fig.colorbar(cmap = True)

fig.show()



krig_model = gstools.Exponential(dim=2, var=np.var(H_results), len_scale=1, )
krig = gstools.krige.Simple(krig_model, mean=40, cond_pos=(lon, lat), cond_val=(H_results))

grid_x = np.arange(np.min(lon) - 0.2, np.max(lon) + 0.2, 0.1)
grid_y = np.arange(np.min(lat) - 0.2, np.max(lat) + 0.2, 0.1)

krig((grid_x, grid_y), mesh_type='structured')


H_krig_da = xr.DataArray(
        krig[0].T,
        coords={"longitude": grid_x, "latitude": grid_y},
        dims=["latitude", "longitude"],
        name="H"
    )


fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [west, east, south, north], projection = 'M15c', frame = ['WSne', 'afg', 'afg'])
    cmap = pygmt.makecpt(cmap = 'jet', series = [30, 60])
    fig.grdimage(grid = H_krig_da, cmap = True)
    fig.plot(x = lon, y = lat, style = 'i0.3c', fill = 'magenta', pen = '1p,black')
    fig.plot(x = joints[:,0], y = joints[:,1], pen = '1p,black')
    fig.text(x = joints[:,0], y = joints[:,1], text = list("ABCD"), font = '12p,Helvetica', fill = 'white', pen = '1p,black')
    fig.colorbar(cmap = True)

fig.show()


krig_model = gstools.Exponential(dim=2, var=np.var(H_std_results), len_scale=1)
krig = gstools.krige.Simple(krig_model, mean=5, cond_pos=(lon, lat), cond_val=(H_std_results))

grid_x = np.arange(np.min(lon) - 0.2, np.max(lon) + 0.2, 0.1)
grid_y = np.arange(np.min(lat) - 0.2, np.max(lat) + 0.2, 0.1)

krig((grid_x, grid_y), mesh_type='structured')


H_std_krig_da = xr.DataArray(
        krig[0].T,
        coords={"longitude": grid_x, "latitude": grid_y},
        dims=["latitude", "longitude"],
        name="H"
    )




model_encoding = {'longitude': {'dtype': 'float32', '_FillValue': None },
            'latitude': {'dtype': 'float32', '_FillValue': None },
            'k': {'dtype': 'float32', '_FillValue': 99999 , 'zlib': False}
            }
k_krig_ds = k_krig_da.to_dataset(name = 'k')
k_krig_ds.to_netcdf('../DATA/MAPPING/nc_models/Krigged_Hk_k.nc', encoding = model_encoding)

model_encoding = {'longitude': {'dtype': 'float32', '_FillValue': None },
            'latitude': {'dtype': 'float32', '_FillValue': None },
            'H': {'dtype': 'float32', '_FillValue': 99999 , 'zlib': False}
            }
H_krig_ds = H_krig_da.to_dataset(name = 'H')
H_krig_ds.to_netcdf('../DATA/MAPPING/nc_models/Krigged_Hk_H.nc', encoding = model_encoding)


model_encoding = {'longitude': {'dtype': 'float32', '_FillValue': None },
            'latitude': {'dtype': 'float32', '_FillValue': None },
            'H_std': {'dtype': 'float32', '_FillValue': 99999 , 'zlib': False}
            }
H_std_krig_ds = H_std_krig_da.to_dataset(name = 'H_std')
H_std_krig_ds.to_netcdf('../DATA/MAPPING/nc_models/Krigged_Hk_H_std.nc', encoding = model_encoding)



#%% Plot H results on cross section


x_lim = [0, 450]
y_lim = [-120, 0]


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
    line = np.array([0,0])  # holding values
    for i in range(len(joints)-1):
        l = np.linspace(joints[i], joints[i+1], int(num_points[i]))

        if i == 0: # append all values for first iteration
            line = np.vstack((line, l))

        else: # append only values after the first so we don't duplicate values
            line = np.vstack((line, l[1:len(l)]))


    line = line[1:len(line)] # remove holding value


    # cummulative distance along line
    d = np.array([0])
    for i in range(len(line)-1):
        d = np.append(d, haversine(line[i,1], line[i,0],line[i+1,1], line[i+1,0]))

    line_dist = np.cumsum(d)

    return joints_dist, line, line_dist

joints = np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_Line.txt', delimiter = ',')
joints_dist, line, line_dist  = Return_Line_Elements(joints)

x = xr.DataArray(line[:,0], dims='Distance_Along_Trend')
y = xr.DataArray(line[:,1], dims='Distance_Along_Trend')

gmt_region = np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_GridArea-ABCD.txt')
north, south, east, west = gmt_region
gmt_region = '{}/{}/{}/{}'.format(west, east, south, north)
grid = pygmt.datasets.load_earth_relief(resolution="01m", region=gmt_region)


fig_x = 20   # length of figure in cm
aspect = 1        # dimensions of x-axis km to y-axis km

xdist = x_lim[1] - x_lim[0]
ydist = y_lim[1] - y_lim[0]

ratio = xdist / fig_x     # ratio of km to cm
fig_y = ydist / (ratio * aspect)


stat_table = pd.read_csv('../DATA/SPREADSHEETS/TANGO_South_PassingStations.csv')
codes = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i]) for i in range(len(stat_table))]
stat_table['Code'] = codes

contributing_stations = pd.read_csv('../DATA/SPREADSHEETS/Contributing_Stations.csv')
stat_table = stat_table[stat_table.Code.isin(contributing_stations.Code)]
stat_table.reset_index(drop = True, inplace = True)


def plot_topo(fig, joints, fig_x, fig_y, Numerals = 'ABCDEFG'):

    joints_dist, line, line_dist  = Return_Line_Elements(joints)


    fig.shift_origin(yshift = '+h')
    with pygmt.config(FONT = '16p', MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p/28p"):
        fig.basemap(region = [x_lim[0], x_lim[1], 0,6], projection = 'X{}c/{}c'.format(fig_x, fig_y / 5), frame = ['WsNe', 'xa50f10+lProfile Distance (km)', 'ya5f1+e+lElev. (km)'])


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

                fig.plot(x=track.p + joints_dist[i],
                         y=stat_elev_interp.values/1000, style='i0.3c', fill='cyan', pen='0.3,black')
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





        fig.text(x = joints_dist, y = np.ones(len(joints_dist)) * 4.9, text = list(Numerals[0:len(joints_dist)]), font = '14.p, Helvetica-Bold', fill = 'white', pen = '0.3p,black', no_clip = True)


    return fig



fig = pygmt.Figure()    # initialize the map
with pygmt.config(FONT="16p", MAP_ANNOT_OFFSET="10p", MAP_LABEL_OFFSET="10p"):

    grid_region = [x_lim[0], x_lim[1], y_lim[0], y_lim[1]]
    prj = 'X{}/{}'.format(fig_x, fig_y)

    fig.basemap(region=grid_region, projection=prj, frame=['Wsne', 'xa50f10g50+lProfile Distance (km)', 'ya20f10g50+lDepth (km)'])

    slab = xr.open_dataset('../DATA/MAPPING/sam_slab2_dep_02.23.18.grd')
    slab.coords['longitude'] = slab.x - 360; slab.coords['latitude'] = slab.y
    slab = slab.swap_dims({'x': 'longitude', 'y': 'latitude'})

    slab_slice = slab.interp(longitude = x, latitude = y, method = 'linear')
    slab_slice = slab_slice.assign_coords({'Distance_Along_Trend': line_dist})

    slab_x = slab_slice['Distance_Along_Trend'].values
    slab_y = slab_slice.z.values
    fig.plot(x=  slab_x, y = slab_y, pen = '1p,black,--')

    dem = grid.interp(lat = y, lon = x, method = 'linear') / 1000
    dem = dem.assign_coords({'Distance_Along_Trend': line_dist})


    H_slice = H_krig_da.interp(longitude = x, latitude = y, method = 'linear', kwargs={"bounds_error": False, "fill_value": np.nan})
    H_slice = H_slice.assign_coords({'Distance_Along_Trend': line_dist})

    H_std_slice = H_std_krig_da.interp(longitude = x, latitude = y, method = 'linear', kwargs={"bounds_error": False, "fill_value": np.nan})
    H_std_slice = H_std_slice.assign_coords({'Distance_Along_Trend': line_dist})

    k_slice = k_krig_da.interp(longitude = x, latitude = y, method = 'linear', kwargs={"bounds_error": False, "fill_value": np.nan})
    k_slice = k_slice.assign_coords({'Distance_Along_Trend': line_dist})
    k_slice = k_slice.expand_dims(depth=np.arange(y_lim[0], y_lim[1], 0.2))
    k_slice = k_slice.where(k_slice.depth >= -H_slice+dem.values)

    cmap = pygmt.makecpt(cmap = 'jet', series = [1.7, 2.0])
    fig.plot(x= H_slice['Distance_Along_Trend'].values, y = -H_slice.values+dem.values, pen = '2p,black,-.-')

    fig.grdimage(grid = k_slice, cmap = True, nan_transparent = True)
    fig.colorbar(cmap = True, position = 'JMR+w{}/0.25c+o0.25c/0c'.format(fig_y), frame = 'xa0.05f0.025')

    fig.plot(x= H_slice['Distance_Along_Trend'].values, y = -H_slice.values+dem.values + H_std_slice, pen = '.5p,black,--')
    fig.plot(x= H_slice['Distance_Along_Trend'].values, y = -H_slice.values+dem.values - H_std_slice, pen = '.5p,black,--')

    # overlying grid
    fig.basemap(region=grid_region, projection=prj, frame=['wsne', 'xa50f10g25+lProfile Distance (km)', 'ya20f10g25+lDepth (km)'])

fig = plot_topo(fig, joints, fig_x, fig_y)




fig.show()
