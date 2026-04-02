# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 17:54:13 2026

@author: 7418888
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

os.chdir('C:/Users/7418888/Documents/TANGO/TANGO_South/SCRIPTING')
# os.chdir('/tango/bradford/TANGO_South/SCRIPTING')

plt.close('all')



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

def Hk_solve(d):
    
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
    Hk_station, H_tr_vals, k_tr_vals = Hk_classic(st, vp = vp_base, w = weights, H = H, k = k)
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


fig_x = 20
fig_sub_x = 5


data = [ 'XM.CSB10', 'XM.CSB14', 'XM.CSB07', 'XM.CSB08', 'ZR',
    'XM.ASB01', 'XM.ASB12', 'XM.ASB02', 'XM.ASB10', 'XM.ASB03', 'XM.ASB04', 'XM.ASB05', 'XM.ASB21', 'XM.ASB09']

vp_base = 6.4
weights = (0.4, 0.3, 0.3)
H = (35, 55)
k = (1.7, 1.9)


fig = pygmt.Figure()
cmap = pygmt.makecpt(cmap = 'jet', series = [0,1])

with pygmt.config(FONT = '14p'):
    
    region = [np.min(H), np.max(H), np.min(k), np.max(k)]
    prj = f'X{fig_sub_x}c/{fig_sub_x}c'

    with fig.subplot(nrows=5, ncols=3, figsize=(f'{fig_x}c', f"{fig_x+10}c"), frame="lrtb"):
        for j in range(len(data)):
            with fig.set_panel(panel=j):
                fig.basemap(region = region,
                            projection = prj, frame = ['WSen', 'xa10fg5', 'yafg.05'])

        
                Hk_station, max_data, H_std, k_std, num_traces, st = Hk_solve(data[j])
                Hk_max = Hk_station.max()
                Hk_min = Hk_station.min()
                Hk_station = (Hk_station / Hk_max)
                
                
                fig.grdimage(grid = Hk_station, cmap = cmap, region = region, projection = prj)
                fig.grdcontour(grid = Hk_station, levels = [Hk_station.values.max() * 0.90], 
                               pen = '0.5p,black', region = region, projection = prj)

                fig.plot(x = max_data.H, y = max_data.k,region = region, projection = prj,
                         style = 'c.2c', fill = 'green', pen = '1p,black')


                fig.plot(x = [max_data.H - H_std, max_data.H + H_std], y = [max_data.k, max_data.k],  region = region, projection = prj, pen = '1p,black')
                fig.plot(x = [max_data.H , max_data.H ], y = [max_data.k - k_std, max_data.k + k_std],region = region, projection = prj,  pen = '1p,black')

                fig.plot(x = [max_data.H, max_data.H], y = [Hk_station.k.values.min(), max_data.k],   region = region, projection = prj, pen = '.5p,white')
                fig.plot(x = [Hk_station.H.values.min(), max_data.H], y = [max_data.k, max_data.k],   region = region, projection = prj, pen = '.5p,white')

                fig.text(x = 49, y = 1.71, text = data[j], fill = 'white', font = '10p,Helvetica', pen = '0.5p,black', region = region, projection = prj,)

                fig.text(x = 50, y = 1.89, text = f'H: {np.round(max_data.H.values, 1)}, k:{np.round(max_data.k.values, 2)}', 
                         font = '10p,Helvetica', fill = 'white', pen = '0.5p,black', no_clip = True, region = region, projection = prj,)

    



    fig.shift_origin(xshift = f'{fig_x+0.5}c')
    
    region = [-1, 30, -1.2, 1.2]
    prj = f'X9c/{fig_sub_x/1.5}c'

    with fig.subplot(nrows=7, ncols=2, figsize=(f'{fig_x}c', f"{fig_x+10}c"), frame="lrtb"):
        for j in range(len(data)):
            with fig.set_panel(panel=j):
                fig.basemap(region = region,
                            projection = prj, frame = ['WSen', 'xaf', 'yaf'])

    
                Hk_station, max_data, H_std, k_std, num_traces, st = Hk_solve(data[j])
                Hk_max = Hk_station.max()
                Hk_min = Hk_station.min()
                Hk_station = (Hk_station / Hk_max)
                
                
                data_stack = st[0].data
                data_time = st[0].times() - 10
                fig.plot(x = data_time, y = data_stack, pen = '0.5p,gray',  region = region, projection = prj,)
            
                rayp = []
                for i in range(1,len(st)):
                    data_stack = np.vstack((data_stack, st[i].data))
                    rayp.append(st[i].stats.sac.user4)
                    fig.plot(x = data_time, y = data_stack[-1,:], pen = '0.5p,gray',  region = region, projection = prj,)
                rf_avg = np.mean(data_stack, axis = 0)
                rf_std = np.std(data_stack, axis = 0)
            
            
                fig.plot(x = data_time, y = rf_avg, pen = '1p,black',  region = region, projection = prj,)
                fig.plot(x = data_time, y = rf_avg + rf_std, pen = '.5p,black,--',  region = region, projection = prj,)
                fig.plot(x = data_time, y = rf_avg - rf_std, pen = '.5p,black,--',  region = region, projection = prj,)

                qb = np.sqrt( (1/(vp_base/max_data.k.values)) **2 - np.average(rayp)**2)
                qa = np.sqrt( (1/vp_base) **2 - np.average(rayp)**2)
    
                t1  = max_data.H.values *( qb - qa)
                t2  = max_data.H.values * (qb + qa)
                t3  = 2*max_data.H.values * qb
    
                fig.plot(x = [t1, t1], y = [-0.5, 0.5], pen = '1p,red',  region = region, projection = prj,)
                fig.plot(x = [t2, t2], y = [-0.5, 0.5], pen = '1p,blue',  region = region, projection = prj,)
                fig.plot(x = [t3, t3], y = [-0.5, 0.5], pen = '1p,green',  region = region, projection = prj,)
                fig.text(x = 25, y = -1, text = data[j], fill = 'white', font = '10p,Helvetica', pen = '0.5p,black',  region = region, projection = prj,)

        
fig.show()

fig.savefig('../FIGURES/SupplementaryFigure5.png', dpi = 600)


