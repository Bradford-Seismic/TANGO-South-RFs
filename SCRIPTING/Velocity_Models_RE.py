#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 13:31:08 2024

@author: bradford
"""


import numpy as np
import pandas as pd
import os
import sys
import subprocess
import shutil
import obspy
import obspy.taup.taup_geo as taup
from obspy.taup import TauPyModel
import matplotlib.pyplot as plt
from matplotlib import image
from obspy.core import UTCDateTime as utc
from obspy.core import Stream, read
from obspy.io.sac import SACTrace
import obspy.taup.taup_geo as taup
from glob import glob
import pygmt
import math
import xarray as xr
import rioxarray
import geopandas as gpd
from shapely.vectorized import contains
from shapely import geometry
from shapely.geometry import Point, Polygon
from math import radians, sin, cos, atan2, sqrt
from scipy.signal import savgol_filter
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

active_dir = 'C:/Users/7418888/Documents/TANGO/TANGO_South/SCRIPTING'
# active_dir = '/tango/bradford/TANGO_South/SCRIPTING/'

os.chdir(active_dir)


"""
    The goal of this script is to develop 3D velocity models using the constraints provided by other
    data, and combine separate models to create a single averaged model


"""





#%% Look at the different datasets


"""
The datasets we want to use include the CSEM2 vp model, our determined H-k
estimates for the backarc, and the vp and vp/vs data from Hicks et al 2014

"""

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




### (1) set up joints for cross section

gmt_region = np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_GridArea.txt')
n,s,e,w = gmt_region

joints =  np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_Line.txt', delimiter = ',')

joints_dist, line, line_dist = Return_Line_Elements(joints)

fig_x = 20
depth = 200
aspect = 1

ratio = line_dist.max() / 20 # km per figure cm
fig_y = depth/ratio

x = xr.DataArray(line[:,0], dims='Distance_Along_Trend')
y = xr.DataArray(line[:,1], dims='Distance_Along_Trend')



slab = xr.open_dataset('../DATA/MAPPING/sam_slab2_dep_02.23.18.grd')
slab.coords['longitude'] = slab.x - 360; slab.coords['latitude'] = slab.y
slab = slab.swap_dims({'x': 'longitude', 'y': 'latitude'})

slab_slice = slab.interp(longitude = x, latitude = y, method = 'linear')
slab_slice = slab_slice.assign_coords({'Distance_Along_Trend': line_dist})

slab_x = slab_slice['Distance_Along_Trend'].values
slab_y = -slab_slice.z.values



### (2) read velocity data in their original data format

m1 = 'CSEM2-SouthAmerica.v2024.12.01.r0.0-n4.nc'
m2 = 'SHicks_Maule_3-D_velocity_model_latlong.xyzv'
m3 = 'Krigged_Hk_k.nc'

vp_min = 5.0



### (3) plot models along cross section


# CSEM2 --> .nc file
fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [0, line_dist.max(), 0, depth], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    with xr.open_dataset(os.path.join('../DATA/MAPPING/nc_models', m1)) as model:

        model_slice = model.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None}).VPV
        model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})
        fig.plot(x=  slab_x, y = slab_y, pen = '1p,black,--')

        grid_color = pygmt.makecpt(cmap = 'jet', series = [vp_min, 12.0, 0.2], reverse = True)
        fig.grdimage(grid = model_slice, cmap = True)

        fig.plot(x=  slab_x, y = slab_y, pen = '1p,black,--')

        fig.text(position = 'BR', text = f'{m1} - vp')




        fig.colorbar(cmap = True, position = 'JMR+w{}/0.25c+o0.25c/0c'.format(fig_y), frame = 'xa1f1+l{}'.format('vp'))

fig.show()

# save the model in dataset format in memory
CSEM2_vp = model.VPV.copy()






# Hicks et al (2014) --> .xyzf file

fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [0, line_dist.max(), 0, depth], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    # read the model information in its native xyzf format, then grab the vp and vp/vs data
    model = pd.read_csv(os.path.join('../DATA/MAPPING/nc_models', m2), skiprows=21, header = None,sep='\\s+',
                        names = ['longitude', 'latitude', 'depth', 'vp', 'vs', 'vp/vs', 'vp_spread', 'vp/vs_spread'])

    model = model[['longitude', 'latitude', 'depth', 'vp', 'vp/vs']]
    model = model[(model.longitude > w) & (model.longitude < e)]
    model = model[(model.latitude > s) & (model.latitude < n)]

    # requires downsampling, otherwise it will take forever
    model = model.sample(frac = .25, random_state = 1)

    model = model.reset_index(drop = True)


    # the data is in an irregular gridded format, converting to a xr dataset
    # will blow the memory space and introduce a lot of nan numbers
    # use scipy interp griddata to convert it into a regular gridded format


    lon_grid = np.arange(model.longitude.min(), model.longitude.max(), 0.1)
    lat_grid = np.arange(model.latitude.min(), model.latitude.max(), 0.1)
    depth_grid = np.arange(model.depth.min(), model.depth.max(), 5)
    LON, LAT, DEPTH = np.meshgrid(
        lon_grid,
        lat_grid,
        depth_grid,
        indexing="ij"
    )

    grid_values = griddata(
        model[['longitude', 'latitude', 'depth']].values,
        model.vp.values,
        (LON, LAT, DEPTH),
        method="linear"
    )

    model = xr.Dataset(
    {"vp": (["depth","longitude", "latitude"], grid_values.transpose(2,0,1))},
    coords={
        "longitude": lon_grid,
        "latitude": lat_grid,
        "depth": depth_grid
    })




    model_slice = model.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None}).vp
    model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})

    grid_color = pygmt.makecpt(cmap = 'jet', series = [vp_min, 12.0, 0.2], reverse = True)
    fig.grdimage(grid = model_slice, cmap = True)
    fig.plot(x=  slab_x, y = slab_y, pen = '1p,black,--')

    fig.text(position = 'BR', text = f'{m2} - vp')

    fig.colorbar(cmap = True, position = 'JMR+w{}/0.25c+o0.25c/0c'.format(fig_y), frame = 'xa1f1+l{}'.format('vp'))

fig.show()

# save the model in dataset format in memory
Hicks_vp = model.vp.copy()

# save the model as a .nc file
model_encoding = {'depth': {'dtype': 'float32', '_FillValue': None },
            'longitude': {'dtype': 'float32', '_FillValue': None },
            'latitude': {'dtype': 'float32', '_FillValue': None },
            'vp': {'dtype': 'float32', '_FillValue': 99999 , 'zlib': False}
            }

Hicks_vp.to_netcdf('../DATA/MAPPING/nc_models/Hicks2014_vp.nc', encoding = model_encoding)




fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [0, line_dist.max(), 0, depth], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    # read the model information in its native xyzf format, then grab the vp and vp/vs data
    model = pd.read_csv(os.path.join('../DATA/MAPPING/nc_models', m2), skiprows=21, header = None,sep='\\s+',
                        names = ['longitude', 'latitude', 'depth', 'vp', 'vs', 'vp/vs', 'vp_spread', 'vp/vs_spread'])

    model = model[['longitude', 'latitude', 'depth', 'vp', 'vp/vs']]
    model = model[(model.longitude > w) & (model.longitude < e)]
    model = model[(model.latitude > s) & (model.latitude < n)]

    model = model.sample(frac = .25, random_state = 1)

    model = model.reset_index(drop = True)


    # the data is in an irregular gridded format, converting to a xr dataset
    # will blow the memory space and introduce a lot of nan numbers
    # use scipy interp griddata to convert it into a regular gridded format


    lon_grid = np.arange(model.longitude.min(), model.longitude.max(), 0.1)
    lat_grid = np.arange(model.latitude.min(), model.latitude.max(), 0.1)
    depth_grid = np.arange(model.depth.min(), model.depth.max(), 5)
    LON, LAT, DEPTH = np.meshgrid(
        lon_grid,
        lat_grid,
        depth_grid,
        indexing="ij"
    )

    grid_values = griddata(
        model[['longitude', 'latitude', 'depth']].values,
        model['vp/vs'].values,
        (LON, LAT, DEPTH),
        method="linear"
    )

    model = xr.Dataset(
    {"vpvs": (["depth","longitude", "latitude"], grid_values.transpose(2,0,1))},
    coords={
        "longitude": lon_grid,
        "latitude": lat_grid,
        "depth": depth_grid
    })




    model_slice = model.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None})['vpvs']
    model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})

    grid_color = pygmt.makecpt(cmap = 'jet', series = [1.6, 2.0, 0.02])
    fig.grdimage(grid = model_slice, cmap = True)
    fig.plot(x=  slab_x, y = slab_y, pen = '1p,black,--')

    fig.text(position = 'BR', text = f'{m2} - vp/vs')

    fig.colorbar(cmap = True, position = 'JMR+w{}/0.25c+o0.25c/0c'.format(fig_y), frame = 'xa.05f.05+l{}'.format('vp/vs'))

fig.show()

# save the model in dataset format in memory
Hicks_vpvs = model['vpvs'].copy()

model_encoding = {'depth': {'dtype': 'float32', '_FillValue': None },
            'longitude': {'dtype': 'float32', '_FillValue': None },
            'latitude': {'dtype': 'float32', '_FillValue': None },
            'vpvs': {'dtype': 'float32', '_FillValue': 99999 , 'zlib': False}
            }

Hicks_vpvs.to_netcdf('../DATA/MAPPING/nc_models/Hicks2014_vpvs.nc', encoding = model_encoding)




# H-k results --> .nc file, but needs depth constraint from CSEM2

fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [0, line_dist.max(), 0, depth], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    with xr.open_dataset(os.path.join('../DATA/MAPPING/nc_models', m3)) as model:


        # restrict the hk data within the gmt_region
        model = model.sel(longitude = slice(w,e), latitude = slice(s,n))

        # pull lat/long data and make gridded area
        lon, lat = model.longitude.values, model.latitude.values


        # now read the CSMEM2 data and also restrict the domain, and interpolate
        with xr.open_dataset('../DATA/MAPPING/nc_models/{}'.format(m1)) as mod:

            mod = mod.sel(longitude = slice(w,e), latitude = slice(s,n))
            mod = mod.interp(longitude = lon, latitude = lat, depth = np.arange(0,200,1), method = 'linear', kwargs={"fill_value": None} )

            # now expand the vpvs model dimension in the CSEM2 depth

            model = model.k.broadcast_like(mod)

            # lastly restrcit the model vpvs in depth by value in CSEM2

            model = model.where(mod.VPV <= 7.2, other=1.79)


        model_slice = model.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None})
        model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})


        grid_color = pygmt.makecpt(cmap = 'jet', series = [1.6, 2.0, 0.02])
        fig.grdimage(grid = model_slice, cmap = True)
        fig.plot(x=  slab_x, y = slab_y, pen = '1p,black,--')

        fig.text(position = 'BR', text = f'{m3} - vp/vs')

        fig.colorbar(cmap = True, position = 'JMR+w{}/0.25c+o0.25c/0c'.format(fig_y), frame = 'xa.05f.05+l{}'.format('vp/vs'))



fig.show()
# save the model in dataset format in memory
Hk_vpvs = model.copy()
Hk_vpvs = Hk_vpvs.rename('vpvs')


model_encoding = {'depth': {'dtype': 'float32', '_FillValue': None },
            'longitude': {'dtype': 'float32', '_FillValue': None },
            'latitude': {'dtype': 'float32', '_FillValue': None },
            'vpvs': {'dtype': 'float32', '_FillValue': 99999 , 'zlib': False}
            }

Hk_vpvs.to_netcdf('../DATA/MAPPING/nc_models/Hk_vpvs.nc', encoding = model_encoding)



#%% Create 3D vp model with blended averaging


# (1) first, interpolate the data into a common domain

lon = np.arange(w, e+0.2, 0.1)
lat = np.arange(s, n+0.2, 0.1)
depth = np.arange(0, 200+5, 2)


LON, LAT = np.meshgrid(lon, lat, indexing="ij")


CSEM2 = CSEM2_vp.interp(longitude = lon,
                           latitude = lat,
                           depth = depth, method = 'linear',
                           kwargs={"fill_value": None})

Hicks = Hicks_vp.interp(longitude = lon,
                           latitude = lat,
                           depth = depth, method = 'linear',
                           kwargs={"fill_value": None})

slab_interp = slab.interp(longitude = lon,
                           latitude = lat,
                           kwargs={"fill_value": None})





# (2) load the spatial domain where we want to use the Hicks data

hicks_mask = gpd.read_file('../DATA/MAPPING/Hicks_model_cutoff.shp')
polygon = hicks_mask.geometry.unary_union
mask2d = contains(polygon, LON, LAT)

mask2d = xr.DataArray(
    mask2d,
    coords={"longitude": lon, "latitude": lat},
    dims=("longitude","latitude")
)

mask3d = mask2d.broadcast_like(Hicks)
Hicks_masked = Hicks.where(mask3d)

Hicks_masked = Hicks_masked.where(Hicks_masked.depth <= -slab_interp.z+10)

Hicks_masked = Hicks_masked.where(Hicks_masked.depth <= 70)



# (3) plot the domain on top of each other to show how they will fit

fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [0, line_dist.max(), 0, 200], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    grid_color = pygmt.makecpt(cmap = 'jet', series = [vp_min, 12.0, 0.2], reverse = True)

    CSEM2_slice = CSEM2.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None})
    CSEM2_slice = CSEM2_slice.assign_coords({'Distance_Along_Trend': line_dist})
    fig.grdimage(grid = CSEM2_slice, cmap = True)

    Hicks_slice = Hicks_masked.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None})
    Hicks_slice = Hicks_slice.assign_coords({'Distance_Along_Trend': line_dist})
    fig.grdimage(grid = Hicks_slice, cmap = True, nan_transparent = True)

    fig.plot(x=  slab_x, y = slab_y, pen = '1p,black,--')

    fig.colorbar(cmap = True, position = 'JMR+w{}/0.25c+o0.25c/0c'.format(fig_y), frame = 'xa1f1+l{}'.format('vp'))

    fig.text(position = 'BR', text = 'Overlapping - vp')

fig.show()




# (4) decompose the grids into masks

# make an array of 0 and 1 values, 1 for where data within Hicks et al does not exist
# and 0 where it does exist
Hicks_mask = Hicks_masked.copy()
Hicks_mask = xr.where(Hicks_mask.isnull(), 1, 0)


fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [0, line_dist.max(), 0, 200], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    grid_color = pygmt.makecpt(cmap = 'jet')


    Hicks_slice = Hicks_mask.interp(longitude = x, latitude = y, method='nearest', kwargs={"fill_value": None})
    Hicks_slice = Hicks_slice.assign_coords({'Distance_Along_Trend': line_dist})
    fig.grdimage(grid = Hicks_slice, cmap = True, nan_transparent = True)

    fig.plot(x=  slab_x, y = slab_y, pen = '1p,black,--')
    fig.colorbar(cmap = True, position = 'JMR+w{}/0.25c+o0.25c/0c'.format(fig_y), frame = 'xa1f1+l{}'.format('mask value'))

    fig.text(position = 'BR', text = 'Hicks 0/1 mask')


fig.show()


# Hicks mask is now a gridded dataset where all spots where we want to remove values from
# CSEM2 are 0, and all areas where we want to keep CSEM2 are 1

# Hicks keep has the original value where we want to keep the Hicks data, and is
# 0 everywhere else
Hicks_keep = Hicks_masked.where(~np.isnan(Hicks_masked), 0)

# Now mask the CSEM2 dataset
CSEM2_masked = CSEM2 * Hicks_mask
fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [0, line_dist.max(), 0, 200], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    grid_color = pygmt.makecpt(cmap = 'jet', series = [vp_min, 12.0, 0.2], reverse = True)

    # note that there are wierd low values around the clipped area
    # this is from the linear interpolation function and not a data issue

    CSEM2_slice = CSEM2_masked.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None})
    CSEM2_slice = CSEM2_slice.assign_coords({'Distance_Along_Trend': line_dist})
    fig.grdimage(grid = CSEM2_slice, cmap = True, nan_transparent = True)

    fig.plot(x=  slab_x, y = slab_y, pen = '1p,black,--')
    fig.colorbar(cmap = True, position = 'JMR+w{}/0.25c+o0.25c/0c'.format(fig_y), frame = 'xa1f1+l{}'.format('vp'))

    fig.text(position = 'BR', text = 'CSEM2 Masked data - vp')

fig.show()




# (5) Combine the two masked gridded data and apply a smoothing filter

merged_data = CSEM2_masked + Hicks_keep

fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [0, line_dist.max(), 0, 200], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    grid_color = pygmt.makecpt(cmap = 'jet', series = [vp_min, 12.0, 0.2], reverse = True)

    # note that there is a smoothness around the clipped area
    # this is from the linear interpolation function and not a data issue

    merged_slice = merged_data.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None})
    merged_slice = merged_slice.assign_coords({'Distance_Along_Trend': line_dist})
    fig.grdimage(grid = merged_slice, cmap = True, nan_transparent = True)

    fig.plot(x=  slab_x, y = slab_y, pen = '1p,black,--')
    fig.colorbar(cmap = True, position = 'JMR+w{}/0.25c+o0.25c/0c'.format(fig_y), frame = 'xa1f1+l{}'.format('vp'))

    fig.text(position = 'BR', text = 'Merged Dataset - vp')


fig.show()

merged_data_smoothed = xr.apply_ufunc(gaussian_filter,
    merged_data, kwargs={"sigma": (.5, .5, 0)})

fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [0, line_dist.max(), 0, 200], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    grid_color = pygmt.makecpt(cmap = 'jet', series = [vp_min, 12.0, 0.2], reverse = True)

    # note that there is a smoothness around the clipped area
    # this is from the linear interpolation function and not a data issue

    merged_slice = merged_data_smoothed.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None})
    merged_slice = merged_slice.assign_coords({'Distance_Along_Trend': line_dist})
    fig.grdimage(grid = merged_slice, cmap = True, nan_transparent = True)

    fig.plot(x=  slab_x, y = slab_y, pen = '1p,black,--')
    fig.colorbar(cmap = True, position = 'JMR+w{}/0.25c+o0.25c/0c'.format(fig_y), frame = 'xa1f1+l{}'.format('vp'))

    fig.text(position = 'BR', text = 'Merged Dataset Smoothed - vp')


fig.show()



# (6) save the merged model as a .nc file that we can use later

merged_data_smoothed = merged_data_smoothed.to_dataset(name = 'vp')

model_encoding = {'depth': {'dtype': 'float32', '_FillValue': None },
            'longitude': {'dtype': 'float32', '_FillValue': None },
            'latitude': {'dtype': 'float32', '_FillValue': None },
            'vp': {'dtype': 'float32', '_FillValue': 99999 , 'zlib': False}
            }


merged_data_smoothed.to_netcdf('../DATA/MAPPING/nc_models/BlendedModel_CSEM2-Hicks2014_vp.nc', encoding = model_encoding)

#%% Repeat with vp/vs data


# (1) first, interpolate the data into a common domain, keep grid data from previous part



Hk= Hk_vpvs.interp(longitude = lon,
                           latitude = lat,
                           depth = depth, method = 'linear',
                           kwargs={"fill_value": None})

# extra insert to cover mantle domain outside of inital model setup
Hk = Hk.where(~np.isnan(Hk), 1.79)

Hicks = Hicks_vpvs.interp(longitude = lon,
                           latitude = lat,
                           depth = depth, method = 'linear',
                           kwargs={"fill_value": None})

slab_interp = slab.interp(longitude = lon,
                           latitude = lat,
                           kwargs={"fill_value": None})





# (2) load the spatial domain where we want to use the Hicks data

hicks_mask = gpd.read_file('../DATA/MAPPING/Hicks_model_cutoff.shp')
polygon = hicks_mask.geometry.unary_union
mask2d = contains(polygon, LON, LAT)

mask2d = xr.DataArray(
    mask2d,
    coords={"longitude": lon, "latitude": lat},
    dims=("longitude","latitude")
)

mask3d = mask2d.broadcast_like(Hicks)
Hicks_masked = Hicks.where(mask3d)

Hicks_masked = Hicks_masked.where(Hicks_masked.depth <= -slab_interp.z+10)

Hicks_masked = Hicks_masked.where(Hicks_masked.depth <= 70)




# (3) plot the domain on top of each other to show how they will fit

fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [0, line_dist.max(), 0, 200], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    grid_color = pygmt.makecpt(cmap = 'jet', series = [1.6, 2.0, 0.02])

    Hk_slice = Hk.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None})
    Hk_slice = Hk_slice.assign_coords({'Distance_Along_Trend': line_dist})
    fig.grdimage(grid = Hk_slice, cmap = True)

    Hicks_slice = Hicks_masked.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None})
    Hicks_slice = Hicks_slice.assign_coords({'Distance_Along_Trend': line_dist})
    fig.grdimage(grid = Hicks_slice, cmap = True, nan_transparent = True)

    fig.plot(x=  slab_x, y = slab_y, pen = '1p,black,--')

    fig.colorbar(cmap = True, position = 'JMR+w{}/0.25c+o0.25c/0c'.format(fig_y), frame = 'xa1f1+l{}'.format('vp'))

    fig.text(position = 'BR', text = 'Overlapping - vp/vs')

fig.show()




# (4) decompose the grids into masks

# make an array of 0 and 1 values, 1 for where data within Hicks et al does not exist
# and 0 where it does exist
Hicks_mask = Hicks_masked.copy()
Hicks_mask = xr.where(Hicks_mask.isnull(), 1, 0)


fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [0, line_dist.max(), 0, 200], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    grid_color = pygmt.makecpt(cmap = 'jet')


    Hicks_slice = Hicks_mask.interp(longitude = x, latitude = y, method='nearest', kwargs={"fill_value": None})
    Hicks_slice = Hicks_slice.assign_coords({'Distance_Along_Trend': line_dist})
    fig.grdimage(grid = Hicks_slice, cmap = True, nan_transparent = True)

    fig.plot(x=  slab_x, y = slab_y, pen = '1p,black,--')
    fig.colorbar(cmap = True, position = 'JMR+w{}/0.25c+o0.25c/0c'.format(fig_y), frame = 'xa1f1+l{}'.format('mask value'))

    fig.text(position = 'BR', text = 'Hicks 0/1 mask')


fig.show()


# Hicks mask is now a gridded dataset where all spots where we want to remove values from
# Hk are 0, and all areas where we want to keep Hk are 1

# Hicks keep has the original value where we want to keep the Hicks data, and is
# 0 everywhere else
Hicks_keep = Hicks_masked.where(~np.isnan(Hicks_masked), 0)

# Now mask the Hk dataset
Hk_masked = Hk * Hicks_mask
fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [0, line_dist.max(), 0, 200], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    grid_color = pygmt.makecpt(cmap = 'jet', series = [1.6, 2.0, 0.02])

    # note that there are wierd low values around the clipped area
    # this is from the linear interpolation function and not a data issue

    Hk_slice = Hk_masked.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None})
    Hk_slice = Hk_slice.assign_coords({'Distance_Along_Trend': line_dist})
    fig.grdimage(grid = Hk_slice, cmap = True, nan_transparent = True)

    fig.plot(x=  slab_x, y = slab_y, pen = '1p,black,--')
    fig.colorbar(cmap = True, position = 'JMR+w{}/0.25c+o0.25c/0c'.format(fig_y), frame = 'xa1f1+l{}'.format('vp'))

    fig.text(position = 'BR', text = 'Hk Masked data - vp/vs')

fig.show()




# (5) Combine the two masked gridded data and apply a smoothing filter

merged_data = Hk_masked + Hicks_keep

fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [0, line_dist.max(), 0, 200], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    grid_color = pygmt.makecpt(cmap = 'jet', series = [1.6, 2.0, 0.02])

    # note that there is a smoothness around the clipped area
    # this is from the linear interpolation function and not a data issue

    merged_slice = merged_data.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None})
    merged_slice = merged_slice.assign_coords({'Distance_Along_Trend': line_dist})
    fig.grdimage(grid = merged_slice, cmap = True, nan_transparent = True)

    fig.plot(x=  slab_x, y = slab_y, pen = '1p,black,--')
    fig.colorbar(cmap = True, position = 'JMR+w{}/0.25c+o0.25c/0c'.format(fig_y), frame = 'xa1f1+l{}'.format('vp'))

    fig.text(position = 'BR', text = 'Merged Dataset - vp/vs')


fig.show()

merged_data_smoothed = xr.apply_ufunc(gaussian_filter,
    merged_data, kwargs={"sigma": (.5, .5, 0)})

fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):
    fig.basemap(region = [0, line_dist.max(), 0, 200], projection = 'X{}c/-{}c'.format(fig_x, fig_y), frame = ['Wsen', 'xa100f10+lProfile Distance (km)', 'ya20f10+lDepth (km)'])

    grid_color = pygmt.makecpt(cmap = 'jet', series = [1.6, 2.0, 0.02])

    # note that there is a smoothness around the clipped area
    # this is from the linear interpolation function and not a data issue

    merged_slice = merged_data_smoothed.interp(longitude = x, latitude = y, method='linear', kwargs={"fill_value": None})
    merged_slice = merged_slice.assign_coords({'Distance_Along_Trend': line_dist})
    fig.grdimage(grid = merged_slice, cmap = True, nan_transparent = True)

    fig.plot(x=  slab_x, y = slab_y, pen = '1p,black,--')
    fig.colorbar(cmap = True, position = 'JMR+w{}/0.25c+o0.25c/0c'.format(fig_y), frame = 'xa1f1+l{}'.format('vp'))

    fig.text(position = 'BR', text = 'Merged Dataset Smoothed - vp/vs')


fig.show()


# (6) save the merged model as a .nc file that we can use later

merged_data_smoothed = merged_data_smoothed.to_dataset(name = 'vpvs')

model_encoding = {'depth': {'dtype': 'float32', '_FillValue': None },
            'longitude': {'dtype': 'float32', '_FillValue': None },
            'latitude': {'dtype': 'float32', '_FillValue': None },
            'vpvs': {'dtype': 'float32', '_FillValue': 99999 , 'zlib': False}
            }


merged_data_smoothed.to_netcdf('../DATA/MAPPING/nc_models/BlendedModel_Hk-Hicks2014_vpvs.nc', encoding = model_encoding)


