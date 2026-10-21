#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 13 11:53:01 2026

@author: jimbradford
"""



import os
import numpy as np
import pandas as pd
import xarray as xr
import pygmt
from math import radians, sin, cos, sqrt, atan2
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import sys
import geopandas as gpd
import shapely
import rioxarray
from scipy.interpolate import griddata




os.chdir('.')

region = 'TANGO_South'






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

def rasterize(file):
    shape = rioxarray.open_rasterio(file)
    shape = shape.sel(band = 1).drop('band')
    shape.values[shape.values == 0] = np.nan

    
    shape_grid = shape.interp(x = grid.lon, y = grid.lat, method = 'nearest', kwargs={"fill_value": None})
    shape_grid.values = shape_grid.values * grid.values

    shape_grid.values = np.clip(shape_grid.values, np.nanmin(shape_grid.values), np.nanmax(shape_grid.values))


    return shape_grid



joints =  np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_Line.txt', delimiter = ',')
joints_dist, line, line_dist = Return_Line_Elements(joints)

# Geographic and depth limits
gmt_region = [joints[0,0], joints[-1,0], -36.5, -34.5, -100.001, 12]
w, e, s, n, base, top = gmt_region


x = xr.DataArray(line[:,0], dims='Distance_Along_Trend')
y = xr.DataArray(line[:,1], dims='Distance_Along_Trend')



dist_x = haversine(s,w,s,e)
figx = 20
ratio = (figx / dist_x)

dist_y = haversine(s,w,n,w)
figy = ratio * dist_y

aspect = 1
figz = ratio * aspect * np.abs(base)







# Plot full grid data


grid_scale = 4
grid = pygmt.datasets.load_earth_relief(resolution="03s", region=[w,e,s,n])
grid.values = (grid.values / 1000)
grid.values[grid.values>0] = grid.values[grid.values>0]*grid_scale
# grid.values[grid.values < 0] = np.nan
Lon, Lat = np.meshgrid(grid.lon.values, grid.lat.values)


for i in range(len(joints)-1):
    lons = np.unique(Lon[(Lon > joints[i,0]) & (Lon < joints[i+1,0])])
    lon_interp_func = interp1d([joints[i,0], joints[i+1,0]],[joints[i,1], joints[i+1,1]] , kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
    lats = lon_interp_func(lons)
 
    for j in range(len(lons)):
        grid.values[(Lon == lons[j]) & (Lat < lats[j])] = np.nan
    
    
# grid.values[(Lon < joints[:,0].min())] = np.nan
# grid.values[(Lon > joints[:,0].max())] = np.nan






## Sea level plane

sea_level = grid.copy()
sea_level.values = grid.values * 0


fig = pygmt.Figure()
with pygmt.config(FONT="10p", MAP_FRAME_TYPE="plain"):
    
    
    perspective = [155, 5]

    fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["WESNZ"], perspective=perspective, zsize="{}c".format(figz))
    fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["xg1", "yg1", "za20f10g20+lDepth (km)", "wesnZ"], perspective=perspective, zsize="{}c".format(figz))

    
    
    
    slab = xr.open_dataset('../DATA/MAPPING/sam_slab2_dep_02.23.18.grd')
    slab.coords['lon'] = slab.x - 360; slab.coords['lat'] = slab.y
    slab = slab.swap_dims({'x': 'lon', 'y': 'lat'})
    slab.z.values[slab.z.values < base] = np.nan
    Lon, Lat = np.meshgrid(slab.lon.values, slab.lat.values)
    for i in range(len(joints)-1):
        lons = np.unique(Lon[(Lon > joints[i,0]) & (Lon < joints[i+1,0])])
        lon_interp_func = interp1d([joints[i,0], joints[i+1,0]],[joints[i,1], joints[i+1,1]] , kind = 'linear', bounds_error = False, fill_value = 'extrapolate')
        lats = lon_interp_func(lons)
     
        for j in range(len(lons)):
            slab.z.values[(Lon == lons[j]) & (Lat > lats[j])] = np.nan

    # slab.z.values[(Lon < joints[:,0].min())] = np.nan
    # slab.z.values[(Lon > joints[:,0].max())] = np.nan
    
    
    pygmt.makecpt(cmap="lightgray", series=[-100, 0, 10], continuous=False)
    fig.grdview(
        grid=slab.z,
        region = gmt_region,
        perspective = True,
        surftype = 's',
        contourpen = '.1p',
        cmap = True,
        transparency = 50

    )


    fig.grdview(
        grid=sea_level,
        region = gmt_region,
        perspective = True,
        surftype="i",
        cmap = 'lightgray'
    )

    dem_color = pygmt.makecpt(cmap = 'bukavu', series = [np.nanmin(grid.values), np.nanmax(grid.values)])

    fig.grdview(
        grid=grid,
        shading = '+a135+nt0.6', 
        region = gmt_region,
        perspective = True,
        surftype="i",
        cmap= True,
    )
    
    
    shape_grid = rasterize('../DATA/MAPPING/Payenia_Volcanic_Field_Raster.tif')
    pygmt.makecpt(cmap="red", series=[np.nanmin(shape_grid.values)-1, np.nanmax(shape_grid.values)])
    fig.grdview(
        grid=shape_grid,
        shading = '+a135+nt0.6', 
        region = gmt_region,
        perspective = True,
        surftype="s",
        cmap= True,
        transparency = 90,
    )

    shape_grid = rasterize('../DATA/MAPPING/LagunaLlancanelo_Raster.tif')
    pygmt.makecpt(cmap="skyblue", series=[np.nanmin(shape_grid.values)-1, np.nanmax(shape_grid.values)])
    fig.grdview(
        grid=shape_grid,
        shading = '+a135+nt0.6', 
        region = gmt_region,
        perspective = True,
        surftype="s",
        cmap= True,
        transparency = 0,
    )

    
    stat_table = pd.read_csv('../DATA/SPREADSHEETS/{}_PassingStations.csv'.format(region))
    stat_table['Code'] = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i]) for i in range(len(stat_table))]

    contributing_stations = pd.read_csv('../DATA/MAPPING/Contributing_Stations_RE.csv')
    stat_table = stat_table[stat_table.Code.isin(contributing_stations.Code)].reset_index(drop = True)
    
    
    fig.plot3d(x = stat_table.stlo, 
               y = stat_table.stla,
               z = stat_table.stel/1000*grid_scale,
               style = 'u0.1c', pen = '0.2p,black', fill = 'gold',
               region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))

    code_category = ['1X', 'ZR', 'XM',]
    code_color = ['cyan', 'lightred', 'magenta']
    for i, code in enumerate(code_category):
        fig.plot3d(x = stat_table.stlo[stat_table.Code.str.contains(code)], 
                   y = stat_table.stla[stat_table.Code.str.contains(code)],
                   z = stat_table.stel[stat_table.Code.str.contains(code)]/1000*grid_scale,
                   style = 'u0.1c', pen = '0.2p,black', fill = code_color[i],
                   region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))


    fig.plot3d(x = joints[:,0], y = joints[:,1], z = np.ones(len(joints))*0,  style = 'o0.01c', pen = '0.2p,black', fill = 'green',
               region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))
    

    reidel = pd.read_csv('../DATA/MAPPING/ReidelHornig2025_EQs.csv')
    reidel = reidel[(reidel.Lat < -30) & (reidel.Lat > -40)][reidel.Lat_Error < 10][reidel.Lon_Error < 10][reidel.Dep_Error < 10]
    reidel.Dep = reidel.Dep
    eq_data = reidel[['Lon', 'Lat', 'Dep', 'id']]
    used_data = []
    for i in range(len(joints) - 1): # constricting the joints cycle so we can see EQ data (no changes to image occur)
        track = pygmt.project(data=eq_data, center=joints[i], endpoint=joints[i+1], width=[
                              -80, 0], length=[0, joints_dist[i+1] - joints_dist[i]], unit=True)
        track = track.rename(columns={
                             0: 'longitude', 1: 'latitude', 2: 'depth', 3: 'eq_id', 4: 'p', 5: 'q', 6: 'r', 7: 's'})

        if len(track) > 0:
            try:
                data_used_check = [track.eq_id[j]
                                   in used_data for j in range(len(track))]
                data_used_check = [not (bool(data_used_check[j]))
                                   for j in range(len(data_used_check))]
                track = track[data_used_check].reset_index(drop=True)
                used_data = np.append(
                    used_data, np.unique(track.eq_id.values))
                
                fig.plot3d(x = track.longitude, y = track.latitude, z = track.depth,
                           style = 'u0.05c', pen = '0.01p,black', fill = 'green',
                           region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))


            except:
                pass
            

    LdM_EQ = pd.read_csv('../DATA/MAPPING/catalog_zr+tango.csv')
    LdM_EQ['id'] = LdM_EQ.index.values
    LdM_EQ = LdM_EQ[(LdM_EQ['lat_err'] < 10) & (LdM_EQ['lon_err'] < 10) * (LdM_EQ['dep_err'] < 10)]
    
    eq_data = LdM_EQ[['longitude', 'latitude', 'depth', 'id']]

    used_data = []
    for i in range(len(joints) - 1):
        track = pygmt.project(data=eq_data, center=joints[i], endpoint=joints[i+1], width=[
                              -40, 0], length=[0, joints_dist[i+1] - joints_dist[i]], unit=True)
        track = track.rename(columns={
                             0: 'longitude', 1: 'latitude', 2: 'depth', 3: 'eq_id', 4: 'p', 5: 'q', 6: 'r', 7: 's'})

        if len(track) > 0:
            try:
                data_used_check = [track.eq_id[j]
                                   in used_data for j in range(len(track))]
                data_used_check = [not (bool(data_used_check[j]))
                                   for j in range(len(data_used_check))]
                track = track[data_used_check].reset_index(drop=True)
                used_data = np.append(
                    used_data, np.unique(track.eq_id.values))

                fig.plot3d(x = track.longitude, y = track.latitude, z = track.depth*-1,
                           style = 'u0.05c', pen = '0.01p,black', fill = 'purple',
                           region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))
            except:
                pass


fig.show()



fig.savefig('../FIGURES/F10_3DInterp_FullData.png', dpi = 800, transparent = True)

#%% Plot Background Frame

fig = pygmt.Figure()
with pygmt.config(FONT="10p", MAP_FRAME_TYPE="plain"):
    
    

    fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["WESNZ"], perspective=perspective, zsize="{}c".format(figz))
    fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["xg1", "yg1", "za20f10g20+lDepth (km)", "wesnZ"], perspective=perspective, zsize="{}c".format(figz))


fig.show()


fig.savefig('../FIGURES/F10_3DInterp_BackgoundFrame.png', dpi = 800, transparent = True)


#%% PLot Frame with Topo

fig = pygmt.Figure()
with pygmt.config(FONT="10p", MAP_FRAME_TYPE="plain"):
    
    

    # fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["WESNZ"], perspective=perspective, zsize="{}c".format(figz))
    fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["x", "y", "za20f10+lDepth (km)", "sZ"], perspective=perspective, zsize="{}c".format(figz))
  
    
  
    fig.grdview(
        grid=sea_level,
        region = gmt_region,
        perspective = True,
        surftype="i",
        cmap = 'lightgray'
    )

    dem_color = pygmt.makecpt(cmap = 'bukavu', series = [np.nanmin(grid.values), np.nanmax(grid.values)])

    fig.grdview(
        grid=grid,
        shading = '+a135+nt0.6', 
        region = gmt_region,
        perspective = True,
        surftype="i",
        cmap= True,
    )


   
    shape_grid = rasterize('../DATA/MAPPING/Payenia_Volcanic_Field_Raster.tif')
    pygmt.makecpt(cmap="red", series=[np.nanmin(shape_grid.values)-1, np.nanmax(shape_grid.values)])
    fig.grdview(
        grid=shape_grid,
        shading = '+a135+nt0.6', 
        region = gmt_region,
        perspective = True,
        surftype="s",
        cmap= True,
        transparency = 90,
    )

    shape_grid = rasterize('../DATA/MAPPING/LagunaLlancanelo_Raster.tif')
    pygmt.makecpt(cmap="skyblue", series=[np.nanmin(shape_grid.values)-1, np.nanmax(shape_grid.values)])
    fig.grdview(
        grid=shape_grid,
        shading = '+a135+nt0.6', 
        region = gmt_region,
        perspective = True,
        surftype="s",
        cmap= True,
        transparency = 0,
    )



fig.savefig('../FIGURES/F10_3DInterp_topo.png', dpi = 800, transparent = True)

    

fig.show()

#%% Plot Frame with Slab

fig = pygmt.Figure()
with pygmt.config(FONT="10p", MAP_FRAME_TYPE="plain"):
    
    

    # fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["WESNZ"], perspective=perspective, zsize="{}c".format(figz))
    fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["x", "y", "za20f10+lDepth (km)", "sZ"], perspective=perspective, zsize="{}c".format(figz))
  
    pygmt.makecpt(cmap="lightgray", series=[-100, 0, 10], continuous=False)
    fig.grdview(
        grid=slab.z,
        region = gmt_region,
        perspective = True,
        surftype = 's',
        contourpen = '.1p',
        cmap = True,
        transparency = 50

    )

fig.show()

fig.savefig('../FIGURES/F10_3DInterp_slab.png', dpi = 800, transparent = True)



#%% Plot frame with LAB

## Plot LAB
LAB = pd.read_csv('../DATA/MAPPING/Tassara_2012_LAB.txt', delimiter = '\s+', names = ['longitude', 'latitude', 'depth'])
    
lon_grid = np.arange(LAB.longitude.min(), LAB.longitude.max(), 0.1)
lat_grid = np.arange(LAB.latitude.min(), LAB.latitude.max(), 0.1)
LON, LAT = np.meshgrid(
    lon_grid,
    lat_grid,
    indexing="ij"
)

grid_values = griddata(
    LAB[['longitude', 'latitude']].values,
    LAB.depth.values,
    (LON, LAT),
    method="linear"
)

LAB = xr.Dataset(
{"depth": (["longitude", "latitude"], grid_values)},
coords={
    "longitude": lon_grid,
    "latitude": lat_grid,
})



LAB_slice = LAB.interp(longitude = x, latitude = y, method = 'linear')
LAB_slice = LAB_slice.assign_coords({'Distance_Along_Trend': line_dist})


LAB_slice2 = LAB.interp(longitude = np.ones(10)*e, latitude = np.linspace(joints[-1,1], n, 10), method = 'linear')



fig = pygmt.Figure()
with pygmt.config(FONT="10p", MAP_FRAME_TYPE="plain"):
    
    
    # fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["WESNZ"], perspective=perspective, zsize="{}c".format(figz))
    fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["x", "y", "za20f10+lDepth (km)", "sZ"], perspective=perspective, zsize="{}c".format(figz))
  
    
    fig.plot3d(x = LAB_slice.longitude, y = LAB_slice.latitude, z = LAB_slice.depth, pen = '0.2p,black',
               region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))


    
    fig.plot3d(x = LAB_slice2.longitude, y = LAB_slice2.latitude, z = LAB_slice2.depth.values[0,:], pen = '0.2p,black',
               region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))

fig.show()

fig.savefig('../FIGURES/F10_3DInterp_LAB.png', dpi = 800, transparent = True)



#%% Plot Frame with seismcity (above slab and in front of frame)

fig = pygmt.Figure()
with pygmt.config(FONT="10p", MAP_FRAME_TYPE="plain"):
    
    

    # fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["wsZ"], perspective=perspective, zsize="{}c".format(figz))
    fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["x", "y", "za20f10+lDepth (km)", "sZ"], perspective=perspective, zsize="{}c".format(figz))

    reidel = pd.read_csv('../DATA/MAPPING/ReidelHornig2025_EQs.csv')
    reidel = reidel[(reidel.Lat < -30) & (reidel.Lat > -40)][reidel.Lat_Error < 10][reidel.Lon_Error < 10][reidel.Dep_Error < 10]
    reidel.Dep = reidel.Dep
    eq_data = reidel[['Lon', 'Lat', 'Dep', 'id']]
    used_data = []
    for i in range(len(joints) - 1): # constricting the joints cycle so we can see EQ data (no changes to image occur)
        track = pygmt.project(data=eq_data, center=joints[i], endpoint=joints[i+1], width=[
                              -100, 0], length=[0, joints_dist[i+1] - joints_dist[i]], unit=True)
        track = track.rename(columns={
                             0: 'longitude', 1: 'latitude', 2: 'depth', 3: 'eq_id', 4: 'p', 5: 'q', 6: 'r', 7: 's'})

        if len(track) > 0:
            try:
                data_used_check = [track.eq_id[j]
                                   in used_data for j in range(len(track))]
                data_used_check = [not (bool(data_used_check[j]))
                                   for j in range(len(data_used_check))]
                track = track[data_used_check].reset_index(drop=True)
                used_data = np.append(
                    used_data, np.unique(track.eq_id.values))
                
                
                slab_at_points = slab.interp(
                    lon=xr.DataArray(track["longitude"].to_numpy(), dims="points"),
                    lat=xr.DataArray(track["latitude"].to_numpy(), dims="points"),
                    )
                
                track["slab_surface"] = slab_at_points.z.values
                track['slab_surface'] = track['slab_surface'].fillna(-999)
                track_above = track[track['depth'] > track['slab_surface']].copy()
                
                
                fig.plot3d(x = track_above.longitude[track_above.latitude>s], y = track_above.latitude[track_above.latitude>s], z = track_above.depth[track_above.latitude>s],
                           style = 'u0.05c', pen = '0.01p,black', fill = 'green',
                           region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))

            except:
                pass
            
            # sys.exit('here')
            


    LdM_EQ = pd.read_csv('../DATA/MAPPING/catalog_zr+tango.csv')
    LdM_EQ['id'] = LdM_EQ.index.values
    LdM_EQ = LdM_EQ[(LdM_EQ['lat_err'] < 10) & (LdM_EQ['lon_err'] < 10) * (LdM_EQ['dep_err'] < 10)]
    
    eq_data = LdM_EQ[['longitude', 'latitude', 'depth', 'id']]

    used_data = []
    for i in range(len(joints) - 1):
        track = pygmt.project(data=eq_data, center=joints[i], endpoint=joints[i+1], width=[
                              -40, 0], length=[0, joints_dist[i+1] - joints_dist[i]], unit=True)
        track = track.rename(columns={
                             0: 'longitude', 1: 'latitude', 2: 'depth', 3: 'eq_id', 4: 'p', 5: 'q', 6: 'r', 7: 's'})

        if len(track) > 0:
            try:
                data_used_check = [track.eq_id[j]
                                   in used_data for j in range(len(track))]
                data_used_check = [not (bool(data_used_check[j]))
                                   for j in range(len(data_used_check))]
                track = track[data_used_check].reset_index(drop=True)
                used_data = np.append(
                    used_data, np.unique(track.eq_id.values))

                fig.plot3d(x = track.longitude, y = track.latitude, z = track.depth*-1,
                           style = 'u0.05c', pen = '0.01p,black', fill = 'purple',
                           region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))
            except:
                pass


fig.show()


fig.savefig('../FIGURES/F10_3DInterp_Seismicity-aboveslab-infront.png', dpi = 800, transparent = True)

#%% Plot Frame with seismcity (below slab and behind frame)

fig = pygmt.Figure()
with pygmt.config(FONT="10p", MAP_FRAME_TYPE="plain"):
    
    

    # fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["wsZ"], perspective=perspective, zsize="{}c".format(figz))
    fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["x", "y", "za20f10+lDepth (km)", "sZ"], perspective=perspective, zsize="{}c".format(figz))

    reidel = pd.read_csv('../DATA/MAPPING/ReidelHornig2025_EQs.csv')
    reidel = reidel[(reidel.Lat < -30) & (reidel.Lat > -40)][reidel.Lat_Error < 10][reidel.Lon_Error < 10][reidel.Dep_Error < 10]
    reidel.Dep = reidel.Dep
    eq_data = reidel[['Lon', 'Lat', 'Dep', 'id']]
    used_data = []
    for i in range(len(joints) - 1): # constricting the joints cycle so we can see EQ data (no changes to image occur)
        track = pygmt.project(data=eq_data, center=joints[i], endpoint=joints[i+1], width=[
                              -100, 0], length=[0, joints_dist[i+1] - joints_dist[i]], unit=True)
        track = track.rename(columns={
                             0: 'longitude', 1: 'latitude', 2: 'depth', 3: 'eq_id', 4: 'p', 5: 'q', 6: 'r', 7: 's'})

        if len(track) > 0:
            try:
                data_used_check = [track.eq_id[j]
                                   in used_data for j in range(len(track))]
                data_used_check = [not (bool(data_used_check[j]))
                                   for j in range(len(data_used_check))]
                track = track[data_used_check].reset_index(drop=True)
                used_data = np.append(
                    used_data, np.unique(track.eq_id.values))
                
                
                slab_at_points = slab.interp(
                    lon=xr.DataArray(track["longitude"].to_numpy(), dims="points"),
                    lat=xr.DataArray(track["latitude"].to_numpy(), dims="points"),
                    )
                
                track["slab_surface"] = slab_at_points.z.values
                track['slab_surface'] = track['slab_surface'].fillna(-999)
                track_below = track[track['depth'] < track['slab_surface']].copy()
                
                
                fig.plot3d(x = track_below.longitude[track_below.latitude>s], y = track_below.latitude[track_below.latitude>s], z = track_below.depth[track_below.latitude>s],
                           style = 'u0.05c', pen = '0.01p,black', fill = 'green', transparency = 70,
                           region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))

            except:
                pass
            
            # sys.exit('here')
            


    LdM_EQ = pd.read_csv('../DATA/MAPPING/catalog_zr+tango.csv')
    LdM_EQ['id'] = LdM_EQ.index.values
    LdM_EQ = LdM_EQ[(LdM_EQ['lat_err'] < 10) & (LdM_EQ['lon_err'] < 10) * (LdM_EQ['dep_err'] < 10)]
    
    eq_data = LdM_EQ[['longitude', 'latitude', 'depth', 'id']]

    used_data = []
    for i in range(len(joints) - 1):
        track = pygmt.project(data=eq_data, center=joints[i], endpoint=joints[i+1], width=[
                              0, 40], length=[0, joints_dist[i+1] - joints_dist[i]], unit=True)
        track = track.rename(columns={
                             0: 'longitude', 1: 'latitude', 2: 'depth', 3: 'eq_id', 4: 'p', 5: 'q', 6: 'r', 7: 's'})

        if len(track) > 0:
            try:
                data_used_check = [track.eq_id[j]
                                   in used_data for j in range(len(track))]
                data_used_check = [not (bool(data_used_check[j]))
                                   for j in range(len(data_used_check))]
                track = track[data_used_check].reset_index(drop=True)
                used_data = np.append(
                    used_data, np.unique(track.eq_id.values))

                fig.plot3d(x = track.longitude, y = track.latitude, z = track.depth*-1,
                           style = 'u0.05c', pen = '0.01p,black', fill = 'purple', transparency = 70,
                           region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))
            except:
                pass


fig.show()


fig.savefig('../FIGURES/F10_3DInterp_Seismicity-belowslab-behind.png', dpi = 800, transparent = True)



#%% Plot frame with stations


fig = pygmt.Figure()
with pygmt.config(FONT="10p", MAP_FRAME_TYPE="plain"):
    
    

    # fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["wsZ"], perspective=perspective, zsize="{}c".format(figz))
    fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["x", "y", "za20f10+lDepth (km)", "sZ"], perspective=perspective, zsize="{}c".format(figz))


    stat_table = pd.read_csv('../DATA/SPREADSHEETS/{}_PassingStations.csv'.format(region))
    stat_table['Code'] = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i]) for i in range(len(stat_table))]

    contributing_stations = pd.read_csv('../DATA/MAPPING/Contributing_Stations_RE.csv')
    stat_table = stat_table[stat_table.Code.isin(contributing_stations.Code)].reset_index(drop = True)
    
    
    fig.plot3d(x = stat_table.stlo, 
               y = stat_table.stla,
               z = stat_table.stel/1000*grid_scale,
               style = 'u0.1c', pen = '0.2p,black', fill = 'gold',
               region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))

    code_category = ['1X', 'ZR', 'XM',]
    code_color = ['cyan', 'lightred', 'magenta']
    for i, code in enumerate(code_category):
        fig.plot3d(x = stat_table.stlo[stat_table.Code.str.contains(code)], 
                   y = stat_table.stla[stat_table.Code.str.contains(code)],
                   z = stat_table.stel[stat_table.Code.str.contains(code)]/1000*grid_scale,
                   style = 'u0.1c', pen = '0.2p,black', fill = code_color[i],
                   region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))


fig.show()

fig.savefig('../FIGURES/F10_3DInterp_stations.png', dpi = 800, transparent = True)



#%% Plot joints

fig = pygmt.Figure()
with pygmt.config(FONT="10p", MAP_FRAME_TYPE="plain"):
    
    

    # fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["wsZ"], perspective=perspective, zsize="{}c".format(figz))
    fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["x", "y", "za20f10+lDepth (km)", "sZ"], perspective=perspective, zsize="{}c".format(figz))

    fig.plot3d(x = joints[:,0], y = joints[:,1], z = np.ones(len(joints))*0,  style = 'o0.01c', pen = '0.2p,black', fill = 'green',
               region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))
    

    fig.plot3d(x = line[:,0], y = line[:,1], z = np.ones(len(line))*0,   pen = '0.2p,black',
               region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))


    fig.plot3d(x = line[:,0], y = line[:,1], z = np.ones(len(line))*base,   pen = '0.2p,black',
               region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))

fig.show()

fig.savefig('../FIGURES/F10_3DInterp_joints.png', dpi = 800, transparent = True)



#%% Plot Frame Boxes

fig = pygmt.Figure()
with pygmt.config(FONT="10p", MAP_FRAME_TYPE="plain"):
    
    

    # fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["wsZ"], perspective=perspective, zsize="{}c".format(figz))
    fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["x", "y", "za20f10+lDepth (km)", "sZ"], perspective=perspective, zsize="{}c".format(figz))



    line_connect_forward = np.vstack((np.vstack((line[0], line)), line[-1]))
    line_connect_backward = np.vstack((np.vstack((line[0], line)), line[-1]))




    fig.plot3d(x = [joints[0,0], joints[0,0], joints[0,0], joints[0,0], joints[0,0]], y = [s, joints[0,1], joints[0,1], s, s], z = [base, base, 0, 0, base], fill = 'lightgray',
               region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))


    fig.plot3d(x = np.vstack((np.vstack((np.vstack((line[0], line)), line[-1])), np.flip(line, axis = 0)))[:,0] , y = np.vstack((np.vstack((np.vstack((line[0], line)), line[-1])), np.flip(line, axis = 0)))[:,1], z = np.append(np.append(np.append( 0, np.ones(len(line))*base), 0), np.ones(len(line))*0),   fill = 'lightgray',     
               region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))


    # fig.plot3d(x = joints[:,0], y = joints[:,1], z = np.ones(len(joints))*0,  style = 'o0.01c', pen = '0.2p,black', fill = 'green',
    #            region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))
    

    fig.plot3d(x = [joints[-1,0], joints[-1,0], joints[-1,0], joints[-1,0], joints[-1,0]], y = [joints[-1,1], n, n, joints[-1,1], joints[-1,1]], z = [base, base, 0, 0, base],  fill = 'lightgray',
               region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))


    
fig.show()

fig.savefig('../FIGURES/F10_3DInterp_FrameBoxes.png', dpi = 800, transparent = True)


#%% Plot Frame Boxes

fig = pygmt.Figure()
with pygmt.config(FONT="10p", MAP_FRAME_TYPE="plain"):
    
    

    # fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["wsZ"], perspective=perspective, zsize="{}c".format(figz))
    fig.basemap(region=gmt_region, projection='X{}c/{}c'.format(figx,figy), frame=["x", "y", "za20f10+lDepth (km)", "sZ"], perspective=perspective, zsize="{}c".format(figz))



    line_connect_forward = np.vstack((np.vstack((line[0], line)), line[-1]))
    line_connect_backward = np.vstack((np.vstack((line[0], line)), line[-1]))




    fig.plot3d(x = [joints[0,0], joints[0,0], joints[0,0], joints[0,0], joints[0,0]], y = [s, joints[0,1], joints[0,1], s, s], z = [base, base, 0, 0, base], pen =' 0.2p,black',
               region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))


    fig.plot3d(x = np.vstack((np.vstack((np.vstack((line[0], line)), line[-1])), np.flip(line, axis = 0)))[:,0] , y = np.vstack((np.vstack((np.vstack((line[0], line)), line[-1])), np.flip(line, axis = 0)))[:,1], z = np.append(np.append(np.append( 0, np.ones(len(line))*base), 0), np.ones(len(line))*0),   pen =' 0.2p,black',     
               region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))


    # fig.plot3d(x = joints[:,0], y = joints[:,1], z = np.ones(len(joints))*0,  style = 'o0.01c', pen = '0.2p,black', fill = 'green',
    #            region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))
    

    fig.plot3d(x = [joints[-1,0], joints[-1,0], joints[-1,0], joints[-1,0], joints[-1,0]], y = [joints[-1,1], n, n, joints[-1,1], joints[-1,1]], z = [base, base, 0, 0, base],   pen =' 0.2p,black',
               region=gmt_region, projection='X{}c/{}c'.format(figx,figy),perspective=perspective, zsize="{}c".format(figz))


    
fig.show()

fig.savefig('../FIGURES/F10_3DInterp_FrameBoxes-outline.png', dpi = 800, transparent = True)





#%% CCP Panels



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


joints =  np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_Line.txt', delimiter = ',')


joints_dist, line, line_dist = Return_Line_Elements(joints)

x = xr.DataArray(line[:,0], dims='Distance_Along_Trend')
y = xr.DataArray(line[:,1], dims='Distance_Along_Trend')

# require a minimum number of bin hits to be present within the bin radius and bin std radius
# to be plotted
bin_hits_limit = 10
std_hits_limit = 1
std_limit = 0.15


fig = pygmt.Figure()
with pygmt.config(FONT = '14p'):


    for k in range(len(joints)-1):
        
        
        aspect = 1
        ratio =  -base / figy   # ratio of km to cm
        fig_x = np.abs( (joints_dist[k+1] - joints_dist[k])/ (ratio * aspect))
        
            
            
            
        grid_region = [joints_dist[k] , joints_dist[k+1], 0, -base]
        prj = 'X{}c/-{}c'.format(fig_x, figy)
        fig.basemap(region=grid_region, projection=prj, frame=['wsne', 'xa50f10g50+lProfile Distance (km)', 'ya20f10g50+lDepth (km)'])

        with xr.open_dataset('../DATA/MAPPING/nc_models/CCP_TANGO_South_G-5.0_FZB_CSEM2-Hicks2014-vp_Hick2014-Hk-vpvs.nc') as model:
            model_slice = model.interp(longitude = x, latitude = y, method = 'linear', kwargs={"fill_value": None})
            model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})
            model_slice['depth'] = -model_slice['depth']
            
            # truncate model_slice values to be within scale range
            model_slice['Amplitude'].values = np.clip(model_slice['Amplitude'].values, -0.5, 0.5)
        
            # nan grid space with bins less than bin hit limit
            model_slice['Amplitude'].values[np.where(model_slice['Bin_Hits'] < bin_hits_limit)] = np.nan
            model_slice['Amplitude'].values[np.where(model_slice['Bin_Hits_std'] < std_hits_limit)] = np.nan
            model_slice['Amplitude'].values[np.where(model_slice['Bootstrap_std'] > std_limit)] = np.nan

        grid_color = pygmt.makecpt(cmap = '../DATA/MAPPING/no_green.cpt',   series = [-0.5, 0.5], continuous = True)
        fig.grdimage(grid = model_slice['Amplitude'])


        fig.basemap(region=grid_region, projection=prj, frame=['wsne', 'xg20+lProfile Distance (km)', 'yg20+lDepth b.s.l. (km)'])
        
        fig.shift_origin(xshift = f'{fig_x+0.5}c')



fig.show()

fig.savefig('../FIGURES/F10_3DInterp_CCP-all.png', dpi = 800, transparent = True)



#%% Plot CCP Panels {individual}


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


joints =  np.loadtxt('../DATA/MAPPING/TANGO_South_CCP_Line.txt', delimiter = ',')


joints_dist, line, line_dist = Return_Line_Elements(joints)

x = xr.DataArray(line[:,0], dims='Distance_Along_Trend')
y = xr.DataArray(line[:,1], dims='Distance_Along_Trend')

# require a minimum number of bin hits to be present within the bin radius and bin std radius
# to be plotted
bin_hits_limit = 10
std_hits_limit = 1
std_limit = 0.15




for k in range(len(joints)-1):
    
    
    aspect = 1
    ratio =  -base / figy   # ratio of km to cm
    fig_x = np.abs( (joints_dist[k+1] - joints_dist[k])/ (ratio * aspect))

    fig = pygmt.Figure()
    with pygmt.config(FONT = '14p'):
        
        grid_region = [joints_dist[k] , joints_dist[k+1], 0, -base]
        prj = 'X{}c/-{}c'.format(fig_x, figy)
        fig.basemap(region=grid_region, projection=prj, frame=['s', 'x', 'y'])
    
        with xr.open_dataset('../DATA/MAPPING/nc_models/CCP_TANGO_South_G-5.0_FZB_CSEM2-Hicks2014-vp_Hick2014-Hk-vpvs.nc') as model:
            model_slice = model.interp(longitude = x, latitude = y, method = 'linear', kwargs={"fill_value": None})
            model_slice = model_slice.assign_coords({'Distance_Along_Trend': line_dist})
            model_slice['depth'] = -model_slice['depth']
            
            # truncate model_slice values to be within scale range
            model_slice['Amplitude'].values = np.clip(model_slice['Amplitude'].values, -0.5, 0.5)
        
            # nan grid space with bins less than bin hit limit
            model_slice['Amplitude'].values[np.where(model_slice['Bin_Hits'] < bin_hits_limit)] = np.nan
            model_slice['Amplitude'].values[np.where(model_slice['Bin_Hits_std'] < std_hits_limit)] = np.nan
            model_slice['Amplitude'].values[np.where(model_slice['Bootstrap_std'] > std_limit)] = np.nan
    
        grid_color = pygmt.makecpt(cmap = '../DATA/MAPPING/no_green.cpt',   series = [-0.5, 0.5], continuous = True)
        fig.grdimage(grid = model_slice['Amplitude'])
    
    
        fig.basemap(region=grid_region, projection=prj, frame=['wsne', 'xg20+lProfile Distance (km)', 'yg20+lDepth b.s.l. (km)'])
    
        fig.show()


    fig.savefig(f'../FIGURES/F10_3DInterp_CCP_Frame-{k}.png', dpi = 800, transparent = True)


