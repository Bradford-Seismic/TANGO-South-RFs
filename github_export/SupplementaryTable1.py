# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:04:37 2026

@author: jimbradford
"""



import pygmt
import obspy
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import hilbert
import geopandas as gpd
import shapely

from obspy import Stream, read
from obspy.clients.fdsn import Client
from obspy import UTCDateTime

import os
import sys
import io

import numpy as np
import pandas as pd
from glob import glob
import xarray as xr
from math import radians, sin, cos, sqrt, atan2

os.chdir('.')




stations = pd.read_csv('../DATA/SPREADSHEETS/SA_Stations.csv')
moveouts = pd.read_csv('../DATA/SPREADSHEETS/Moveout_Summary.csv')
downloads = pd.read_csv('../DATA/SPREADSHEETS/Download_Summary.csv')
snr = pd.read_csv('../DATA/SPREADSHEETS/SNR_Summary.csv')
itd = pd.read_csv('../DATA/SPREADSHEETS/Iterdecon_Summary.csv', keep_default_na=False)

region = 'TANGO_South'

stat_table = pd.read_csv('../DATA/SPREADSHEETS/{}_PassingStations.csv'.format(region))
stat_table['Code'] = ['{}-{}'.format(stat_table.net[i], stat_table.stat[i]) for i in range(len(stat_table))]

contributing_stations = pd.read_csv('../DATA/SPREADSHEETS/Contributing_Stations_ABCD.csv')


used_stats = stations[stations['Code'].isin(contributing_stations['Code'])]
used_stats = used_stats.drop_duplicates(subset = 'Code', keep = 'first').reset_index(drop = True)



sys.exit('here')


network = []
num_stations = []
datacenter = []
starttime = []
endtime = []
citation =[]

for i in range(len(used_stats)):
    net = used_stats.Network[i]
    if net not in network:
        network.append(net)
        num_stations.append(len(used_stats[used_stats.Network == net]))
        datacenter.append(used_stats.DataCenter[i])
        starttime.append(used_stats.StartTime[used_stats.Network == net].min())
        endtime.append(used_stats.EndTime[used_stats.Network == net].min())




#### Post-add network citations
citation = ['GFZ Potsdam, 2001',
            'GFZ Potsdam, 2010',
            'Roecker and Russo, 2010',
            'Universidad de Chile, 1991',
            'Universidad de Chile, 2012',
            'Thurber, 2019',
            'Beck et al., 2022',
            'Beck et al., 2022']
        




stat_data = pd.DataFrame({'Network': network, 'Num. Stations': num_stations,
                          'Data Center': datacenter, 'Start': starttime, 
                          'End': endtime, 'Citation': citation})


stat_data.to_csv('../DATA/SPREADSHEETS/Station_Data-Citations.csv')


mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["font.sans-serif"] = ["Helvetica"]

fig, ax = plt.subplots(figsize=(20/2.54, 20/2.54))
ax.axis("off")
table = pd.plotting.table(
    ax, stat_data, loc="center", cellLoc="center", colWidths=np.ones(len(stat_data.columns))*0.2
)

# Set font size (points)
table.auto_set_font_size(True)



# fig.savefig('../FIGURES/SupplementaryTable1.png', dpi = 600)


