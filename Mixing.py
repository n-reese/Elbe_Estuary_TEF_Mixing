#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 13:45:09 2021

@author: nina

Mixing analysis in isohaline framework
using code from Burchard et al. (2021) [1]
Also plots spring-neap averaged salinity distribution along thalweg line (j=158)

Script yields results presented in Section 6.1, 6.3 in thesis [2]
--> Fig. 6.2 b,d
--> Fig. 6.5
--> Fig. 6.6

[1] Burchard, H., U. Gräwe, K. Klingbeil, N. Koganti, X. Lange, and M. Lorenz, 2021:
    Effective Diahaline Diffusivities in Estuaries. Journal of Advances in Modeling Earth
    Systems, 13 (2), doi:https://doi.org/10.1029/2020MS002307, url: https://agupubs.
    onlinelibrary.wiley.com/doi/abs/10.1029/2020MS002307.
[2] Reese, Nina: "Salinity Mixing in the Elbe Estuary". [Master's thesis, Universität Rostock].
    Universität Rostock, Rostock, 2021.
"""


import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import xarray
plt.rcParams.update({       #Use LaTeX for plotting
    "text.usetex": True})


#==============================================================================

# Manual Input
month = 'April'

start = '2013-04-01 00:00:00'
stop = '2013-04-29 12:48:00' #2 spring-neap cycles

date = '20130401'

#Path to river runoff file:
river_path = './ElbeAbfluss.txt'

#Path to GETM output in 'Mean' file:
mean_base = '/silod6/reese/tools/getm/setups/elbe_realistic/store/exp03/OUT_116/' + date + '/' 
mean_file_name = 'Mean_all.' + date + '.nc'

#Path to GETM output in 'TEF' file:
tef_base = '/silod6/reese/tools/getm/setups/elbe_realistic/store/exp03/OUT_116/' + date + '/' 
tef_file_name = 'TEF_Mean_all.' + date + '.nc'


#==============================================================================
#%%  RIVER RUNOFF
#==============================================================================
#import observed river runoff data (which was also used in the model!):
#river runoff can be retrieved from file "rivers.nc" in the setup directory

#Load Measured data
f = open(river_path, 'r') # 'r' = read
meas_data = np.genfromtxt(f, skip_header=18, skip_footer=3, delimiter='\t', dtype=str)
f.close()

mestart = start
mestop = stop[:-9] + ' 00:00:00' #river runoff is given in daily values: round down
start_ind = np.where([(mestart in m) for m in meas_data[:,0]])[0][0] #first index of the start date
stop_ind = np.where([(mestop in m) for m in meas_data[:,0]])[0][0] + 1 #first index of the start date
Q_r = meas_data[start_ind:stop_ind,1] #select data in requested time span
Q_r = np.nanmean(Q_r.astype(float)) #temporal average


#==============================================================================
#%%  LOAD DATA
#==============================================================================
#%% Then load TEF stuff from model output. First: Mean

path = mean_base + mean_file_name #full path to file

time = xarray.open_mfdataset(path)['time'][:]
salt = xarray.open_mfdataset(path)['salt'][:]
bathy = xarray.open_mfdataset(path)['bathymetry'][:]
h = xarray.open_mfdataset(path)['hn'][:]
dx_tot = xarray.open_mfdataset(path)['dxc'][:]
dy = xarray.open_mfdataset(path)['dyc'][:]
etac = xarray.open_mfdataset(path)['etac'][:]
xic = xarray.open_mfdataset(path)['xic'][:]
dA = xarray.open_mfdataset(path)['areaC'][:]

#convert from xarray to numpy array
time = np.asarray(time.loc[start:stop])
salt = np.asarray(salt.loc[start:stop])
salt[salt<0] = np.nan
bathy = np.asarray(bathy)
h = np.asarray(h.loc[start:stop])
print(np.shape(h))
dx_tot = np.asarray(dx_tot)
dx_tot[np.isnan(dx_tot)] = 1
dy = np.asarray(dy)
dy[np.isnan(dy)] = 0
etac = np.asarray(etac)
xic = np.asarray(xic)
print(np.shape(xic))
dA = np.asarray(dA)
dA[np.isnan(dA)] = 0
dA[dx_tot == 0] = 0
dx_tot[dx_tot == 0] = 1


#Along-channel distance in km (for plotting):
dx = np.asarray(dx_tot)[157,:] #just need dx along thalweg
dx[454:509] = dx_tot[160,454:509] #fill Hamburg area with Norderelbe distances
dx[452] = 351.272
dx[453] = 356.421
dx[509] = 256.211
dx[510] = 219.797

x = np.cumsum(dx[::-1])[::-1]/1000 #distance along thalweg j=157 in km, with upstream end at 0km


#================
# Isohalines in the area:
# computes the minimum salinity that leaves part of the estuarine volume bounded by the transect
# with index i at least once in the considered time span
    
i = 79 #oceanside transect for the longitudinal TEF analysis
i2 = 1 #full domain; for the mixing analysis
test_salt = np.copy(salt)
test_salt[test_salt<0.5] = np.nan
print('Minimum salinity not fully inside i=' + str(i) + ': ' + str(np.nanmin(test_salt[:,:,:,i-1])))
print('Minimum salinity not fully inside i=' + str(i2) + ': ' + str(np.nanmin(test_salt[:,:,:,i2-1])))


#%% Next: load TEF data

path = tef_base + tef_file_name #full path to file

hpmS_s = xarray.open_mfdataset(path)['hpmS_s'][:]  #physical mixing
hnmS_s = xarray.open_mfdataset(path)['hnmS_s'][:]  #numerical mixing
flags_s = xarray.open_mfdataset(path)['flags_s'][:] #flags
h_s = xarray.open_mfdataset(path)['h_s'][:]

#convert from xarray to numpy array
hpmS_s = np.asarray(hpmS_s.loc[start:stop])
hnmS_s = np.asarray(hnmS_s.loc[start:stop])
flags_s = np.asarray(flags_s.loc[start:stop])
h_s = np.asarray(h_s.loc[start:stop])

salt_s = np.linspace(0,35,176) #salt bins
ds =(salt_s[1]-salt_s[0])



#%%
# =============================================================================
# Temporal averaging of Mixing variables and layer height in the Salinity classes
# =============================================================================

print('Temporal averaging...')
hpmS_s_mean = np.mean(hpmS_s[:,:,:,:],axis=0) #Physical Mixing
hnmS_s_mean = np.mean(hnmS_s[:,:,:,:],axis=0) #Numerical Mixing
h_s_mean = np.mean(h_s[:,:,:,:],axis=0) #layer height



#%%
# =============================================================================
# Integration along X and Y; Division by ds to get variables per salinity class
# =============================================================================

print('Integration along X and Y...')
mms_phy = np.zeros(len(salt_s))
mms_num = np.zeros(len(salt_s))
vol_s = np.zeros(len(salt_s))

mms_phy = np.sum(np.sum(hpmS_s_mean*dA,axis=1),axis=1)/ds
mms_num = np.sum(np.sum(hnmS_s_mean*dA,axis=1),axis=1)/ds
vol_s = np.sum(np.sum(h_s_mean*dA,axis=1),axis=1)/ds

mms_total = mms_phy+mms_num  #this is now the total mixing per salinity class



#%%
# =============================================================================
# Integration only along Y; Division by ds to get variables per salinity class
# =============================================================================

print('Integration along Y...')
mms_phy_x = np.zeros((len(salt_s), len(dx_tot[0,:])))
mms_num_x = np.zeros((len(salt_s), len(dx_tot[0,:])))

##This is only integrated in z and y, not in x:
mms_phy_x = np.sum(hpmS_s_mean*dA/dx_tot,axis=1)/ds #/ np.sum(dy, axis=0)
mms_num_x = np.sum(hnmS_s_mean*dA/dx_tot,axis=1)/ds #/ np.sum(dy, axis=0)

mms_total_x = mms_phy_x+mms_num_x  #this is now the total mixing per salinity class



#%%
# =============================================================================
# Spring-neap averaged salinities
# =============================================================================

thalweg = 157 #thalweg cell center line : no. 158
# temporally averaged salinity along thalweg:
salt_avg = np.nanmean(salt[:,:,int(thalweg),:], axis=0)

#Just for plotting:
z = np.cumsum(np.mean(h[:,::-1,thalweg,:], axis=0), axis=0) #temporal averaging included
salt_avg[z<-2] = 0
z[z<-2] = -2 
xt = np.tile( x, (len(h[0,:,0,0]),1) )

#interpolate depths to regular spacing
depths_interp = (-1)*np.arange(1.5,28,dtype=float)

# Coordinates for plotting:
Xz, Z = np.meshgrid( x, depths_interp )
Xz = Xz.T
Z = Z.T

#Interpolate salinity onto coordinate grid
S_interp = griddata((xt.flatten(),(-z[::-1,:]).flatten()), salt_avg.flatten(), (Xz, Z), method='cubic')



#%%
# =============================================================================
# Plot results
# =============================================================================

#================
# Universal Law of Estuarine Mixing:
# Fig. 6.6

fig, ax = plt.subplots(figsize=(4,3), tight_layout=True)

ax.plot(salt_s, mms_total, '-k', label='$m_{total}(S)$')
ax.plot(salt_s, 2*salt_s*Q_r, '--k', alpha=0.5, label='$2S\\langle Q_r\\rangle$')
ax.set_xlabel('$S$ (g/kg)')
ax.set_ylabel('$m(S)$ (m$^3$(g/kg)s$^{-1})$')
ax.legend()
ax.set_title(month + ' 2013')

fig.savefig('plots/' + date + '/isohaline_mixing_' + date + '.pdf')
plt.show()


#================
# Mixing along the channel:
# Fig. 6.2 b,d

#coordinates for plotting:
S, X = np.meshgrid(salt_s, x)
X = X.T
S = S.T  

#First: Total mixing (not used in thesis, but nice-to-have) 
fig, ax = plt.subplots(figsize=(6,2), tight_layout=True)

cax = ax.contourf(X, S, mms_total_x, levels=np.linspace(0,3.5,71), cmap='magma_r')
ax.set_xlabel('Elbe model-km')
ax.set_ylabel('$S$ (g/kg)')
ax.set_xlim([x[78], x[253]])
ax.set_ylim([30,0])
cbar = fig.colorbar(cax)
cbar.set_label('$m_{tot, loc}(x,S)$ (m$^2$(g/kg)s$^{-1})$')
ax.set_title(month + ' 2013')

fig.savefig('plots/' + date + '/local_total_mixing_' + date + '.png', dpi=300)
plt.show()


#Then: Just physical mixing (this is Fig. 6.2 b,d)
fig, ax = plt.subplots(figsize=(6,2), tight_layout=True)

cax = ax.contourf(X, S, mms_phy_x, levels=np.linspace(0,3,61), cmap='magma_r')
ax.set_xlabel('Elbe model-km')
ax.set_ylabel('$S$ (g/kg)')
ax.set_xlim([x[78], x[253]])
ax.set_ylim([30,0])
cbar = fig.colorbar(cax)
cbar.set_label('$m_{loc}(x,S)$ (m$^2$(g/kg)s$^{-1})$')
ax.set_title(month + ' 2013')

fig.savefig('plots/' + date + '/local_physical_mixing_' + date + '.png', dpi=300)
plt.show()


#================
# Temporally averaged salinities along the channel, depth-interpolated:
# Fig. 6.5

fig, ax = plt.subplots(figsize=(9,3), tight_layout=True)

cax = ax.contourf(Xz, Z, S_interp, levels=np.linspace(0,35,71), cmap='magma')
c = ax.contour(Xz, Z, S_interp, levels=np.linspace(0, 34, 18), colors='w')
c2 = ax.contour(Xz, Z, S_interp, levels=np.linspace(1, 35, 18), colors='w', linestyles='--', linewidths=0.8, alpha=0.5)

cbar = fig.colorbar(cax)
cbar.set_label('Salinity (g/kg)')
ax.clabel(c, levels=np.linspace(0, 34, 18), inline = 1, fmt ='% 2d', fontsize = 12)

ax.plot()
ax.set_xlabel('Elbe model-km')
ax.set_ylabel('z (m)')
ax.set_title('Average salinity along thalweg (j=' + str(thalweg+1) + ') in ' + month + ' 2013')
ax.set_xlim([x[0], x[253]])
ax.set_ylim([-26,0])
ax.fill_between(x, -30, -bathy[thalweg,:], facecolor='grey', zorder=300)
ax.plot(x, -bathy[thalweg,:], color='k')

fig.savefig('plots/' + date + '/isohalines_' + date + '.png', dpi=300)
plt.show()
