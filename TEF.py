#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Wed Sep 29 14:21:39 2021
@author: nina

Python script for computing longitudinal and temporal Total Exchange Flow (TEF)
variability from GETM output, see MacCready (2011)[1], Wang et al. (2017)[2].
Script yields results presented in Section 6.1, 6.2 in thesis [3]
--> Fig. 6.2 a,c
--> Fig. 6.3
--> Fig. 6.4

[1] MacCready, P., 2011: Calculating estuarine exchange flow using isohaline coordinates.
    Journal of Physical Oceanography, 41 (6), 1116–1124, doi:10.1175/2011JPO4517.1,
    url: https://journals.ametsoc.org/view/journals/phoc/41/6/2011jpo4517.1.xml.
[2] Wang, T., W. R. Geyer, and P. MacCready, 2017: Total Exchange Flow, Entrainment, and
    Diffusive Salt Flux in Estuaries. Journal of Physical Oceanography, 47 (5), 1205–1220,
    doi:10.1175/JPO-D-16-0258.1, url: https://journals.ametsoc.org/view/journals/phoc/47/5/jpo-d-16-0258.1.xml.
[3] Reese, Nina: "Salinity Mixing in the Elbe Estuary". [Master's thesis, Universität Rostock].
    Universität Rostock, Rostock, 2021.
"""


import numpy as np
from scipy import signal
import xarray

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams.update({       #Use LaTeX for plotting
    "text.usetex": True})


#==============================================================================

# Manual Input
month = 'June'

start = '2013-06-01 00:00:00'
stop = '2013-06-29 12:48:00' #2 spring-neap cycles

date = '20130601'

#Path to river runoff file:
river_path = './ElbeAbfluss.txt'

#Path to GETM output for model-km:
km_base = '/silod6/reese/tools/getm/setups/elbe_realistic/store/exp03/OUT_116/' + date + '/' 
km_file_name = 'Mean_all.' + date + '.nc'

#Path to GETM output for TEF analysis:
tef_base = '/silod6/reese/tools/getm/setups/elbe_realistic/store/exp03/OUT_116/' + date + '/'  
tef_file_name = 'Elbe_TEF_all.' + date + '.nc'


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
Q_r = meas_data[start_ind:stop_ind,1]  #select data in requested time span
Q_r_plt = Q_r.astype(float) #convert from str to float
Q_r = np.nanmean(Q_r.astype(float)) #temporal average

#time (for plotting):
tt = np.linspace(1,29,29,dtype=int)
timestamps = [start[:8] + str(t).zfill(2) + ' 00:00:00' for t in tt]
Qr_time = np.asarray([np.datetime64(ts) for ts in timestamps])



#==============================================================================
#%%  LOAD Model-km
#==============================================================================
#Find distance along the thalweg cell-center line j=158 in km
#with upstream end at 0km (Geesthacht) (i.e., compute model-km)

path = km_base + km_file_name #full path to file
dx_tot = xarray.open_mfdataset(path)['dxc'][:]
dx = np.asarray(dx_tot)[157,:] #just need dx along thalweg
dx[454:509] = dx_tot[160,454:509] #fill Hamburg area with Norderelbe distances
dx[452] = 351.272
dx[453] = 356.421
dx[509] = 256.211
dx[510] = 219.797
x = np.cumsum(dx[::-1])[::-1]/1000 #distance along thalweg j=158 in km, with upstream end at 0km
x = x[78:253] #just need x in the section considered here...


#==============================================================================
#%%  LOAD DATA
#==============================================================================
#import getm output / load TEF stuff

path = tef_base + tef_file_name #full path to file

time = xarray.open_mfdataset(path)['time'][:]    #time
salt = xarray.open_mfdataset(path)['salt_s'][:]  #salinity classes
etac = xarray.open_mfdataset(path)['etac'][:]    #grid index j
xic = xarray.open_mfdataset(path)['xic'][:]      #grid index i
uu_s = xarray.open_mfdataset(path)['uu_s'][:,:,:,:]
Sfluxu_s = xarray.open_mfdataset(path)['Sfluxu_s'][:,:,:,:]
dyu = xarray.open_mfdataset(path)['dyu'][:,:]

#convert from xarray to numpy array (because I'm a noob, sorry T.T)
time = np.asarray(time.loc[start:stop])
salt = np.asarray(salt)
etac = np.asarray(etac)
xic = np.asarray(xic)
uu_s = np.asarray(uu_s.loc[start:stop])
Sfluxu_s = np.asarray(Sfluxu_s.loc[start:stop])
dyu = np.asarray(dyu)

#step size for salinity classes:
delta_s = salt[1] - salt[0]



#==============================================================================
#%%  TEF ANALYSIS: q(s), Q_in, s_in, Q_out, s_out
#==============================================================================
#Compute TEF stuff from simulated data 

Q_transect = np.zeros((len(salt), len(xic))) #total Q for each full transect
q_transect = np.zeros((len(salt), len(xic))) #total q for each full transect
q_transect_s = np.zeros((len(salt), len(xic))) #total q for each full transect

#Initialise
Q_in = np.zeros((len(xic))) #volume inflow
Q_out = np.zeros((len(xic))) #volume outflow
Q_sin = np.zeros((len(xic))) #salt inflow
Q_sout = np.zeros((len(xic))) #salt outflow

#replace NaNs with 0, since NaN would lead to NaN in cumsum and sum
dyu[np.isnan(dyu)] = 0

#loop through along-channel (x) indices (i.e. through transects)
for ii in range(len(xic)):
    
    Q_transect_inst = np.zeros((len(time), len(salt))) #total Q for full transect
    q_transect_inst = np.zeros((len(time), len(salt))) #total q for full transect
    
    #loop through cross-channel (y) indices along the transect
    for jj in range(len(etac)): 
        Q = np.cumsum(uu_s[:,::-1,jj,ii]*dyu[jj,ii], axis=1) #cumulative sum over all salinity classes S; starting w/ largest S
        q = ( (uu_s[:,:,jj,ii]*dyu[jj,ii]) / delta_s ) #sign seems to be correct ??
        Q_transect_inst[:,:] += Q #add Q for single y to full transect Q
        q_transect_inst[:,:] += q
    
    #temporal averaging after integration over area:
    Q_transect[:,ii] = np.nanmean(Q_transect_inst[:,:], axis=0)
    q_transect[:,ii] = np.nanmean(q_transect_inst[:,:], axis=0)
    q_transect_s[:,ii] = q_transect[:,ii] * salt
      
    in_idx = np.where(q_transect[:,ii] >= 0) #indices of S where q(S) >= 0
    out_idx = np.where(q_transect[:,ii] < 0) #indices of S where q(S) < 0

    
    if(len(in_idx[0]>0)): #if list not empty
        #integration to find inflows Q_in, Q_sin
        Q_in[ii] = np.sum( q_transect[in_idx,ii] * delta_s, axis=1)[0] 
        Q_sin[ii] = np.sum( q_transect_s[in_idx,ii] * delta_s, axis=1 )[0]
    else:
        Q_in[ii] = np.nan
        Q_sin[ii] = np.nan
    if(len(out_idx[0]>0)):
        #integration to find outflows Q_out, Q_sout
        Q_out[ii] = np.sum( q_transect[out_idx,ii] * delta_s, axis=1 )[0]
        Q_sout[ii] = np.sum( q_transect_s[out_idx,ii] * delta_s, axis=1 )[0]
    else: 
        Q_out[ii] = np.nan
        Q_sout[ii] = np.nan
        

#Compute transport-averaged inflow and outflow salinities    
s_in = Q_sin / Q_in
s_out = Q_sout / Q_out

#Stratification:
d_s = abs(s_in - s_out) #w/o abs(), this might become negative for very small s_in, s_out

#Not used... because I kinda did smth wrong I guess
#salt_q = salt - delta_s/2 #for plotting q(s)
#salt_q = salt_q[salt_q>=0] #remove negative salt values (only first index, actually)
#salt_Q = salt #for plotting Q(s)


#==============================================================================
#%%  PLOTTING
#==============================================================================

#%%Fig. 6.2 a,c
#Colormap Plot for q along longitudinal transect (cross-channel & temporal average)

# Coordinates for plotting:
S, X = np.meshgrid(salt, x)
X = X.T
S = S.T            

#For plotting, set transport in empty salinity classes from 0 to NaN
q_transect[np.where(q_transect==0)] = np.nan


plt.figure(figsize=(6,2), tight_layout=True)

cax = plt.contourf(X, S, q_transect, levels=np.linspace(-800,800,33), cmap='twilight_shifted', extend='both') #'PuOr_r'
plt.xlim([x[0], x[-1]])
plt.ylim([30,0])
plt.xlabel('Elbe model-km')
plt.ylabel('$S$ (g/kg)')
cbar = plt.colorbar(cax, extend='both')
cbar.set_label('$q(S)$ (m$^3$s$^{-1}$(g/kg)$^{-1}$)')
plt.title(month + ' 2013')

plt.savefig('plots/' + date + '/Long_TEF_' + date + '.png', dpi=400)
plt.show()



#%% Fig. 6.3 a,b
# Plot of Q_in, Q_out, and Q_in+Q_out=Q_r (in case of volume conserv.; should be constant along x!!!)

plt.figure(figsize=(4.0,3.2), tight_layout=True)

plt.hlines(0, x[0], x[-1], colors='k', linestyles=':', linewidth=1, alpha=0.5)
plt.hlines(-Q_r, x[0], x[-1], colors='orange', linestyles='--', linewidth=1, label='$\\langle Q_r\\rangle $')
plt.plot(x, Q_in, color='k', alpha=1, label='$Q_{in}$')
plt.plot(x, Q_out, color='k', alpha=0.66, label='$Q_{out}$')
plt.plot(x, Q_in+Q_out, color='k', alpha=0.33, label='$Q_{in} + Q_{out}$')
plt.xlabel('Elbe model-km', fontsize=12)
plt.ylabel('$Q$ (m$^3$/s)', fontsize=12)
plt.legend(fontsize=12, loc=6, ncol=2) #loc=4 for April
#plt.grid(True)
plt.xlim([x[0], x[-1]])
plt.ylim([-4600, 2200])
plt.title(month + ' 2013')

plt.savefig('plots/' + date + '/Q_in-Q_out_' + date + '.pdf')
plt.show()



#%% Fig. 6.3 c,d
# Plot of s_in, s_out

plt.figure(figsize=(4.0,3.2), tight_layout=True)

plt.plot(x, s_in, color='k', alpha=1, label='$s_{in}$')
plt.plot(x, s_out, color='k', alpha=0.3, label='$s_{out}$')
plt.plot(x, d_s, color='orange', alpha=1, label='$\Delta s$')
plt.xlim([x[0], x[-1]])
plt.ylim([-1, 25])
plt.xlabel('Elbe model-km', fontsize=12)
plt.ylabel('$S$ (g/kg)', fontsize=12)
plt.legend(fontsize=12)
plt.title(month + ' 2013')

plt.savefig('plots/' + date + '/s_in-s_out_' + date + '.pdf')
plt.show()



#%% Fig. 6.3 e,f
# Plot checking the Knudsen relations without storage terms, stuff should be close to 0...

a = np.zeros((len(d_s)))
b = np.zeros((len(d_s)))
for ii in range(len(d_s)):
    if d_s[ii]<0:
        a[ii] = b[ii] = 0
    else: 
        a[ii] = (s_out[ii]/d_s[ii])
        b[ii] = (s_in[ii]/d_s[ii])
k1 = Q_in - a*Q_r
k2 = Q_out + b*Q_r

plt.figure(figsize=(4.0,3.2), tight_layout=True)
plt.plot(x, Q_in, color='k', alpha=1, label='$Q_{in}$')
plt.plot(x, a*Q_r, color='k', alpha=0.3, label='$(s_{out}/\Delta s) \\langle Q_r \\rangle$')
plt.plot(x, k1, color='orange', alpha=1, label='$Q_{in} - (s_{out}/\Delta s) \\langle Q_r\\rangle$')
plt.hlines(0, x[0], x[-1], colors='k', linewidth=1, linestyles=':', alpha=0.5)
plt.xlim([x[0], x[-1]])
plt.ylim([-2900, 3000])
plt.xlabel('Elbe model-km', fontsize=12)
plt.ylabel('$Q$ (m$^3$/s)', fontsize=12)
plt.legend(fontsize=12)
plt.title(month + ' 2013')

plt.savefig('plots/' + date + '/Knudsen_' + date + '.pdf')

plt.show()




#%%
#==============================================================================
# Make temporal TEF analysis
#==============================================================================  

ii = 60 #index of transect for which temporal TEF should be done
print(x[ii])

#Initialise:
Q_in = np.zeros((len(time))) #volume inflow
Q_out = np.zeros((len(time))) #volume outlow
Q_sin = np.zeros((len(time))) #salt inflow
Q_sout = np.zeros((len(time))) #salt outflow

Q_transect_inst = np.zeros((len(time), len(salt))) #total Q for the full transect
q_transect_inst = np.zeros((len(time), len(salt))) #total q for the full transect

#replace NaNs with 0, since NaN would lead to NaN in cumsum and sum
dyu[np.isnan(dyu)] = 0

#Loop through j-indices (y) of the transect:
for jj in range(len(etac)):
    Q = np.cumsum(uu_s[:,::-1,jj,ii]*dyu[jj,ii], axis=1) #cumulative sum
    q = ( (uu_s[:,:,jj,ii]*dyu[jj,ii]) / delta_s ) #sign seems to be correct ??
    Q_transect_inst[:,:] += Q #add Q for single y to full transect Q
    q_transect_inst[:,:] += q

q_transect_s = q_transect_inst[:,:] * salt #q_s(S)

#remove tidal variation with a low-pass filter:
q_trans_lp = np.copy(q_transect_inst)
q_trans_s_lp = np.copy(q_transect_s)
for ss in range(len(salt)):
    b, a = signal.butter(3, 1/(3600*30)/(1/3600/2),  #30 h low-pass
                         btype='low', analog=False)
    q_trans_lp[:,ss] = signal.filtfilt(b, a, q_trans_lp[:,ss])
    q_trans_s_lp[:,ss] = signal.filtfilt(b, a, q_trans_s_lp[:,ss])


#Loop through temporal indices
for tt in range(len(time)):
      
    in_idx = np.where(q_trans_lp[tt,:] >= 0) #indices of S where q(S) >= 0
    out_idx = np.where(q_trans_lp[tt,:] < 0) #indices of S where q(S) < 0

    
    if(len(in_idx[0]>0)): #if list not empty
        #integration to find inflows Q_in, Q_sin
        Q_in[tt] = np.sum( q_trans_lp[tt,in_idx] * delta_s, axis=1 )[0]
        Q_sin[tt] = np.sum( q_trans_s_lp[tt,in_idx] * delta_s, axis=1 )[0]
    else: 
        Q_in[tt] = np.nan
        Q_sin[tt] = np.nan
    if(len(out_idx[0]>0)):
        #integration to find outflows Q_out, Q_sout
        Q_out[tt] = np.sum( q_trans_lp[tt,out_idx] * delta_s, axis=1 )[0]
        Q_sout[tt] = np.sum( q_trans_s_lp[tt,out_idx] * delta_s, axis=1 )[0]
    else: 
        Q_out[tt] = np.nan
        Q_sout[tt] = np.nan
        

#Compute transport-averaged inflow and outflow salinities 
s_in = Q_sin / Q_in
s_out = Q_sout / Q_out

#Stratification:
d_s = abs(s_in - s_out) #w/o abs(), this might become negative for very small s_in, s_out

#not used
#salt_q = salt - delta_s/2 #for plotting q(s)
#salt_q = salt_q[salt_q>=0] #remove negative salt values (only first index, actually)
#salt_Q = salt #for plotting Q(s)


#%%
#==============================================================================
# PLOT temporal TEF analysis
#==============================================================================  

#Fig. 6.4
fig, ax = plt.subplots(figsize=(8,3.2))

#plot Q_in, Q_r
ax.plot(time, Q_in, color='k', alpha=1, label='$Q_{in}$')
ax.plot(Qr_time, Q_r_plt, color='k', alpha=0.3, label='$Q_r$')
ax.set_ylabel('$Q$ (m$^3$/s)', fontsize=14)
ax.set_xlim([time[23], time[-23]])
ax.set_ylim([0, 4200])
ax.set_title(month + ' 2013')

#now add another y-axis for salt; plot stratification
ax_s=ax.twinx()
ax_s.plot(time, d_s, color='orange', alpha=1, label='$\Delta s$')
ax_s.set_ylabel('$\Delta s$ (g/kg)', fontsize=14)
ax_s.set_ylim([0,9])

custom_lines = [Line2D([0], [0], color='k', lw=1),
		Line2D([0], [0], color='k', lw=1, alpha=0.3),
		Line2D([0], [0], color='darkorange', lw=1)]
ax.legend(custom_lines, ['$Q_{in}$', '$Q_r$', '$\Delta s$'], 
               fontsize=14, facecolor='white', loc=1, ncol=2)

fig.savefig('plots/' + date + '/Temp_TEF_' + date + '.pdf')
plt.show()
