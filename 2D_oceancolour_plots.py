#!/usr/bin/env python
# coding: utf-8

# ## This Jupyter notebook will show you to perform basic calculations and plots with 2 dimensional data (matrices, )
# 
# ## We will compare two images:
# ### * MODIS-AQUA - 31st August 2005
# ### * MODIS-AQUA - 16th Feburary 2017
# 
# Now, we will need to import several packages/toolboxes that are essential for nearly every scientific work in Python.

# In[1]:


import os #change folders
import numpy as np # perform calculations and basic math
import matplotlib.pyplot as plt # plot data
import pandas as pd # work with dataframes,tables, spreadsheets, etc.
import netCDF4 as nc4 # work with netcdf files, the standard file for satellite 2D and 3D data


# ## Now, lets load each image using the netCDF4 module.

# In[2]:


# Let's open the first image (31st August 2005)
file = 'A2005243140500.L2_LAC_OC.x 2.hdf' #write the name of the file
modis_31august2005 = nc4.Dataset(file, mode='r') #open the file in python
print(modis_31august2005) #print full details of the image


# In[3]:


# You can also use fh.variables to read information only on the variables
print(modis_31august2005.variables)


# ## Notice that you have the following variables:
# ### * Time information
#     * Year
#     * Day of the Year
#     * Milliseconds of Day
# ### * Scan line information
#     * Tilt angle for scan line
#     * Scan start-pixel longitude
#     * Scan center-pixel longitude
#     * Scan end-pixel longitude
#     * Scan start-pixel latitude
#     * Scan center-pixel latitude
#     * Scan end-pixel latitude
#     * (...)
# ### * Remote Sensing Reflectances
# ## * **Latitude**
# ## * **Longitude**
# ## * **Chl-a (OC3 algorithm)**
# ### * Aerosol optical thickness
# ### * CDOM
# ### * PAR
# ### * Particulate Organic Carbon

# In[5]:


# Extracting variables
longitude = np.array(modis_31august2005['longitude'])
print(longitude)


# In[ ]:


# Extracting variables
longitude = np.array(modis_31august2005['longitude'])
latitude = np.array(modis_31august2005['latitude'])
mld = np.array(fh['mlotst'])
mld[mld == 32767] = np.nan
mld = np.swapaxes(np.swapaxes(mld, 0, 2), 0, 1)
time = np.array(fh['time'])


pixel1 = pd.read_csv('pixel1_monthly.csv')
pixel2 = pd.read_csv('pixel2_monthly.csv')
pixel3 = pd.read_csv('pixel3_monthly.csv')


# Let's print one of the datasets to check the structure

# In[ ]:


print(pixel1)


# You will notice the data corresponds to monthly-averaged Chl-a concentrations.
# 
# Let's extract the data from each dataset and calculate the mean, min, max, standard deviation

# In[ ]:


pixel1_chla = pixel1['Chl-a'].values
pixel2_chla = pixel2['Chl-a'].values
pixel3_chla = pixel3['Chl-a'].values

# Pixel 1
pixel1_mean = np.nanmean(pixel1_chla)
pixel1_min = np.nanmin(pixel1_chla)
pixel1_max = np.nanmax(pixel1_chla)
pixel1_stdev = np.nanstd(pixel1_chla)

# Pixel 2
pixel2_mean = np.nanmean(pixel2_chla)
pixel2_min = np.nanmin(pixel2_chla)
pixel2_max = np.nanmax(pixel2_chla)
pixel2_stdev = np.nanstd(pixel2_chla)

# Pixel 3
pixel3_mean = np.nanmean(pixel3_chla)
pixel3_min = np.nanmin(pixel3_chla)
pixel3_max = np.nanmax(pixel3_chla)
pixel3_stdev = np.nanstd(pixel3_chla)

print('The Chl-a dataset of pixel 1 has:',
      'mean = {:.2f} mg.m/3, minimum = {:.2f} mg.m/3, maximum = {:.2f} mg.m/3 and standard deviation = {:.2f} mg.m/3 \n'.format(pixel1_mean, pixel1_min, pixel1_max, pixel1_stdev))

print('The Chl-a dataset of pixel 2 has:',
      'mean = {:.2f} mg.m/3, minimum = {:.2f} mg.m/3, maximum = {:.2f} mg.m/3 and standard deviation = {:.2f} mg.m/3 \n'.format(pixel2_mean, pixel2_min, pixel2_max, pixel2_stdev))

print('The Chl-a dataset of pixel 3 has:',
      'mean = {:.2f} mg.m/3, minimum = {:.2f} mg.m/3, maximum = {:.2f} mg.m/3 and standard deviation = {:.2f} mg.m/3 \n'.format(pixel3_mean, pixel3_min, pixel3_max, pixel3_stdev))


# ## Other simple to calculate and useful calculations using numpy are:
# ``` python
# np.ptp(array) # Calculates range (maximum - minimum)
# np.percentile(array) # Calculates the q-th percentile
# np.quantile(array) # Calculates the q-th quantile
# np.median(array) # Calculates the median
# ```
# ## Now say we want to plot each dataset

# In[ ]:


print('Pixel 1 Plot')
plt.plot(pixel1_chla)


# In[ ]:


print('Pixel 2 Plot')
plt.plot(pixel2_chla)


# In[ ]:


print('Pixel 3 Plot')
plt.plot(pixel3_chla)


# They all seem different but let's compare put them in the same plot for comparison.

# In[ ]:


plt.plot(pixel1_chla)
plt.plot(pixel2_chla)
plt.plot(pixel3_chla)


# We can use matplotlib options to improve our plot.

# In[ ]:


plt.figure(figsize=(12,6))
plt.plot(pixel1_chla, c='r', label='Pixel 1')
plt.plot(pixel2_chla, c='b', linestyle='--', label='Pixel 2')
plt.plot(pixel3_chla, c='k', linestyle=':', label='Pixel 3')
plt.xlabel('Years', fontsize=14)
plt.ylabel('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
plt.xticks(ticks=np.arange(0, len(pixel1_chla), 12), labels=np.arange(1998, 2021))
plt.xlim(0,len(pixel1_chla))
plt.ylim(0, 2)
plt.title('Pixel Chl-$\it{a}$ comparison', fontsize=18)
plt.legend(loc=0, fontsize=14)
#plt.tight_layout()


# ## Other types of plots you can do to compare one dimensional datasets!
# * Scatter plots
# * Histograms
# * Boxplots
# * etc.

# In[ ]:


plt.figure()
plt.scatter(pixel1_chla, pixel2_chla, s=10)
plt.xlabel('Pixel 1 Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
plt.ylabel('Pixel 2 Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
plt.title('Scatter Plot - Pixel 1 vs. Pixel 2', fontsize=18)
plt.figure()
plt.scatter(pixel1_chla, pixel3_chla, s=10, c='grey')
plt.xlabel('Pixel 1 Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
plt.ylabel('Pixel 3 Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
plt.title('Scatter Plot - Pixel 1 vs. Pixel 3', fontsize=18)


# In[ ]:


plt.figure()
plt.hist(pixel1_chla, color='r')
plt.xlabel('Pixel 1 Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
plt.ylabel('N', fontsize=14)
plt.title('Histogram - Pixel 1', fontsize=18)
plt.xlim(0,2)
plt.ylim(0,150)
plt.figure()
plt.hist(pixel2_chla, color='b')
plt.xlabel('Pixel 2 Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
plt.ylabel('N', fontsize=14)
plt.title('Histogram - Pixel 2', fontsize=18)
plt.xlim(0,2)
plt.ylim(0,150)
plt.figure()
plt.hist(pixel3_chla, color='b')
plt.xlabel('Pixel 3 Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
plt.ylabel('N', fontsize=14)
plt.title('Histogram - Pixel 3', fontsize=18)
plt.xlim(0,2)
plt.ylim(0,150)


# In[ ]:


pixel1_chla_nonans = pixel1_chla[~np.isnan(pixel1_chla)] # Remove missing values
plt.figure()
bplot = plt.boxplot([pixel1_chla_nonans, pixel2_chla, pixel3_chla], notch = True, patch_artist=True, vert=True)
# fill with colors
colors = ['r', 'b', 'k']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
for patch, color in zip(bplot['medians'], colors):
    patch.set_color('w')
    patch.set_linewidth(2)
plt.xlabel('Pixels', fontsize=14)
plt.ylabel('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
plt.title('Boxplot Comparison', fontsize=18)


# ## Last but not least, how to save an image.
# 
# Let's use the boxplots image as an example

# In[ ]:


pixel1_chla_nonans = pixel1_chla[~np.isnan(pixel1_chla)] # Remove missing values
plt.figure()
bplot = plt.boxplot([pixel1_chla_nonans, pixel2_chla, pixel3_chla], notch = True, patch_artist=True, vert=True)
# fill with colors
colors = ['r', 'b', 'k']
for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)
for patch, color in zip(bplot['medians'], colors):
    patch.set_color('w')
    patch.set_linewidth(2)
plt.xlabel('Pixels', fontsize=14)
plt.ylabel('Chl-$\it{a}$ (mg.m$^{-3}$)', fontsize=14)
plt.title('Boxplot Comparison', fontsize=18)
#plt.show()
# Save in .png
plt.savefig('boxplots_TP4.png',format = 'png', bbox_inches = 'tight', dpi = 100)
# Save in .jpeg
plt.savefig('boxplots_TP4.jpeg',format = 'jpeg', bbox_inches = 'tight', dpi = 100)
# Save in .pdf
plt.savefig('boxplots_TP4.pdf',format = 'pdf', bbox_inches = 'tight', dpi = 100)

