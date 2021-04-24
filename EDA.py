# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.8.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# numpy
import numpy as np

# pandas
import pandas as pd
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# plotting
import json
import matplotlib.pyplot as plt
import seaborn as sns
colorMap = sns.light_palette("blue", as_cmap=True)
from haversine import haversine, Unit

#data reader
import dabl
import datatable as dt

# system
import warnings
warnings.filterwarnings('ignore')
# -

# #  Exploratory data analysis

# %%time
#parse_dates = ['CreatedDate', 'RequestedDate', 'AccTime', 'PupTime', 'DelTime']
#A = pd.read_csv('Booking_Dispatch_training_set.csv', parse_dates = parse_dates, 
#date_parser = lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M:%S"))
#data_parsed = A.to_csv('data_parsed.csv')

# %%time
train_data_datatable = dt.fread('data_parsed.csv')
data = train_data_datatable.to_pandas()
data = data.iloc[:,1:]

data.shape

data.head()

# check datatypes
data.dtypes

# check statistics of the features
stats = data[['AccDistanceFromJob','DelJobDistance','DelJobTime',
     'ChargesPrice','ChargesExtras','ChargesFlagfall','Tolls','Tips','Discount']]
stats.describe().round(3)

#FinalDispatchStatus
finalstatus_count = data['FinalDispatchStatus'].value_counts().rename_axis('unique_values').reset_index(name='counts')
finalstatus_count

#target variable
target_count = data['TargetVariable'].value_counts().rename_axis('unique_values').reset_index(name='counts')
target_count

sns.barplot(x='unique_values', y = 'counts',data=target_count)

#target variable
round(data['TargetVariable'].value_counts()/data.shape[0],3)

#check misssing data
print(data.isnull().sum())

# ## 1 Univariate Analysis

# ## 1.1 Time

# Changing the pickup_datetime and dropoff_datetime from object to datetime datatype
data['CreatedDate']=pd.to_datetime(data['CreatedDate'])
data['RequestedDate']=pd.to_datetime(data['RequestedDate'])
data['PupTime']=pd.to_datetime(data['PupTime'])
data['DelTime']=pd.to_datetime(data['DelTime'])

# Creating features based on day
data['create_by_day'] = data['CreatedDate'].dt.day_name()
data['request_by_day'] = data['RequestedDate'].dt.day_name()
data['PupTime_by_day'] = data['PupTime'].dt.day_name()
data['DelTime_by_day'] = data['DelTime'].dt.day_name()

# Creating features based on Hour
data['create_by_hour'] = data['CreatedDate'].dt.hour
data['request_by_hour'] = data['RequestedDate'].dt.hour
data['PupTime_by_hour'] = data['PupTime'].dt.hour
data['DelTime_by_hour'] = data['DelTime'].dt.hour
#Morning, which starts at 6:01 am and ends at 12:oopm
#Afternoon, which starts at 12:01 pm and ends at 18:00pm
#Evening, which starts at 18:01 and ends at 21:00pm
#Night, which start at 21:01 and ends at 6:00am

def part_of_day (t):
    if t in range (6,12):
        return "Morning"
    elif t in range (12,18):
        return "Afternoon"
    elif t in range (18,21):
        return "Evening"
    else:
        return "Night"


data['create_part_of_day']=data['create_by_hour'].apply(part_of_day)
data['request_part_of_day']=data['request_by_hour'].apply(part_of_day)
data['PupTime_part_of_day']=data['PupTime_by_hour'].apply(part_of_day)
data['DelTime_part_of_day']=data['DelTime_by_hour'].apply(part_of_day)

# create_part_of_day
(data['TargetVariable']
 .groupby(data['create_part_of_day'])
 .value_counts(normalize=True)
 .rename('proportion')
 .reset_index()
 .pipe((sns.barplot, "data"), x ='TargetVariable', y ='proportion', hue='create_part_of_day'))

# request_part_of_day
(data['TargetVariable']
 .groupby(data['request_part_of_day'])
 .value_counts(normalize=True)
 .rename('proportion')
 .reset_index()
 .pipe((sns.barplot, "data"), x ='TargetVariable', y ='proportion', hue='request_part_of_day'))

#PupTime_part_of_day
(data['TargetVariable']
 .groupby(data['PupTime_part_of_day'])
 .value_counts(normalize=True)
 .rename('proportion')
 .reset_index()
 .pipe((sns.barplot, "data"), x ='TargetVariable', y ='proportion', hue='PupTime_part_of_day'))

#DelTime_part_of_day
(data['TargetVariable']
 .groupby(data['DelTime_part_of_day'])
 .value_counts(normalize=True)
 .rename('proportion')
 .reset_index()
 .pipe((sns.barplot, "data"), x ='TargetVariable', y ='proportion', hue='DelTime_part_of_day'))

# ## 1.2 Location

sns.catplot(x="TargetVariable", y='AccDistanceFromJob', kind="point", data=data)

sns.catplot(x="TargetVariable", y='DelJobDistance', kind="point", data=data)

destlat = data['DestLat'].dropna()
destlong = data['DestLong'].dropna()

# +
# reading training data
latlong = np.array(destlat, destlong)

# cut off long distance trips
lat_low, lat_hgh = np.percentile(destlat, [2,98])
lon_low, lon_hgh = np.percentile(destlong, [2, 98])

# +
# create image
bins =250
lat_bins = np.linspace(lat_low, lat_hgh, bins)
lon_bins = np.linspace(lon_low, lon_hgh, bins)
H2, _, _ = np.histogram2d(destlat, destlong, bins=(lat_bins, lon_bins))

img = np.log(H2[::-1, :] + 1)

plt.figure()
ax = plt.subplot(1,1,1)
plt.imshow(img)
plt.axis('off')
plt.title('Taxi trip end points')
plt.savefig("taxi_trip_end_points.png")
# -

# ## 1.3 No destination records

# categorize if there is a destination record
loc_target = data[['DestLat','TargetVariable']]
loc_target['Null'] = loc_target['DestLat'].isnull()

(loc_target['TargetVariable']
 .groupby(loc_target['Null'])
 .value_counts(normalize=True)
 .rename('proportion')
 .reset_index()
 .pipe((sns.barplot, "data"), x ='TargetVariable', y ='proportion', hue='Null'))

# ## 1.4 DelJobDistance

sns.catplot(x="TargetVariable", y='DelJobDistance', kind="point", data=data)

# ## 1.5 DelJobTime

sns.catplot(x="TargetVariable", y='DelJobTime', kind="point", data=data)

# ## 1.6 ChargesPrice	

sns.catplot(x="TargetVariable", y='ChargesPrice', kind="point", data=data)

# ## 1.7 ChargesExtras

sns.catplot(x="TargetVariable", y='ChargesExtras', kind="point", data=data)

# ## 1.8 ChargesFlagfall

sns.catplot(x="TargetVariable", y='ChargesFlagfall', kind="point", data=data)

# ## 1.9 Tolls	

sns.catplot(x="TargetVariable", y='Tolls', kind="point", data=data)

# ## 1.11 Tips	

sns.catplot(x="TargetVariable", y='Tips', kind="point", data=data)

# ## 1.12 Discount

sns.catplot(x="TargetVariable", y='Discount', kind="point", data=data)

# ## 1.13 BookingSource

data['BookingSource'].value_counts()

(data['TargetVariable']
 .groupby(data['BookingSource'])
 .value_counts(normalize=True)
 .rename('proportion')
 .reset_index()
 .pipe((sns.barplot, "data"), x ='TargetVariable', y ='proportion', hue='BookingSource'))

# ## 1.14 Conditions    

data['Conditions'].value_counts().head(30)
# 64 Endorsed Driver Driver is qualified for account work
# 4194304 Premium $11 Booking Fee
# 262144  Wheelchair Accredited Zero200 Wheelchair Accredited
# 16777216 5 seater 5 Seat Vehicle Required


# ## 1.15 HasCondition 

(data['TargetVariable']
 .groupby(data['HasCondition'])
 .value_counts(normalize=True)
 .rename('proportion')
 .reset_index()
 .pipe((sns.barplot, "data"), x ='TargetVariable', y ='proportion', hue='HasCondition'))

# ## 1.16 IsAccountBooking 

(data['TargetVariable']
 .groupby(data['IsAccountBooking'])
 .value_counts(normalize=True)
 .rename('proportion')
 .reset_index()
 .pipe((sns.barplot, "data"), x ='TargetVariable', y ='proportion', hue='IsAccountBooking'))

# ## 1.17 BookingFleet

#target variable
booking_fleet = data['BookingFleet'].value_counts().rename_axis('unique_values').reset_index(name='counts')
booking_fleet

(data['TargetVariable']
 .groupby(data['BookingFleet'])
 .value_counts(normalize=True)
 .rename('proportion')
 .reset_index()
 .pipe((sns.barplot, "data"), x ='TargetVariable', y ='proportion', hue='BookingFleet'))

# ## 1.18 DispatchFleet 

Dispatch_fleet = data['DispatchFleet'].value_counts().rename_axis('unique_values').reset_index(name='counts')
Dispatch_fleet

(data['TargetVariable']
 .groupby(data['DispatchFleet'])
 .value_counts(normalize=True)
 .rename('proportion')
 .reset_index()
 .pipe((sns.barplot, "data"), x ='TargetVariable', y ='proportion', hue='DispatchFleet'))

# ## 1.19 PuSuburb

pick = data['PuSuburb'].value_counts().tail(191).keys().tolist() #<4
pick

data['pick'] = np.where(data['PuSuburb'].isin(pick), True, False)

(data['TargetVariable']
 .groupby(data['pick'])
 .value_counts(normalize=True)
 .rename('proportion')
 .reset_index()
 .pipe((sns.barplot, "data"), x ='TargetVariable', y ='proportion', hue='pick'))

# ## 1.20 DestSuburb

dest = data['DestSuburb'].value_counts().tail(334).keys().tolist() #<1
dest

data['dest'] = np.where(data['PuSuburb'].isin(dest), True, False)

(data['TargetVariable']
 .groupby(data['dest'])
 .value_counts(normalize=True)
 .rename('proportion')
 .reset_index()
 .pipe((sns.barplot, "data"), x ='TargetVariable', y ='proportion', hue='dest'))

# ## 2 Multivariate Analysis

# Calculating the correlation
corr =data.corr()

# Visualizing the relationship based on correlation
sns.heatmap(corr)
