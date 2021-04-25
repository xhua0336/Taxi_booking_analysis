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

# general packages
import numpy as np
import pandas as pd
import datatable as dt

# ## Data preposessing

# %%time
#parse_dates = ['CreatedDate', 'RequestedDate', 'AccTime', 'PupTime', 'DelTime']
#A = pd.read_csv('Booking_Dispatch_training_set.csv', parse_dates = parse_dates, 
#date_parser = lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M:%S"))
#data_parsed = A.to_csv('data_parsed.csv')

# %%time
#parse_dates = ['CreatedDate', 'RequestedDate']
#B = pd.read_csv('Booking_Dispatch_training_set.csv', parse_dates = parse_dates, 
#date_parser = lambda x: pd.to_datetime(x).strftime("%Y-%m-%d %H:%M:%S"))
#data_parsed = B.to_csv('val_data_parsed.csv')

# %%time
train_data_datatable = dt.fread('data_parsed.csv') 
data = train_data_datatable.to_pandas()
data = data.iloc[:,1:]

# %%time
val_data_datatable = dt.fread('val_data_parsed.csv')
val_data = val_data_datatable.to_pandas()
val_data = val_data.iloc[:,1:]

data.shape

val_data.shape

#subtract the features and target variable for modeling 
train_features = data[['BookingSource','Conditions',
     'HasCondition','IsAccountBooking','CreatedDate','RequestedDate',
     'BookingFleet','DispatchFleet','AreaNumber','PuPlace','PuAddress',
     'PuSuburb','PuLat','PuLong','TargetVariable']]

val_features = val_data[['BookingSource','Conditions',
     'HasCondition','IsAccountBooking','CreatedDate','RequestedDate',
     'BookingFleet','DispatchFleet','AreaNumber','PuPlace','PuAddress',
     'PuSuburb','PuLat','PuLong','TargetVariable']]

train_features.head()

val_features.head()

#concatenate training data and validation data for feature engineering
features = pd.concat([train_features, val_features], axis=0)

features.head()

# check datatypes
features.dtypes

#BookingSource variable
Conditions_count = data['Conditions'].value_counts().rename_axis('unique_values').reset_index(name='counts')
Conditions_count

feature1 = features

# ## Feature engineering

# BookingSource 
dummies = pd.get_dummies(features1['BookingSource'], drop_first=False)
feature1 = feature1.join(dummies)
feature1 = feature1.drop('room_type', axis= 1)





# ## Modeling








