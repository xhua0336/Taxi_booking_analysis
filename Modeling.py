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
train_data_datatable = dt.fread('data_parsed.csv')
data = train_data_datatable.to_pandas()
data = data.iloc[:,1:]

# %%time
val_data_datatable = dt.fread('Booking_Dispatch_training_set.csv')
val_data = val_data_datatable.to_pandas()
val_data = val_data.iloc[:,1:]

data.shape

val_data.shape

train_features = data[['BookingSource','Conditions',
     'HasCondition','IsAccountBooking','CreatedDate','RequestedDate',
     'BookingFleet','DispatchFleet','AreaNumber','PuPlace','PuAddress',
     'PuSuburb','PuLat','PuLong','TargetVariable']]

val_features = val_data[['BookingSource','Conditions',
     'HasCondition','IsAccountBooking','CreatedDate','RequestedDate',
     'BookingFleet','DispatchFleet','AreaNumber','PuPlace','PuAddress',
     'PuSuburb','PuLat','PuLong','TargetVariable']]

train_features
