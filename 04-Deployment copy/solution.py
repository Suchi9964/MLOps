#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[2]:


get_ipython().system('python -V')


# In[ ]:


import pickle
import pandas as pd


# In[ ]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[ ]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


# In[ ]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_????-??.parquet')


# In[ ]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[ ]:




