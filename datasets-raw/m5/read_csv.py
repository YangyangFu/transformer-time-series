# %% [markdown]
# ## Purpose
# This is to read raw data and merge/organize for dataloader.
# The output format of sales is expected to be a dataframe with rows for timesteps, and columns as the time series id that needs to be predicted. 
# Thus, the sales dataframe should be 1941x30490 for item-by-item organization. 
# If add aggregations, the shape should be 1941x42840.
# 
# To distinguish the timeseries, we need save another dataframe as a header to map the timeseries id to its `state_id, store_id, cat_id, dept_id, item_id`.
# 
# 
# This type of organization is good for channel-independent algorithms, which considers each time-series as an independent channel.

# %%
import logging
import pathlib
import joblib
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from util import INDEX_COLUMNS, init, reduce_mem_usage

# %%
def dump(df, name, path):
    df = reduce_mem_usage(df)
    save_dir = pathlib.Path(path)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    joblib.dump(df, save_dir / f'{name}.joblib', compress=True)

dump_dir = os.path.join('./data', 'individual')

# %%
data_dir = "./raw-data/m5-forecasting-uncertainty/"

sales_file = "sales_train_evaluation.csv"
calendar_file = "calendar.csv"
price_file = "sell_prices.csv"


# %%
sales = pd.read_csv(os.path.join(data_dir, sales_file))
calendar = pd.read_csv(os.path.join(data_dir, calendar_file), parse_dates=["date"])
price = pd.read_csv(os.path.join(data_dir, price_file))

# %%
print(sales.columns)
print(calendar.columns)
print(price.columns)
print(sales.shape, calendar.shape, price.shape)

# %% [markdown]
# ## Global time features and time series

# %%
id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']


# %%
s = sales.drop(id_cols, axis=1)
s.index = sales['id']
s = s.transpose()
s = s.reset_index()
s = s.rename(columns={'index': 'd'})
s.head()

# %%
s = s.merge(calendar[['d', 'date']], how = 'outer')
s = s.drop(['d'], axis=1)
s.index = s['date']
s = s.drop(['date'], axis=1)

# %%
s.shape, s.columns

# %%
s.head()

# %%
dump(s, 'ts', dump_dir)

# %%
del s

# %% [markdown]
# ## Local features

# %% [markdown]
# ### Time-invariant local features
# Time-invariant local features refer to the features that are dependent on the time series itself but not on time, such as `item_id, store_id`, etc.

# %%
items = sales[id_cols]
items.head()

# %% [markdown]
# Need convert to integers using label encoding

# %%
label_encoders = {}
for col in items.columns:
    encoder = LabelEncoder()
    items[col] = encoder.fit_transform(items[col])
    label_encoders[col] = encoder

items.head()

# %%
dump(items, 'local_invariant', dump_dir)

# %% [markdown]
# ### Time-variant local features
# 
# local features that is time variant are `SNAP`, `item sell price`, etc

# %% [markdown]
# snap is a local feature based on states. it will be different for each time series.
# 
# the targets are each item in each store in each state, thus, each item should have their own SNAP feature based on the states it locates.
# 
# here I create a snap table for each time series, so that during training, we can grab the snap for corresponding target.

# %%
snap = calendar[['date', 'snap_CA', 'snap_TX', 'snap_WI']]
snap = snap.rename(columns={'snap_CA':'CA',
             'snap_TX':'TX',
             'snap_WI':'WI'})
snap = pd.concat([snap, pd.DataFrame(columns=items['id'])])
for idx, state in zip(items['id'], items['state_id']):
    snap[idx] = snap[state]


# %%
snap.index = snap['date']

# %%
snap = snap.drop(['CA', 'TX', 'WI', 'date'], axis=1)

# %%
snap.shape, snap.columns

# %%
snap.head()

# %%
dump(snap, 'local_variant_snap', dump_dir)

# %%
del snap

# %% [markdown]
# Price for each item, should serve as a numerical time-variant local feature.
# 
# The price is represented on a weekly basis, while the sales is represent on a daily basis. 
# Thus, we need manipulate the weekly price to daily price.

# %%
price.head(), price.shape

# %%
# some items are only for sale after a specific date.
# here we have a release date to consider this effect
releases = price.groupby(['store_id','item_id'])['wm_yr_wk'].min().reset_index()
releases.columns = ['store_id','item_id','wm_yr_wk']
weekday = calendar.groupby('wm_yr_wk')['date'].min().reset_index()
releases = releases.merge(weekday)
releases.columns = ['store_id','item_id','release_week', 'release_date']
releases.drop('release_week', axis=1, inplace=True)
releases.head()

# %%
price.columns
pr = price.merge(releases)
pr.columns, pr.head()

# %%
pr = pr.merge(calendar[['wm_yr_wk', 'd', 'date']])
pr = pr.merge(items)
pr.columns, pr.head()

# %%
#pr['released'] = (pr['date'] >= pr['release_date']).astype(int)
pr = pr.drop(['wm_yr_wk', 'item_id', 'store_id', 'sell_price', 'dept_id', 'cat_id','state_id'], axis=1)
pr = pr.pivot(index='date', columns='id', values='release_date')

# %%
pr.head()

# %% [markdown]
# there are some `NaT` in the columns, checkout why???

# %%
pr.shape, pr.columns

# %%
pr = pr[items['id']]

# %%
pr = pr.apply(lambda x: x <= pr.index, axis=0)
pr = pr.astype(int)

# %%
pr.head(), pr.shape

# %%
dump(pr, 'local_variant_release', dump_dir)
del pr

# %%
price.head()

# %%
price = price.merge(calendar[['wm_yr_wk', 'd', 'date']])

# %%
price.head(), price.shape

# %%
price = price.merge(items)

# %%
price.head(), price.shape

# %%
price = price.pivot(index='date', columns='id', values='sell_price')

# %%
price.head(), price.shape

# %%
price = price[items['id']]
price.head()

# %%
price = price.fillna(value=0)
dump(price, 'local_variant_price', dump_dir)

# %%
del price


# %% [markdown]
# Note some items are not for sales within the given time period, thus the prices are `NAN`. 

# %% [markdown]
# As noticed, there are a lot of zeros/NAN in the price, which basically due to "out of stock" or "not released". However, we dont have information to indicate if a zero/NAN price is due to out of stock. 
# 
# Based on the original price data, we can see some items only have price info after specific date. We will set that date as the release date. 
# 
# The NANs for testing during `d1942-d1969` are due to the fact that the testing data is not openly accessible during competition.


