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

from tsl.utils.utils import seed_everything, reduce_mem_usage

# %%
def dump(df, name, path):
    df = reduce_mem_usage(df)
    save_dir = pathlib.Path(path)
    if not save_dir.exists():
        save_dir.mkdir(parents=True)
    joblib.dump(df, save_dir / f'{name}.joblib', compress=True)

dump_dir = os.path.join('./ETT-small', 'ETTh1')

# %%
data_dir = "./ETT-small"

ts_file = 'ETTh1.csv'


# %%
ts = pd.read_csv(os.path.join(data_dir, ts_file), index_col=0, parse_dates=True)


# %%
ts_cols = ts.columns

# %% [markdown]
# ## Global time features and time series
# %%
dump(ts, 'ts', dump_dir)

# %%
del ts

# %% [markdown]
# ## Local features
# local features for the index of each column, this is necessary with dataloader that batch on time series.
id = pd.DataFrame({'id': [i for i in range(len(ts_cols))]})

dump(id, 'id', dump_dir)


# %%
import shutil
import os
 
# path to destination directory
dest_dir = '../datasets/ETTh1'
 
# getting all the files in the source directory
files = os.listdir(dump_dir)

shutil.copytree(dump_dir, dest_dir, dirs_exist_ok=True)

