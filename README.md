# Transformer Timeseries in Tensorflow
A library that implements various transformer and nontransformer models on typical time series prediction, imputation and abnormality detection.

## Installation

Direct installation from github:
```bash
pip install git+https://github.com/YangyangFu/transformer-time-series@main
```

## Usage

Take an example of using `PatchTST` model to predict the ETTh-1 dataset. A detailed example file can be referred to `/examples/train_patchtst.ipynb`.
```Python
import tensorflow as tf
# import data loader that batch on time
from tsl.dataloader.batch_on_time  import DataLoader
# import transformer model
from tsl.transformers.patch import PatchTST
from tsl.utils.utils import seed_everything

# seed everything
seed_everything(2000)

# experiment settings
embed_dim = 16
source_seq_len = 336 #512
target_seq_len = 96
pred_len = 96
target_cols=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT']
n_targets = len(target_cols)

# load data
dataloader = DataLoader(
        data_path,
        ts_file,
        num_cov_global_file=None,
        cat_cov_global_file=None,
        num_cov_local_variant_file=[],
        cat_cov_local_variant_file=[],
        num_cov_local_invariant_file=[],
        cat_cov_local_invariant_file=[],
        num_cov_local_variant_names=[],
        cat_cov_local_variant_names=[],
        target_cols=target_cols,
        train_range=(0, 24*30*12),
        val_range=(24*30*12, 24*30*16),
        test_range=(24*30*16, 24*30*20),
        hist_len=source_seq_len,
        token_len=target_seq_len-pred_len,
        pred_len=pred_len,
        batch_size=128,
        freq='H',
        normalize=True,
        use_time_features=False,
        use_holiday=False,
        use_holiday_distance=False,
        normalize_time_features=False,
        use_history_for_covariates=False
)

# generate dataset for training, validation and testing
train_ds = dataloader.generate_dataset(mode="train", shuffle=True, seed=1)
val_ds = dataloader.generate_dataset(mode="validation", shuffle=False, seed=1)
test_ds = dataloader.generate_dataset(mode="test", shuffle=False, seed=1)

# create PatchTST model
target_cols_index = dataloader.target_cols_index
model = PatchTST(pred_len = pred_len,
                target_cols_index = target_cols_index,
                embedding_dim = embed_dim,
                num_layers = 3,
                num_heads = 4,
                ffn_hidden_dim = 128,
                patch_size = 16,
                patch_strides = 8, 
                patch_padding = "end", 
                dropout_rate = 0.3,
                linear_head_dropout_rate=0.0)

```

## Library Strucutre
```text
- dataloader
    - batch_on_time
    - batch_on_ts
- convolutions
    - timenet
- transformers
    - vanilla
    - informer
    - patch
- tide

```

### General Data Structure

For a general multi-variate timeseries forcasting problem, the time series data can be categorized into three types: time series, global covariates and local covariates. We will use a M5 example to illustrate the data structure.
In general, the M5 competition is to predict the sales of 3049 products in 10 stores across 3 states in the US. 
1. **timeseries**: the time series that are to be predicted themselves. In M5, this is the sales of each product in each store. The shape of the raw data is $(N, M)$, where $N$ is the number of time steps, $M$ is the number of time series.
2. **global covariates**: covariates that are indepedent of the time series themselves, such as time features, time series covariates etc. They are the covariates that are shared across all time series. 
    - `time_features`: time features based on time series data time index, shape is $(N, D_t)$, where $N$ is the number of time steps, $D_t$ is the dimension of time features. 
    - `cat_cov_global`: categorical global features other than time features such as external features, shape is $(N, D_{gc})$, where $D_{gc}$ is the dimension of categorical global features. 
    - `num_cov_global`: numerical global features other than time features such as Dowjones index when predicting a particular item demand. The shape is $(N, D_{gn})$, where $D_{gn}$ is the dimension of numerical global features.
3. **local covariates**: covariates that are dependent on the time series themselves, such as the attributes of the time series (e.g., index of the series in a multi-variate prediction problem, store_id of the product prices etc).
We can further categorize the local covariates into two types: time-variant and time-invariant, depending on whether the covariates change over time. 
    - `cat_cov_local_invariant`: local time-invariant categorical covariates, whose raw data shape is $(M, D_{li,c})$, where $M$ is number of time series, $D_{li,c}$ is the number of time-invariant categorical covariates. In M5, this could be the `id`, `store_id`, `state_id` for each time series.
    - `num_cov_local_invariant`: local time-invariant numerical covariates, whose raw data shape is $(M, D_{li,n})$.
    - `cat_cov_local_variant`: local time-variant categorical covariates, whose raw data shape is $(D_{lv,c}, N, M)$, such as the SNAP status of the state the store is located, etc.
    - `num_cov_local_variant`: local time-variant numerical covariates, whoe raw data shape is $(D_{lv,n}, N, M)$,  such as the price of an item.


### Generalized Dataloader
We implemented a generalized data loader for time series data based on the above structure organization. 
Based on the needs of different modeling and training strategies, there are basically two different ways to organize the data.
- **batch on time**: the data is batched on time index, e.g., $N$. 
In this case, the data loader yields the following shapes:
    - time series: $(B, N_b, M)$, where $B$ is the batch size, $M$ is the number of time series. $N_b$ is the number of time steps in each batch, which can represent the `window size (lookback + prediction horizon)` of the time series. 
    Because the batch index is time, the data means that at the $b-th$ step, we prepare the time series data from $b-L$ to $b+H$, where $L$ is the lookback and $H$ is the prediction horizon.
    - global covariates: 
        - `time_features`: $(B, N_b, D_t)$
        - `cat_cov_global`: $(B, N_b, D_{gc})$
        - `num_cov_global`: $(B, N_b, D_{gn})$
    - local covariates:
        - `cat_cov_local_invariant`: $(M, D_{li,c})$. Because this is time-invariant, the features are the same across all time steps.
        - `num_cov_local_invariant`: $(M, D_{li,n})$. Because this is time-invariant, the features are the same across all time steps.
        - `cat_cov_local_variant`: $(B, D_{lv,c}, N_b, M)$
        - `num_cov_local_variant`: $(B, D_{lv,n}, N_b, M)$
- **batch on time series**: the data is batched on the time series index, e.g., $M$. In this case, the data yields the following shapes:
    - time series: $(B, N_b)$, where $B$ is the batch size on the time series $M$. $N_b$ is the number of time steps in each batch, which can represent the `window size (lookback + prediction horizon)` of the time series.
    Because the batch index is time series, the data means that `for each time step``, for the $b-th$ time series, we prepare the time series data from $i-L$ to $i+H$, where $L$ is the lookback and $H$ is the prediction horizon.
    - global covariates: 
        - `time_features`: $(B, N_b, D_t)$
        - `cat_cov_global`: $(B, N_b, D_{gc})$
        - `num_cov_global`: $(B, N_b, D_{gn})$
    - local covariates:
        - `cat_cov_local_invariant`: $(B, D_{li,c})$. Because this is time-invariant, the features are the same across all time steps.
        - `num_cov_local_invariant`: $(B, D_{li,n})$. Because this is time-invariant, the features are the same across all time steps.
        - `cat_cov_local_variant`: $(B, N_b, D_{lv,c})$
        - `num_cov_local_variant`: $(B, N_b, D_{lv,n})$

### Transformers
The following transformer models are implemented in this library:
- [ ] Non-stationary Transformer
- [ ] Autoformer
- [x] [Informer](https://arxiv.org/abs/2012.07436)
- [ ] Temporal Fusion Transformer
- [x] [PatchTST](https://arxiv.org/abs/2211.14730)

### Convs
- [x] [TimesNet](https://arxiv.org/abs/2210.02186): uses `Inception` network as backbone to extract 2-D information for 1-D timeseries

### MLP
- [x] [TiDE](https://arxiv.org/abs/2304.08424): pure MLP encoder/decoder archtiecture
- [ ] DLinear

## Examples

A list of examples are provided in the `examples` folder.
Colab is recommended to run the notebook examples.


##  Citation
If you find this library useful, please cite our work as follows:
```text
@software{fu2023transformer,
      title={Transformer for Timeseries in Tensorflow}, 
      author={Yangyang Fu},
      url = {https://github.com/YangyangFu/transformer-time-series},
      version = {0.1.0},
      date = {2013-09-15},
      year={2023},
}
```

