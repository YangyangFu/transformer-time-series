# Processor for Informer model
# Embedding for categorical features
# Normalization for numerical features
# Global time feature embedding
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf 
import numpy as np
 
from .time_features import TimeCovariates

""" Assume cvs data format with the first column as datetime column"""

class DataLoader():
    """ Data Loader for Informer model"""
    def __init__(self,
                 data_path,
                 target_cols,
                 ts_cov_cols=None,
                 num_cov_cols=None,
                 cat_cov_cols=None,
                 train_range=None,
                 val_range=None,
                 test_range=None,
                 hist_len=24,
                 token_len=4,
                 pred_len=4,
                 batch_size=32,
                 freq='H',
                 normalize=True,
                 use_time_features=False,
                 use_holiday_distance=False,
                 use_which_holiday=True,
                 ):
        """_summary_

        Args:
            data_path (_type_): _description_
            datetime_col (_type_): _description_
            target_cols (_type_): _description_
            ts_cov_cols (_type_): timeseries covariate columns as covariate features
            num_cov_cols (_type_): _description_
            cat_cov_cols (_type_): _description_
            train_range (_type_): _description_
            val_range (_type_): _description_
            test_range (_type_): _description_
            hist_len (_type_): _description_
            pred_len (_type_): _description_
            batch_size (int, optional): _description_. Defaults to 32.
            freq (str, optional): _description_. Defaults to 'H'.
            normalized (bool, optional): _description_. Defaults to True.
            max_epochs (int, optional): _description_. Defaults to 10.
            holiday (bool, optional): _description_. Defaults to True.
            permute (bool, optional): _description_. Defaults to True.
        """
        self.target_cols = target_cols
        self.train_range = train_range
        self.cat_cov_cols = cat_cov_cols
        self.num_cov_cols = num_cov_cols
        self.val_range = val_range
        self.test_range = test_range
        self.hist_len = hist_len # historical length used for encoder
        self.token_len = min(token_len, hist_len) # token length (previous history) used for decoder
        self.pred_len = pred_len # prediction length used for decoder
        self.batch_size = batch_size
        self.use_time_features = use_time_features
        self.use_holiday_distance = use_holiday_distance
        self.use_which_holiday = use_which_holiday
        
        self.window_size = self.hist_len + self.pred_len
        # read data
        self.data_df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        # resample to frequency
        self.data_df = self.data_df.resample(freq).mean()
        
        # time series covariate columns
        if ts_cov_cols is None or len(ts_cov_cols) == 0:
            ts_cov_cols = target_cols
        
        # isolate categorical columns and numerical columns before normalization
        self.data_cat_cov = None
        self.n_cat_cov = 0
        if self.cat_cov_cols is not None and len(self.cat_cov_cols) > 0:
            self.data_cat_cov = self.data_df[self.cat_cov_cols].values
            self.n_cat_cov = len(self.cat_cov_cols)
            
        self.data_num_cov = None
        self.n_num_cov = 0
        if self.num_cov_cols is not None and len(self.num_cov_cols) > 0:
            self.data_num_cov = self.data_df[self.num_cov_cols].values
            self.n_num_cov = len(self.num_cov_cols)
            
        # time features dataframe
        print("Generating time features.................")
        if use_time_features:
            self.time_features = TimeCovariates(
                self.data_df.index, 
                use_holiday_distance=use_holiday_distance,
                use_which_holiday=use_which_holiday,
                normalized=False,
            ).get_covariates()
            self.time_features_cols = self.time_features.columns
        
        # normalize numerical columns
        if normalize:
            self._normalize_numeric_data()
        
        # concatenate all covariate features
        #self.data_cov = self.data_num_cov
        #self.data_cov_cols = self.num_cov_cols
        #if self.data_cat_cov is not None:
        #    self.data_cov = np.concatenate((self.data_cat_cov, self.data_num_cov), axis=1)
        #    self.data_cov_cols = self.cat_cov_cols + self.num_cov_cols
        
        # get target info
        self.target_cols_index = [self.num_cov_cols.index(col) for col in target_cols]
        
    def _normalize_numeric_data(self):
        self.scaler = StandardScaler()
        train_num_col = self.data_num_cov[self.train_range[0]:self.train_range[1],:]
        self.scaler = self.scaler.fit(train_num_col)
        self.data_num_cov = self.scaler.transform(self.data_num_cov)
    
    def _split_window(self, window, extract_target=True):
        """_summary_

        Args:
            window (_type_): _description_

        Returns:
            _type_: _description_
        """
        # history window for encoder
        history = window[:, :self.hist_len, :]
        # future window for decoder, including token length and prediction length
        future = window[:, -(self.token_len+self.pred_len):, :]
        
        if extract_target and self.target_cols is not None:
            future = tf.stack(
                [future[:,:, ind] for ind in self.target_cols_index],
                axis=-1
            )
        
        return history, future
    
    def generator(self, start_idx, end_idx, shuffle=True, seed=0):
        """_summary_

        Args:
            data (_type_): _description_
            shuffle (bool, optional): _description_. Defaults to True.

        Yields:
            _type_: _description_
        """
        # create a dataset of indices
        start = start_idx
        end = end_idx - self.window_size + 1
        idx = tf.data.Dataset.range(start, end)

        # shuffle indices
        if shuffle:
            idx = idx.shuffle(
                buffer_size=(end-start),
                seed=seed)
        
        # batch indices
        idx = idx.batch(self.batch_size)
        
        # map indices to data at each batch    
        # map covariate data
        for batch_idx in idx:
            num_batch, cat_batch, time_batch = self._gather_all_features(batch_idx)
            num_cov_enc, targets = self._split_window(num_batch, extract_target=True)
            
            cat_cov_enc = None
            if cat_batch is not None:
                cat_cov_enc, _ = self._split_window(cat_batch, extract_target=False)
            
            time_features_enc, time_features_dec = None, None
            if self.use_time_features:
                time_features_enc, time_features_dec = self._split_window(time_batch, extract_target=False)
            
            yield num_cov_enc, cat_cov_enc, time_features_enc, time_features_dec, targets

    def generate_dataset(self, mode="train", shuffle=False, seed=0):
        # get range [start, end)
        if mode == "train":
            start_idx, end_idx = self.train_range[0], self.train_range[1]
        elif mode == "validation":
            start_idx, end_idx = self.val_range[0], self.val_range[1]
        elif mode == "test":
            start_idx, end_idx = self.test_range[0], self.test_range[1]
        else:
            raise ValueError("mode must be one of 'train', 'validation', 'test'")
        
        ds = tf.data.Dataset.from_generator(
            self.generator, 
            args=[start_idx, end_idx, shuffle, seed], 
            output_types=(tf.float32, tf.float32, tf.float32, tf.float32, tf.float32))
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        
        return ds 
    
    def _gather_all_features(self, batch_idx):
        """ Gahter all features from given time series data 
        """
        # features: numeric covariates, categorical covariates, time features  
        # batch for numeric features
        numeric_batch = np.stack(
                [self.data_num_cov[i:i+self.window_size, :] for i in batch_idx.numpy()],
                axis = 0
            )
        # batch for categorical features
        cat_batch = None
        if self.data_cat_cov is not None:
            cat_batch = np.stack(
                [self.data_cat_cov[i:i+self.window_size, :] for i in batch_idx.numpy()],
                axis = 0
            )
        # batch for time features
        time_batch = None
        if self.use_time_features:
            time_batch = np.stack(
                    [self.time_features.values[i:i+self.window_size, :] for i in batch_idx.numpy()],
                    axis = 0
                )
        return numeric_batch, cat_batch, time_batch

if __name__=="__main__":
    
    data = pd.read_csv('ETTh1.csv', index_col=0, parse_dates=True)
    print(data.head())
    print(type(data.index))

    #dataset = tf.keras.utils.timeseries_dataset_from_array(data=data,
    #                                                       targets=None, #data['OT'],
    #                                                       sequence_length=Nx+Ny,
    #                                                       sampling_rate=1,
    #                                                       sequence_stride=1,
    #                                                       shuffle=False,
    #                                                       batch_size=batch_size)
    #time_feature dataset: zip(dataset, time_features) 
    dataloader = DataLoader(data_path='ETTh1.csv',
                            target_cols=['OT'],
                            num_cov_cols=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'],
                            train_range=(0, 10),
                            hist_len=3,
                            pred_len=2,
                            batch_size=5
                            )
    print(dataloader.time_features.head())
    train_ds = dataloader.generate_dataset(mode="train", shuffle=True, seed=1)
    for batch in train_ds:
        num_cov_enc, cat_cov_enc, time_features_enc, time_features_dec, targets = batch
        print(cat_cov_enc)
        print(num_cov_enc.shape, time_features_enc.shape, time_features_dec.shape, targets.shape)
