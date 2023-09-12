import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf 
import numpy as np
import joblib 
import os
 
from tsl.utils.time_features import TimeCovariates

""" Assume joblib data format with the first column as datetime column"""

class DataLoader():
    """ Data Loader for Informer model"""
    def __init__(self,
                 data_path,
                 ts_file,
                 num_cov_global_file=None,
                 cat_cov_global_file=None,
                 num_cov_local_variant_file=[],
                 cat_cov_local_variant_file=[],
                 num_cov_local_invariant_file=None,
                 cat_cov_local_invariant_file=None,
                 num_cov_local_variant_names = [],
                 cat_cov_local_variant_names = [],
                 target_cols=None,
                 train_range=None,
                 val_range=None,
                 test_range=None,
                 hist_len=28,
                 token_len=7,
                 pred_len=28,
                 batch_size=32,
                 freq='D',
                 normalize=True,
                 use_time_features=False,
                 use_holiday=False,
                 use_holiday_distance=False,
                 normalize_time_features=False,
                 use_history_for_covariates=True
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
        self.num_cov_global = num_cov_global_file
        self.cat_cov_global = cat_cov_global_file
        self.num_cov_local_variant = num_cov_local_variant_file
        self.cat_cov_local_variant = cat_cov_local_variant_file 
        self.num_cov_local_invariant = num_cov_local_invariant_file
        self.cat_cov_local_invariant = cat_cov_local_invariant_file
        self.num_cov_local_variant_names = num_cov_local_variant_names
        self.cat_cov_local_variant_names = cat_cov_local_variant_names
        self.val_range = val_range
        self.test_range = test_range
        self.normalize = normalize
        self.hist_len = hist_len # historical length used for encoder
        self.token_len = min(token_len, hist_len) # token length (previous history) used for decoder
        self.pred_len = pred_len # prediction length used for decoder
        self.batch_size = batch_size
        self.freq = freq
        self.use_time_features = use_time_features # whether to use time features or not
        self.use_holiday = use_holiday # whether to use holiday is the time features or not
        self.use_holiday_distance = use_holiday_distance # whether to use holiday distance as the holiday feature or not
        self.use_holiday_index = True if use_holiday and not use_holiday_distance else False # whether to use which holiday as the holiday feature or not
        # whether to use hisotry and future for covariates just like for time series
        self.normalize_time_features = normalize_time_features
        self.use_history_for_covariates = use_history_for_covariates
        
        self.window_size = self.hist_len + self.pred_len
        
        ## read ts and features        
        # read ts data
        self._ts = joblib.load(os.path.join(data_path, ts_file))
        self._ts = self._ts.resample(freq).mean()
        
        # read global features
        if self.num_cov_global:
            self._num_cov_global = joblib.load(os.path.join(data_path, self.num_cov_global))
            self._num_cov_global_cols = self._num_cov_global.columns
            
        if self.cat_cov_global:
            self._cat_cov_global = joblib.load(os.path.join(data_path, self.cat_cov_global))
            self._cat_cov_global_cols = self._cat_cov_global.columns
        
        # read local features
        # read local time-variant; (Nl, T, M)
        if self.num_cov_local_variant:
            if isinstance(self.num_cov_local_variant, str):
                self.num_cov_local_variant = [self.num_cov_local_variant]
            if len(num_cov_local_variant_names) != len(self.num_cov_local_variant):
                raise ValueError('The length of the names of local time-variant numeric covariates should be equal to the length of data set !!!!')
            # constructing a multi-index dataframe if there are multiple covariates
            ts_cols = self._ts.columns
            lvs = []
            for local_variant in self.num_cov_local_variant:
                lv = joblib.load(os.path.join(data_path, local_variant))
                lv = lv[ts_cols]
                lvs.append(lv)
            
            multi_index = pd.MultiIndex.from_product([num_cov_local_variant_names, ts_cols])
            self._num_cov_local_variant = pd.DataFrame(pd.concat(lvs, axis=1).values, columns=multi_index, index=self._ts.index)
            self._num_cov_local_variant = np.stack([self._num_cov_local_variant[cov].values for cov in num_cov_local_variant_names], axis=0)
            
        if self.cat_cov_local_variant:
            if isinstance(self.cat_cov_local_variant, str):
                self.cat_cov_local_variant = [self.cat_cov_local_variant]
            if len(cat_cov_local_variant_names) != len(self.cat_cov_local_variant):
                raise ValueError('The length of the names of local time-variant categorical covariates should be equal to the length of data set !!!!')
            # constructing a multi-index dataframe if there are multiple covariates
            ts_cols = self._ts.columns
            lvs = []
            for local_variant in self.cat_cov_local_variant:
                lv = joblib.load(os.path.join(data_path, local_variant))
                lv = lv[ts_cols]
                lvs.append(lv)
            
            multi_index = pd.MultiIndex.from_product([cat_cov_local_variant_names, ts_cols])
            self._cat_cov_local_variant = pd.DataFrame(pd.concat(lvs, axis=1).values, columns=multi_index, index=self._ts.index)        

            # get categorical size for embedding use
            self.cat_cov_local_variant_sizes = []
            for col in cat_cov_local_variant_names:
                self.cat_cov_local_variant_sizes.append(len(pd.unique(pd.melt(self._cat_cov_local_variant[col], value_name=col)[col])))

            # reshape to 3-d tensor (n, T, M)
            self._cat_cov_local_variant = np.stack([self._cat_cov_local_variant[cov].values for cov in cat_cov_local_variant_names], axis=0)        
        
        # read local time-invariant 
        if self.num_cov_local_invariant:
            self._num_cov_local_invariant = joblib.load(os.path.join(data_path, self.num_cov_local_invariant))

        if self.cat_cov_local_invariant:
            self._cat_cov_local_invariant = joblib.load(os.path.join(data_path, self.cat_cov_local_invariant))
            self.cat_cov_local_invariant_names = self._cat_cov_local_invariant.columns 
            self.cat_cov_local_invariant_sizes = [len(self._cat_cov_local_invariant[col].unique()) for col in self.cat_cov_local_invariant_names]
            
        # time features dataframe
        print("Generating time features.................")
        if self.use_time_features:
            self.time_features = TimeCovariates(
                self._ts.index, 
                normalized = self.normalize_time_features,
                use_holiday=self.use_holiday,
                use_holiday_distance=self.use_holiday_distance,
                freq = self.freq
            ).get_covariates()
            self.time_features_names = self.time_features.columns
            
            # TODO: get size of categorical time features
        ## default settings
        if self.target_cols is None:
            self.target_cols = self._ts.columns
        self.target_cols_index = [list(self._ts.columns).index(col) for col in self.target_cols]    
        
        # normalize numerical columns: global numeric features and local time variant numeric features
        if self.normalize:
            self._normalize_numeric_data()
        
        # concatenate all covariate features
        #self.data_cov = self.data_num_cov
        #self.data_cov_cols = self.num_cov_cols
        #if self.data_cat_cov is not None:
        #    self.data_cov = np.concatenate((self.data_cat_cov, self.data_num_cov), axis=1)
        #    self.data_cov_cols = self.cat_cov_cols + self.num_cov_cols
        
    def _normalize_numeric_data(self):
        """ Normalize numeric features
        """
        # numeric scaler for global features
        if self.num_cov_global:
            self.global_scaler = StandardScaler()
            num_cov_global_train = self._num_cov_global.iloc[self.train_range[0]:self.train_range[1],:]
            cols = num_cov_global_train.columns
            self.global_scaler = self.global_scaler.fit(num_cov_global_train)
            self._num_cov_global = self.global_scaler.transform(self._num_cov_global)
            self._num_cov_global = pd.DataFrame(self._num_cov_global, columns=cols)
             
        # numeric scaler for local features
        if self.num_cov_local_variant:
            n = len(self.num_cov_local_variant_names)
            self.local_scaler = [StandardScaler()] * n
            # scaler for all numeric features
            for i in range(n):
                num_cov_local_train = self._num_cov_local_variant[i, self.train_range[0]:self.train_range[1],:]
                self.local_scaler[i] = self.local_scaler[i].fit(num_cov_local_train)
            # fir for all data
            self._num_cov_local_variant = np.stack([self.local_scaler[i].transform(self._num_cov_local_variant[i,:,:]) for i in range(n)], 
                                                   axis=0)
            

    def _split_window(self, window, extract_target=True):
        """_summary_

        Args:
            window (_type_): _description_

        Returns:
            _type_: _description_
        """
        # history window for encoder
        #history = window[:, :self.hist_len, :]
        history = np.take(window, range(self.hist_len), axis=-2)
        # future window for decoder, including token length and prediction length
        #future = window[:, -(self.token_len+self.pred_len):, :]
        future = np.take(window, range(-(self.token_len+self.pred_len), 0), axis=-2)
        
        #if extract_target and self.target_cols is not None:
        #    future = tf.stack(
        #        [future[:,:, ind] for ind in self.target_cols_index],
        #        axis=-1
        #    )
        
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
            (ts_batch, 
            num_global_batch, 
            cat_global_batch, 
            num_local_variant_batch, 
            cat_local_variant_batch, 
            num_local_invariant_batch, 
            cat_local_invariant_batch, 
            time_batch ) = self._gather_all_features(batch_idx)
            
            # use historical ts to predict future
            # (B, L, M), (B, H, M) <- (B, L+H, M)
            ts_enc, ts_dec = self._split_window(ts_batch, extract_target=True)

            # split global feature into history and future if needed
            num_global_enc, num_global_dec = None, None
            if num_global_batch is not None:
                if num_global_batch.shape[-2] == self.window_size:
                    # (B, L, n), (B, H, n) <- (B, L+H, n)
                    num_global_enc, num_global_dec = self._split_window(num_global_batch, extract_target=False)
                elif num_global_batch.shape[-2] == 1:
                    # (B, 1, n), None <- (B, 1, n)
                    num_global_enc, num_global_dec = num_global_batch, None
                    
            cat_global_enc, cat_global_dec = None, None
            if cat_global_batch is not None:
                if cat_global_batch.shape[-2] == self.window_size:
                    # (B, L, n), (B, H, n) <- (B, L+H, n)
                    cat_global_enc, cat_global_dec = self._split_window(cat_global_batch, extract_target=False)
                elif cat_global_batch.shape[-2] == 1:
                    # (B, 1, n), None <- (B, 1, n)
                    cat_global_enc, cat_global_dec = cat_global_batch, None
                
            # split local variant 
            num_local_variant_enc, num_local_variant_dec = None, None
            if num_local_variant_batch is not None:
                # (B, n, L, M), (B, n, H, M) <- (B, n, L+H, M)
                if num_local_variant_batch.shape[-2] == self.window_size:
                    num_local_variant_enc, num_local_variant_dec = self._split_window(num_local_variant_batch)
                # (B, n, 1, M), None <- (B, n, 1, M)
                elif num_local_variant_batch.shape[-2] == 1:
                    num_local_variant_enc, num_local_variant_dec = num_local_variant_batch, None
                
            cat_local_variant_enc, cat_local_variant_dec = None, None
            if cat_local_variant_batch is not None:
                if cat_local_variant_batch.shape[-2] == self.window_size:
                    # (B, n, L, M), (B, n, H, M) <- (B, n, L+H, M)
                    cat_local_variant_enc, cat_local_variant_dec = self._split_window(cat_local_variant_batch)
                elif cat_local_variant_batch.shape[-2] == 1:
                    # (B, n, 1, M), None <- (B, n, 1, M)
                    cat_local_variant_enc, cat_local_variant_dec = cat_local_variant_batch, None
            
            # split local invariant: local invariant is not dependent on time
            # there is no need to split but just for consistency
            num_local_invariant_enc, num_local_invariant_dec = None, None
            if num_local_invariant_batch is not None:
                # (B, M, n), None <- (B, M, n)
                num_local_invariant_enc, num_local_invariant_dec = num_local_invariant_batch, None
            
            cat_local_invariant_enc, cat_local_invariant_dec = None, None
            if cat_local_invariant_batch is not None:
                # (B, M, n), None <- (B, M, n)
                cat_local_invariant_enc, cat_local_invariant_dec = cat_local_invariant_batch, None                
            
            # split time features
            time_features_enc, time_features_dec = None, None
            if self.use_time_features:
                # (B, L, n), (B, H, n) <- (B, L+H, n)
                time_features_enc, time_features_dec = self._split_window(time_batch)
            
            # output
            enc = (ts_enc, num_global_enc, cat_global_enc, 
                num_local_variant_enc, cat_local_variant_enc, 
                num_local_invariant_enc, cat_local_invariant_enc, time_features_enc)
            
            dec = (ts_dec, num_global_dec, cat_global_dec, 
                num_local_variant_dec, cat_local_variant_dec, 
                num_local_invariant_dec, cat_local_invariant_dec, time_features_dec)
            
            yield enc, dec

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
            output_types=((tf.float32, tf.float32, tf.float32, 
                          tf.float32, tf.float32,
                          tf.float32, tf.float32, tf.float32), 
                          (tf.float32, tf.float32, tf.float32, 
                          tf.float32, tf.float32,
                          tf.float32, tf.float32, tf.float32))
        )
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
        
        return ds 
    
    def _gather_all_features(self, batch_idx):
        """ Gahter all features from given time series data 

            return:
                time_feature_batch, num_cov_global_batch, cat_cov_global_batch, 
                num_local_variant_batch, cat_local_variant_batch, num_local_invariant_batch, cat_local_invariant_batch
                
        """
        # features: numeric covariates, categorical covariates, time features
        # time series batch: (B, L+H, M)
        ts_batch = np.stack(
                [self._ts.values[i:i+self.window_size, :] for i in batch_idx.numpy()],
                axis = 0
            ) 
        
        # global numeric: (B, L+H, Ngn) or (B, 1, Ngn)
        num_global_batch = None
        
        if self.num_cov_global:
            if self.use_history_for_covariates:
                num_global_batch = np.stack(
                        [self._num_cov_global.values[i:i+self.window_size, :] for i in batch_idx.numpy()],
                        axis = 0
                    )
            else:
            # if no history for global features is need, we only use current time index for predictions
            # Note: t_idx is actually the previous hist_len step, the current step should be (t_idx + hist_len - 1)
                num_global_batch = np.stack(
                        [self._num_cov_global.values[[i+self.hist_len-1], :] for i in batch_idx.numpy()],
                        axis = 0
                    )
                                
        # global categorical: (B, L+H, Ngc)
        cat_global_batch = None
        if self.cat_cov_global:
            if self.use_history_for_covariates:
                cat_global_batch = np.stack(
                        [self._cat_cov_global.values[i:i+self.window_size, :] for i in batch_idx.numpy()],
                        axis = 0
                )
            else:
                cat_global_batch = np.stack(
                        [self._cat_cov_global.values[[i+self.hist_len-1], :] for i in batch_idx.numpy()],
                        axis = 0
                )
                
        # local numeric time variant: (B, Nln, L+H, M)
        num_local_variant_batch = None 
        if self.num_cov_local_variant:
            if self.use_history_for_covariates:
                num_local_variant_batch = np.stack(
                        [self._num_cov_local_variant[:, i:i+self.window_size, :] for i in batch_idx.numpy()],
                        axis = 0
                )
            else:
                num_local_variant_batch = np.stack(
                        [self._num_cov_local_variant[:, [i+self.hist_len-1], :] for i in batch_idx.numpy()],
                        axis = 0
                )                

        # local categorical time variant: (B, Nlc, L+H, T)
        cat_local_variant_batch = None
        if self.cat_cov_local_variant:
            if self.use_history_for_covariates:
                cat_local_variant_batch = np.stack(
                        [self._cat_cov_local_variant[:, i:i+self.window_size, :] for i in batch_idx.numpy()],
                        axis = 0
                )
            else:
                cat_local_variant_batch = np.stack(
                        [self._cat_cov_local_variant[:, [i+self.hist_len-1], :] for i in batch_idx.numpy()],
                        axis = 0
                )
                
        # local numeric time invariant: (B, M, Nlci)
        num_local_invariant_batch = None
        if self.num_cov_local_invariant:
            # repeat B times
            num_local_invariant_batch = np.stack(
                [self._num_cov_local_invariant for _ in batch_idx.numpy()]
            )
        
        # local categorical time invariant
        cat_local_invariant_batch = None
        if self.cat_cov_local_invariant:
            # repeat B times
            cat_local_invariant_batch = np.stack(
                [self._cat_cov_local_invariant for _ in batch_idx.numpy()]
            )               
        
        # batch for time features: (B, L+H, Ntf)
        time_batch = None
        if self.use_time_features:
            time_batch = np.stack(
                    [self.time_features.values[i:i+self.window_size, :] for i in batch_idx.numpy()],
                    axis = 0
                )
        return (ts_batch, 
                num_global_batch, 
                cat_global_batch, 
                num_local_variant_batch, 
                cat_local_variant_batch, 
                num_local_invariant_batch, 
                cat_local_invariant_batch, 
                time_batch
                )

if __name__=="__main__":
    
    data_path = './data/individual'
    ts_file = 'ts.joblib'
    num_cov_local_variant_file = ['local_variant_price.joblib', 'local_variant_snap.joblib', 'local_variant_release.joblib']
    num_cov_local_variant_names = ['price', 'snap', 'release']
    
    loader = DataLoader(
            data_path,
            ts_file,
            num_cov_global_file=None,
            cat_cov_global_file=None,
            num_cov_local_variant_file=num_cov_local_variant_file,
            cat_cov_local_variant_file=[],
            num_cov_local_invariant_file=None,
            cat_cov_local_invariant_file=None,
            num_cov_local_variant_names = num_cov_local_variant_names,
            cat_cov_local_variant_names = [],
            target_cols=None,
            train_range=[0, 1913],
            val_range=[1913, 1941],
            test_range=[1941, 1969],
            hist_len=28,
            token_len=7,
            pred_len=28,
            batch_size=32,
            freq='D',
            normalize=True,
            use_time_features=False,
            use_holiday=False,
            use_holiday_distance=False,
            normalize_time_features=False,
    )

    
