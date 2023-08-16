# Processor for Informer model
# Embedding for categorical features
# Normalization for numerical features
# Global time feature embedding
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf 
import numpy as np 

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
                 pred_len=4,
                 batch_size=32,
                 freq='H',
                 normalize=True,
                 max_epochs=10,
                 holiday=True,
                 permute=True,
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
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.batch_size = batch_size
        
        self.window_size = self.hist_len + self.pred_len
        # read data
        self.data_df = pd.read_csv(data_path, index_col=0, parse_dates=True)

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
            
        # TODO: add time features
        # time features dataframe
        
        # normalize numerical columns
        if normalize:
            self._normalize_numeric_data()
        
        # concatenate all covariate features
        self.data_cov = self.data_num_cov
        self.data_cov_cols = self.num_cov_cols
        if self.data_cat_cov is not None:
            self.data_cov = np.concatenate((self.data_cat_cov, self.data_num_cov), axis=1)
            self.data_cov_cols = self.cat_cov_cols + self.num_cov_cols
        
        # get target info
        self.target_cols_index = [self.data_cov_cols.index(col) for col in target_cols]
        
        
           
    def _normalize_numeric_data(self):
        self.scaler = StandardScaler()
        train_num_col = self.data_num_cov[self.train_range[0]:self.train_range[1],:]
        self.scaler = self.scaler.fit(train_num_col)
        self.data_num_cov = self.scaler.transform(self.data_num_cov)
    
    def _split_window(self, window):
        """_summary_

        Args:
            window (_type_): _description_

        Returns:
            _type_: _description_
        """
        inputs = window[:, :self.hist_len, :]
        # cannot indexing like this
        #labels = window[:, self.hist_len:, self.target_cols_index]
        labels = window[:, self.hist_len:, :]
        if self.target_cols is not None:
            labels = tf.stack(
                [labels[:,:, ind] for ind in self.target_cols_index],
                axis=-1
            )
        
        return inputs, labels
    
    def make_dataset(self, data):
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets = None,
            sequence_length=self.window_size,
            sequence_stride=1,
            shuffle=False,
            batch_size=self.batch_size
        )
        
        ds = ds.map(self._split_window)

        return ds 
    
    def generate_train(self):
        
        train_data = self.data_cov[self.train_range[0]:self.train_range[1], :]
        train_ds = self.make_dataset(train_data)
        
        return train_ds
    
    def generate_val(self):
            
        val_data = self.data_cov[self.val_range[0]:self.val_range[1], :]
        val_ds = self.make_dataset(val_data)
        
        return val_ds
    
    
    def generate_test(self):
                
        test_data = self.data_cov[self.test_range[0]:self.test_range[1], :]
        test_ds = self.make_dataset(test_data)
        
        return test_ds
    
    
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
                            batch_size=2
                            )
    train_dataset = dataloader.train()
    
    for inputs, labels in train_dataset.take(1):
        print(inputs.shape)
        print(labels.shape)
        print(inputs)
        print(labels)
        break
        
    

    
                                                               
                                                           
    