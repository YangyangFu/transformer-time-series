# Embedding layer, whose outputs are passed to the encoder and decoder

import tensorflow as tf
import numpy as np 

class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, 
                 embedding_dim=64,
                 max_timescale=10000, 
                 seq_axis=1,
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.max_timesclae = max_timescale
        self._seq_axis = seq_axis
        
    def get_angles(self, pos, i):
        """ Get the angles for the positional encoding

        Args:
            pos: column vector containing the positions
            i: row vector containing the dimension span
            embedding_dim: encoding size

        Returns:
            angles: matrix of shape (pos, embedding_dim)
        """
        angles = 1 / tf.math.power(self.max_timesclae, (2 * (i // 2)) / self.embedding_dim)
        return pos * angles
    
    def call(self, inputs):
        """ Call the layer

        Args:
            inputs: input tensor of shape (batch_size, seq_len, embedding_dim)

        Returns:
            output: positional encoding of shape (batch_size, seq_len, embedding_dim)
        """
        seq_len = tf.shape(inputs)[self._seq_axis]
        angles = self.get_angles(
            np.arange(seq_len, dtype=np.float32)[:, np.newaxis],
            np.arange(self.embedding_dim, dtype=np.float32)[np.newaxis, :]
        )
        angles_numpy = angles.numpy()
        angles_numpy[:, 0::2] = np.sin(angles[:, 0::2])
        angles_numpy[:, 1::2] = np.cos(angles[:, 1::2])
        angles = tf.convert_to_tensor(angles_numpy)
        pos_encoding = angles[np.newaxis, ...]
        return pos_encoding[:, :seq_len, :]

class TemporalEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, freq='H', use_holiday=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.freq = freq
        self.use_holiday = use_holiday
        
        # size of embedding
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13
        week_size = 53
        holiday_size = 19 # 18 holidays + 1 for no holiday
        
        # embedding layers
        if freq=="t":
            self.moh_embedding = tf.keras.layers.Embedding(minute_size, self.embedding_dim)
        self.hod_embedding = tf.keras.layers.Embedding(hour_size, self.embedding_dim)
        self.dom_embedding = tf.keras.layers.Embedding(day_size, self.embedding_dim)
        self.dow_embedding = tf.keras.layers.Embedding(weekday_size, self.embedding_dim)
        self.moy_embedding = tf.keras.layers.Embedding(month_size, self.embedding_dim)
        self.woy_embedding = tf.keras.layers.Embedding(week_size, self.embedding_dim)
        if self.use_holiday:
            self.holiday_embedding = tf.keras.layers.Embedding(holiday_size, self.embedding_dim)
        
    def call(self, time_features):
        """ Call the layer

        Args:
            time_features: input tensor of shape (batch_size, seq_len, num_features)

        Returns:
            output: temporal encoding of shape (batch_size, seq_len, embedding_dim)
        """
        # get the embedding for each time feature
        
        moh_embedding = self.moh_embedding(time_features[:,:,0]) if hasattr(self, "moh_embedding") else 0
        hod_embedding = self.hod_embedding(time_features[:,:,1])
        dom_embedding = self.dom_embedding(time_features[:,:,2])
        dow_embedding = self.dow_embedding(time_features[:,:,3])
        moy_embedding = self.moy_embedding(time_features[:,:,4])
        woy_embedding = self.woy_embedding(time_features[:,:,5])
        holiday_embedding = self.holiday_embedding(time_features[:,:,6]) if hasattr(self, "holiday_embedding") else 0
        
        # sum the embeddings
        x = moh_embedding + hod_embedding + dow_embedding + dom_embedding + moy_embedding + woy_embedding + holiday_embedding

        
        return x

class CategoricalEmbedding(tf.keras.layers.Layer):
    
    """ Embedding for categorical features
    - concatenate the embeddings of all categorical features
    - output after a linear layer
    """
    def __init__(self, 
                 num_embedding, 
                 embedding_size, 
                 embedding_dim,
                 output_dim,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_embedding = num_embedding
        # need check that embedding_size should be a list
        self.embedding_size = [embedding_size] if num_embedding == 1 else embedding_size
        self.embedding_dim = embedding_dim
                
        self.embedding = [
            tf.keras.layers.Embedding(embedding_size[i], embedding_dim) for i in range(self.num_embedding)
        ]
        self.linear = tf.keras.layers.Dense(output_dim)
        
    def call(self, inputs):
        # inputs: (batch_size, seq_len, num_embedding)
        x = tf.concat([
                self.embedding[i](inputs[:,:,i]) for i in range(self.num_embedding)
            ], axis=-1)
        x = self.linear(x)
        
        return x
        
if __name__ == "__main__":
    from dataloader import DataLoader
    
    ds = DataLoader(data_path='ETTh1.csv',
                    target_cols=['OT'],
                    num_cov_cols=['HUFL','HULL','MUFL','MULL','LUFL','LULL','OT'],
                    train_range=(0, 10),
                    hist_len=3,
                    pred_len=2,
                    batch_size=6
                    )

    train = ds.generate_dataset(mode="train", 
                                shuffle=False)
    embedding_dim = 64
    
    time_emb = TemporalEmbedding(embedding_dim=embedding_dim, 
                                 freq='H', 
                                 use_holiday=True)
    cat_emb = CategoricalEmbedding(num_embedding=2,
                                embedding_size=[2, 3],
                                embedding_dim=embedding_dim,
                                output_dim=16)
    for batch in train:
        num_covs, cat_covs, time_enc, time_dec, target = batch
        time_enc_out = time_emb(time_enc)
        print(time_enc_out.shape)
        # fake categorical features
        cat_covs = tf.random.uniform(shape=(6, 3, 2), minval=0, maxval=2, dtype=tf.int32)
        cat_enc_out = cat_emb(cat_covs)
        print(cat_enc_out.shape)
