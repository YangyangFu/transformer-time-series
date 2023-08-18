# Embedding layer, whose outputs are passed to the encoder and decoder

import tensorflow as tf
import numpy as np 

class PositionalEncoder(tf.keras.layers.Layer):
    def __init__(self, ):
        super().__init__()

    def get_angles(self, pos, i, embedding_dim):
        """ Get the angles for the positional encoding

        Args:
            pos: column vector containing the positions
            i: row vector containing the dimension span
            embedding_dim: encoding size

        Returns:
            angles: matrix of shape (pos, embedding_dim)
        """
        angles = 1 / np.power(10000, (2 * (i // 2)) / embedding_dim)
        return pos * angles
    
    def call(self, inputs):
        """ Call the layer

        Args:
            inputs: input tensor of shape (batch_size, seq_len, embedding_dim)

        Returns:
            output: positional encoding of shape (batch_size, seq_len, embedding_dim)
        """
        seq_len = tf.shape(inputs)[1]
        embedding_dim = tf.shape(inputs)[2]
        angles = self.get_angles(
            np.arange(seq_len, dtype=np.float32)[:, np.newaxis],
            np.arange(embedding_dim, dtype=np.float32)[np.newaxis, :],
            embedding_dim
        )
        angles[:, 0::2] = np.sin(angles[:, 0::2])
        angles[:, 1::2] = np.cos(angles[:, 1::2])
        pos_encoding = angles[np.newaxis, ...]
        return pos_encoding[:, :seq_len, :]

class PositionalEmbedding(tf.keras.layers.Layer):
    """ Embedding + Positional Encoding

    Args:
        inputs: input tensor of shape (batch_size, seq_len)
    """
    
    def __init__(self, vocab_size, embedding_dim, dropout_rate=0.1):
        super().__init__()
        self.embedding_dim = embedding_dim
        # mask_zero=True to support variable length sequences using masking
        # padding mask is added in the encoder
        # Tensorflow will propagate the mask through the layers automatically
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.pos_encoding = PositionalEncoder()
        # drop
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    # this is needed for keras to automatically propagate the mask
    # compute_mask() is called by keras to generate mask, and default generation is propagating
    def compute_mask(self, *args, **kwargs):
        # computer padding mask: mask all the 0s in the input
        return self.embedding.compute_mask(*args, **kwargs)
    
    def call(self, inputs):

        # embedding: (B, seq, embedding_dim)
        x = self.embedding(inputs)
        # scale the embedding by sqrt(embedding_dim)
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        # add positional embedding: (1, seq, embedding_dim)
        x += self.pos_encoding(x)
        # pass the encoded embedding through a dropout layer
        x = self.dropout(x)
        
        return x 

class TemporalEmbedding(tf.keras.layers.Layer):
    def __init__(self, d_model, freq='H', use_holiday=False):
        super().__init__()
        self.d_model = d_model
        self.freq = freq
        self.use_holiday = use_holiday
        
        # size of embedding
        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 31
        month_size = 12
        week_size = 53
        holiday_size = 19 # 18 holidays + 1 for no holiday
        
        # embedding layers
        if freq=="t":
            self.moh_embedding = tf.keras.layers.Embedding(minute_size, self.d_model)
        self.hod_embedding = tf.keras.layers.Embedding(hour_size, self.d_model)
        self.dom_embedding = tf.keras.layers.Embedding(day_size, self.d_model)
        self.dow_embedding = tf.keras.layers.Embedding(weekday_size, self.d_model)
        self.moy_embedding = tf.keras.layers.Embedding(month_size, self.d_model)
        self.woy_embedding = tf.keras.layers.Embedding(week_size, self.d_model)
        if self.use_holiday:
            self.holiday_embedding = tf.keras.layers.Embedding(holiday_size, self.d_model)
        
    def call(self, time_features):
        """ Call the layer

        Args:
            time_features: input tensor of shape (batch_size, seq_len, num_features)

        Returns:
            output: temporal encoding of shape (batch_size, seq_len, d_model)
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
    d_model = 64
    
    time_emb = TemporalEmbedding(d_model=d_model, freq='H', use_holiday=True)
    for batch in train:
        num_covs, cat_covs, time_enc, time_dec, target = batch
        time_enc_out = time_emb(time_enc)
        print(time_enc_out.shape)