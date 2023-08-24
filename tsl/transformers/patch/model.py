""" implementation of PatchTST

    reference:
        - a time series is worth of 64 words: long-term forcasting with transformers 
"""
import tensorflow as tf 

class LearnablePositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, 
                 embedding_dim=64,
                 **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        # assume the last dim is seq_len
        self.pos_embedding = self.add_weight(
            shape=(input_shape[-1], self.embedding_dim),
            initializer=tf.keras.initializers.RandomUniform(minval=-0.2, maxval=0.2),
            trainable=True,
        )
    
    def call(self, inputs):
        # inputs: [seq_len, embedding_dim]
        return self.pos_embedding    
    
class InstanceNormalization(tf.keras.layers.Layer):
    """ Instance Normalization Layer (https://arxiv.org/abs/1607.08022). 
    """
    def __init__(self, num_features, epsilon=1e-5, learnable=False, substraction="mean", **kwargs):
        """_summary_

        Args:
            num_features (int): number of features to normalize/denormalize
            epsilon (float, optional): small tolerance for numerical stablity. Defaults to 1e-5.
            learnable (bool, optional): weather the layer is learnable. If not, statistics is calculated based on the data on the fly. Defaults to False.
            substraction (str, optional): normalization mode. Defaults to "mean".
        """
        super(InstanceNormalization, self).__init__()
        self.num_features = num_features
        self.eps = epsilon
        self.substraction = substraction
        self.learnable = learnable
    
    def _normalize(self, x):
        # x: [batch_size, seq_len, num_features]
        if self.substraction == "mean":
            self._mean = tf.reduce_mean(x, axis=[1], keepdims=True)
            self._std = tf.math.reduce_std(x, axis=[1], keepdims=True)
            x = (x - self._mean) / (self._std + self.eps)
        
        return x    
    
    def _denormalize(self, x):
        # x: [batch_size, seq_len, num_features]
        if self.substraction == "mean":
            x = x * (self._std + self.eps) + self._mean
        
        return x
    
    def call(self, x, mode="norm"):
        if mode == "norm":
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        
        return x
        
class Patching(tf.keras.layers.Layer):
    def __init__(self, patch_size, patch_strides, patch_padding, name='patch'):
        super(Patching, self).__init__(name=name)
        self.patch_size = patch_size
        self.patch_strides = patch_strides
        self.patch_padding = patch_padding
    
    def call(self, inputs):
        # input: [batch_size, seq_len, feature_dim] -> [batch_size, feature_dim, seq_len]
        x = tf.transpose(inputs, [0, 2, 1])
        
        # padding
        if self.patch_padding == "end":
            # padding to the end of sequence with the last value of sequence
            last_value = tf.reshape(x[:, :, -1], [-1, x.shape[1], 1])
            paddings = tf.tile(last_value, [1, 1, self.patch_strides])
            # x: [batch_size, feature_dim, seq_len + patch_strides]
            x = tf.concat([x, paddings], axis=-1)
        
        # do patching
        # x: [batch_size, feature_dim, seq_len + patch_strides, 1]
        x = tf.expand_dims(x, axis=-1)
        
        # x: [batch_size, feature_dim, patch_num, patch_size]
        x = tf.image.extract_patches(
            images=x,
            sizes=[1, 1, self.patch_size, 1],
            strides=[1, 1, self.patch_strides, 1],
            rates=[1, 1, 1, 1],
            padding='VALID',
        )
        
        # x: [batch_size, feature_dim, patch_num, patch_size]
        return x

class EncoderInputEmbedding(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, pos_type="learn", dropout_rate=0.2, name='encoder_input_embedding'):
        super(EncoderInputEmbedding, self).__init__(name=name)
        self.embedding_dim = embedding_dim
        self.pos_type = pos_type
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        # input_shape: [batch_size, feature_dim, patch_num, patch_size]
        # projection: [batch_size, feature_dim, patch_num, embedding_dim]
        self.linear = tf.keras.layers.Dense(self.embedding_dim)
        # position embedding: [patch_num, embedding_dim]
        if self.pos_type == "learn":
            self.pos_embedding = LearnablePositionalEmbedding(self.embedding_dim)
        elif self.pos_type == "sin":
            pass 
        
        self.add = tf.keras.layers.Add()
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        
    def call(self, inputs):
        # inputs: [batch_size, feature_dim, patch_num, patch_size]
        # project patch_size dimension to embedding dim
        x = self.linear(inputs)
        
        # position embedding
        pos = self.pos_embedding(tf.transpose(x, [0,1,3,2]))
        
        # add: we dont use + to support mask propagation if any
        #pos = tf.tile(tf.reshape(pos, [-1,1,pos.shape[0], pos.shape[1]]), 
        #              [x.shape[0], x.shape[1], 1, 1])
        #x = self.add([x, pos]) 
        x = self.dropout(x+pos)
        
        return x

class FeedForwardBlock(tf.keras.layers.Layer):
    def __init__(self, out_dim, hidden_dim, activation='relu', dropout_rate=0.2, **kwargs):
        super(FeedForwardBlock, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(self.hidden_dim, activation=self.activation),
            tf.keras.layers.Dense(self.out_dim),
            tf.keras.layers.Dropout(self.dropout_rate),
        ])
        self.add = tf.keras.layers.Add()
        self.norm = tf.keras.layers.LayerNormalization()

    def call(self, inputs):
        x = self.ffn(inputs)
        x = self.add([x, inputs])
        x = self.norm(x)
        return x

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 num_heads, 
                 key_dim, 
                 value_dim, 
                 dropout_rate=0.2, 
                 **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.num_heads,
                                                      key_dim=self.key_dim,
                                                      value_dim=self.value_dim,
                                                      dropout=self.dropout_rate)
        self.add = tf.keras.layers.Add()
        # research has shown that batch norm is better than layer norm in transformers for time-series
        self.norm = tf.keras.layers.BatchNormalization()

    def call(self, inputs):
        x = self.mha(query = inputs, 
                            value = inputs,
                            key = inputs)
        x = self.add([x, inputs])
        x = self.norm(x)
        return x

class EncoderLayer(tf.keras.layers.Layer):
    """ A vanilla transformer encoder layer"""
    def __init__(self, d_model, num_heads, ffn_hidden_dim, activation='gelu', dropout_rate = 0.2, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.key_dim = d_model // num_heads
        self.value_dim = d_model // num_heads
        self.ffn_hidden_dim = ffn_hidden_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        self.mha = AttentionBlock(num_heads=self.num_heads,
                                key_dim=self.key_dim,
                                value_dim=self.value_dim,
                                dropout_rate=self.dropout_rate)
        self.ffn = FeedForwardBlock(out_dim = self.d_model,
                                    hidden_dim = self.ffn_hidden_dim,
                                    activation = self.activation,
                                    dropout_rate = self.dropout_rate)
    
    def call(self, inputs):
        x = self.mha(inputs)
        x = self.ffn(x)
        return x

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, ffn_hidden_dim, dropout_rate, activation='gelu', **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_hidden_dim = ffn_hidden_dim
        self.activation = activation
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.layers = [EncoderLayer(d_model=self.d_model,
                                    num_heads=self.num_heads,
                                    ffn_hidden_dim=self.ffn_hidden_dim,
                                    activation = self.activation,
                                    dropout_rate=self.dropout_rate) 
                       for _ in range(self.num_layers)] 
    
    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

class LinearHead(tf.keras.layers.Layer):
    def __init__(self, out_dim, dropout_rate=0.2, **kwargs):
        super(LinearHead, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        # input_shape: [BxM, patch_num, d_model]
        # output_shape: [B, pred_len, M]
        self.linear = tf.keras.layers.Dense(self.out_dim)
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)
        
    def call(self, inputs):
        # inputs: [BxM, patch_num, d_model]
        # flatten the first dimension
        # [B, M, patch_num, d_model]
        #x = tf.reshape(inputs, [-1, self.num_targets, inputs.shape[-2], inputs.shape[-1]])
        # [B, M, patch_num*d_model]
        batch_size = inputs.shape[0]
        num_features = inputs.shape[1]
        x = tf.reshape(inputs, [batch_size, num_features, -1])
        # [B, M, pred_len]
        x = self.linear(x)
        x = self.dropout(x)
        
        # get output shape
        # [B, pred_len, M]
        x = tf.transpose(x, perm=[0, 2, 1])
        
        return x

class PatchTST(tf.keras.Model):
    def __init__(self,
                 pred_len = 96,
                 target_cols_index = [-1], 
                 embedding_dim = 16,
                 num_layers = 3,
                 num_heads = 4,
                 ffn_hidden_dim = 128,
                 patch_size = 16, 
                 patch_strides = 8, 
                 patch_padding="end", 
                 dropout_rate=0.2,
                 linear_head_dropout_rate = 0.0,  
                 **kwargs):
        super(PatchTST, self).__init__(**kwargs)
        self.pred_len = pred_len
        self.target_cols_index = target_cols_index
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_hidden_dim = ffn_hidden_dim
        self.patch_size = patch_size
        self.patch_strides = patch_strides
        self.patch_padding = patch_padding
        self.dropout_rate = dropout_rate
        self.linear_head_dropout_rate = linear_head_dropout_rate
        
    def build(self, input_shape):
        self.ins_norm = InstanceNormalization(num_features=input_shape[-1])
        self.patching = Patching(patch_size=self.patch_size, 
                                 patch_strides=self.patch_strides, 
                                 patch_padding=self.patch_padding)
        self.input_embedding = EncoderInputEmbedding(embedding_dim=self.embedding_dim)
        self.encoder = Encoder(num_layers=self.num_layers, 
                               d_model=self.embedding_dim, 
                               num_heads=self.num_heads, 
                               ffn_hidden_dim=self.ffn_hidden_dim, 
                               dropout_rate=self.dropout_rate)
        self.linear_head = LinearHead(out_dim=self.pred_len, 
                                      dropout_rate=self.linear_head_dropout_rate)
    
    def call(self, inputs):
        # inputs: [batch_size, seq_len, feature_dim]
        # instance normalization
        x = self.ins_norm(inputs, mode="norm")
        
        # patching
        # [batch_size, feature_dim, patch_num, patch_size]
        x = self.patching(inputs)
        
        # input embedding
        # [batch_size, feature_dim, patch_num, embedding_dim]
        x = self.input_embedding(x)
        
        # encoder
        x = tf.reshape(x, [-1, x.shape[-2], x.shape[-1]])
        # [batch_size * feature_dim, patch_num, embedding_dim]
        x = self.encoder(x)
        # [batch_size , feature_dim, patch_num, embedding_dim]
        x = tf.reshape(x, [inputs.shape[0], -1, x.shape[-2], x.shape[-1]])
        
        # linear head
        # [batch_size, pred_len, feature_dim]
        x = self.linear_head(x)
        
        # instance normalization
        x = self.ins_norm(x, mode="denorm")
        
        # [batch_size, pred_len, num_target]
        x = tf.gather(x, self.target_cols_index, axis=-1)
        
        return x
    
if __name__ == '__main__':
    
    pred_len = 96
    hist_len = 512
    num_features = 8
    target_cols_index = [6,7]
    embedding_dim = 128
    num_layers = 3
    num_heads = 4
    ffn_hidden_dim = 128
    patch_size = 16 
    patch_strides = 8 
    patch_padding = "end" 
    dropout_rate = 0.2

    batch_size = 32

    # generate fake data
    x = tf.random.normal((batch_size, hist_len, num_features))
    y = tf.random.normal((batch_size, pred_len, len(target_cols_index)))
    
    # instance normalization
    ins_norm = InstanceNormalization(num_features=num_features)
    ins_norm_out = ins_norm(x, mode="norm")
    print("shape after instance normalization: ", ins_norm_out.shape)
    denorm_out = ins_norm(ins_norm_out, mode="denorm")

    # patching
    print("shape before patching: ", x.shape)
    patching = Patching(patch_size=patch_size,
                        patch_strides=patch_strides,
                        patch_padding=patch_padding)
    patch_out = patching(x)
    print("shape after patching: ", patch_out.shape)
    
    # input embedding
    input_emb = EncoderInputEmbedding(embedding_dim=embedding_dim)
    input_enc = input_emb(patch_out)
    print("shape afte input embeding: ", input_enc.shape)
    
    # encoder
    encoder = Encoder(num_layers=num_layers,
                        d_model=embedding_dim,
                        num_heads=num_heads,
                        ffn_hidden_dim=ffn_hidden_dim,
                        dropout_rate=dropout_rate)
    enc_out = encoder(input_enc)
    print("shape after encoder: ", enc_out.shape)
    
    # linear head
    head = LinearHead(out_dim=pred_len)
    head_out = head(enc_out)
    print("shape after linear head: ", head_out.shape)
    print("shape of target: ", y.shape)
                      