import tensorflow as tf 

from multihead_attention import MultiHeadAttention
from multihead_probsparse_attention import MultiHeadProbSparseAttention

class ProbSpareAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 num_heads, 
                 key_dim, 
                 value_dim, 
                 output_dim, 
                 ffn_hidden_dim,
                 factor=4, 
                 use_bias=True, 
                 dropout_rate=0.1, 
                 **kwargs):
        super().__init__(**kwargs)
        self.mha = MultiHeadProbSparseAttention(num_heads=num_heads,
                                                key_dim=key_dim,
                                                value_dim=value_dim,
                                                factor=factor,
                                                output_dim=output_dim,
                                                use_bias=use_bias)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.add = tf.keras.layers.Add()
        self.ffn = tf.keras.layers.Dense(ffn_hidden_dim, activation='gelu', use_bias=use_bias)
        
    def call(self, inputs, training=True, **kwargs):
        # query: (batch_size, target_seq_len, output_dim)
        # key: (batch_size, source_seq_len, output_dim)
        # value: (batch_size, source_seq_len, output_dim)
        query, key, value = inputs
        
        # mha: (batch_size, target_seq_len, output_dim)
        attn_output, attn_weights = self.mha([query, key, value], 
                                             use_causal_mask=False,
                                             return_attention_scores=True,
                                             training=training)
        self.last_attn_weights = attn_weights
        # residual, layernorm, dropout
        # (batch_size, target_seq_len, output_dim)
        x = self.add([query, attn_output])
        x = self.layernorm(x, training=training)
        x = self.dropout(x, training=training)
        x_skip = x
        
        # pointwise feed forward
        x = self.ffn(x)
        
        # residual, layernorm, dropout
        x = self.add([x_skip, x])
        x = self.layernorm(x, training=training)
        x = self.dropout(x, training=training)
                
        return x, attn_weights
        

class DistillingBlock(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim):
        self.conv1d = tf.keras.layers.Conv1D(filters=output_dim, 
                                             kernel_size=3, 
                                             strides=1, 
                                             padding='same',
                                             input_shape=(input_dim, None))
        self.batch_norm = tf.keras.layers.BatchNormalization(output_dim)
        self.activation = tf.keras.layers.Activation('elu')
        self.maxpool1d = tf.keras.layers.MaxPool1D(pool_size=3,
                                                   strides=2,
                                                   padding='same')
        
    def call(self, x, training=True, **kwargs):
        x = self.conv1d(x)
        x = self.batch_norm(x, training=training)
        x = self.activation(x)
        x = self.maxpool1d(x)
        
        # (batch_size, seq_len/2, output_dim)
        return x
    

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        pass 

class Encoder(tf.keras.layers.Layer):
    def __init__(self):
        pass

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        pass

class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        pass
