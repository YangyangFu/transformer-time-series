import tensorflow as tf 

from multihead_attention import MultiHeadAttention
from multihead_probsparse_attention import MultiHeadProbSparseAttention

""" Should not pass padding mask to the attention layer because the sequence length is not fixed.
"""

class ProbSpareAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, 
                 num_heads, 
                 key_dim, 
                 value_dim, 
                 output_dim, 
                 hidden_dim,
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
        self.layernorm = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
        # pointwise feed forward: note the original paper uses conv1d instead
        #self.ffn = tf.keras.layers.Dense(ffn_hidden_dim, activation='gelu', use_bias=use_bias)
        #self.out = tf.keras.layers.Dense(output_dim, use_bias=use_bias)
        self.conv1d_1 = tf.keras.layers.Conv1D(filters=hidden_dim,kernel_size=1, strides=1)
        self.conv1d_2 = tf.keras.layers.Conv1D(filters=output_dim, kernel_size=1, strides=1)
        self.activation = tf.keras.layers.Activation('elu')
        
    def call(self, inputs, use_causal_mask=False, training=True, **kwargs):
        # query: (batch_size, target_seq_len, output_dim)
        # key: (batch_size, source_seq_len, output_dim)
        # value: (batch_size, source_seq_len, output_dim)
        query, key, value = inputs
        
        # mha: (batch_size, target_seq_len, output_dim)
        attn_output, attn_weights = self.mha(inputs, 
                                             use_causal_mask=use_causal_mask,
                                             return_attention_scores=True,
                                             training=training)
        self.last_attn_weights = attn_weights
        # residual, layernorm, dropout
        # (batch_size, target_seq_len, output_dim)
        x = self.dropout(attn_output, training=training)
        x = self.add([query, x])
        out = x = self.layernorm(x, training=training)

        # pointwise feed forward, dropout, add and norm
        x = self.conv1d_1(x)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        
        
        x = self.conv1d_2(x)
        x = self.dropout(x, training=training)
        x = self.add([out, x])
        x = self.layernorm(x, training=training)

        return x
        
class DistillingBlock(tf.keras.layers.Layer):
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.conv1d = tf.keras.layers.Conv1D(filters=output_dim, 
                                             kernel_size=3, 
                                             strides=1, 
                                             padding='same',
                                             input_shape=(input_dim, None))
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.activation = tf.keras.layers.Activation('elu')
        self.maxpool1d = tf.keras.layers.MaxPool1D(pool_size=3,
                                                   strides=2,
                                                   padding='same')
        self.supports_masking = True
        
    def call(self, x, training=True, **kwargs):
        x = self.conv1d(x)
        x = self.batch_norm(x, training=training)
        x = self.activation(x)
        x = self.maxpool1d(x)
        
        # (batch_size, seq_len/2, output_dim)
        return x
    
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, 
                 num_heads,
                 key_dim,
                 value_dim,
                 output_dim,
                 hidden_dim,
                 factor=4,
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.psa = ProbSpareAttentionBlock(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            factor=factor,
            dropout_rate=dropout_rate)
        
        self.distilling = DistillingBlock(
            input_dim=output_dim, 
            output_dim=output_dim)
        
    def call(self, x, use_causal_mask=False, training=True, **kwargs):
        # x: (batch_size, seq_len, output_dim)
        x = self.psa([x, x, x], 
                    use_causal_mask=use_causal_mask,
                    training=training)
        x = self.distilling(x, training=training)
        return x
    
class Encoder(tf.keras.layers.Layer):
    def __init__(self, 
                num_layers, 
                num_heads,
                key_dim,
                value_dim,
                output_dim,
                hidden_dim,
                factor=4,
                dropout_rate=0.1,
                **kwargs):
        super().__init__(**kwargs)
        self.enc_layers = [EncoderLayer(num_heads=num_heads, 
                                        key_dim=key_dim, 
                                        value_dim=value_dim, 
                                        output_dim=output_dim, 
                                        hidden_dim=hidden_dim, 
                                        factor=factor, 
                                        dropout_rate=dropout_rate) for _ in range(num_layers)]
    
    def call(self, x, use_causal_mask=True, training=True, **kwargs):
        # x: (batch_size, seq_len, output_dim)
        
        for enc_layer in self.enc_layers:
            x = enc_layer(x, use_causal_mask=use_causal_mask, training=training)
            
        return x
        

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self):
        pass

class Decoder(tf.keras.layers.Layer):
    def __init__(self):
        pass

if __name__ == '__main__':
    
    d_model = 512
    source_seq_len = 64
    num_heads = 8
    num_encoder_layers = 4
    
    x = tf.random.normal((32, source_seq_len, d_model))
    padding_mask = tf.random.uniform((32, source_seq_len), minval=0, maxval=2, dtype=tf.int32)
    padding_mask = tf.cast(padding_mask, tf.bool)
    use_mask = False
    if use_mask:
        x._keras_mask = padding_mask
    
    # attention block
    psa = ProbSpareAttentionBlock(num_heads=num_heads,
                                key_dim = d_model//num_heads, 
                                value_dim = d_model//num_heads,
                                output_dim=d_model,
                                hidden_dim=2048)
    
    y = psa([x, x, x], 
            use_causal_mask=True)
    attn_weights = psa.last_attn_weights
    print(y.shape, attn_weights.shape)
    
    # distilling block
    dis = DistillingBlock(input_dim=d_model, output_dim=d_model)
    y = dis(y)
    print("shape after distilling", y.shape)

    # encoder layer
    enc_layer = EncoderLayer(num_heads=num_heads,
                             key_dim=d_model//num_heads,
                             value_dim=d_model//num_heads,
                             output_dim=d_model,
                             hidden_dim=2048)
    y = enc_layer(x, use_causal_mask=True)
    print("shape after encoder layer", y.shape)

    # encoder
    encoder = Encoder(num_layers=num_encoder_layers,
                    num_heads=num_heads,
                    key_dim=d_model//num_heads,
                    value_dim=d_model//num_heads,
                    output_dim=d_model,
                    hidden_dim=2048)
    y = encoder(x, use_causal_mask=False)
    print(y.shape)
    
            