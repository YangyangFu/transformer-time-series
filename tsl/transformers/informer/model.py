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
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()
        self.layernorm2 = tf.keras.layers.LayerNormalization()

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
        out = x = self.layernorm1(x, training=training)

        # pointwise feed forward, dropout, add and norm
        x = self.conv1d_1(x)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        
        
        x = self.conv1d_2(x)
        x = self.dropout(x, training=training)
        x = self.add([out, x])
        x = self.layernorm2(x, training=training)

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

class CrossAttentionBlock(tf.keras.layers.Layer):
    def __init__(self, 
                num_heads,
                key_dim,
                value_dim,
                output_dim,
                hidden_dim,
                dropout_rate=0.1,
                **kwargs):
        super().__init__(**kwargs)
        self.mha = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            output_shape=output_dim,
            dropout=dropout_rate)
        self.last_attn_weights = None
        
        # dropout, residual, layernorm
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.add = tf.keras.layers.Add()
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        
        # pointwise feed forward
        self.conv1d_1 = tf.keras.layers.Conv1D(filters=hidden_dim, kernel_size=1, strides=1)
        self.conv1d_2 = tf.keras.layers.Conv1D(filters=output_dim, kernel_size=1, strides=1)
        self.activation = tf.keras.layers.Activation('elu')
        
    def call(self, x, context, use_causal_mask=True, training=True, **kwargs):
        # x: (B, T, D)
        # context: (B, S, D)
        x_new, attn_weights = self.mha(query = x,
                     key = context,
                     value = context,
                     return_attention_scores=True,
                     use_causal_mask=use_causal_mask,
                     training = training)
        self.last_attn_weights = attn_weights 
        
        # dropout, residual, layernorm
        # mha from tensorflow.keras already implemented dropout
        x = self.add([x, x_new])
        x_skip = x = self.layernorm1(x, training=training)
        
        # pointwise feed forward
        x = self.conv1d_1(x)
        x = self.activation(x)
        x = self.dropout(x, training=training)
        
        x = self.conv1d_2(x)
        x = self.dropout(x, training=training)
        x = self.add([x_skip, x])
        x = self.layernorm2(x, training=training)
        
        return x
        
class DecoderLayer(tf.keras.layers.Layer):
    
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
        self.mask_ps_attn = ProbSpareAttentionBlock(num_heads=num_heads,
                                                key_dim=key_dim,
                                                value_dim=value_dim,
                                                output_dim=output_dim,
                                                hidden_dim=hidden_dim,
                                                factor=factor,
                                                dropout_rate=dropout_rate)
        self.cross_attn = CrossAttentionBlock(num_heads=num_heads,
                                             key_dim=key_dim,
                                             value_dim=value_dim,
                                             output_dim=output_dim,
                                             hidden_dim=hidden_dim,
                                             dropout_rate=dropout_rate)
        
    def call(self, x, context, **kwargs):
        # masked self probsparse attention 
        x = self.mask_ps_attn([x, x, x])
        # cross attention
        x = self.cross_attn(x, context)
        return x 
                     
class Decoder(tf.keras.layers.Layer):
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
        self.dec_layers = [DecoderLayer(num_heads=num_heads,
                                        key_dim=key_dim,
                                        value_dim=value_dim,
                                        output_dim=output_dim,
                                        hidden_dim=hidden_dim,
                                        factor=factor,
                                        dropout_rate=dropout_rate) for _ in range(num_layers)]
    
    def call(self, x, context, **kwargs):
        
        for dec_layer in self.dec_layers:
            x = dec_layer(x, context)
        
        return x
        

if __name__ == '__main__':
    
    d_model = 512
    source_seq_len = 64
    target_seq_len = 128
    num_heads = 16
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
    y_enc = encoder(x, use_causal_mask=False)
    print("shape after encoder with 4 layers:", y.shape)
    
    
    # decoder layer
    num_heads_dec = 8
    
    y = tf.random.normal((32, target_seq_len, d_model))
    
    # cross attention block
    cross_attn = CrossAttentionBlock(num_heads=num_heads_dec,
                                    key_dim=d_model//num_heads_dec,
                                    value_dim=d_model//num_heads_dec,
                                    hidden_dim=2048,
                                    output_dim=d_model)
    y_out = cross_attn(y, y_enc)
    print("shape after cross attention", y_out.shape)
    
    # decoder layer
    dec_layer = DecoderLayer(num_heads=num_heads_dec,
                             key_dim=d_model//num_heads_dec,
                             value_dim=d_model//num_heads_dec,
                             output_dim=d_model,
                             hidden_dim=2048)
    y_out = dec_layer(y, y_enc)                          
    print("shape after one decoder layer:", y_out.shape)
    
    # decoder
    dec = Decoder(num_layers=2,
                num_heads=num_heads_dec,
                key_dim=d_model//num_heads_dec,
                value_dim=d_model//num_heads_dec,
                output_dim=d_model,
                hidden_dim=2048)