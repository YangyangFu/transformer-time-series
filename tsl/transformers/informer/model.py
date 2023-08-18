import tensorflow as tf 

from multihead_attention import MultiHeadAttention
from multihead_probsparse_attention import MultiHeadProbSparseAttention
from preprocessor import TemporalEmbedding, PositionalEmbedding, CategoricalEmbedding

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
        
    def call(self, x, training=True, **kwargs):
        # no need to use causal mask in encoder
        # x: (batch_size, seq_len, output_dim)
        x = self.psa([x, x, x], 
                    use_causal_mask=False,
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
    
    def call(self, x, training=True, **kwargs):
        # x: (batch_size, seq_len, output_dim)
        
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training=training)
            
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
        # cross attention with causal mask
        x = self.cross_attn(x, context, use_causal_mask=True)
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

class EncoderInputEmbedding(tf.keras.layers.Layer):
    def __init__(self, 
                 seq_len,
                 embedding_dim,
                 num_cat_cov=None,
                 cat_cov_embedding_size=None,
                 cat_cov_embedding_dim=None,
                 dropout_rate=0.1,
                 **kwargs):
        super().__init__(**kwargs)
        
        # preprocessor for numeric covariate features
        # conv1d treats the second last dim as time dimension, and the last dim as channel dimension
        self.num_cov_embedding = tf.keras.layers.Conv1D(filters=embedding_dim, 
                                                        kernel_size=3,
                                                        padding='same', 
                                                        strides=1)
        # preporcessfor for categorical covariate features
        if num_cat_cov is not None:
            assert len(cat_cov_embedding_size) == num_cat_cov
            self.cat_cov_embedding = CategoricalEmbedding(
                num_embedding=num_cat_cov, 
                embedding_size = cat_cov_embedding_size, 
                embedding_dim = cat_cov_embedding_dim,
                output_dim = embedding_dim,
                dropout_rate=dropout_rate,)
        # embedding for time features
        self.time_embedding = TemporalEmbedding(embedding_dim=embedding_dim, freq="H", use_holiday=True)
        # sequence relative position embedding
        self.pos_embedding = PositionalEmbedding(embedding_dim=embedding_dim)
        # add and dropout
        self.add = tf.keras.layers.Add()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs, **kwargs):
        # inputs
        # num_cov_enc: (batch_size, seq_len, num_num_cov)
        # cat_cov_enc: (batch_size, seq_len, num_cat_cov)
        # time_enc: (batch_size, seq_len, num_time_features)
        num_cov_enc, cat_cov_enc, time_enc = inputs
        
        num_cov = self.num_cov_embedding(num_cov_enc)
        if hasattr(self, 'cat_cov_embedding'):
            cat_cov = self.cat_cov_embedding(cat_cov_enc)
        
        pos = self.pos_embedding(num_cov_enc)
        time = self.time_embedding(time_enc)
        x = self.add([num_cov, cat_cov, pos, time])
        x = self.dropout(x)
        
        # (batch_size, seq_len, embedding_dim)
        return x
        
class Informer(tf.keras.Model):
    def __init__(self, 
                 output_dim,
                 num_layers_encoder=4, 
                 num_heads_encoder=16, 
                 key_dim_encoder=32, 
                 value_dim_encoder=32, 
                 output_dim_encoder=512, 
                 hidden_dim_encoder=2048, 
                 factor_encoder=4,
                 num_layers_decoder=2, 
                 num_heads_decoder=8, 
                 key_dim_decoder=64, 
                 value_dim_decoder=64, 
                 output_dim_decoder=512, 
                 hidden_dim_decoder=2048, 
                 factor_decoder=4, 
                 dropout_rate=0.1, 
                 **kwargs):
        """ Informer model for time series forecasting.

        Args:
            output_dim (int): model output dimension
            num_layers_encoder (_type_): number of encoder layers
            num_heads_encoder (_type_): number of heads in each encoder layer
            key_dim_encoder (_type_): key dimension in each head of encoder layer
            value_dim_encoder (_type_): value dimension in each head of encoder layer
            output_dim_encoder (_type_): output dimension of each encoder layer
            hidden_dim_encoder (_type_): hidden dimension of the fully connected network in each encoder layer
            factor_encoder (_type_): factor to determine the number of selected keys in each encoder layer
            num_layers_decoder (_type_): number of decoder layers
            num_heads_decoder (_type_): number of heads in each decoder layer
            key_dim_decoder (_type_): key dimension in each head of decoder layer
            value_dim_decoder (_type_): value dimension in each head of decoder layer
            output_dim_decoder (_type_): output dimension of each decoder layer
            hidden_dim_decoder (_type_): hidden dimension of the fully connected network in each decoder layer
            factor_decoder (_type_): factor to determine the number of selected keys in each decoder layer
            dropout_rate (float, optional): _description_. Defaults to 0.1.
        
        """
        
        
        super().__init__(**kwargs)
        
        # (B, S, D)
        self.encoder = Encoder(
            num_layers=num_layers_encoder,
            num_heads=num_heads_encoder,
            key_dim=key_dim_encoder,
            value_dim=value_dim_encoder,
            output_dim=output_dim_encoder,
            hidden_dim=hidden_dim_encoder,
            factor=factor_encoder,
            dropout_rate=dropout_rate)
        
        # (B, T, D)
        self.decoder = Decoder(
            num_layers=num_layers_decoder,
            num_heads=num_heads_decoder,
            key_dim=key_dim_decoder,
            value_dim=value_dim_decoder,
            output_dim=output_dim_decoder,
            hidden_dim=hidden_dim_decoder,
            factor=factor_decoder,
            dropout_rate=dropout_rate)
        # (B, T, O)
        self.final_fc = tf.keras.layers.Dense(units=output_dim)
    
    def call(self, x_enc, x_dec):
        
        enc_out = self.encoder(x_enc)
        dec_out = self.decoder(x_dec, enc_out)
        dec_out = self.final_fc(dec_out)
        
        return dec_out
    
if __name__ == '__main__':
    
    out_model = 72
    embed_dim = 512
    source_seq_len = 64
    target_seq_len = 128
    
    x_enc = tf.random.normal((32, source_seq_len, embed_dim))
    x_dec = tf.random.normal((32, target_seq_len, embed_dim))
    padding_mask = tf.random.uniform((32, source_seq_len), minval=0, maxval=2, dtype=tf.int32)
    padding_mask = tf.cast(padding_mask, tf.bool)
    use_mask = False
    if use_mask:
        x_enc._keras_mask = padding_mask
    
    # attention block
    model = Informer(output_dim=out_model)
    y = model(x_enc, x_dec)
    print(y.shape)
    print(model.summary())
    