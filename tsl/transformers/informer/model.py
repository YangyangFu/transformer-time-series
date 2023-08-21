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
        
    def call(self, x, context, use_causal_mask=False, training=True):
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
        x = self.cross_attn(x, context, use_causal_mask=False)
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
                 embedding_dim,
                 num_cat_cov=0,
                 cat_cov_embedding_size=[],
                 cat_cov_embedding_dim=16,
                 use_holiday=True,
                 freq='H',
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
        if num_cat_cov > 0:
            assert len(cat_cov_embedding_size) == num_cat_cov
            self.cat_cov_embedding = CategoricalEmbedding(
                num_embedding=num_cat_cov, 
                embedding_size = cat_cov_embedding_size, 
                embedding_dim = cat_cov_embedding_dim,
                output_dim = embedding_dim)
        # embedding for time features
        self.time_embedding = TemporalEmbedding(embedding_dim=embedding_dim, freq=freq, use_holiday=use_holiday)
        # sequence relative position embedding
        self.pos_embedding = PositionalEmbedding(embedding_dim=embedding_dim)
        # add and dropout
        self.add = tf.keras.layers.Add()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs):
        # inputs
        # num_cov_enc: (batch_size, seq_len, num_num_cov)
        # cat_cov_enc: (batch_size, seq_len, num_cat_cov)
        # time_enc: (batch_size, seq_len, num_time_features)
        num_cov_enc, cat_cov_enc, time_enc = inputs
        
        batch_size = tf.shape(num_cov_enc)[0]
        
        num_cov = self.num_cov_embedding(num_cov_enc)
        cat_cov = self.cat_cov_embedding(cat_cov_enc) if hasattr(self, 'cat_cov_embedding') else tf.zeros_like(num_cov)
        # (seq_len, embedding_dim)
        pos = self.pos_embedding(num_cov_enc)
        # (batch_size, seq_len, embedding_dim)
        pos = tf.tile(tf.expand_dims(pos, axis=0), [batch_size, 1, 1])
        time = self.time_embedding(time_enc)
        x = self.add([num_cov, cat_cov, pos, time])
        x = self.dropout(x)
        
        # (batch_size, seq_len, embedding_dim)
        return x

class DecoderInputEmbedding(tf.keras.layers.Layer):
    """ Decoder input embedding
    
        Assume target(s) are numeric values
        
    """
    def __init__(self, embedding_dim, freq="H", use_holiday=True, dropout_rate=0.1, **kwargs):
        
        super().__init__(**kwargs)

        #self.supports_masking = True
        #(B, Lt+Ly, D)
        self.token_embedding = tf.keras.layers.Conv1D(filters=embedding_dim,kernel_size=3, padding='same', strides=1)
        #
        self.time_embedding = TemporalEmbedding(embedding_dim=embedding_dim, freq=freq, use_holiday=use_holiday)
        self.pos_embedding = PositionalEmbedding(embedding_dim=embedding_dim)
        self.add = tf.keras.layers.Add()
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def call(self, inputs):
        # inputs: cat(token,zeros)
        # time_dec: (batch_size, seq_len, num_time_features)
        # token_dec: (batch_size, seq_len, num_targets)
        time_dec, token_dec = inputs
        batch_size = tf.shape(time_dec)[0]
        
        token = self.token_embedding(token_dec)
        time = self.time_embedding(time_dec)
        # (seq_len, embedding_dim)
        pos = self.pos_embedding(token_dec)
        # (batch_size, seq_len, embedding_dim)
        pos = tf.tile(tf.expand_dims(pos, axis=0), [batch_size, 1, 1])
        
        x = self.add([token, time, pos])
        x = self.dropout(x)
        
        return x
        
class Informer(tf.keras.Model):
    def __init__(self, 
                 output_dim,
                 pred_len,
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
                 num_cat_cov=0,
                 cat_cov_embedding_size=[],
                 cat_cov_embedding_dim=16,
                 freq='H',
                 use_holiday=True,
                 dropout_rate=0.1, 
                 **kwargs):
        """ Informer model for time series forecasting.

        Args:
            output_dim (int): output dimension
            pred_len (int): prediction length
            num_layers_encoder (int, optional): number of encoder layers. Defaults to 4.
            num_heads_encoder (int, optional): number of heads in encoder. Defaults to 16.
            key_dim_encoder (int, optional): key dimension in encoder. Defaults to 32.
            value_dim_encoder (int, optional): value dimension in encoder. Defaults to 32.
            output_dim_encoder (int, optional): output dimension in encoder. Defaults to 512.
            hidden_dim_encoder (int, optional): hidden dimension in encoder. Defaults to 2048.
            factor_encoder (int, optional): factor in encoder. Defaults to 4.
            num_layers_decoder (int, optional): number of decoder layers. Defaults to 2.
            num_heads_decoder (int, optional): number of heads in decoder. Defaults to 8.
            key_dim_decoder (int, optional): key dimension in decoder. Defaults to 64.
            value_dim_decoder (int, optional): value dimension in decoder. Defaults to 64.
            output_dim_decoder (int, optional): output dimension in decoder. Defaults to 512.
            hidden_dim_decoder (int, optional): hidden dimension in decoder. Defaults to 2048.
            factor_decoder (int, optional): factor in decoder. Defaults to 4.
            num_cat_cov (int, optional): number of categorical covariates. Defaults to 0.
            cat_cov_embedding_size (list, optional): embedding size for each categorical covariate. Defaults to [].
            cat_cov_embedding_dim (int, optional): embedding dimension for categorical covariates. Defaults to 16.
            freq (str, optional): frequency of time series. Defaults to 'H'.
            use_holiday (bool, optional): whether to use holiday embedding. Defaults to True.
            dropout_rate (float, optional): dropout rate. Defaults to 0.1.
        
        """
        super().__init__(**kwargs)
        self.pred_len = pred_len
        
        # encoder input
        # (B, S, D)
        self.encoder_input_embedding = EncoderInputEmbedding(
            embedding_dim=output_dim_encoder,
            num_cat_cov=num_cat_cov,
            cat_cov_embedding_size=cat_cov_embedding_size,
            cat_cov_embedding_dim=cat_cov_embedding_dim,
            freq=freq,
            use_holiday=use_holiday,
            dropout_rate=dropout_rate)
                                                             
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
        
        # decoder input
        self.decoder_input_embedding = DecoderInputEmbedding(
            embedding_dim=output_dim_decoder,
            freq=freq,
            use_holiday=use_holiday,
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
        enc_in = self.encoder_input_embedding(x_enc)
        enc_out = self.encoder(enc_in)
        
        dec_in = self.decoder_input_embedding(x_dec)
        dec_out = self.decoder(dec_in, enc_out)
        dec_out = self.final_fc(dec_out)
        
        return dec_out[:, -self.pred_len:, :]
    
if __name__ == '__main__':
    
    from dataloader import DataLoader
    
    embed_dim = 512
    source_seq_len = 64
    target_seq_len = 128
    pred_len = 96
    n_num_covs = 7
    n_targets = 1
    
    # attention block
    model = Informer(output_dim=n_targets, 
                    pred_len=pred_len,
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
                    num_cat_cov=0,
                    cat_cov_embedding_size=[],
                    cat_cov_embedding_dim=16,
                    freq='H',
                    use_holiday=True,
                    dropout_rate=0.1,)
    
    # take a batch   
    num_covs = tf.random.uniform(shape=(32, source_seq_len, n_num_covs))
    cat_covs = None 
    time_enc = tf.random.uniform(shape=(32, source_seq_len, 7))
    
    # zero for target 
    token_dec = tf.random.uniform(shape=(32, target_seq_len-pred_len, n_targets))
    zeros = tf.zeros(shape=(32, pred_len, n_targets))
    target_dec = tf.concat([token_dec, zeros], axis=1)
    time_dec = tf.random.uniform(shape=(32, target_seq_len, 7))
        
    # feed model
    x_enc = [num_covs, cat_covs, time_enc]
    x_dec = [time_dec, target_dec]
    y = model(x_enc, x_dec)
    print(y.shape)
    print(model.summary())
    
    # check cross attention
    print(model.decoder.dec_layers[0].cross_attn.last_attn_weights.shape)
    
    # check prob sparse attention
    print(model.decoder.dec_layers[0].mask_ps_attn.last_attn_weights[0,0,:,:])
    
    

    

    