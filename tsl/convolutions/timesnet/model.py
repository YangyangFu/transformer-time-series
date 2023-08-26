import tensorflow as tf 
from tsl.transformers.informer import TimeFeatureEmbedding, TemporalEmbedding, PositionalEmbedding, CategoricalEmbedding
from tsl.transformers.patch import InstanceNormalization 

class FFTPeriods(tf.keras.layers.Layer):
    def __init__(self, k, **kwargs):
        super(FFTPeriods, self).__init__(**kwargs)
        self.k = k 
        
    def call(self, inputs):
        # inputs: (batch_size, seq_len, d_model)
        # outputs: 
        #       periods: (k,1), 
        #       weights: (batch_size, k)
        
        # need decompose on time axis
        # (batch_size, n_rfft, d_model)
        xf = tf.transpose(tf.signal.rfft(tf.transpose(inputs, perm=[0,2,1])),
                          perm=[0,2,1])
        
        # find k periods by amplitude
        # (n_rfft, )
        frequency_list = tf.reduce_mean(tf.reduce_mean(abs(xf), axis=0), axis=-1)
        #frequency_list[0] = 0.0
        frequency_list = tf.tensor_scatter_nd_update(frequency_list, 
                                                     indices = tf.constant([[0]]), 
                                                     updates = tf.constant([0.0]))                             
        # find the top k frequency
        # (k, )
        _, top_k_indx = tf.math.top_k(frequency_list, k=self.k)
        
        # calculate periods based on top k frequency
        # (k, )
        periods = tf.cast(tf.math.divide(tf.shape(inputs)[1], top_k_indx), tf.int32)
        
        # calculate weights based on top k frequency
        # (batch_size, k)
        weights = tf.gather(tf.reduce_mean(abs(xf), axis=-1), top_k_indx, axis=-1)
        
        return periods, weights

class InceptionBlockV1(tf.keras.layers.Layer):
    """ Implementation of Inception V1 """
    def __init__(self, out_channels, num_kernels, initializer="he_normal", **kwargs):
        super(InceptionBlockV1, self).__init__(**kwargs)
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        self.initializer = initializer
        
    def build(self, input_shape):
        # input_shape: (batch_size, d_model, seq_len, period) 
        self.kernels = [tf.keras.layers.Conv2D(filters=self.out_channels, 
                                                    kernel_size=2*i+1, 
                                                    padding='same',
                                                    kernel_initializer=self.initializer)
                        for i in range(self.num_kernels)]
            
    def call(self, inputs):
        # inputs: (batch_size, d_model, seq_len, period)
        # channels last
        # (batch_size, seq_len, period, d_model)
        x = tf.transpose(inputs, perm=[0,2,3,1])
        # [(batch_size, out_channel, seq_len, period)]
        out = []
        for i in range(self.num_kernels):
            xo = self.kernels[i](x)
            xo = tf.transpose(xo, perm=[0,3,1,2])
            out.append(xo)
    
        #out = [tf.transpose(self.kernels[i](x), perm=[0,3,1,2]) for i in range(self.num_kernels)]
        # stack and mean on the period axis
        out = tf.reduce_mean(tf.stack(out, axis=-1), axis=-1)
        
        # (batch_size, out_chanel, seq_len, period)
        return out 

class InceptionV1(tf.keras.layers.Layer):
    def __init__(self, d_model, out_channels, num_kernels, **kwargs):
        super(InceptionV1, self).__init__(**kwargs)
        self.d_model = d_model
        self.out_channels = out_channels
        self.num_kernels = num_kernels

    def build(self, input_shape):
        # input_shape: (batch_size, seq_len, d_model)
        self.inp1 = InceptionBlockV1(out_channels=self.out_channels, num_kernels=self.num_kernels)
        self.activation = tf.keras.layers.Activation('gelu')
        self.inp2 = InceptionBlockV1(out_channels=self.d_model, num_kernels=self.num_kernels)

    def call(self, inputs, *args, **kwargs):
        # inputs: (batch_size, seq_len, d_model)
        x = self.inp1(inputs)
        x = self.activation(x)
        x = self.inp2(x)
        return x
    
class TimesBlock(tf.keras.layers.Layer):
    def __init__(self, k, conv_hidden_dim, num_kernels, **kwargs):
        super(TimesBlock, self).__init__(**kwargs)
        self.k = k
        self.conv_hidden_dim = conv_hidden_dim
        self.num_kernels = num_kernels
        
    def build(self, input_shape):
        # input_shape: (batch_size, seq_len, d_model)
        D = input_shape[-1]
        self.fft = FFTPeriods(k = self.k)
        self.softmax = tf.keras.layers.Softmax()
        self.conv = [InceptionV1(d_model = D, 
                                out_channels=self.conv_hidden_dim, 
                                num_kernels=self.num_kernels)
                    for _ in range(self.k)]
        
    def call(self, inputs):
        # inputs: (batch_size, seq_len, d_model)
        B = tf.shape(inputs)[0]
        L = tf.shape(inputs)[1]
        D = tf.shape(inputs)[2]
        
        # (k, ), (batch_size, k)
        periods, weights = self.fft(inputs)

        # reshape for each period
        x_cat = []
        for i in range(self.k):
            period = periods[i]
            # padding so that seq_len % period == 0
            x = inputs
            if L % period != 0:
                x = tf.pad(x, [[0,0], [0, period - L % period], [0, 0]])
            
            length = tf.shape(x)[1]
            # reshape to (batch_size, d_model, seq_len // period, period) 
            x = tf.transpose(tf.reshape(x, [B, length // period, period, D]),
                                perm=[0,3,1,2])
            
            # conv block
            # (batch_size, d_model, seq_len // period, period)
            x = self.conv[i](x)
            # reshape back
            # (batch_size, seq_len, d_model)
            x = tf.reshape(tf.transpose(x, perm=[0,2,3,1]), [B, length, D])
            x = tf.gather(x, tf.range(L), axis=1)
            
            # append for each period
            x_cat.append(x)
            
        # concat
        # (batch_size, seq_len, d_model, k)
        x = tf.stack(x_cat, axis=-1)
        
        # weighted aggregation
        # (batch_size, k)
        w = self.softmax(weights)
        # (batch_size, 1, 1, k)
        w = tf.expand_dims(tf.expand_dims(w, axis=1), axis=1)
        # (batch_size, seq_len, d_model, k)
        w = tf.tile(w, [1, L, D, 1])
        # weighted sum: (batch_size, seq_len, d_model)
        x = tf.reduce_sum(tf.multiply(x, w), axis=-1)
                
        return x

class InputEmbedding(tf.keras.layers.Layer):
    def __init__(self, 
                 embedding_dim,
                 num_cat_cov=0,
                 cat_cov_embedding_size=[],
                 cat_cov_embedding_dim=16,
                 time_embedding_type="time2vec",
                 use_holiday=True,
                 freq='H',
                 dropout_rate=0.1,
                 **kwargs):
        """ Data input embedding
        
        Args:
            embedding_dim (_type_): embedding dimension
            num_cat_cov (int, optional): number of categorical variates in the features. Defaults to 0.
            cat_cov_embedding_size (list, optional): embedding size for each categorical variates. Defaults to [].
            cat_cov_embedding_dim (int, optional): embedding dimension for each categorical variates. Defaults to 16.
            time_embedding_type (str, optional): embedding type for time features: "time2vec" or "temporal". Defaults to "time2vec".
            use_holiday (bool, optional): weather to use holiday embeddings. Defaults to True.
            freq (str, optional): _description_. Defaults to 'H'.
            dropout_rate (float, optional): _description_. Defaults to 0.1.
        """
        
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
        self.time_embedding = TemporalEmbedding(embedding_dim=embedding_dim, freq=freq, use_holiday=use_holiday) if time_embedding_type == "temporal" else TimeFeatureEmbedding(embedding_dim=embedding_dim)
        
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

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, topk, conv_hidden_dim, num_kernels, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.topk = topk
        self.conv_hidden_dim = conv_hidden_dim
        self.num_kernels = num_kernels

    def build(self, input_shape):
        self.blocks = []
        for i in range(self.num_layers):
            self.blocks.append(TimesBlock(k=self.topk, 
                                          conv_hidden_dim=self.conv_hidden_dim, 
                                          num_kernels=self.num_kernels))
        self.add = tf.keras.layers.Add()
        self.norm = tf.keras.layers.LayerNormalization()
        
    def call(self, inputs):
        # inputs: (batch_size, seq_len, d_model)
        x = inputs
        for i in range(self.num_layers):
            x = self.add([x, self.blocks[i](x)])
            x = self.norm(x)
        return x
    
class TimesNet(tf.keras.Model):
    def __init__(self,
                 target_cols_index,
                 pred_len,
                 hist_len,
                 num_layers,
                 embedding_dim, 
                 topk,
                 cov_hidden_dim,
                 num_kernels,
                 num_cat_cov=0, 
                 cat_cov_embedding_size=[], 
                 cat_cov_embedding_dim=4,
                 time_embedding_type="time2vec", 
                 use_holiday=True, 
                 freq='H', 
                 dropout_rate=0.1, 
                 **kwargs):
        
        super().__init__(**kwargs)
        self.target_cols_index = target_cols_index
        self.num_targets = len(target_cols_index)
        self.pred_len = pred_len
        self.hist_len = hist_len
        self.num_layers = num_layers
        self.topk = topk
        self.cov_hidden_dim = cov_hidden_dim
        self.num_kernels = num_kernels
        self.embedding_dim = embedding_dim
        self.num_cat_cov = num_cat_cov
        self.cat_cov_embedding_size = cat_cov_embedding_size
        self.cat_cov_embedding_dim = cat_cov_embedding_dim
        self.time_embedding_type = time_embedding_type
        self.use_holiday = use_holiday
        self.freq = freq
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        # input_shape: (batch_size, source_seq_len, num_features)
        
        # instance norm 
        self.ins_norm = InstanceNormalization()

        # embedding
        # (batch_size, seq_len, embedding_dim)
        self.input_embedding = InputEmbedding(embedding_dim=self.embedding_dim,
                                              num_cat_cov=self.num_cat_cov,
                                              cat_cov_embedding_dim=self.cat_cov_embedding_dim,
                                              cat_cov_embedding_size=self.cat_cov_embedding_size,
                                              time_embedding_type = self.time_embedding_type,
                                              use_holiday=self.use_holiday,
                                              freq=self.freq,
                                              dropout_rate=self.dropout_rate)
        # linear prediction
        # (batch_size, pred_len + hist_len, embedding_dim)
        self.linear = tf.keras.layers.Dense(units=self.pred_len + self.hist_len)
        
        # (batch_size, seq_len, embedding_dim)
        self.enc = Encoder(num_layers=self.num_layers,
                                 topk=self.topk,
                                 conv_hidden_dim=self.cov_hidden_dim,
                                 num_kernels=self.num_kernels)
        
        # linear output
        self.dec = tf.keras.layers.Dense(units=self.num_targets)
        
    
    def call(self, inputs):
        x_num, x_cat, x_time = inputs

        # instance norm on numeric features
        x_num = self.ins_norm(x_num, mode='norm')
        
        # embedding
        # (batch_size, seq_len, embedding_dim)
        x = self.input_embedding([x_num, x_cat, x_time])

        # linear prediction
        # (batch_size, pred_len + hist_len, embedding_dim)
        x = tf.transpose(x, perm=[0,2,1])
        x = self.linear(x)
        x = tf.transpose(x, perm=[0,2,1])
        
        # encoder
        x = self.enc(x)

        # linear output
        x = self.dec(x)

        #print(ss)
        return x[:, -self.pred_len:, :]