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
    pass

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
    def __init__(self, embedding_dim, pos_type="learn", name='encoder_input_embedding'):
        super(EncoderInputEmbedding, self).__init__(name=name)
        self.embedding_dim = embedding_dim
        self.pos_type = pos_type
        
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
        x += pos
        
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
    def __init__(self, num_heads, key_dim, value_dim, dropout_rate=0.2, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
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
        self.norm = tf.keras.layers.LayerNormalization()
    
    def call(self, inputs):
        x = self.mha(query = inputs, 
                            value = inputs,
                            key = inputs)
        x = self.add([x, inputs])
        x = self.norm(x)
        return x

class EncoderLayer(tf.keras.layers.Layer):
    """ A vanilla transformer encoder layer"""
    def __init__(self, d_model, num_heads, ffn_hidden_dim, dropout_rate = 0.2, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.key_dim = d_model // num_heads
        self.value_dim = d_model // num_heads
        self.ffn_hidden_dim = ffn_hidden_dim
        self.dropout_rate = dropout_rate
    
    def build(self, input_shape):
        self.mha = AttentionBlock(num_heads=self.num_heads,
                                key_dim=self.key_dim,
                                value_dim=self.value_dim,
                                dropout=self.dropout_rate)
        self.ffn = FeedForwardBlock(out_dim = self.d_model,
                                    hidden_dim = self.ffn_hidden_dim,
                                    dropout_rate = self.dropout_rate)
    
    def call(self, inputs):
        x = self.mha(inputs)
        x = self.ffn(x)
        return x

class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, ffn_hidden_dim, dropout_rate, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.ffn_hidden_dim = ffn_hidden_dim
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
        self.layers = [EncoderLayer(d_model=self.d_model,
                                    num_heads=self.num_heads,
                                    ffn_hidden_dim=self.ffn_hidden_dim,
                                    dropout_rate=self.dropout_rate) 
                       for _ in range(self.num_layers)] 
    
    def call(self, inputs):
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs

class LinearHead(tf.keras.layers.Layer):
    def __init__(self, out_dim, num_targets, **kwargs):
        super(LinearHead, self).__init__(**kwargs)
        self.out_dim = out_dim
        self.num_targets = num_targets
        
    def build(self, input_shape):
        # input_shape: [BxM, patch_num, d_model]
        # output_shape: [B, M, out_dim]
        self.linear = tf.keras.layers.Dense(self.out_dim)
    
    def call(self, inputs):
        # inputs: [BxM, patch_num, d_model]
        # flatten the first dimension
        # [B, M, patch_num, d_model]
        #x = tf.reshape(inputs, [-1, self.num_targets, inputs.shape[-2], inputs.shape[-1]])
        # [B, M, patch_num*d_model]
        x = tf.reshape(x, [-1, self.num_targets, inputs.shape[-2]*inputs.shape[-1]])
        x = self.linear(inputs)
        return x

class PatchTST(tf.keras.Model):
    def __init__(self,
                 pred_len,
                 num_targets, 
                 embedding_dim,
                 num_layers,
                 num_heads,
                 ffn_hidden_dim,
                 patch_size, 
                 patch_strides, 
                 patch_padding="end", 
                 dropout_rate=0.2,  
                 **kwargs):
        super(PatchTST, self).__init__(**kwargs)
        self.pred_len = pred_len
        self.num_targets = num_targets
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.ffn_hidden_dim = ffn_hidden_dim
        self.patch_size = patch_size
        self.patch_strides = patch_strides
        self.patch_padding = patch_padding
        self.dropout_rate = dropout_rate
        
    def build(self, input_shape):
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
                                      num_targets=self.num_targets)
    
    def call(self, inputs):
        # inputs: [batch_size, seq_len, feature_dim]
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
        
        # linear head
        # [batch_size, feature_dim, pred_len]
        x = self.linear_head(x)
        
        return x
    
if __name__ == '__main__':
    
    x = tf.random.normal([2, 10, 3])
    emb_patch = Patching(patch_size=2, patch_strides=1, patch_padding="end")
    patch_out = emb_patch(x)
    print(patch_out.shape)
    
    d_model = 6
    emb_pos = LearnablePositionalEmbedding(embedding_dim=d_model)
    pos_out = emb_pos(tf.transpose(patch_out, [0,1,3,2]))
    print(pos_out.shape)
    
    emb_input = EncoderInputEmbedding(embedding_dim=d_model)
    emb_out = emb_input(patch_out)
    print(emb_out.shape)
    
    