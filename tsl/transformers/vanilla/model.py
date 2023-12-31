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

class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        # donot use '+' as it will not propagate the mask
        self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention): 
    def call(self, x, enc_output):
        attn_output, attn_scores = self.mha(
            query = x, 
            key = enc_output,
            value = enc_output, 
            return_attention_scores=True)
        # save the attention scores for visualization
        self.last_attn_scores = attn_scores
        
        # add and norm
        x = self.add([x, attn_output])
        x = self.layer_norm(x)
        
        return x

class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        attn_output = self.mha(
            query = x,
            value = x,
            key = x
        )
        
        # add and norm
        x = self.add([x, attn_output])
        x = self.layer_norm(x)

        return x
    
class CausalSelfAttention(BaseAttention):
    # the new tensorflow version has the causal mask built in
    # no need to manually create the mask
    def call(self, x):
        attn_output = self.mha(
            query = x,
            key = x,
            value = x,
            use_causal_mask=True
        )
        
        x = self.add([x, attn_output])
        x = self.layer_norm(x)
        
        return x
        
class FeedForward(tf.keras.layers.Layer):
    def __init__(self, embedding_dim, ffn_hidden_dim, dropout_rate=0.1, **kwargs):
        super().__init__()
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ffn_hidden_dim, activation='relu'),
            tf.keras.layers.Dense(embedding_dim),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.ffn(x)])
        x = self.layer_norm(x)
        return x

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, ffn_hidden_dim, dropout_rate=0.1, **kwargs):
        super().__init__()
        self.self_attention = GlobalSelfAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads, 
            dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model, ffn_hidden_dim, dropout_rate)
        
    def call(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x

class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, ffn_hidden_dim, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model 
        self.num_layers = num_layers
        
        # positional embedding        
        self.pos_embedding = PositionalEmbedding(vocab_size, d_model, dropout_rate)
        
        # encoder layers
        self.enc_layers = [
            EncoderLayer(d_model, num_heads, ffn_hidden_dim, dropout_rate) 
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
    
    def call(self, x):
        # x: (B, seq_len) -> (B, seq_len, d_model)
        x = self.pos_embedding(x)
        
        # add a dropout layer
        x = self.dropout(x)
        
        # for each encoder layer
        # (B, seq_len, d_model) -> (B, seq_len, d_model)
        for i in range(self.num_layers):
            x = self.enc_layers[i](x)
        
        return x
    
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, d_model, num_heads, ffn_hidden_dim, dropout_rate=0.1):
        super().__init__()
        self.causal_self_attention = CausalSelfAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads, 
            dropout=dropout_rate
        )
        self.cross_attention = CrossAttention(
            num_heads=num_heads, 
            key_dim=d_model // num_heads, 
            dropout=dropout_rate
        )
        self.ffn = FeedForward(d_model, ffn_hidden_dim, dropout_rate)
        
    def call(self, x, enc_output):
        # causal self attention
        x = self.causal_self_attention(x)
        
        # cross attention
        x = self.cross_attention(x, enc_output)
        
        # cache the last attention scores for visualization
        self.last_attn_scores = self.cross_attention.last_attn_scores
        
        # ffn
        x = self.ffn(x)
        
        return x

class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, d_model, num_heads, ffn_hidden_dim, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.d_model = d_model 
        self.num_layers = num_layers
        
        # positional embedding        
        self.pos_embedding = PositionalEmbedding(vocab_size, d_model, dropout_rate)
        
        # decoder layers
        self.dec_layers = [
            DecoderLayer(d_model, num_heads, ffn_hidden_dim, dropout_rate) 
            for _ in range(num_layers)
        ]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

        self.last_attn_scores = None
    
    def call(self, x, enc_output):
        # x: (B, target_seq_len) -> (B, target_seq_len, d_model)
        x = self.pos_embedding(x)
        
        # add a dropout layer
        x = self.dropout(x)
        
        # for each decoder layer
        # (B, seq_len, d_model) -> (B, seq_len, d_model)
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, enc_output)
        
        self.last_attn_scores = self.dec_layers[-1].last_attn_scores
        
        return x

class Transformer(tf.keras.Model):
    def __init__(self, *, num_layers, d_model, num_heads, ffn_hidden_dim, input_vocab_size, target_vocab_size, dropout_rate=0.1):
        super().__init__()
        self.encoder = Encoder(
            num_layers=num_layers, 
            d_model=d_model, 
            num_heads=num_heads, 
            dff=ffn_hidden_dim, 
            vocab_size=input_vocab_size, 
            dropout_rate=dropout_rate
        )
        self.decoder = Decoder(
            num_layers=num_layers, 
            d_model=d_model, 
            num_heads=num_heads, 
            dff=ffn_hidden_dim, 
            vocab_size=target_vocab_size, 
            dropout_rate=dropout_rate
        )
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        
    def call(self, inputs):
        # To use a Keras model with `.fit` you must pass all your inputs in the
        # first argument.
        x, y = inputs
        # x: (B, seq_len) -> (B, seq_len, d_model)
        enc_output = self.encoder(x)
        
        # y: (B, target_seq_len) -> (B, target_seq_len, d_model)
        dec_output = self.decoder(y, enc_output)
        
        # (B, target_seq_len, d_model) -> (B, target_seq_len, vocab_size)
        final_output = self.final_layer(dec_output)
        
        # drop the keras mask
        try:
            del final_output._keras_mask
        except AttributeError:
            pass
        
        return final_output

   
if __name__ == "__main__":
    
    embed = PositionalEmbedding(vocab_size=1000, embedding_dim=50)

    seq = tf.random.uniform((64, 10), maxval = 1000, dtype=tf.int32)
    out = embed(seq)
    print(seq.shape, out.shape)
    print(out._keras_mask)

    # cross attention
    sample_ca = CrossAttention(num_heads=2, key_dim=512)

    out2 = out + 0.1
    print(sample_ca(out, out2).shape)

    # ffn
    sample_ffn = FeedForward(50, 2048)
    print(sample_ffn(out).shape)
    
    # encoder layer
    enc_layer = EncoderLayer(50, 10, 2048)
    print(enc_layer(out).shape)
    
    # encoder 
    encoder = Encoder(num_layers=2, d_model=50, num_heads=10, ffn_hidden_dim=2048, vocab_size=1000)
    print(encoder(seq, training=True).shape)
    
    # test multihead attention: k_dim = d_model // num_heads
    mha = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=512)
    target = tf.keras.Input(shape=[8, 50])
    source = tf.keras.Input(shape=[10, 50])
    output, att_score = mha(query=target, value=source, key=source, return_attention_scores=True)
    print(output.shape, att_score.shape)
    print(mha.trainable_variables[0].shape)