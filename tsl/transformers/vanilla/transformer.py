import tensorflow as tf
import numpy as np

class PositionalEncoder(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

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
    
    def __init__(self, vocab_size, embedding_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        # mask_zero=True to support variable length sequences using masking
        # padding mask is added in the encoder
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.pos_encoding = PositionalEncoder()
        # drop
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        
    def compute_mask(self, *args, **kwargs):
        # computer padding mask: mask all the 0s in the input
        return self.embedding.compute_mask(*args, **kwargs)
    
    def call(self, inputs, training):

        # embedding: (B, seq, embedding_dim)
        x = self.embedding(inputs)
        # scale the embedding by sqrt(embedding_dim)
        x *= tf.math.sqrt(tf.cast(self.embedding_dim, tf.float32))
        # add positional embedding: (1, seq, embedding_dim)
        x += self.pos_encoding(x)
        # pass the encoded embedding through a dropout layer
        x = self.dropout(x, training = training)
        
        return x 

class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        # donot use '+' as it will not propagate the mask
        self.add = tf.keras.layers.Add()

class CrossAttention(BaseAttention): 
    def call(self, x, context):
        attn_output, attn_scores = self.mha(
            query = x, 
            key = context,
            value = context, 
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
    def __init__(self, embedding_dim, dff, dropout_rate=0.1, **kwargs):
        super().__init__()
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(embedding_dim),
            tf.keras.layers.Dropout(dropout_rate)
        ])
        self.add = tf.keras.layers.Add()
        self.layer_norm = tf.keras.layers.LayerNormalization()

    def call(self, x):
        x = self.add([x, self.ffn(x)])
        x = self.layer_norm(x)
        return x



if __name__ == "__main__":
    
    embed = PositionalEmbedding(vocab_size=1000, embedding_dim=50)

    seq = tf.random.uniform((64, 10), maxval = 1000, dtype=tf.int32)
    out = embed(seq)
    print(seq.shape, out.shape)
    #print(embed._keras_mask)

    # cross attention
    sample_ca = CrossAttention(num_heads=2, key_dim=512)

    out2 = out + 0.1
    print(sample_ca(out, out2).shape)

    # ffn
    sample_ffn = FeedForward(50, 2048)
    print(sample_ffn(out).shape)