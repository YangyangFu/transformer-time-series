import collections
import tensorflow as tf

class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self,
        num_heads,
        key_dim,
        value_dim=None,
        dropout=0.0,
        use_bias=True,
        output_dim=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim is not None else key_dim
        self.use_bias = use_bias
        # TODO: how to make the output_dim equal to the input_dim, which is usually referred from given data? -> use build() method
        self.output_dim = output_dim if output_dim is not None else self.value_dim*self.num_heads
        
        # input projection
        self.query_projection = tf.keras.layers.Dense(units=self.num_heads * self.key_dim, use_bias=self.use_bias)
        self.key_projection = tf.keras.layers.Dense(units=self.num_heads * self.key_dim, use_bias=self.use_bias)
        self.value_projection = tf.keras.layers.Dense(units=self.num_heads * self.value_dim, use_bias=self.use_bias)
        
        # dropout
        self.dropout = tf.keras.layers.Dropout(dropout)
        
        # output
        self.output_projection = tf.keras.layers.Dense(units = self.output_dim, use_bias=self.use_bias)
        
            
    def _compute_causal_mask(self, query, value=None):
        """Computes a causal mask (e.g., for masked self-attention layers).

        For example, if query and value both contain sequences of length 4,
        this function returns a boolean `Tensor` equal to:

        ```
        [[[True,  False, False, False],
          [True,  True,  False, False],
          [True,  True,  True,  False],
          [True,  True,  True,  True]]]
        ```

        Args:
            query: query `Tensor` of shape `(B, T, ...)`.
            value: value `Tensor` of shape `(B, S, ...)` (optional, defaults to
                query).

        Returns:
            mask: a boolean `Tensor` of shape [1, T, S] containing a lower
                triangular matrix of shape [T, S].
        """
        q_seq_length = tf.shape(query)[1]
        v_seq_length = q_seq_length if value is None else tf.shape(value)[1]
        return tf.linalg.band_part(  # creates a lower triangular matrix
            tf.ones((1, q_seq_length, v_seq_length), tf.bool), -1, 0
        )
    
    def _compute_attention_mask(self, query, value=None, key=None, attention_mask=None, use_causal_mask=False):

        # attention propagated from previous layer such as embedding layer
        # attention_mask shape: [B, T], or [B, S]
        query_mask = getattr(query, "_keras_mask", None)
        value_mask = getattr(value, "_keras_mask", None)
        key_mask = getattr(key, "_keras_mask", None)    
        auto_mask = None
        
        if query_mask is not None:
            query_mask = tf.cast(query_mask, tf.bool)  # defensive casting
            # B = batch size, T = max query length
            auto_mask = query_mask[:, :, tf.newaxis]  # shape is [B, T, 1]
        if value_mask is not None:
            value_mask = tf.cast(value_mask, tf.bool)  # defensive casting
            # B = batch size, S == max value length
            mask = value_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask
        if key_mask is not None:
            key_mask = tf.cast(key_mask, tf.bool)  # defensive casting
            # B == batch size, S == max key length == max value length
            mask = key_mask[:, tf.newaxis, :]  # shape is [B, 1, S]
            auto_mask = mask if auto_mask is None else auto_mask & mask        
        
        if use_causal_mask:
            # the shape of causal mask is [1, T, S]
            mask = self._compute_causal_mask(query, value)
            # update auto_mask
            auto_mask = mask if auto_mask is None else auto_mask & mask
        
        if auto_mask is not None:
            # merge attention mask and auto_mask to shape [B, T, S]
            attention_mask = auto_mask if attention_mask is None else tf.cast(attention_mask, tf.bool) & auto_mask

        return attention_mask
    
    def _compute_attention_raw_score(self, query, key, scale=True):
        """ scaled dot product score

        Args:
            query (_type_): _description_
            key (_type_): _description_
        """
        # query shape: [B, T, H, D]
        # key shape: [B, S, H, D]
        # score shape: [B, H, T, S]
        d_k = tf.shape(query)[-1]
        # scale query
        if scale:
            query = tf.multiply(query, 1.0 / tf.math.sqrt(tf.cast(d_k, query.dtype)))
        
        #
        scores = tf.matmul(
            tf.transpose(query, perm=[0,2,1,3]), 
            tf.transpose(key, perm=[0,2,3,1]))

        
        return scores
        
    def _compute_masked_softmax(self, scores, attention_mask):

        # score shape: [B, H, T, S]
        # mask shape: [B, T, S] or None
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, scores.dtype)
            # check the shape of attention_mask
            if attention_mask.shape != scores.shape:    
                # warning
                Warning(
                    f"Attention mask shape {tf.shape(attention_mask)} "
                    f"not compatible with attention logits shape "
                    f"{tf.shape(scores)}."
                )
                
                # reshape attention_mask to [B, 1, T, S]
                attention_mask = tf.expand_dims(attention_mask, axis=1)
                
            # add a very small value to attention_mask where it is 0 (no attention at that position)
            scores += -10e9 * (1.0 - attention_mask)
            
        # add softmax to scores
        return tf.nn.softmax(scores, axis=-1)
            
    def _compute_attention(self, scores, value):
        """compute attention using scores and value

        Args:
            scores (_type_): _description_
            value (_type_): _description_
        """
        # scores shape: [B, H, T, S]
        # value shape: [B, S, H, D]
        # context shape: [B, H, T, D]
        context = tf.matmul(scores, 
                            tf.transpose(value, perm=[0,2,1,3])
                            )
        return context
    
    def call(self, 
             inputs, 
             scale=True, 
             attention_mask=None, 
             use_causal_mask=True, 
             return_attention_scores=True, 
             training=True
             ):
        
        # inputs: [query, key, value]
        # query shape: [B, T, Dk]
        # key shape: [B, S, Dk]
        # value shape: [B, S, Dv]
        query, key, value = inputs
        _, T, _ = query.shape
        _, S, _ = key.shape

        # linear projection
        # (B, T, Dk) -> (B, T, H*dk) -> (B, T, H, dk)
        query = self.query_projection(query)
        # reshape operation drops the mask, so we need to save it and restore it after the reshape
        # this is currently hard-coded under the assumption that the mask shape is [B, T], and the reshape will not change the first two dimensions
        # TODO: current tensorflow version does not support _keras_mask attribute for tf.Tensor
        # need a fresh implementation.
        query_mask = getattr(query, "_keras_mask", None)
        query = tf.reshape(query, (-1, T, self.num_heads, self.key_dim))
        query._keras_mask = query_mask
        
        # (B, S, Dk) -> (B, S, H*dk) -> (B, S, H, dk)
        key = self.key_projection(key)
        key_mask = getattr(key, "_keras_mask", None)
        key = tf.reshape(key, (-1, S, self.num_heads, self.key_dim))
        key._keras_mask = key_mask
        
        # (B, S, Dv) -> (B, S, H*dv) -> (B, S, H, dv)
        value = self.value_projection(value)
        value_mask = getattr(value, "_keras_mask", None)
        value = tf.reshape(value, (-1, S, self.num_heads, self.value_dim))
        value._keras_mask = value_mask
        
        # compute attention mask
        # (B, T, S)
        attention_mask = self._compute_attention_mask(query, value, key, attention_mask, use_causal_mask)

        # compute dot scale product attention
        # (B, H, T, S)
        attention_scores = self._compute_attention_raw_score(query, key, scale)

        # compute masked softmax
        # (B, H, T, S)
        attention_scores = self._compute_masked_softmax(attention_scores, attention_mask)
        attention_scores._keras_mask = attention_mask
 
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # (B, H, T, S)
        attention_scores_dropout = self.dropout(attention_scores, training=training)
        
        # compute attention
        # (B, H, T, Dv)
        attentions = self._compute_attention(attention_scores_dropout, value)
        
        # output projection
        # (B, H, T, Dv) -> (B, T, H*Dv) -> (B, T, O)
        attentions = tf.reshape(
                tf.transpose(attentions, perm=[0,2,1,3]),
                (-1, T, self.num_heads * self.value_dim)
                )
        attentions._keras_mask = query._keras_mask
        attentions = self.output_projection(attentions)
        
        if return_attention_scores:
            return attentions, attention_scores
        
        return attentions