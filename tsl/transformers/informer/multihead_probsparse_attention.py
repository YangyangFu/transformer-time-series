import collections
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
import numpy as np 

class MultiHeadProbSparseAttention(tf.keras.layers.Layer):
    def __init__(self,
        num_heads,
        key_dim,
        value_dim=None,
        factor = 4,
        use_bias=True,
        output_dim=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim if value_dim is not None else key_dim
        # force factor to be int16
        self.factor = tf.cast(factor, tf.int32)
        self.use_bias = use_bias
        # TODO: how to make the output_dim equal to the input_dim, which is usually referred from given data? -> use build() method
        self.output_dim = output_dim if output_dim is not None else self.value_dim*self.num_heads
        
        # input projection
        self.query_projection = tf.keras.layers.Dense(units=self.num_heads * self.key_dim, use_bias=self.use_bias)
        self.key_projection = tf.keras.layers.Dense(units=self.num_heads * self.key_dim, use_bias=self.use_bias)
        self.value_projection = tf.keras.layers.Dense(units=self.num_heads * self.value_dim, use_bias=self.use_bias)
        
        # dropout: no dropout in the paper
        
        # output
        self.output_projection = tf.keras.layers.Dense(units = self.output_dim, use_bias=self.use_bias)
        
        # support propagation of mask
        self.supports_masking = True
            
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
        # shape: [B, T], or [B, S]
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
            # to (B, T, S)
            mask = tf.tile(mask, [tf.shape(query)[0], 1, 1])
            # update auto_mask
            auto_mask = mask if auto_mask is None else auto_mask & mask
        
        if auto_mask is not None:
            # merge attention mask and auto_mask to shape [B, T, S]
            attention_mask = auto_mask if attention_mask is None else tf.cast(attention_mask, tf.bool) & auto_mask

        # check dimensions
        
        return attention_mask
    
    def _compute_s0(self, query, value, attention_mask):
        # query: (B, T, H, dk)
        # cannot use tf.shape() to get the shape of a tensor in eager mode
        S = tf.shape(value)[1]
        T = tf.shape(query)[1]
        
        if attention_mask is not None:
            assert (S == T) # require S == T for self-attention
            # (B, S, H, dv)
            context = tf.cumsum(value, axis=1)
        else:
            # (B, 1, H, dv)
            context = tf.reduce_mean(value, axis=1, keepdims=True)
            context = tf.tile(context, [1, T, 1, 1])
        
        # (B, T, H, dv)
        return context
    
    def _compute_s1(self, context, q_top_k, top_index, attention_mask, scale=True):
        
        # context: (B, T, H, dv) T=S for self-attention
        # q_top_k: (B, H, n_top, S)
        # index: (B, n_top, H)
        # value: (B, S, H, dv)
        # attention_mask: (B, T, S)
        B = tf.shape(context)[0]
        H = tf.shape(context)[2]

        if scale:
            q_top_k = tf.multiply(q_top_k, 1.0 / tf.math.sqrt(tf.cast(self.key_dim, q_top_k.dtype)))
            
        if attention_mask is not None:
            attention_mask = tf.cast(attention_mask, q_top_k.dtype)
            # check the shape of attention_mask
            if attention_mask.shape != q_top_k.shape:    
                # warning
                Warning(
                    f"Attention mask shape {tf.shape(attention_mask)} "
                    f"not compatible with attention logits shape "
                    f"{tf.shape(q_top_k)}."
                )
                
                # get attention mask for q_top_k: (B, H, sample_k, S)
                # (B, T, S) -> (B, H, T, S)
                attention_mask = tf.tile(tf.expand_dims(attention_mask, axis=1), [1, H, 1, 1])
                # (B, H, T, S) -> (B, H, top_n, S)
                attention_mask = attention_mask[tf.range(B)[:, tf.newaxis, tf.newaxis], 
                                                tf.range(H)[tf.newaxis,:, tf.newaxis],
                                                tf.transpose(top_index, perm=[0,2,1]), 
                                                :]
                
            # add a very small value to attention_mask where it is 0 (no attention at that position)
            # (B, H, n_top, S)
            q_top_k += -10e9 * (1.0 - attention_mask)
            
        # add softmax to scores
        return tf.nn.softmax(q_top_k, axis=-1)        
            
    def _compute_attention(self, query, key, value, sample_k, n_top, attention_mask, scale=True):
        """compute attention using scores and value

        Args:
            scores (_type_): _description_
            value (_type_): _description_
        """
        # scores shape: [B, H, T, S]
        # value shape: [B, S, H, D]
        # context shape: [B, H, T, D]
        B = tf.shape(query)[0]
        T = tf.shape(query)[1]
        H = tf.shape(query)[2]
        S = tf.shape(key)[1]
        
        # (B, H, n_top, S), (B, n_top, H)
        q_top_k, top_index = self._prob_QK(query, key, sample_k, n_top)
        # (B, T, H, dv)
        context = self._compute_s0(query, value, attention_mask)
        # (B, H, n_top, S)
        scores = self._compute_s1(context, q_top_k, top_index, attention_mask, scale)
        # (B, H, n_top, dv)
        attention = tf.matmul(scores, 
                              tf.transpose(value, perm=[0,2,1,3])
                            )
        # fill attention into context: mutation is not supported in tensorflow
        # (B, H, T, dv)
        context = tf.transpose(context, perm=[0,2,1,3])
        # fill attention of shape (B, H, n_top, dv) into context of shape (B, H, T, dv) given top_index
        #context[tf.range(B)[:, tf.newaxis, tf.newaxis],
        #        tf.range(H)[tf.newaxis,:, tf.newaxis],
        #        tf.transpose(top_index, perm=[0,2,1]), :] = attention
        # the above code wont work due to immutable tensor. 
        context = self._update_at_index(context, attention, tf.transpose(top_index, perm=[0,2,1]))
        
        # fill scores with shape (B, H, T, S)
        final_scores = (tf.ones((B,H,T,S))/S).astype(scores.dtype)
        final_scores = self._update_at_index(final_scores, scores, tf.transpose(top_index, perm=[0,2,1]))
        final_scores._keras_mask = attention_mask
        
        return context, final_scores
    
    # step 2 - 5 in the paper
    def _prob_QK(self, query, key, sample_k, n_top):
        # query shape: [B, T, H, dk]
        # key shape: [B, S, H, dk]
        # sample_k: number of samples for key
        # n_top: number of top samples for query
        B = tf.shape(query)[0]
        T = tf.shape(query)[1]
        H = tf.shape(query)[2]
        S = tf.shape(key)[1]
        
        # randomly sample keys
        # (B, S, H, dk) -> (B, 1, S, H, dk)
        key_expand = tf.expand_dims(key, axis=1)
        # (B, 1, S, H, dk) -> (B, T, S, H, dk)
        key_expand = tf.tile(key_expand, [1, T, 1, 1, 1])
        # (T, sample_k)
        index_sample = tf.random.uniform(shape=[T, sample_k], minval=0, maxval=S, dtype=tf.int32)
        
        # get sample_k keys for each query
        #K_sample = tf.gather_nd(key_expand, index_sample, axis=2, batch_dims=1)
        # (T,1)
        arange_T = tf.reshape(tf.range(T), (-1,1))
        # (B, T, sample_k, H, dk) 
        key_sample = key_expand[:, arange_T, index_sample, :, :]
        
        # compute sample score
        # (B, T, H, dk) -> (B, T, H, 1, dk)
        q = tf.expand_dims(query, axis=-2)
        # (B, T, sample_k, H, dk) -> (B, T, H, sample_k, dk)
        k_sample = tf.transpose(key_sample, perm=[0,1,3,2,4])
        # sample score: (B, T, H, sample_k) 
        qk_sample = tf.matmul(q, tf.transpose(k_sample, perm=[0,1,2,4,3]))
        qk_sample = tf.squeeze(qk_sample, axis=-2)
        
        # compute measurement
        # (B, T, H)
        M = tf.reduce_max(qk_sample, axis=-1) - tf.reduce_mean(qk_sample, axis=-1)
        # (B, H, n_top)
        M_top = tf.math.top_k(tf.transpose(M, perm=[0,2,1]), 
                              k=n_top, 
                              sorted=False).indices
        # (B, n_top, H)
        M_top = tf.transpose(M_top, perm=[0,2,1])

        # reduced Q from the top M scores
        # (B, n_top, H, dk)
        query_reduced = query[tf.range(B)[:, tf.newaxis, tf.newaxis], 
                              M_top, 
                              tf.range(H)[tf.newaxis,tf.newaxis,:], 
                              :]
        # (B, H, n_top, S)
        qk = tf.matmul(tf.transpose(query_reduced, perm=[0,2,1,3]), 
                        tf.transpose(key, perm=[0,2,3,1]))
        
        return qk, M_top
    
    def _update_at_index(self, context, updates, index):
        """ Update context with given updates at index. This is to perform scatter update in tensorflow.
            
            context[index] = attention     

        Args:
            context (B, H, T, d): _description_
            updates (B, H, n, d): _description_
            index (B, H, n): _description_
        """
        B, H = context.shape[:2]

        # Create the indices for scatter update
        b_indices = tf.range(B)[:, tf.newaxis, tf.newaxis, tf.newaxis]
        h_indices = tf.range(H)[tf.newaxis, :, tf.newaxis, tf.newaxis]

        # Broadcast top_index to match the dimensions
        top_expanded = index[:, :, :, tf.newaxis]

        # Combine indices to form the index tensor
        indices = tf.concat([b_indices + tf.zeros_like(top_expanded),
                            h_indices + tf.zeros_like(top_expanded),
                            top_expanded], axis=-1)

        # Scatter update
        updated_context = tf.tensor_scatter_nd_update(context, indices, updates)
        
        return updated_context

    def call(self, 
             inputs, 
             scale=True, 
             attention_mask=None, 
             use_causal_mask=False, 
             return_attention_scores=True, 
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

        # compute attention and scores
        U_part = tf.cast(tf.cast(self.factor, tf.float32) * tf.math.log(tf.cast(S, tf.float32)),tf.int32) # c*ln(L_k)
        u = tf.cast(tf.cast(self.factor, tf.float32) * tf.math.log(tf.cast(T, tf.float32)),tf.int32) # c*ln(L_q) 

        U_part = U_part if U_part < S else S
        u = u if u < T else T
        
        attentions, attention_scores = self._compute_attention(query, key, value, U_part, u, attention_mask, scale)
        
        # output projection
        # (B, H, T, Dv) -> (B, T, H*Dv) -> (B, T, O)
        attentions = tf.reshape(
                tf.transpose(attentions, perm=[0,2,1,3]),
                (-1, T, self.num_heads * self.value_dim)
                )
        # propagating mask
        if self.supports_masking:
            attentions._keras_mask = query._keras_mask
        else:
            attentions._keras_mask = None
            
        attentions = self.output_projection(attentions)
        
        if return_attention_scores:
            return attentions, attention_scores
        
        return attentions


if __name__ == "__main__":
    
    # create zero padding 
    source_sequence = tf.constant([[1, 2, 3, 0, 0], [1, 2, 3, 4, 0]])
    target_sequence = tf.constant([[1, 2, 1, 1, 0], [1, 2, 3, 1, 1]])

    # create embedding layer
    emb = tf.keras.layers.Embedding(input_dim=10, output_dim=4, mask_zero=True)

    pro_mha = MultiHeadProbSparseAttention(num_heads=2, 
                                           key_dim=2, 
                                           value_dim=2,
                                           factor=2, 
                                           output_dim=4)
    
    # (2, 3, 4)
    query = emb(target_sequence)
    # (2, 5, 4)
    key = value = emb(source_sequence)
    
   # compute attention
    attentions, scores = pro_mha((query, key, value),
             scale=True, 
             attention_mask=None, 
             use_causal_mask=True, 
             return_attention_scores=True) 
   
    print(tf.shape(attentions), tf.shape(scores))
    print(attentions)
    print(scores)
   