# coding=utf-8

import numpy as np
import tensorflow as tf
from tqdm import tqdm 

class MLPResidualBlock(tf.keras.layers.Layer):
    def __init__(self, hidden_dim, output_dim, layer_norm=False, dropout_rate=0.0):
        super(MLPResidualBlock, self).__init__()
        self.layer_norm = layer_norm
        self.dense1 = tf.keras.layers.Dense(hidden_dim, activation='relu')
        self.dense2 = tf.keras.layers.Dense(output_dim, activation=None)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.dense_residual = tf.keras.layers.Dense(output_dim, activation=None)
        if self.layer_norm:
            self.norm = tf.keras.layers.LayerNormalization()
    # TODO: is this training flag necessary/
    def call(self, inputs, training=True):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        x_res = self.dense_residual(inputs)
        if self.layer_norm:
            return self.norm(x + x_res)
            
        return x + x_res
    
class MLPResidualStack(tf.keras.Model):
    def __init__(self, hidden_dims, output_dims, layer_norm=False, dropout_rate=0.0):
        """ Stack of MLPResidualBlock to represent encoder, decoder, time encoder and time decoder.

        Args:
            hidden_dims (List): dims of hidden layers in MLPResidualBlock, e.g., [256, 128] for a two-layer MLPResidualBlock
            output_dims (List): dims of output layers in MLPResidualBlock, e.g., [128, 64] for a two-layer MLPResidualBlock
            layer_norm (bool, optional): _description_. Defaults to False.
            dropout_rate (float, optional): _description_. Defaults to 0.0.
        """
        super(MLPResidualStack, self).__init__()
        self.num_layers = len(hidden_dims)
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims
        self.layer_norm = layer_norm
        self.dropout_rate = dropout_rate
        
        self.mlp_residual_blocks = tf.keras.Sequential([
            MLPResidualBlock(self.hidden_dims[i], self.output_dims[i], self.layer_norm, self.dropout_rate) 
            for i in range(self.num_layers)
            ])
    
    def call(self, inputs, training=True):
        return self.mlp_residual_blocks(inputs, training=training)

class Preprocessor(tf.keras.Model):
    """ Preprocessor for TIDE model 

    Args:
        tf (_type_): _description_
    """
    def __init__(self, 
                hist_len,
                pred_len,
                hidden_dims_time_encoder,
                output_dims_time_encoder,
                local_invariant_vocab_sizes,
                local_invariant_emb_sizes,
                global_vocab_sizes,
                global_emb_sizes,
                local_variant_vocab_sizes,
                local_variant_emb_sizes,
                layer_norm, 
                dropout_rate):
        
        super(Preprocessor, self).__init__()
        self.hist_len = hist_len
        self.pred_len = pred_len
        self.local_invariant_vocab_sizes = local_invariant_vocab_sizes
        self.local_invariant_emb_sizes = local_invariant_emb_sizes
        self.global_vocab_sizes = global_vocab_sizes    
        self.global_emb_sizes = global_emb_sizes
        self.local_variant_vocab_sizes = local_variant_vocab_sizes
        self.local_variant_emb_sizes = local_variant_emb_sizes
        
        # local invariant categorical features
        self.local_invariant_embedding = []
        if self.local_invariant_vocab_sizes:
            for i, vocab_size in enumerate(self.local_invariant_vocab_sizes):
                self.local_invariant_embedding.append(
                    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=self.local_invariant_emb_sizes[i])
                    )
        
        # global categorical features
        self.global_embedding = []
        if self.global_vocab_sizes:
            for i, vocab_size in enumerate(self.global_vocab_sizes):
                self.global_embedding.append(
                    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=self.global_emb_sizes[i])
                    )
        
        # local variant categorical features
        self.local_variant_embedding = []
        if self.local_variant_vocab_sizes:
            for i, vocab_size in enumerate(self.local_variant_vocab_sizes):
                self.local_variant_embedding.append(
                    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=self.local_variant_emb_sizes[i])
                    )

        # time encoder for global features (per time step)
        self.time_encoder_global = MLPResidualBlock(
            hidden_dims_time_encoder, 
            output_dims_time_encoder, 
            layer_norm, 
            dropout_rate
        )
        
        # time encoder for local variant features (per time step)
        self.time_encoder_local = MLPResidualBlock(
            hidden_dims_time_encoder,
            output_dims_time_encoder,
            layer_norm,
            dropout_rate
        )
        
    def call(self, inputs, training=True):        
        (ts_hist, # (B, L)
         num_cov_local_invaraint, # (B, Nlin) 
         cat_cov_local_invariant, # (B, Nlic)
         time_features, # (L+H, Nt)
         num_cov_global, # (L+H, Ngn) 
         cat_cov_global, # (L+H, Ngc)
         num_cov_local_variant, # (B, L+H, Nlvn)
         cat_cov_local_variant # (B, L+H, Nlvc)
        ) = inputs
        
        B = tf.shape(ts_hist)[0]
        # ts features: pass

        # local invariant features: (B, Nli) -> (B, Nli')
        local_invariant = [tf.reshape((), (B, 0))] # empty
        if num_cov_local_invaraint is not None and len(num_cov_local_invaraint.shape) > 0:
            local_invariant.append(num_cov_local_invaraint)
        for i, emb in enumerate(self.local_invariant_embedding):
            local_invariant.append(emb(cat_cov_local_invariant[:, i]))
        if len(local_invariant) > 1:
            local_invariant = tf.concat(local_invariant, axis=1)
        else:
            local_invariant = None
                
        # global features 
        # concatenate: (L+H, Ng)
        global_features = [tf.reshape((), (self.hist_len+self.pred_len, 0))] 
        if time_features is not None and len(time_features.shape) > 0:
            global_features.append(time_features)
        if num_cov_global is not None and len(num_cov_global.shape) > 0:
            global_features.append(num_cov_global)
        for i, emb in enumerate(self.global_embedding):
            global_features.append(emb(cat_cov_global[:, i]))
            
        if len(global_features) > 1:
            global_features = tf.concat(global_features, axis=1)
            # time encoder for global features: (L+H, Ng) -> (L+H, Ng'')
            global_features = self.time_encoder_global(global_features, training=training)
        else:
            global_features = None

        
        # local variant features: (B, L+H, Nlv) -> (B, L+H, Nlv')
        # concatenate: (B, L+H, Nlv)
        local_variant = [tf.reshape((), (B, self.hist_len+self.pred_len, 0))]
        if num_cov_local_variant is not None and len(num_cov_local_variant.shape) > 0:
            local_variant = [num_cov_local_variant]
        for i, emb in enumerate(self.local_variant_embedding):
            local_variant.append(emb(cat_cov_local_variant[:, :, i]))
        if len(local_variant) > 1:
            local_variant = tf.concat(local_variant, axis=2)
            # time encoder for local variant features: (B, L+H, Nlv) -> (B, L+H, Nlv'')
            local_variant = self.time_encoder_local(local_variant, training=training)
        else:
            local_variant = None
            

        
        return (ts_hist, # (B, L)
                local_invariant, # (B, Nli')
                global_features, # (L+H, Ng')
                local_variant # (B, L+H, Nlv'')
        )
        
        
class TIDE(tf.keras.Model):
    def __init__(self, 
                hist_length,
                pred_length,
                hidden_dims_encoder, 
                output_dims_encoder, 
                hidden_dims_decoder, 
                output_dims_decoder, 
                hidden_dims_time_encoder,
                output_dims_time_encoder,
                hidden_dims_time_decoder,
                local_invariant_vocab_sizes,
                local_invariant_emb_sizes,
                global_vocab_sizes,
                global_emb_sizes,
                local_variant_vocab_sizes,
                local_variant_emb_sizes,
                layer_norm=False, 
                dropout_rate=0.):
        
        super(TIDE, self).__init__()
        self.hist_length = hist_length
        self.pred_length = pred_length
        self.hidden_dims_encoder = hidden_dims_encoder
        self.output_dims_encoder = output_dims_encoder
        self.hidden_dims_decoder = hidden_dims_decoder
        self.output_dims_decoder = output_dims_decoder
        self.hidden_dims_time_encoder = hidden_dims_time_encoder
        self.output_dims_time_encoder = output_dims_time_encoder
        self.hidden_dims_time_decoder = hidden_dims_time_decoder
        self.output_dims_time_decoder = 1
        self.local_invariant_vocab_sizes = local_invariant_vocab_sizes if local_invariant_vocab_sizes else []
        self.local_invariant_emb_sizes = local_invariant_emb_sizes if local_invariant_emb_sizes else []
        self.global_vocab_sizes = global_vocab_sizes if global_vocab_sizes else []
        self.global_emb_sizes = global_emb_sizes if global_emb_sizes else []
        self.local_variant_vocab_sizes = local_variant_vocab_sizes if local_variant_vocab_sizes else []
        self.local_variant_emb_sizes = local_variant_emb_sizes if local_variant_emb_sizes else []
        self.layer_norm = layer_norm
        self.dropout_rate = dropout_rate
        
        # encoder
        self.encoder = MLPResidualStack(
            self.hidden_dims_encoder, 
            self.output_dims_encoder, 
            self.layer_norm, 
            self.dropout_rate)
        
        # decoder
        self.decoder = MLPResidualStack(
            self.hidden_dims_decoder, 
            self.output_dims_decoder, 
            self.layer_norm, 
            self.dropout_rate)
        
        # time decoder to project features at each step
        self.time_decoder = MLPResidualBlock(
            self.hidden_dims_time_decoder, 
            self.output_dims_time_decoder, 
            self.layer_norm, 
            self.dropout_rate)
        
        # global residual connection
        self.global_residual = tf.keras.layers.Dense(
            self.pred_length, 
            activation=None)
        
        # preprocessor layer
        #self.cat_embs = []
        #for cat_size in cat_sizes:
        #    self.cat_embs.append(
        #    tf.keras.layers.Embedding(input_dim=cat_size, output_dim=cat_emb_size)
        #)
        #self.ts_embs = tf.keras.layers.Embedding(input_dim=num_ts, output_dim=16)
        self.preprocessor = Preprocessor(
                hist_len=self.hist_length,
                pred_len=self.pred_length,
                hidden_dims_time_encoder=self.hidden_dims_time_encoder,
                output_dims_time_encoder=self.output_dims_time_encoder,
                local_invariant_vocab_sizes=self.local_invariant_vocab_sizes,
                local_invariant_emb_sizes=self.local_invariant_emb_sizes,
                global_vocab_sizes=self.global_vocab_sizes,
                global_emb_sizes=self.global_emb_sizes,
                local_variant_vocab_sizes=self.local_variant_vocab_sizes,
                local_variant_emb_sizes=self.local_variant_emb_sizes,
                layer_norm=self.layer_norm,
                dropout_rate=self.dropout_rate
                )
    #@tf.function
    def get_encoder_input(self, ts_hist, local_invariant, global_features, local_variant):
        """ Concatenate all features to get encoder input.

        Args:
            ts_hist (_type_): (B, L) 
            local_invariant (_type_): (B, Nli')
            global_features (_type_): (L+H, Ng')
            local_variant (_type_): (B, L+H, Nlv'')
        """
        B = tf.shape(ts_hist)[0]
        if global_features is not None and len(global_features.shape) > 0:
            global_features = tf.reshape(tf.tile(tf.expand_dims(global_features, axis=0), [B, 1, 1]), (B, -1))
        if local_variant is not None and len(local_variant.shape) > 0:
            local_variant = tf.reshape(local_variant, (B, -1))
        else:
            local_variant = tf.reshape((), (B, 0))
        out = tf.concat([ts_hist, local_invariant, global_features, local_variant], axis=-1)
        
        # (B, L+Nli'+Ng'+Nlv'')
        return out
    
    def call(self, inputs, training=True):
        # inputs:
        # ts_hist, # (B, L)
        # num_cov_local_invaraint, # (B, Nlin) 
        # cat_cov_local_invariant, # (B, Nlic)
        # time_features, # (L+H, Nt)
        # num_cov_global, # (L+H, Ngn) 
        # cat_cov_global, # (L+H, Ngc)
        # num_cov_local_variant, # (B, L+H, Nlvn)
        # cat_cov_local_variant # (B, L+H, Nlvc)
        
        # preprocess to get encoder inputs
        (ts_hist, # (B, L)
        local_invariant, # (B, Nli')
        global_features, # (L+H, Ng')
        local_variant # (B, L+H, Nlv'')
        ) = self.preprocessor(inputs, training=training)
        
        batch_size = tf.shape(ts_hist)[0]
        enc_inputs = self.get_encoder_input(ts_hist, local_invariant, global_features, local_variant)
        
        # encoder
        # (B, output_dims_encoder) 
        enc_outputs = self.encoder(enc_inputs, training=training)
        
        # decoder
        # (B, output_dims_decoder)
        dec_outputs = self.decoder(enc_outputs, training=training)
        
        # unflatten decoer
        # (B, H, p)
        dec_outputs_unflat = tf.reshape(dec_outputs, [batch_size, self.pred_length, -1])
        
        # stack
        # (B, H, p+Ng'+Nlv') 
        global_features_future = tf.tile(tf.expand_dims(global_features[-self.pred_length:, :], axis=0), [batch_size, 1, 1])
        
        if local_variant is not None and len(local_variant.shape) > 0:
            local_variant_future = tf.tile(local_variant[:,-self.pred_length:, :], [1, 1, 1])
        else:
            local_variant_future = tf.reshape((), (batch_size, self.pred_length, 0))
            
        stack_outputs = tf.concat([dec_outputs_unflat, global_features_future, local_variant_future], axis=-1)
        
        # time decoder
        # (B, H)
        out = self.time_decoder(
            stack_outputs, training=training)
        out = tf.squeeze(out, axis=-1)
        
        # global residual
        # (B, H)
        glob_residual_outputs = self.global_residual(ts_hist)
        
        # skip connection
        # (B, H)
        out += glob_residual_outputs
        
        return out