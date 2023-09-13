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

class TIDE(tf.keras.Model):
    def __init__(self, 
                 pred_length,
                 hidden_dims_encoder, 
                 output_dims_encoder, 
                 hidden_dims_decoder, 
                 output_dims_decoder, 
                 hidden_dims_time_encoder,
                 output_dims_time_encoder,
                 hidden_dims_time_decoder,
                 cat_sizes,
                 cat_emb_size,
                 num_ts,
                 layer_norm=False, 
                 dropout_rate=0.):
        super(TIDE, self).__init__()
        self.pred_length = pred_length
        
        # encoder
        self.encoder = MLPResidualStack(hidden_dims_encoder, output_dims_encoder, layer_norm, dropout_rate)
        
        # decoder
        self.decoder = MLPResidualStack(hidden_dims_decoder, output_dims_decoder, layer_norm, dropout_rate)
        
        # time encoder to project features at each step
        self.time_encoder = MLPResidualBlock(hidden_dims_time_encoder, output_dims_time_encoder, layer_norm, dropout_rate)
        
        # time decoder to project features at each step
        self.time_decoder = MLPResidualBlock(hidden_dims_time_decoder, 1, layer_norm, dropout_rate)
        
        # global residual connection
        self.global_residual = tf.keras.layers.Dense(pred_length, activation=None)
        
        # embedding layers for categorical features and time series index (different targets)
        self.cat_embs = []
        for cat_size in cat_sizes:
            self.cat_embs.append(
            tf.keras.layers.Embedding(input_dim=cat_size, output_dim=cat_emb_size)
        )
        self.ts_embs = tf.keras.layers.Embedding(input_dim=num_ts, output_dim=16)
        
    @tf.function
    def _assemble_feats(self, feats, cfeats):
        """assemble all features.

        Args:
            feats; (B, L)
            cfeats: (nc, L)
        
        """
        all_feats = [feats]
        for i, emb in enumerate(self.cat_embs):
            all_feats.append(tf.transpose(emb(cfeats[i, :])))
        return tf.concat(all_feats, axis=0)
    
    def call(self, inputs, training=True):
        # unpack inputs: past_data, future_features, tsidx
        # past_data [(B, L), (nx, L), (ny, L)]
        # future_features [(nx, L), (ny, L)]
        
        past_data = inputs[0]
        future_features = inputs[1]
        # attributes of time series: (B, 1)
        tsidx = inputs[2]
        
        # batch size B
        batch_size = past_data[0].shape[0]
        
        # (B, L)
        past_ts = past_data[0]
        # (nx, L)
        past_feats = self._assemble_feats(past_data[1], past_data[2])
        # (nx, H)
        future_feats = self._assemble_feats(future_features[0], future_features[1])
        
        ## Modeling
        # time encoder: encode feature per time step
        # (nx', L)
        enc_past_feats = tf.transpose(self.time_encoder(
            tf.transpose(past_feats), training = training))
        # (nx', H)
        enc_future_feats = tf.transpose(self.time_encoder(
            tf.transpose(future_feats), training = training))
        
        # attributes embedding
        # (B, ne) <- (B,)
        ts_embs = self.ts_embs(tsidx) 
        
        # encoder
        # (B, nx'*L) <-(B, nx', L) <- (nx', L)
        enc_past = tf.repeat(tf.expand_dims(enc_past_feats, axis=0), batch_size, axis=0)
        enc_past = tf.reshape(enc_past, [batch_size, -1]) 
        # (B, nx'*H) <-(B, nx', H) <- (nx', H)
        enc_future_tmp = tf.repeat(tf.expand_dims(enc_future_feats, axis=0), batch_size, axis=0)
        enc_future = tf.reshape(enc_future_tmp, [batch_size, -1])
        # 
        # (B, L + nx'*L + nx'*H + ne) 
        enc_inputs = tf.concat([past_ts, enc_past, enc_future, ts_embs], axis=1)
        # (B, output_dims_encoder) 
        enc_outputs = self.encoder(enc_inputs, training=training)
        
        # decoder
        # (B, output_dims_decoder)
        dec_outputs = self.decoder(enc_outputs, training=training)
        
        # unflatten decoer
        # (B, p, H)
        dec_outputs_unflat = tf.reshape(dec_outputs, [batch_size, -1, self.pred_length])
        
        # stack
        # (B, p+nx', H) 
        stack_outputs = tf.concat([dec_outputs_unflat, enc_future_tmp], axis=1)
        
        # time decoder
        # (B, H)
        out = self.time_decoder(
            tf.transpose(stack_outputs, perm=[0,2,1]), training=training)
        out = tf.squeeze(out, axis=-1)
        
        # global residual
        # (B, H)
        glob_residual_outputs = self.global_residual(past_ts)
        
        # skip connection
        # (B, H)
        out += glob_residual_outputs
        
        return out
    
    @tf.function
    def train_step(self, past_data, future_features, ytrue, tsidx, optimizer, train_loss):
        """One step of training."""
        with tf.GradientTape() as tape:
            all_preds = self((past_data, future_features, tsidx), training=True)
            loss = train_loss(ytrue, all_preds)

        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

# add test
if __name__ == "__main__":
    batch_size = 16
    pred_length = 720
    past_length = 720
    num_ts = 21
    
    tide = TIDE(pred_length=pred_length,
                 hidden_dims_encoder=[256], 
                 output_dims_encoder=[256], 
                 hidden_dims_decoder=[256], 
                 output_dims_decoder=[4*pred_length], 
                 hidden_dims_time_encoder=64,
                 output_dims_time_encoder=4,
                 hidden_dims_time_decoder=64,
                 cat_sizes=[1],
                 cat_emb_size=4,
                 num_ts=num_ts)
    
    past_data = (tf.random.uniform(shape=(batch_size, past_length), minval=0, maxval=1),
                 tf.random.uniform(shape=(8, past_length), minval=0, maxval=1),
                 tf.random.uniform(shape=(1, past_length), minval=0, maxval=1))
        
    feature_feats = (tf.random.uniform(shape=(8, pred_length), minval=0, maxval=1),
                     tf.random.uniform(shape=(1, pred_length), minval=0, maxval=1))
    tsindx = tf.random.uniform(shape=(batch_size,), minval=0, maxval=num_ts, dtype=tf.int32)
    inputs = (past_data, feature_feats, tsindx)
    # check model summary
    out = tide(inputs)
    print(tide.summary())
    print(out.shape)