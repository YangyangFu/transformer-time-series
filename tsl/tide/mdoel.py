# coding=utf-8
# Contributor:  Yangyang Fu 

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
        x = inputs
        #for i in range(self.num_layers):
        #    x = self.mlp_residual_blocks[i](x, training=training)
        self.mlp_residual_blocks(x, training=training)
        return x
        
class Decoder(tf.keras.Model):
    def __init__(self, hidden_dims, output_dims, layer_norm=False, dropout_rate=0.0):
        super(Decoder, self).__init__()
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
        x = inputs
        #for i in range(self.num_layers):
        #    x = self.mlp_residual_blocks[i](x, training=training)
        x = self.mlp_residual_blocks(x, training=training)
        return x
        
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
                 layer_norm=False, 
                 dropout_rate=0.):
        super(TIDE, self).__init__()
        
        # encoder
        self.encoder = MLPResidualBlock(hidden_dims_encoder, output_dims_encoder, layer_norm, dropout_rate)
        
        # decoder
        self.decoder = MLPResidualStack(hidden_dims_decoder, output_dims_decoder, layer_norm, dropout_rate)
        
        
        # time encoder to project features at each step
        self.time_encoder = MLPResidualStack(hidden_dims_time_encoder, output_dims_time_encoder, layer_norm, dropout_rate)
        
        # time decoder to project features at each step
        self.time_decoder = MLPResidualBlock(hidden_dims_time_decoder, 1, layer_norm, dropout_rate)
        
        # global residual connection
        self.global_residual = tf.keras.layers.Dense(pred_length, activation=None)
        
    
    def call(self, inputs):
        pass
        