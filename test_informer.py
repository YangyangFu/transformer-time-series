import tsl
from tsl.transformers.informer import Informer 
from tsl.transformers.informer import DataLoader
from tsl.transformers.informer import PositionalEmbedding, TemporalEmbedding, CategoricalEmbedding
from tsl.transformers.informer import MultiHeadProbSparseAttention, MultiHeadAttention

model = Informer(output_dim=1, 
                pred_len=96,
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