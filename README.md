# Transformers for Time-series in Tensorflow
A library to benchmark various transformer and nontransformer models on typical time series prediction, imputation and abnormality detection

## Tasks
- Long-term forcasting
- Probalistic Long-term forcasting
- Representation learning
 

## Environment
- cuda 11.8

## Algorithms
- [x] Baseline Model
- [x] ARIMA
- [x] XGBoost
- [x] LSTM
- [ ] Encoder-Decoder RNN

### Transformers
- [ ] DLinear
- [ ] Non-stationary Transformer
- [ ] Autoformer
- [x] Informer
- [ ] Temporal Fusion Transformer
- [x] PatchTST

### Convs
- [x] 'TimesNet': uses Inception network as backbone to extract 2-D information for 1-D timeseries

### MLP
- [x] TiDE: pure MLP encoder/decoder archtiecture
- [ ] DLinear

## Notes
1. How padding mask and causal mask work in transformer?
2. Is padding mask implemented for each attention layer or just the first layer in both encoder and decoder?
3. How does tensorflow automatically propagate mask to each attention layer?
4. How does tensorflow automatically propage mask to cross-attention layer with different masks from source and target?
5. tensorflow reshape() API or layer drops the mask as _keras_mask. How to keep the mask while reshaping?
