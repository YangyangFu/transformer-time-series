# transformer-time-series
A library to benchmark various transformer and nontransformer models on typical time series prediction, imputation and abnormality detection

## Tasks
Assume we have two years of historical electricity price data from energy market, we want to predict the next 24-hour ahead prices.
- deterministic prediction
  - 24-hour ahead
- probablistic prediction
  - 24-hour ahead

## Environment
- cuda 11.8


## Algorithms
- [ ] Baseline Model
- [ ] ARIMA
- [ ] XGBoost
- [ ] LSTM
- [ ] Encoder-Decoder RNN
- [ ] Transformer


## Transformers
- [ ] TimesNet
- [ ] DLinear
- [ ] Non-stationary Transformer
- [ ] Autoformer
- [x] Informer
- [ ] Temporal Fusion Transformer
- [x] TiDE
- [ ] PatchTST

## Notes
1. How padding mask and causal mask work in transformer?
2. Is padding mask implemented for each attention layer or just the first layer in both encoder and decoder?
3. How does tensorflow automatically propagate mask to each attention layer?
4. How does tensorflow automatically propage mask to cross-attention layer with different masks from source and target?
5. tensorflow reshape() API or layer drops the mask as _keras_mask. How to keep the mask while reshaping?