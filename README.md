## 02-01-2025
- `accurate_masks.py` is the accurate implementation of masks.
- `effcient_masks.py` is an alternative implementation of masks, which theoretically equivalent to the accurate implementation, but due to some numerical imprecisions of pytorch, the inference results of a Merger class would be different from inference results of an actual merged model.
- Specifically, `effcient_masks.py` actually masks `input`, while `accurate_masks.py` masks `weights`. 
  ```
  accurate: output = x @ (mask_a * w_a + mask_b * w_b).T
  effcient: output = (mask_a * x) @ w_a.T + (mask_b * x) @ w_b.T
  These two implementations, theoretically, are equivalent.
  ```
- `debug-training-2.ipynb` is a rough implementation of training loop: customized `compute_loss` with KL divergence loss, automatic logits' coefficent calculation (prioritize logits of models with higher levels of confidence, aka lower entropies) and customized `collator` to process batching.
- `train.py` is a `.py` equivalent of `debug-training-2.ipynb` for easier viewing. untested.
- migrating to h100
