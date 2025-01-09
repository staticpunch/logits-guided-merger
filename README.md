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

## 07-01-2025
- the method somewhat works. the loss is converging. the performance is acceptable with smol testcases.
- trying distributed training. disappointed that it failed. zero2 & zero3 don't seem to work, as they incur more memory (probably due to communication) and training time compared to pure DDP.

### set up:
- data A: rewrite text task -> model A.
- data B: summarization task -> model B.
- data train mask: 30k samples gồm 1/2 task A và 1/2 task B.
- hiện tại tôi đang train cho logits_merged gần với logits_a nếu như data thuộc task A, ngược lại với task B.
- other hyperparams: 3 epoch, batch size = 128 (accumulated), ctx length = 2048. khởi tạo all masks = 0.5 cho model A, all masks = 0.5 cho model B.
- behavior: training run thuận lợi, loss giảm đều, gpu chiếm khoảng 35GB.

### result
- tôi test merged model với data task A nó làm được, với task B nó cũng làm được. trước đây nếu merge ngẫu nhiên ấy, thì merged model nó sẽ có hiện tượng ko phân biệt được task. ví dụ như là, hỏi task A lại trả lời task B (hỏi summarize lại text thì model lại rewrite text).

## plan until 15/02/2025.

Assumption: Model A performs well on Task A but poorly on Task B. Therefore, it is necessary to train Model A to perform well on Task B, resulting in Model B. However, Model B performs well on Task B but poorly on Task A. Conclusion: There is a need to merge the two models.

Sufficient conditions for ACL papers:
- Core method should work on at least 2 languages, hit pre-defined criteria (not too deteritory from model A, gain much performance from model B). => method robustness.
- If introduce new loss/regularizer: requires many experiments to prove the loss function is superior compared to existing notions.
- Intuition / Motivation (in abstract & intro).

Experiments Milestones:
1. Translate `Evol-Instruct` to at least 3 languages (SEA langs, japan, korean) (https://huggingface.co/datasets/WizardLMTeam/WizardLM_evol_instruct_V2_196k)
2. Train model A on translated versions of data. Make sure the assumption holds true. (even requires dirty tricks)
- Minimum: English benchmark -> translate to target languages.
- Expected: Distinct benchmark 
3. Ablation studies: regularization of weights, initialization, loss (selective loss, entropy based, v..v..), scale data (10k, 20k, 50k)

## 08-01-2025
Memory behaviors:
- I used DeepSpeed ZeRO-2 & ZeRO-3 but surprisingly they did not save any memory at all, and even incur addtional memory overhead for communication! This is disappointing.
- I suspected the memory high usage comes inference of component models. However this is wrong, as I disabled the component models inference and only let merger run forward, the memory is still the same. This means that running inference with component models costs nothing.
- In another note, I suspected high memory is because of the computational graph. I want the autograd only track gradient of trainable params, which are masks, and ignore every other weights. I explicitly tried tricks like calling `detach()` when operating with model weights, using in-place operations like `_add`. Nothing works.
- I tried to reduce the context length. The memory reduced significanly. Would investigate further from here.
- Context length -> Memory: 2048 -> 37GB, 128 -> 25GB, 1024 -> 29GB.

New trick to try:
```py
in_features = 4
out_features = 10

x = torch.rand(in_features)
W = torch.rand(out_features, in_features)

mask_in = torch.rand(in_features)
mask_out = torch.rand(out_features)
```
```
x @ (mask_in * W).T == (mask_in * x) @ W.T
```
This means that masking the weight with `mask_in` is equivalent to masking the input.

```
remask_out = mask_out.unsqueeze(0).T ~ (out_features, 1)
x @ (remask_out * W).T == x @ (W.T * mask_out) == (x @ W.T) * mask_out
```
This means that masking the weight with `mask_out` is equivalent to masking the output.

I tried to move the matrix multiplication under no_grad() (decouple at outputs). Conceptually:
```py
partial_outputs = []
with torch.no_grad():
    for linear in self.linears:
        # standard forward: y = xW^T + b
        y = linear(x)
        partial_outputs.append(y)
```
This works horribly. Check `masks_output.py`. This does nothing but increases memory to 60GB and slower converging loss. Maybe the buffer reserved for multiple linear forwards is too much.
- move to a100
