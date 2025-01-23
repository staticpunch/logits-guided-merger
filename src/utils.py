from transformers import AutoTokenizer
import math
from typing import List, Optional, Tuple, Union
import torch
import logging
import numpy as np
import json
from transformers import GenerationConfig, TextStreamer

def generate(
    prompt, model, tokenizer,
    repetition_penalty=1.13,
    top_p=0.95,
    top_k=50,
    max_new_tokens=1024,
    temperature=0.4,
    eos_token_id=None,
    do_sample=False,
    use_cache=True,
    return_dict_in_generate=True,
    output_attentions=False,
    output_hidden_states=False,
    output_scores=False,
    streaming=True
):
    input_ids = tokenizer(
        prompt, return_tensors="pt"
    )["input_ids"].to(model.device)
    eos_token_id = (eos_token_id if eos_token_id is not None 
                    else tokenizer.eos_token_id)
    model.eval()
    with torch.no_grad():
        generation_config = GenerationConfig(
            repetition_penalty=repetition_penalty,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            # bos_token_id=tokenizer.bos_token_id,
            # eos_token_id=tokenizer.eos_token_id,
            # eos_token_id=0, # for open-end generation.
            eos_token_id=eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=do_sample,
            use_cache=use_cache,
            return_dict_in_generate=return_dict_in_generate,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_scores=output_scores,
        )
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        generated = model.generate(
            inputs=input_ids,
            generation_config=generation_config,
            streamer=streamer if streaming else None,
        )
        
    gen_tokens = generated["sequences"].cpu()[:, len(input_ids[0]):]
    output = tokenizer.batch_decode(gen_tokens)[0]
    output = output.split(tokenizer.eos_token)[0]
    return output.strip()

def load_layer(path, layer_idx=8):
    state_dict = {}
    shard_paths = [f for f in os.listdir(path) if f.endswith('.safetensors')]
    for shard_path in sorted(shard_paths, key=lambda x: int(x.split('-')[1])):
        apath = os.path.join(path, shard_path)
        with safe_open(apath, framework="pt", device="cpu") as f:
            for key in f.keys():
                if f"layers.{str(layer_idx)}." in key:
                    state_dict[key] = f.get_tensor(key)
    return state_dict

def free_memory(logger):
    if not torch.cuda.is_available():
        logger.info("CUDA is not available. No GPU memory to free.")
        return
        
    initial_memory = torch.cuda.memory_allocated()
    logger.info(f"Initial GPU memory allocated: {initial_memory / 1024**3:.2f} GB")
    gc.collect()
    torch.cuda.empty_cache()

    final_memory = torch.cuda.memory_allocated()
    logger.info(f"Final GPU memory allocated: {final_memory / 1024**3:.2f} GB")

    freed_memory = initial_memory - final_memory
    logger.info(f"Freed GPU memory: {freed_memory / 1024**3:.2f} GB")

def get_hf_token(secret_file="SECRET.json"):
    hf_token = json.load(open(secret_file))["hf_token"]
    return hf_token

## -------------------------------------------------------- ##
##                       DEBUG UTILS                        ##  
## -------------------------------------------------------- ##
def get_logits(text, model, tokenizer):
    input_ids = tokenizer(text, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        logits = model(**input_ids).logits
    return logits

def get_hidden_states(text, model, tokenizer):
    input_ids = tokenizer(text, return_tensors="pt").to(model.device)
    model.eval()
    with torch.no_grad():
        outputs = model(**input_ids, output_hidden_states=True, use_cache=False)
    return outputs

def init_input(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    past_key_values = None
    cache_position = None
    position_ids = None
    output_hidden_states = True
    output_attentions = False
    use_cache = False
    return_dict = True
    model.eval()
    
    with torch.no_grad():
        return_legacy_cache = False
        inputs_embeds = model.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() 
                if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens, 
                past_seen_tokens + inputs_embeds.shape[1], 
                device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = model._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, 
            past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = model.rotary_emb(hidden_states, position_ids)

    return dict(
        hidden_states=hidden_states,
        attention_mask=causal_mask,
        position_ids=position_ids,
        past_key_value=past_key_values,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )

def mlp_forward(mlp, x: torch.Tensor):
    """
    ref: self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))
    """
    steps = {}
    steps.update({"step 0 (input)": x})
    
    gate = mlp.gate_proj(x)
    steps.update({"step 1 (gate)": gate})
    
    up = mlp.up_proj(x)
    steps.update({"step 2 (up)": up})
    
    act = mlp.act_fn(gate)
    steps.update({"step 3 (activation)": act})
    
    act_up = act * up
    steps.update({"step 4 (act_up)": act_up})

    down = mlp.down_proj(act_up)
    steps.update({"step 5 (down - output)": down})
    
    return dict(
        outputs=down,
        steps=steps
    )

def decoder_forward(
    decoder,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:

    steps = {}
    # logger.warning(f"-------- Logging hidden_states in decoder forward:")
    residual = hidden_states
    # logger.warning(f" hidden_states step 1 (as input): {hidden_states}")
    steps.update({"step 1": hidden_states})

    hidden_states = decoder.input_layernorm(hidden_states)
    # logger.warning(f" hidden_states step 2 (after input_layernorm): {hidden_states}")
    steps.update({"step 2": hidden_states})
    # Self Attention
    hidden_states, self_attn_weights, present_key_value = decoder.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        cache_position=cache_position,
        position_embeddings=position_embeddings,
    )
    # logger.warning(f" hidden_states step 3 (after self_attn): {hidden_states}")
    steps.update({"step 3": hidden_states})
    
    hidden_states = residual + hidden_states
    # logger.warning(f" hidden_states step 4 (after first skip connection): {hidden_states}")
    steps.update({"step 4": hidden_states})
    # Fully Connected
    residual = hidden_states
    hidden_states = decoder.post_attention_layernorm(hidden_states)
    # logger.warning(f" hidden_states step 5 (after post_attention_layernorm): {hidden_states}")
    steps.update({"step 5": hidden_states})
    
    hidden_states = decoder.mlp(hidden_states)
    # logger.warning(f" hidden_states step 6 (after mlp): {hidden_states}")
    steps.update({"step 6": hidden_states})
    
    hidden_states = residual + hidden_states
    # logger.warning(f" hidden_states step 7 (after second skip connection): {hidden_states}")
    steps.update({"step 7": hidden_states})

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    outputs
    return dict(
        outputs=outputs,
        steps=steps
    )

def model_forward(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    past_key_values = None
    cache_position = None
    position_ids = None
    output_hidden_states = True
    output_attentions = False
    use_cache = False
    return_dict = True
    #############
    
    model.eval()
    with torch.no_grad():

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        inputs_embeds = model.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() 
                if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens, 
                past_seen_tokens + inputs_embeds.shape[1], 
                device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        causal_mask = model._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, 
            past_key_values, output_attentions
        )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = model.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None
        all_decoder_steps = ()

        for i, decoder_layer in enumerate(model.layers[:2]):   
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
          
            debug_outputs = decoder_forward(
                decoder_layer,
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            layer_outputs = debug_outputs["outputs"]

            hidden_states = layer_outputs[0]
            steps = debug_outputs["steps"]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            all_decoder_steps += (steps,)

        hidden_states = model.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        if not return_dict:
            return tuple(v for v in [
                hidden_states, next_cache, all_hidden_states, all_self_attns
            ] if v is not None)
            
        outputs = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

        return dict(
            outputs=outputs,
            steps=all_decoder_steps
        )