from dataclasses import dataclass
from typing import Dict, List, Optional
import torch.nn as nn
import torch.nn.functional as F
import torch
import yaml
import logging

from transformers import Trainer
# Configure logger
from logging_config import configure_logging
configure_logging()
logger = logging.getLogger(__name__)

def selective_logits_target(logits_components, data_source):
    """Select appropriate logits based on data source."""

    logits_target = torch.empty_like(logits_components[0])
    for idx, source in enumerate(data_source):
        logits_target[idx] = logits_components[source][idx]

    return logits_target

def compute_kl_div(logits_a, logits_b, effective_idxs, temperature=1.0):
    kl_fct = nn.KLDivLoss(reduction="none")
    diff = (
        kl_fct(
            F.log_softmax(logits_b / temperature, dim=-1),
            F.softmax(logits_a / temperature, dim=-1)
        )
        * (temperature**2)
    )
    
    # Calculate final loss
    loss = (diff.sum(-1) * effective_idxs).sum() / effective_idxs.sum()
    return loss

def compute_entropy(logits, effective_idxs, temperature=1.0):
    softmax = F.softmax(logits / temperature, dim=-1)
    log_softmax = F.log_softmax(logits / temperature, dim=-1)
    entropy = (- softmax * log_softmax) * (temperature**2)
    loss = (entropy.sum(-1) * effective_idxs).sum() / effective_idxs.sum()
    return loss

class MergerTrainer(Trainer):
    """
    Hello it's Nguyen Thanh Do.
    """
    def __init__(self, *args, loss_func_name="kl_div", mask_decay=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_func_name = loss_func_name
        if self.args.should_save:
            assert self.loss_func_name in ("entropy", "kl_div"), (
                f"Loss function {self.loss_func_name} is not supported. "
                f"Please select a loss function of type `entropy` or `kl_div`."
            )
            loss_func_full_name = (
                "Minimizing Entropy loss (AdaMerging)" if loss_func_name == "entropy"
                else "Disentangled KL Divergence loss" if loss_func_name == "kl_div"
                else "Not yet supported loss"
            )
            logger.info(f"You are training masks with {loss_func_full_name}.")
            
        self.mask_decay = mask_decay
        self.track_masks_params(print_param_count=True)
        
    def track_masks_params(self, print_param_count=False):
        params_a, params_b = [], []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if "masks.0." in name:
                    params_a.append(param.flatten())
                elif "masks.1." in name:
                    params_b.append(param.flatten())
                    
        statistics =  {
            "params_a": torch.cat(params_a),
            "params_b": torch.cat(params_b)
        }
        
        if print_param_count and self.args.should_save:
            count_millions = (
                statistics["params_a"].numel() 
                + statistics["params_b"].numel()
            ) / 1_000_000
            logger.info(f"Trainable parameters (masks): {count_millions:.2f}M")
            
        return statistics
            
        
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        with torch.no_grad():
            all_params = self.track_masks_params()
            params_a, params_b = all_params["params_a"], all_params["params_b"]

            logs["mean_a"] = params_a.mean().item()
            logs["mean_b"] = params_b.mean().item()
        
        if self.state.epoch is not None:
            logs["epoch"] = self.state.epoch
        if self.args.include_num_input_tokens_seen:
            logs["num_input_tokens_seen"] = self.state.num_input_tokens_seen
            if start_time is not None:
                speed_metrics("train", start_time, num_tokens=self.state.num_input_tokens_seen)

        output = {**logs, **{"step": self.state.global_step}}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(self.args, self.state, self.control, logs)
    
    def compute_kl_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        data_source = inputs.pop("data_source")
        effective_idxs = (labels != -100).float()
        
        outputs = model(**inputs)
        logits_merged = outputs["merger_outputs"].logits
        # logits_target = logits_merged
        logits_components = [x.logits.detach() for x in outputs["components_outputs"]]

        # Compute target logits and KL divergence
        logits_target = selective_logits_target(logits_components, data_source)
        loss = compute_kl_div(logits_merged, logits_target, effective_idxs)
        return loss

    def compute_entropy_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        effective_idxs = (labels != -100).float()
        
        outputs = model(**inputs)
        logits_merged = outputs["merger_outputs"].logits

        loss = compute_entropy(logits_merged, effective_idxs)
        return (loss, outputs) if return_outputs else loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ## Debug: Before computing loss.
        # ------------------------------------------------------
        for name, param in model.named_parameters():
            if param.requires_grad and torch.isnan(param).any():
                logger.info(f"NaN detected in parameter {name}")
                import pdb; pdb.set_trace()
        # ------------------------------------------------------
        
        loss_func = (
            self.compute_kl_loss if self.loss_func_name == "kl_div"
            else self.compute_entropy_loss
        )
        loss = loss_func(model, inputs, return_outputs, num_items_in_batch)
        if self.mask_decay is not None:
            params_a = self.track_masks_params()["params_a"]
            params_b = self.track_masks_params()["params_b"]

            mean_a = torch.mean(params_a**2)
            mean_b = torch.mean(params_b**2)
            loss_w = self.mask_decay * (mean_b / (mean_a + mean_b))
            loss += loss_w
            
        ## Debug: After computing loss.
        # ------------------------------------------------------
        if torch.isnan(loss):
            logger.info(f"Loss became NaN")
            import pdb; pdb.set_trace()
        # ------------------------------------------------------

        return (loss, outputs) if return_outputs else loss
        
@dataclass
class TrainingConfig:
    """Configuration for training loaded from YAML."""
    # Model configuration
    model_paths: List[str]
    mode: str
    constrain_mode: str
    mask_init: dict
    
    # Dataset configuration
    dataset_configs: Dict[str, int]  # Path to dataset -> number of samples
    source_keys: List[int]
    train_split: str
    max_length: int
    
    # Training parameters
    output_dir: str
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: int
    save_steps: int
    eval_steps: int
    logging_steps: int
    eval_strategy: str
    report_to: str
    remove_unused_columns: bool = False
    logging_first_step: bool = True
    bf16: bool = True
    gradient_checkpointing: bool = False
    validation_split: Optional[str] = None
    loss_func_name: str = "kl_div"
    mask_decay: Optional[float] = None
    train_on_inputs: bool = False
    lr_scheduler: str = "constant"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TrainingConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
            # Convert learning rate to float if it's a string
            if isinstance(config_dict['learning_rate'], str):
                config_dict['learning_rate'] = float(config_dict['learning_rate'])
        return cls(**config_dict)