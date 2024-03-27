# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/LITA/blob/main/LICENSE

from dataclasses import dataclass, field
from typing import Optional, List
import transformers


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")
    #
    num_frames: int = 100
    num_time_tokens: int = 100
    input_type: str = 'video'
    video_arch: str = 'temporal'

        
@dataclass
class DataArguments:
    # data_path points to dataset root directory for all datasets for hybrid datasest

    task_sample_rate: list[float]
    dvc_sample_rate: list[float] 
    event_loc_sample_rate: list[float] 
    imgqa_sample_rate: list[float]
    vidqa_sample_rate: list[float]
    temporal_reasoning_sample_rate: list[float]
    
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})  
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)  # depricate for hybrid dataset
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)

    tasks: str = 'dvc'  # 'dvc||event_loc||imgqa||vidqa'
    dvc_data: str = 'activitynet||youcook2||vitt'
    event_loc_data: str = 'activitynet'
    imgqa_data: str = 'llava'
    vidqa_data: str = 'videochat'
    temporal_reasoning_data: str = 'activitynet'
    samples_per_epoch: int = 15000 # 500 * 8 * 2 * 10


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
