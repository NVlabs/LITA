# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/LITA/blob/main/LICENSE

import os
import warnings

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch

from lita.model import *
from lita.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, TIME_TOKEN_TEMPLATE


def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto"):
    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    if 'lita' not in model_name.lower():
        warnings.warn("this function is for loading LITA models")
    if 'lora' in model_name.lower():
        warnings.warn("lora is currently not supported for LITA")
    if 'mpt' in model_name.lower():
        warnings.warn("mpt is currently not supported for LITA")

    if model_base is not None:
        print('Loading LITA from base model...')
        tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
        cfg_pretrained = AutoConfig.from_pretrained(model_path)
        model = LitaLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)

        mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        mm_projector_weights = {k: v.to(torch.float16) for k, v in mm_projector_weights.items() if 'mm_projector' in k}
        model.load_state_dict(mm_projector_weights, strict=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        model = LitaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", False)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device='cuda', dtype=torch.float16)
    image_processor = vision_tower.image_processor

    # time tokens and embeddings
    num_time_tokens = getattr(model.config, "num_time_tokens", 0)
    if num_time_tokens > 0:
        time_tokens = [TIME_TOKEN_TEMPLATE.format(t=x) for x in range(num_time_tokens)]
        num_new_tokens = tokenizer.add_tokens(time_tokens)

        if model_base is None:
            assert num_new_tokens == 0, "time tokens should already be in the tokenizer for full finetune model"

        if num_new_tokens > 0:
            warnings.warn("looking for weights in mm_projector.bin")
            assert num_new_tokens == num_time_tokens
            model.resize_token_embeddings(len(tokenizer))
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data
            weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            assert 'model.embed_tokens.weight' in weights and 'lm_head.weight' in weights
            
            dtype = input_embeddings.dtype
            device = input_embeddings.device
            
            tokenizer_time_token_ids = tokenizer.convert_tokens_to_ids(time_tokens)
            time_token_ids = getattr(model.config, 'time_token_ids', tokenizer_time_token_ids)
            input_embeddings[tokenizer_time_token_ids] = weights['model.embed_tokens.weight'][time_token_ids].to(dtype).to(device)
            output_embeddings[tokenizer_time_token_ids] = weights['lm_head.weight'][time_token_ids].to(dtype).to(device)
            
    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
            
    return tokenizer, model, image_processor, context_len
