# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/LITA/blob/main/LICENSE

import einops
import numpy as np

import torch
import torch.nn as nn

from llava.model.llava_arch import LlavaMetaForCausalLM
from lita.constants import TIME_TOKEN_TEMPLATE


class LitaMetaForCausalLM(LlavaMetaForCausalLM):
    def images_to_tokens(self, images):
        if type(images) is list or images.ndim == 5:
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            image_features = [x.flatten(0, 1) for x in image_features]
        else:
            image_features = self.encode_images(images)

        return image_features

    def videos_to_tokens(self, images):
        assert images.ndim == 5, "multiple videos per sample not supported yet"
        # BTCHW
        b = len(images)
        images = einops.rearrange(images, 'b t c h w -> (b t) c h w')
        tokens = self.encode_images(images)  # (b t) s d
        tokens = einops.rearrange(tokens, '(b t) s d -> b t s d', b=b)
        
        video_arch = getattr(self.config, 'video_arch', 'temporal')
        
        if video_arch == 'temporal':
            tokens = einops.reduce(tokens, 'b t s d -> b t d', 'mean')
        elif video_arch == 'spatial':
            tokens = einops.reduce(tokens, 'b t s d -> b s d', 'mean')
        elif video_arch == 'temporal_spatial':
            t_tokens = einops.reduce(tokens, 'b t s d -> b t d', 'mean')
            s_tokens = einops.reduce(tokens, 'b t s d -> b s d', 'mean')
            tokens = torch.cat([t_tokens, s_tokens], dim=1)
        elif video_arch == 'temporal_spatial_pool' or video_arch == 'spatial_pool':
            pool_size = 2
            selected_frames = np.round(np.linspace(0, tokens.shape[1] - 1, pool_size * pool_size)).astype(int)
            s_tokens = tokens[:, selected_frames, ...]
            s_tokens = einops.rearrange(s_tokens, 'b t (h w) d -> (b t) d h w', h=16, w=16)
            s_tokens = nn.functional.avg_pool2d(s_tokens, kernel_size=pool_size)
            s_tokens = einops.rearrange(s_tokens, '(b t) d h w -> b (t h w) d', b=b)
            
            if video_arch == 'temporal_spatial_pool':
                t_tokens = einops.reduce(tokens, 'b t s d -> b t d', 'mean')
                tokens = torch.cat([t_tokens, s_tokens], dim=1)
            elif video_arch == 'spatial_pool':
                tokens = s_tokens
        else:
            raise ValueError(f"unknown video arch {video_arch}")
            
        return tokens

    def visual_to_tokens(self, images):
        input_type = getattr(self.config, 'input_type', 'image')
        if input_type == 'image':
            return self.images_to_tokens(images)
        elif input_type == 'video':
            visual_tokens = self.videos_to_tokens(images)
            return visual_tokens

    def initialize_time_tokenizer(self, model_args, tokenizer):
        num_time_tokens = model_args.num_time_tokens
        time_tokens = [TIME_TOKEN_TEMPLATE.format(t=x) for x in range(num_time_tokens)]
        num_new_tokens = tokenizer.add_tokens(time_tokens)
        assert len(time_tokens) == num_new_tokens

        self.resize_token_embeddings(len(tokenizer))
        
        time_token_ids = tokenizer.convert_tokens_to_ids(time_tokens)
        self.config.time_token_ids = time_token_ids

