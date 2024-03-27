# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/LITA/blob/main/LICENSE

import os
import json
import torch
from lita.dataset.base_dataset import BaseDataset


class LlavaDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(LlavaDataset, self).__init__(data_path, tokenizer, data_args)
        
    def set_params(self):
        self.image_folder = os.path.join(self.data_path, 'coco', 'train2017')

    def get_sources(self, i):
        return self.list_data_dict[i]
    
    def get_visual(self, sources):
        image_path = os.path.join(self.image_folder, sources['image'])
        image = self.load_image(image_path)
        return torch.stack([image] * self.data_args.num_frames, dim=0)
    
    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'LLaVA-Instruct-150K', 'llava_instruct_150k.json')
        self.list_data_dict = json.load(open(data_path, "r"))
    