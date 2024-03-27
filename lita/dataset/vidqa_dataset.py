# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/LITA/blob/main/LICENSE

import os
import json
import numpy as np

from lita.dataset.base_dataset import BaseDataset
from lita.constants import DEFAULT_IMAGE_TOKEN, TIME_TOKEN_TEMPLATE


class VidQADataset(BaseDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(VidQADataset, self).__init__(data_path, tokenizer, data_args)
        
    def get_sources(self, i):
        vqas = self.list_data_dict[i]
        return self.format_vqas(vqas)
    
    def get_visual(self, sources):
        if self.visual_data_type == 'video':
            return self.load_video(sources['image'], self.data_args.num_frames)
    
    def format_vqas(self, vqas):
        out = {}
        out['image'] = os.path.join(self.image_folder, vqas['video'])
        
        convo = []
        for i, vqa in enumerate(vqas['QA']):
            if i == 0:
                gpt_prompt = DEFAULT_IMAGE_TOKEN + '\n'
            else:
                gpt_prompt = ""
                
            question = vqa['q']
            answer = vqa['a']
            
            question = question.strip()
            if len(question) > 1:
                question = question[0].upper() + question[1:]
            if len(self.task_prompt) > 0 and not question.endswith('?'):
                question += '?'
            
            gpt_prompt += question
            gpt_prompt += ' ' + self.task_prompt    
            
            gpt_value = answer
            
            convo.append({"from": "human", "value": gpt_prompt.strip()})
            convo.append({"from": "gpt", "value": gpt_value.strip()})
            
        out['conversations'] = convo
        
        return out
    
    
class VidQADataset_msvdqa(VidQADataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(VidQADataset_msvdqa, self).__init__(data_path, tokenizer, data_args)
        
    def set_params(self):
        self.image_folder = os.path.join(self.data_path, 'msvdqa', 'YouTubeClips')
        self.visual_data_type = 'video'
        self.task_prompt = "Answer the question using a single word or phrase."

    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'msvdqa', 'train_processed.json')
        self.list_data_dict = json.load(open(data_path, "r"))
        
        
class VidQADataset_msrvttqa(VidQADataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(VidQADataset_msrvttqa, self).__init__(data_path, tokenizer, data_args)
        
    def set_params(self):
        self.image_folder = os.path.join(self.data_path, 'msrvttqa', 'TrainValVideo')
        self.visual_data_type = 'video'
        self.task_prompt = "Answer the question using a single word or phrase."

    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'msrvttqa', 'train_processed.json')
        self.list_data_dict = json.load(open(data_path, "r"))

        
class VidQADataset_nextqa(VidQADataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(VidQADataset_nextqa, self).__init__(data_path, tokenizer, data_args)
        
    def set_params(self):
        self.image_folder = os.path.join(self.data_path, 'nextqa', 'NExTVideo')
        self.visual_data_type = 'video'
        self.task_prompt = "Answer the question using a short phrase."

    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'nextqa', 'train_processed.json')
        self.list_data_dict = json.load(open(data_path, "r"))
        
        
class VidQADataset_videochat(VidQADataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(VidQADataset_videochat, self).__init__(data_path, tokenizer, data_args)
        
    def set_params(self):
        self.image_folder = os.path.join(self.data_path, 'videochat_instruct_11k', 'videos')
        self.visual_data_type = 'video'
        self.task_prompt = ""

    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'videochat_instruct_11k', 'videochat_instruct_11k.json')
        self.list_data_dict = json.load(open(data_path, "r"))
    
        
        
        