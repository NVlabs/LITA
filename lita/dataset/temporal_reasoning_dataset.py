# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/LITA/blob/main/LICENSE

import os
import glob
import json
import numpy as np
import re

from lita.dataset.base_dataset import BaseDataset
from lita.constants import DEFAULT_IMAGE_TOKEN, TIME_TOKEN_TEMPLATE


class TemporalReasoningDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(TemporalReasoningDataset, self).__init__(data_path, tokenizer, data_args)
        
    def get_sources(self, i):
        vqas = self.list_data_dict[i]
        return self.format_temporal_reasoning(vqas)
    
    def get_visual(self, sources):
        if self.visual_data_type == 'video_frames':
            return self.load_video_frames(sources['image'])
        elif self.visual_data_type == 'video':
            return self.load_video(sources['image'], self.data_args.num_frames)
        
    def format_temporal_reasoning(self, vqas):
        out = {}
        vid = vqas['id']
        out['id'] = vid
        
        if self.visual_data_type == 'video_frames':
            frames = sorted(glob.glob(os.path.join(self.image_folder, vid, '*'+ self.ext)))
            idx = np.round(np.linspace(0, len(frames) - 1, self.data_args.num_frames)).astype(int)
            out['image'] = list(np.array(frames)[idx])
        elif self.visual_data_type == 'video':
            out['image'] = os.path.join(self.image_folder, captions['image'])
            
        convo = []
        duration = vqas['duration']
        max_offset = float(self.data_args.num_time_tokens - 1)
        for i, vqa in enumerate(vqas['QA']):
            if i == 0:
                gpt_prompt = DEFAULT_IMAGE_TOKEN + '\n'
            else:
                gpt_prompt = ""
                
            question = vqa['q']
            answer = vqa['a']
            
            gpt_prompt += question.strip()
            
            # process answer
            timestamp_pattern = '\<(?: (?: \d* \. \d+ ) | (?: \d+ \.? ) )\>'
            rx = re.compile(timestamp_pattern, re.VERBOSE)
            timestamp_tokens = []
            new_answer = ""
            prev_end = 0
            for m in rx.finditer(answer):
                start = m.start(0)
                end = m.end(0)
                timestamp = float(m.group(0)[1:-1])
                timestamp_time = int(np.round(max_offset * (timestamp / duration)))
                timestamp_token = TIME_TOKEN_TEMPLATE.format(t=timestamp_time)
                new_answer += answer[prev_end:start]
                new_answer += timestamp_token
                prev_end = end
            new_answer += answer[prev_end:]
          
            gpt_value = new_answer.strip()
            convo.append({"from": "human", "value": gpt_prompt.strip()})
            convo.append({"from": "gpt", "value": gpt_value.strip()})
            
        out['conversations'] = convo
        
        return out
                
                
class TemporalReasoningDataset_activitynet(TemporalReasoningDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(TemporalReasoningDataset_activitynet, self).__init__(data_path, tokenizer, data_args)
    
    def set_params(self):
        self.image_folder = os.path.join(self.data_path, 'activitynet-captions', 'activitynet_frames')
        self.visual_data_type = 'video_frames'
        self.ext = '.jpg'

    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'temporal_reasoning', 'activitynet_train_gpt-4-0613_temp_6_f10009.json')
        data_dict = json.load(open(data_path, "r"))
        for vid in data_dict:
            data = data_dict[vid]            
            for vqa in data['QA']:
                out = {}
                out['id'] = vid
                out['duration'] = data['duration']
                out['QA'] = [vqa]
                self.list_data_dict.append(out)
                
            