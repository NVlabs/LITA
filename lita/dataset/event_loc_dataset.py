# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/LITA/blob/main/LICENSE

import os
import glob
import json
import numpy as np
import random

from lita.dataset.base_dataset import BaseDataset
from lita.constants import DEFAULT_IMAGE_TOKEN, TIME_TOKEN_TEMPLATE


class EventLocDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(EventLocDataset, self).__init__(data_path, tokenizer, data_args)
        
        self.desc_prompts = [
            "When does \"%s\" happen in the video?",
            "At what point in the video does \"%s\" happen?",
            "When is \"%s\" depicted in the video?",
            "At what time in the video does \"%s\" take place?",
        ] 
        self.time_prompts = [
            "Answer the question only using start and end timestamps.",
            "Provide a response using only start and end timestamps.",
            "Convey your answer using start and end timestamps exclusively.",
        ]
        
    def get_sources(self, i):
        captions = self.list_data_dict[i]
        return self.sample_event_loc(captions)
    
    def get_visual(self, sources):
        if self.visual_data_type == 'video_frames':
            return self.load_video_frames(sources['image'])
        elif self.visual_data_type == 'video':
            return self.load_video(sources['image'], self.data_args.num_frames)
        
    def get_prompt(self, sentence):
        desc_prompt = random.choice(self.desc_prompts)
        time_prompt = random.choice(self.time_prompts)
        sentence = sentence.strip().rstrip('.')
        if len(sentence) > 1:
            sentence = sentence[0].lower() + sentence[1:]
        task_prompt = (desc_prompt % sentence) + ' ' + time_prompt
        
        return DEFAULT_IMAGE_TOKEN + '\n' + task_prompt 
    
    def sample_event_loc(self, captions):    
        out = {}
        vid = captions['id']
        out['id'] = vid
        
        if self.visual_data_type == 'video_frames':
            frames = sorted(glob.glob(os.path.join(self.image_folder, vid, '*'+ self.ext)))
            idx = np.round(np.linspace(0, len(frames) - 1, self.data_args.num_frames)).astype(int)
            out['image'] = list(np.array(frames)[idx])
        elif data_args.visual_data_type == 'video':
            out['image'] = os.path.join(self.image_folder, vid + self.ext)
            
        rng = np.random.RandomState()  # local rng independent of global
        event_idx = rng.choice(list(range(len(captions['timestamps']))))
        
        duration = captions['duration']
        timestamp = captions['timestamps'][event_idx]
        sentence = captions['sentences'][event_idx]
        max_offset = float(self.data_args.num_time_tokens - 1)
        start, end = float(timestamp[0]), float(timestamp[1])

        start_time = int(np.round(max_offset * (start / duration)))
        end_time = int(np.round(max_offset * (end / duration)))
        start_token = TIME_TOKEN_TEMPLATE.format(t=start_time)
        end_token = TIME_TOKEN_TEMPLATE.format(t=end_time)
        
        gpt_value = f"{start_token} {end_token}"
        human_value = self.get_prompt(sentence)
        
        convo = []
        convo.append({"from": "human", "value": human_value.strip()})
        convo.append({"from": "gpt", "value": gpt_value.strip()})  
        out['conversations'] = convo
        
        return out
        
        
class EventLocDataset_activitynet(EventLocDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(EventLocDataset_activitynet, self).__init__(data_path, tokenizer, data_args)
    
    def set_params(self):
        self.image_folder = os.path.join(self.data_path, 'activitynet-captions', 'activitynet_frames')
        self.visual_data_type = 'video_frames'
        self.ext = '.jpg'

    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'activitynet-captions', 'train.json')
        data_dict = json.load(open(data_path, "r"))
        for k in data_dict:
            v = data_dict[k]
            v['id'] = k
            self.list_data_dict.append(v)
            
            
class EventLocDataset_youcook2(EventLocDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(EventLocDataset_youcook2, self).__init__(data_path, tokenizer, data_args)
        
    def set_params(self):
        self.image_folder = os.path.join(self.data_path, 'youcook2', 'youcook2_frames')
        self.visual_data_type = 'video_frames'
        self.ext = '.jpg'

    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'VidChapters', 'YouCook2', 'train.json')
        data_dict = json.load(open(data_path, "r"))
        for k in data_dict:
            v = data_dict[k]
            v['id'] = k
            vid_path = os.path.join(self.image_folder, k)
            if os.path.exists(vid_path):
                self.list_data_dict.append(v)

                
class EventLocDataset_vitt(EventLocDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(EventLocDataset_vitt, self).__init__(data_path, tokenizer, data_args)
        
    def set_params(self):
        self.image_folder = os.path.join(self.data_path, 'vitt', 'vitt_frames')
        self.visual_data_type = 'video_frames'
        self.ext = '.jpg'

    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'VidChapters', 'ViTT', 'train.json')
        data_dict = json.load(open(data_path, "r"))
        for k in data_dict:
            v = data_dict[k]
            v['id'] = k
            vid_path = os.path.join(self.image_folder, k)
            if os.path.exists(vid_path):
                self.list_data_dict.append(v)