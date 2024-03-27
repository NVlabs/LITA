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


class DVCDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(DVCDataset, self).__init__(data_path, tokenizer, data_args)
        # self.image_folder
        # self.ext
        # self.visual_data_type
        self.desc_prompts = [
            "Provide a detailed description of the given video.",
            "Describe the provided video in detail.",
            "Summarize the visual content of the video.",
            "Write a informative summary of the video."
        ] 
        self.time_prompts = [
            "Each sentence should begin with the start and end timestamps.",
            "At the beginning of each sentence, include the start and end timestamps.",
            "Prepend each sentence with its start and end timestamps."
        ]

    def get_sources(self, i):
        captions = self.list_data_dict[i]
        return self.format_dense_video_captions(captions)
    
    def get_visual(self, sources):
        if self.visual_data_type == 'video_frames':
            return self.load_video_frames(sources['image'])
        elif self.visual_data_type == 'video':
            return self.load_video(sources['image'], self.data_args.num_frames)

    def get_prompt(self):
        task_prompt = random.choice(self.desc_prompts) + ' ' + random.choice(self.time_prompts)

        return DEFAULT_IMAGE_TOKEN + '\n' + task_prompt 

    def format_dense_video_captions(self, captions):
        # captions: list
        # id: video id
        # duration: video length 
        # sentences: list of captions
        # timestamps: list of start and end times

        out = {}
        vid = captions['id']
        out['id'] = vid

        if self.visual_data_type == 'video_frames':
            frames = sorted(glob.glob(os.path.join(self.image_folder, vid, '*'+ self.ext)))
            # if torch.bernoulli(torch.tensor(data_args.temp_aug_prob)):
            #     captions, frames = temporal_augmentation(captions, frames, data_args.temp_aug_min_len)
            idx = np.round(np.linspace(0, len(frames) - 1, self.data_args.num_frames)).astype(int)
            out['image'] = list(np.array(frames)[idx])
        elif self.visual_data_type == 'video':
            out['image'] = os.path.join(self.image_folder, captions['image'])  # TODO: update json so key is not 'image'

        duration = captions['duration']
        # timestamps = captions['timestamps'][:max_events]  # TODO: max_events
        timestamps = captions['timestamps']
        max_offset = float(self.data_args.num_time_tokens - 1)
        gpt_value = ""
        for i, (start, end) in enumerate(timestamps):
            start, end = float(start), float(end)

            start_time = int(np.round(max_offset * (start / duration)))
            end_time = int(np.round(max_offset * (end / duration)))
            start_token = TIME_TOKEN_TEMPLATE.format(t=start_time)
            end_token = TIME_TOKEN_TEMPLATE.format(t=end_time)
                
            seg_caption = captions['sentences'][i].strip()
            gpt_value += f"{start_token} {end_token} {seg_caption} "
        convo = []
        convo.append({"from": "human", "value": self.get_prompt()})
        convo.append({"from": "gpt", "value": gpt_value.strip()})      
        out['conversations'] = convo

        return out


class DVCDataset_activitynet(DVCDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(DVCDataset_activitynet, self).__init__(data_path, tokenizer, data_args)
    
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
            
            
class DVCDataset_howto100m(DVCDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(DVCDataset_howto100m, self).__init__(data_path, tokenizer, data_args)
    
    def set_params(self):
        self.image_folder = os.path.join(self.data_path, 'howto100m', 'raw_videos')
        self.visual_data_type = 'video'

    def init_list_data_dict(self):
        self.list_data_dict = []
        data_path = os.path.join(self.data_path, 'howto100m', 'howto100m_dvc_filter_25.json')
        data_dict = json.load(open(data_path, "r"))
        for k in data_dict:
            v = data_dict[k]
            v['id'] = k
            self.list_data_dict.append(v)

            
class DVCDataset_youcook2(DVCDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(DVCDataset_youcook2, self).__init__(data_path, tokenizer, data_args)
        
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

                
class DVCDataset_vitt(DVCDataset):
    def __init__(self, data_path, tokenizer, data_args):
        super(DVCDataset_vitt, self).__init__(data_path, tokenizer, data_args)
        
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
            

