# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/LITA/blob/main/LICENSE

import os
from PIL import Image
import glob
import numpy as np
import torch
import decord
from decord import VideoReader


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def load_image(image_file, processor, image_aspect_ratio='square'):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    if image_aspect_ratio == 'pad':
        image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
    image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    return image


def load_video_frames(visual_path, processor, num_frames, image_aspect_ratio='square'):
    if type(visual_path) is str:
        frame_paths = sorted(glob.glob(os.path.join(visual_path, '*.jpg')))
        idx = np.round(np.linspace(0, len(frame_paths) - 1, num_frames)).astype(int)
        frame_paths = list(np.array(frame_paths)[idx])
    else:
        assert type(visual_path) is list
        frame_paths = visual_path
    
    frames = []
    for frame_path in frame_paths:
        frame = load_image(frame_path, processor, image_aspect_ratio=image_aspect_ratio)
        frames.append(frame)
    return torch.stack(frames, dim=0)


def load_video(video_path, processor, num_frames, return_vid_len=False):
    decord.bridge.set_bridge("torch")
    video_reader = VideoReader(uri=video_path)
    
    idx = np.round(np.linspace(0, len(video_reader) - 1, num_frames)).astype(int)
    frames = video_reader.get_batch(idx)
    
    frames = processor.preprocess(frames, return_tensors='pt')['pixel_values']
    
    if return_vid_len:
        fps = video_reader.get_avg_fps()
        num_frames = len(video_reader)
        if fps > 0:
            vid_len = float(num_frames) / fps
        else:
            vid_len = 0.0
        return frames, vid_len
    else:   
        return frames
