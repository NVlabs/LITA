# Copyright (c) 2024, NVIDIA Corporation & Affiliates. All rights reserved. 
# 
# This work is made available under the Nvidia Source Code License-NC. 
# To view a copy of this license, visit 
# https://github.com/NVlabs/LITA/blob/main/LICENSE

import argparse
import shutil

import torch
import gradio as gr
import os
import tempfile

from llava.conversation import conv_templates, SeparatorStyle, Conversation
from lita.utils import load_video
from lita.serve.gradio_utils import tos_markdown, learn_more_markdown, title_markdown, block_css, Chat


def generate(textbox_in, first_run, state, state_, image_tensor, vid_len):

    if not textbox_in:
        gr.Warning("Please enter instructions")
        return (state, state_, state.to_gradio_chatbot(), first_run, gr.update(value=None, interactive=True), image_tensor, vid_len)
    
    if len(image_tensor) == 0:
        gr.Warning("Please upload a video")
        return (state, state_, state.to_gradio_chatbot(), first_run, gr.update(value=None, interactive=True), image_tensor, vid_len)
    
    print(image_tensor.shape, vid_len)
    
    first_run = False if len(state.messages) > 0 else True
    
    # process textbox_in if needed
    text_en_in = textbox_in
    
    state_, outputs_timestamp = handler.generate(image_tensor, text_en_in, first_run=first_run, state=state_, vid_len=vid_len)
    
    # process text_en_out if needed
    textbox_out = outputs_timestamp
    
    state.append_message(state.roles[0], textbox_in)
    state.append_message(state.roles[1], textbox_out)

    return (state, state_, state.to_gradio_chatbot(), False, gr.update(value=None, interactive=True), image_tensor, vid_len)

def initialize(state, state_):
    state = conv_templates[conv_mode].copy()
    state_ = conv_templates[conv_mode].copy()
    return (gr.update(value=None, interactive=True),
        gr.update(value=None, interactive=True),\
        True, state, state_, state.to_gradio_chatbot(), [], 0.0)

def upload_video(video, image_tensor, vid_len):
    print(video)
    if video is not None:
        image_tensor, vid_len = load_video(video, handler.image_processor, handler.model.config.num_frames, return_vid_len=True)
        print(image_tensor.shape)
        print(vid_len)
        return (gr.update(interactive=False), image_tensor, vid_len)
    else:
        return video, image_tensor, vid_len      

def main(args):

    if not os.path.exists("tmp"):
        os.makedirs("tmp")

    textbox = gr.Textbox(show_label=False, placeholder="Enter text and press ENTER", container=False)
    
    with gr.Blocks(title='LITA', theme=gr.themes.Default(), css=block_css) as demo:
        gr.Markdown(title_markdown)
        state = gr.State()
        state_ = gr.State()
        first_run = gr.State()
        image_tensor = gr.State()
        vid_len = gr.State()

        with gr.Row():
            with gr.Column(scale=3):
                video = gr.Video(label="Input Video")
                upload_btn = gr.Button(value="Upload Video", interactive=True)
                # TODO: examples
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(label="LITA", bubble_full_width=True).style(height=750)
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox.render()
                    with gr.Column(scale=1, min_width=50):
                        submit_btn = gr.Button(
                            value="Send", variant="primary", interactive=True
                        )
                with gr.Row(elem_id="buttons") as button_row:
                    clear_btn = gr.Button(value="Reset", interactive=True)
                    
        gr.Markdown(tos_markdown)
        gr.Markdown(learn_more_markdown)

        submit_btn.click(generate, [textbox, first_run, state, state_, image_tensor, vid_len],
                         [state, state_, chatbot, first_run, textbox, image_tensor, vid_len])
        textbox.submit(generate, [textbox, first_run, state, state_, image_tensor, vid_len],
                         [state, state_, chatbot, first_run, textbox, image_tensor, vid_len])
        clear_btn.click(initialize, [state, state_],
                        [video, textbox, first_run, state, state_, chatbot, image_tensor, vid_len])
        upload_btn.click(upload_video, [video, image_tensor, vid_len], [video, image_tensor, vid_len])
        demo.load(initialize, [state, state_],
                        [video, textbox, first_run, state, state_, chatbot, image_tensor, vid_len])
    
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    
    conv_mode = "llava_v1"
    handler = Chat(args.model_path, args.model_base, conv_mode=conv_mode, load_8bit=args.load_8bit, load_4bit=args.load_4bit, device='cuda')
    
    main(args)
