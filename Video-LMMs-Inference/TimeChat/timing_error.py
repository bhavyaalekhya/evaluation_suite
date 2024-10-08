import argparse
import os
import random
import json
import numpy as np
import wandb
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import torch
import torch.backends.cudnn as cudnn
import torchshow as ts
from timechat.common.config import Config
from timechat.common.dist_utils import get_rank
from timechat.common.registry import registry
from timechat.conversation.conversation_video import Chat, Conversation, default_conversation, SeparatorStyle, conv_llava_llama_2
import decord
import cv2
import time
import subprocess
from decord import VideoReader
from timechat.processors.video_processor import ToTHWC, ToUint8, load_video
decord.bridge.set_bridge('torch')

# imports modules for registration
from timechat.datasets.builders import *
from timechat.models import *
from timechat.processors import *
from timechat.runners import *
from timechat.tasks import *

import random as rnd
from transformers import StoppingCriteria, StoppingCriteriaList
from PIL import Image
import gradio as gr

wandb.init(
    project="Task_Verification",
    entity="vsbhavyaalekhya"
)

def parse_args():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--cfg_path", default='eval_configs/timechat.yaml', help='path to configuration file.')
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to the model.")
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--text_query", default="What is he doing?", help="question the video")
    parser.add_argument("--video_path", default='example/hotdog.mp4', help='path to the video files directory')
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args(args=[])
    return args

def load_file(path):
    with open(path, 'r') as file:
        op = json.load(file)
    return op

def ground_truth(self, name, video, normal_annot, questions):
    gt = []
    steps = video['steps']
    normal = name + '_x'
    n_steps = normal_annot[normal]['steps']
    n_steps_desc = []

    for step in n_steps:
        n_steps_desc.append(step['description'])

    video_steps_desc = [step['description'] for step in steps]
    common_steps = list(set(n_steps_desc).intersection(video_steps_desc))
    
    gt = [0] * len(questions)

    for step in steps:
        if step['description'] in common_steps:
            index = common_steps.index(step['description'])
            if step['has_errors'] and "Timing Error" in step['errors']:
                gt[index] = 1

    return gt

def question_index(related_questions):
    question_to_index = {}
    index_counter = 0
    for question in related_questions:
        question_to_index[question['q']] = index_counter
        if 'followup' in question.keys():
            for followup in question['followup']:
                question_to_index[followup] = index_counter
        index_counter += 1
    return question_to_index

def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]

def ask_question(args, chat, chat_state, img_list, q):
    text_input = f"You are given a cooking video from the Captain Cook dataset. Please watch the video and answer the question: {q} Return the answer in the format of Yes or No."
    print(text_input)

    chat.ask(text_input, chat_state)
    num_beams = args.num_beams
    temperature = args.temperature
    llm_message = chat.answer(conv=chat_state, img_list=img_list, num_beams=num_beams, temperature=temperature, max_new_tokens=300, max_length=5000)[0]

    print(llm_message)
    output = llm_message.lower()

    return output

def op_val(ans):
    if 'yes' in ans or 'not' not in ans:
        return 0
    else:
        return 1

def write_recur(op_file, name, data):
    content = f"\n{name}: Pred: {data}"

    with open(op_file, 'a') as file:
        file.write(content)

def inference(args, chat):
    video_dir = '/data/rohith/captain_cook/videos/gopro/resolution_360p/'
    qs = load_file('/data/bhavya/task_verification/CVVREvaluation/cvvr_evaluation_suite/Video-LMMs-Inference/TimeChat/error_prompts/timing_error.json')
    gt_dict = load_file('/data/bhavya/task_verification/Video-LLaVA/step_annotations.json')
    normal_annot = load_file('/data/bhavya/task_verification/Video-LLaVA/normal_videos.json')
    output_file = './timechat_metrics/timing_error.txt'
    op_file = 'error_outputs/timing_recur.txt'
    prediction = []
    gt = []

    for v in os.listdir(video_dir):
        args.video_path = os.path.join(video_dir, v)
        name = v.split('_')
        q_name = name[0] + '_x'
        
        video, _ = load_video(video_path=args.video_path, n_frms=30, sampling='uniform', return_msg=True)
        questions = qs[q_name]['questions']
        g_truth = ground_truth(name[0], gt_dict[name[0] + '_' + name[1]], normal_annot, questions)
        gt.append(g_truth)
        pred = []
        #print(video.size())
        C, T, H, W = video.shape
        #ts.show(video.transpose(0,1))

        # Setup chat system
        img_list = []
        chat_state = conv_llava_llama_2.copy()
        chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
        msg = chat.upload_video_without_audio(video_path=args.video_path, conv=chat_state, img_list=img_list, n_frms=96)

        question_ind = question_index(questions)
        # Response from chat
        for steps in questions:
            inp1 = steps['q'] 
            pred = ask_question(args, chat, chat_state, img_list, inp1)
            pred = pred.lower()
            print(pred)
            pred[question_ind[inp1]] = op_val(pred)
            if 'followup' in steps.keys():
                for question in steps['followup']:
                    inp2 = question 
                    pred2 = ask_question(args, chat, chat_state, img_list, inp2)
                    print(pred2)
                    pred[question_ind[inp2]] = op_val(pred2)
        
        write_recur(op_file, v, pred)
        prediction.append(pred)

    gt = flatten(gt)
    prediction = flatten(prediction)

    print("Ground Truth: {gt} \nPredicted: {prediction}".format(
        gt = gt,
        prediction = prediction
    ))

    content = "Ground Truth: {ground_truth} \nPredicted: {predicted}".format(
        ground_truth = gt,
        predicted = prediction
    )

    with open(output_file, 'w') as file:
        file.write(content)

    print(f'Output saved in {output_file}')

def main():
    # Initialize chat
    print('Initializing Chat')
    args = parse_args()
    args.cfg_path = '/data/bhavya/task_verification/CVVREvaluation/cvvr_evaluation_suite/Video-LMMs-Inference/TimeChat/eval_configs/timechat.yaml'
    cfg = Config(args)
    DIR = 'ckpt/TimeChat-7b'
    MODEL_DIR = f'{DIR}/timechat_7b.pth'
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_config.ckpt = MODEL_DIR
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
    model.eval()
    wandb.watch(model, log="all")
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    chat = Chat(model, vis_processor, device='cuda: {}'.format(args.gpu_id))
    print("Initialization finished")

    inference(args, chat)

if __name__ == '__main__':
    main()
