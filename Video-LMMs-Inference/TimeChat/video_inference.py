import argparse
import os
import random
import json
import numpy as np
import wandb
import pandas as pd
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

def ground_truth(name, video, normal_annot, questions):
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
            if not step['has_errors']:
                gt[index] = 1

    return gt

def flatten(nested_list):
    return [item for sublist in nested_list for item in sublist]

def accuracy(gt_flat, pred_flat):
    precision = precision_score(gt_flat, pred_flat, average='micro')
    recall = recall_score(gt_flat, pred_flat, average='micro')
    f1 = f1_score(gt_flat, pred_flat, average='micro')
    accuracy = accuracy_score(gt_flat, pred_flat)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy
    }

def data_file(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, sep=',', mode='a+')

def inference(args, chat):
    video_dir = '/data/rohith/captain_cook/videos/gopro/resolution_360p/'
    qs = load_file('/data/bhavya/task_verification/Video-LLaVA/questions.json')
    gt_dict = load_file('/data/bhavya/task_verification/Video-LLaVA/step_annotations.json')
    normal_annot = load_file('/data/bhavya/task_verification/Video-LLaVA/normal_videos.json')
    output_file = 'timechat_proc_metrics.txt'
    op_file = 'timechat_metrics.csv'
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
        print(video.size())
        C, T, H, W = video.shape
        ts.show(video.transpose(0,1))

        # Setup chat system
        img_list = []
        chat_state = conv_llava_llama_2.copy()
        chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
        msg = chat.upload_video_without_audio(video_path=args.video_path, conv=chat_state, img_list=img_list, n_frms=96)
        # Response from chat
        for q in questions:
            text_input = f"You are given a cooking video from the Captain Cook dataset. Please watch the video and answer the question: {q} Return the answer in the format of Yes or No."
            print(text_input)

            chat.ask(text_input, chat_state)
            num_beams = args.num_beams
            temperature = args.temperature
            llm_message = chat.answer(conv=chat_state, img_list=img_list, num_beams=num_beams, temperature=temperature, max_new_tokens=300, max_length=5000)[0]

            print(llm_message)
            output = llm_message.lower()
            if 'yes' in output:
                pred.append(1)
            else:
                pred.append(0)

        if len(g_truth)==len(pred):
            video_metrics = accuracy(g_truth, pred)
            met = {'v': v, 'a': video_metrics['accuracy'], 'r': video_metrics['recall'], 'p': video_metrics['precision'], 'f1': video_metrics['f1_score'], 'gt': g_truth, 'pred': pred}
            data_file(met, op_file)
            wandb.log({'video':v, 'accuracy': video_metrics['accuracy'], 'recall': video_metrics['recall'], 'f1_score': video_metrics['f1_score'], 'precision': video_metrics['precision']})

        else:
            wandb.log({'video': v})

        prediction.append(pred)

    gt = flatten(gt)
    prediction = flatten(prediction)
    print(gt)
    print(prediction)

    metrics = accuracy(gt, prediction)

    print("Accuracy: {accuracy} \n F1: {f1_score} \n Recall: {recall} \n Precision: {precision}".format(
        accuracy=metrics['accuracy'],
        f1_score=metrics['f1_score'],
        recall=metrics['recall'],
        precision=metrics['precision']
    ))

    wandb.log({'value':'total dataset', 'accuracy': metrics['accuracy'], 'recall': metrics['recall'], 'f1_score': metrics['f1_score'], 'precision': metrics['precision']})

    content = "Accuracy: {accuracy} \nF1: {f1_score} \nRecall: {recall} \nPrecision: {precision} \nGround Truth: {ground_truth} \nPredicted: {predicted}".format(
        accuracy=metrics['accuracy'],
        f1_score=metrics['f1_score'],
        recall=metrics['recall'],
        precision=metrics['precision'],
        ground_truth = gt,
        predicted = prediction
    )

    with open(output_file, 'w') as file:
        file.write(content)

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
