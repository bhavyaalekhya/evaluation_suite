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

class Model:
    @staticmethod
    def load_file(path):
        with open(path, 'r') as file:
            op = json.load(file)
        return op
    
    @staticmethod
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
    
    @staticmethod
    def initialize_model():
        print('Initializing Chat')
        args =Model.parse_args()
        args.cfg_path = '/data/bhavya/task_verification/CVVREvaluation/cvvr_evaluation_suite/Video-LMMs-Inference/TimeChat/eval_configs/timechat.yaml'
        cfg = Config(args)
        DIR = 'ckpt/TimeChat-7b'
        MODEL_DIR = f'{DIR}/timechat_7b.pth'
        model_config = cfg.model_cfg
        args.gpu_id = 0
        model_config.device_8bit = args.gpu_id
        model_config.ckpt = MODEL_DIR
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
        model.eval()
        wandb.watch(model, log="all")
        vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
        vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

        chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
        print("Initialization finished")
        
        return args, chat
    
    @staticmethod
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
    
    @staticmethod
    def flatten(nested_list):
        return [item for sublist in nested_list for item in sublist]

    @staticmethod
    def save_data(output_file, gt, pred):
        print("Ground Truth: {gt} \nPredicted: {prediction}".format(
            gt = gt,
            prediction = pred
        ))

        content = "Ground Truth: {gt} \nPredicted: {predicted}".format(
            gt = gt,
            predicted = pred
        )

        with open(output_file, 'w') as file:
            file.write(content)

        print(f"File has been saved at: {output_file}")

    @staticmethod
    def write_recur(file, video, data):
        content = "{video}: Pred: {data}",format(video = video, data = data)

        with open(file, 'a') as f:
            f.write(content)

class Preparation_Error():
    def __init__(self, args, chat, video_dir, gt_dict, normal_annot):
        self.args = args
        self.chat = chat
        self.video_dir = video_dir
        self.gt_dict = gt_dict
        self.normal_annot = normal_annot

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
        q = len(questions)
        
        gt = [0] * q

        for step in steps:
            if step['description'] in common_steps:
                index = n_steps_desc.index(step['description'])
                if index < q:
                    if step['has_errors'] and "Preparation Error" in step['errors']:
                        gt[index] = 1

        return gt
    
    def op_val(self, ans, correctans, q_type):
        if q_type == 'yes_no':
            if 'yes' in ans or 'not' not in ans:
                return 0
            else:
                return 1
        elif q_type == 'option':
            if correctans in ans:
                return 0
            else:
                return 1
        
    def question_index(self, related_questions):
        question_to_index = {}
        index_counter = 0
        for question in related_questions:
            question_to_index[question['q']] = index_counter
            if 'followup' in question.keys():
                for followup in question['followup']:
                    question_to_index[followup] = index_counter
            index_counter += 1
        return question_to_index
    
    def preparation_inference(self):
        video_dir = self.video_dir
        qs = Model.load_file('/data/bhavya/task_verification/CVVREvaluation/cvvr_evaluation_suite/Video-LMMs-Inference/TimeChat/error_prompts/preparation_error.json')
        gt_dict = self.gt_dict
        normal_annot = self.normal_annot
        output_file = 'error_outputs/preparation_error.txt'
        op_file = 'error_outputs/preparation_recur.txt'
        prediction = []
        gt = []

        for v in tqdm(os.listdir(video_dir), desc="Processing videos"):
            self.args.video_path = os.path.join(video_dir, v)
            name = v.split('_')
            q_name = name[0] + '_x'
            
            video, _ = load_video(video_path=self.args.video_path, n_frms=30, sampling='uniform', return_msg=True)
            questions = qs[q_name]['questions']
            g_truth = self.ground_truth(name[0], gt_dict[name[0] + '_' + name[1]], normal_annot, questions)
            gt.append(g_truth)
            pred = [0] * len(g_truth)
            #print(video.size())
            C, T, H, W = video.shape
            ts.show(video.transpose(0,1))

            # Setup chat system
            img_list = []
            chat_state = conv_llava_llama_2.copy()
            chat_state.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
            msg = self.chat.upload_video_without_audio(video_path=self.args.video_path, conv=chat_state, img_list=img_list, n_frms=96)

            question_ind = self.question_index(questions)
            # Response from chat
            for steps in questions:
                inp1 = steps['q']
                if 'correctans' not in steps.keys():
                    q_type = 'yes_no'
                else:
                    correctans = steps['correctans']
                    q_type = 'option'
                output = Model.ask_question(self.args, self.chat, chat_state, img_list, inp1)
                print(output)
                pred[question_ind[inp1]] = self.op_val(pred, correctans, q_type)
                if 'followup' in steps.keys():
                    for q in steps['followup']:
                        inp2 = q
                        pred2 = Model.ask_question(self.args, self.chat, chat_state, img_list, inp2)
                        print(pred2)
                        pred[question_ind[inp2]] = self.op_val(pred2, correctans, q_type)
            
            Model.write_recur(op_file, v, pred)
            prediction.append(pred)

        gt = Model.flatten(gt)
        prediction = Model.flatten(prediction)
        
        Model.save_data(output_file, gt, prediction)

def main():
    #initialize video dir, gt_dict, normal_annot
    tc = Model()
    video_dir = '/home/ptg/ptg/rohith/resolution_360p/'
    gt_dict = tc.load_file('./step_annotations.json')
    normal_annot = tc.load_file('./normal_videos.json')

    # Initialize chat
    args, chat = tc.initialize_model()

    #preparation error inference
    print("Preparation Error Inference: \n")
    preparation_error = Preparation_Error(args, chat, video_dir, gt_dict, normal_annot)
    preparation_error.preparation_inference()


if __name__ == '__main__':
    main()