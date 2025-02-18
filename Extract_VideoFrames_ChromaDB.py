#This script extract image frames from video and then stores the selected frames in the vector database chroma db
########################################################################################
#Convert Videos into Frames
#As mentioned, we'll be treating videos as a collection of frames. To do this we must split the videos into those frames. The script below operates as such:

#The video is read frame by frame. For each frame, the script checks if it should be saved based on the following conditions:

#The frame is the first frame of the video (frame_number == 0).
#The frame is exactly every 5 seconds (frame_number % int(fps * 5) == 0).
#The frame is the last frame of the video (frame_number == frame_count - 1).
#If a frame meets any of the above conditions, its timestamp in seconds is calculated and used to name the saved image file (e.g., frame_15.jpg for a frame at 15 seconds).

#This will prepare all our videos into frames (with some loss) that will be indexed, and point back towards the video file
########################################################################################

import IPython
from IPython.display import HTML, display, Image, Markdown, Video, Audio
from typing import Optional, Sequence, List, Dict, Union

import soundfile as sf

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#from google.colab import userdata

from sentence_transformers import SentenceTransformer
from transformers import ClapModel, ClapProcessor
from datasets import load_dataset

import sys
############# importing sqlite >3.35 before chroma db which requires ir
import pysqlite3
sys.modules['sqlite3'] = pysqlite3
import sqlite3
print(sqlite3.sqlite_version)
#################
import chromadb
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader
from chromadb.api.types import Document, Embedding, EmbeddingFunction, URI, DataLoader

import numpy as np
import torchaudio
import base64
import torch
import json
import cv2
import os



########################################################################################
#Instantiate ChromaDB
#For this project, we'll be using the open source vector database ChromaDB, fully taking advantage of its open source nature to create some of our own custom features
########################################################################################
path = "RagBasedVideoAnalysis/mm_vdb"
client = chromadb.PersistentClient(path=path)
########################################################################################

########################################################################################
#Convert Videos into Frames
########################################################################################

def extract_frames(video_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_filename in os.listdir(video_folder):
        if video_filename.endswith('.mp4'):
            video_path = os.path.join(video_folder, video_filename)
            video_capture = cv2.VideoCapture(video_path)
            fps = video_capture.get(cv2.CAP_PROP_FPS)
            frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            print("duration:",duration)
            print("frame_count:",frame_count)
            print("fps:",fps)

            output_subfolder = os.path.join(output_folder, os.path.splitext(video_filename)[0])
            if not os.path.exists(output_subfolder):
                os.makedirs(output_subfolder)

            success, image = video_capture.read()
            frame_number = 0
            while success:
                if frame_number == 0 or frame_number % int(fps * 5) == 0 or frame_number == frame_count - 1:
                    frame_time = frame_number / fps
                    output_frame_filename = os.path.join(output_subfolder, f'frame_{int(frame_time)}.jpg')
                    cv2.imwrite(output_frame_filename, image)

                success, image = video_capture.read()
                frame_number += 1

            video_capture.release()

video_folder_path = '/home/pgupt60/scripts/CPU_ConvertedVideos/subset/'#Scenario2_Ipad1_05.mp4'
output_folder_path = 'RagBasedVideoAnalysis/StockVideos-CC0-frames'

########################################################################################
# Function call
########################################################################################
extract_frames(video_folder_path, output_folder_path)
print("################################# Done Extract Frames #################################")
