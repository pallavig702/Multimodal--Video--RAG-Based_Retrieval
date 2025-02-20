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

'''
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

extract_frames(video_folder_path, output_folder_path)
print("################################# Done Extract Frames #################################")
############################################################################################
'''
############################################################################################
#Step 2: Creating Video Collection
#Since we're technically processing images, we'll use the Image Loader and Clip Embedding Function from before for our video collection.
############################################################################################
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
from chromadb.utils.data_loaders import ImageLoader


# Create an instance of OpenCLIPEmbeddingFunction
embedding_function = OpenCLIPEmbeddingFunction()
# Create an instance of ImageLoader
image_loader = ImageLoader()

video_collection = client.get_or_create_collection(
    name='video_collection',
    embedding_function=embedding_function,
    data_loader=image_loader
)
print("################################# Done video Collection #################################")

############################################################################################
#Adding Video Frames to Collection
#We iterate over the frame folders and embed them into the database, with specific metadata that links back to the video file that the frame comes from
############################################################################################
def add_frames_to_chromadb(video_dir, frames_dir):
    # Dictionary to hold video titles and their corresponding frames
    video_frames = {}

    # Process each video and associate its frames
    for video_file in os.listdir(video_dir):
        if video_file.endswith('.mp4'):
            video_title = video_file[:-4]
            frame_folder = os.path.join(frames_dir, video_title)
            if os.path.exists(frame_folder):
                # List all jpg files in the folder
                video_frames[video_title] = [f for f in os.listdir(frame_folder) if f.endswith('.jpg')]

    # Prepare ids, uris and metadatas
    ids = []
    uris = []
    metadatas = []

    for video_title, frames in video_frames.items():
        video_path = os.path.join(video_dir, f"{video_title}.mp4")
        for frame in frames:
            frame_id = f"{frame[:-4]}_{video_title}"
            frame_path = os.path.join(frames_dir, video_title, frame)
            ids.append(frame_id)
            uris.append(frame_path)
            metadatas.append({'video_uri': video_path})

    video_collection.add(ids=ids, uris=uris, metadatas=metadatas)

# Running it
video_dir = '/home/pgupt60/scripts/CPU_ConvertedVideos/subset/'
frames_dir = 'RagBasedVideoAnalysis/StockVideos-CC0-frames'

add_frames_to_chromadb(video_dir, frames_dir)
print("########################## Done adding frames to chroma DB ###############################")


############################################################################################
#Video retrieval testing Function
#Now that all of the frames of every video are embedded into the collection, and point back to their respective video file, we can test out video retrieval!
############################################################################################

def display_videos(query_text, max_distance=None, max_results=5, debug=False):
    # Deduplication set
    displayed_videos = set()

    # Query the video collection with the specified text
    results = video_collection.query(
        query_texts=[query_text],
        n_results=max_results,  # Adjust the number of results if needed
        include=['uris', 'distances', 'metadatas']
    )

    # Extract URIs, distances, and metadatas from the result
    uris = results['uris'][0]
    distances = results['distances'][0]
    metadatas = results['metadatas'][0]

    # Display the videos that meet the distance criteria
    for uri, distance, metadata in zip(uris, distances, metadatas):
        video_uri = metadata['video_uri']

        # Check if a max_distance filter is applied and the distance is within the allowed range
        if (max_distance is None or distance <= max_distance) and video_uri not in displayed_videos:
            if debug:
              print(f"URI: {uri} - Video URI: {video_uri} - Distance: {distance}")
            display(Video(video_uri, embed=True, width=300))
            displayed_videos.add(video_uri)  # Add to the set to prevent duplication
        else:
            if debug:
              print(f"URI: {uri} - Video URI: {video_uri} - Distance: {distance} (Filtered out)")

# Running it
#display_videos("Trees", max_distance=1.55, debug=True)
#display_videos("human sitting on a wheel chair", max_distance=1.55, debug=True)
display_videos("human using laptop", max_distance=1.55, debug=True)
print("########################## Done Displaying Videos  ###############################")

## Multimodal RAG retrieval
#query = "Women"
#display(Markdown("# Video(s) Retrieved: \n"))
#display_videos(query, max_distance=2.00, debug=True)
     
########################################################################################
#Putting the 'AG' in Multimodal RAG
#But of course, the retrieval is only half the part of a Retrieval Augmented Generation system. We want to pass the retrieved data through a language model to do some sort of response generation
#
#So, we'll define some new functions that can output the retrieved information into a format that we can use not just for displaying, but language model processing

#Note: At the time of making this, there's no audio capabilities on language models yet the same way that vision and text are widely available, so we'll be displaying the audio file but will be unable to process it with an LLM. Maybe one day 🤔
########################################################################################

#Video Retrieval
#Takes in the query, returns the retrieved frames.

def frame_uris(query_text, max_distance=None, max_results=5):
    results = video_collection.query(
        query_texts=[query_text],
        n_results=max_results,
        include=['uris', 'distances']
    )

    filtered_uris = []
    seen_folders = set()

    for uri, distance in zip(results['uris'][0], results['distances'][0]):
        if max_distance is None or distance <= max_distance:
            folder = os.path.dirname(uri)
            if folder not in seen_folders:
                filtered_uris.append(uri)
                seen_folders.add(folder)

        if len(filtered_uris) == max_results:
            break

    return filtered_uris

# Example usage:
vid_uris = frame_uris("human showing signs of mobility disability", max_distance=1.55)
print(vid_uris)
     

#####################################################################
#LLM Setup
#####################################################################

############################################################################################
############################## STEP 1: LOAD THE MODEL ######################################
############################################################################################
from transformers import BitsAndBytesConfig, LlavaNextVideoForConditionalGeneration, LlavaNextVideoProcessor #for pre-trained model and processor.
import torch #for tensor-based computation.
import csv
import av #video decoding.
import numpy as np

# Model configuration
# Explanation: Configures the model to load in 4-bit quantization (reducing memory footprint) and uses 16-bit floating point precision for computations.
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Model and Processor Initialization:
# Processor: Prepares inputs for the LLaVA-NeXT model.
# Model: Loads the video-conditioned text generation model with pre-trained weights.

processor = LlavaNextVideoProcessor.from_pretrained("llava-hf/LLaVA-NeXT-Video-7B-hf")
model = LlavaNextVideoForConditionalGeneration.from_pretrained(
    "llava-hf/LLaVA-NeXT-Video-7B-hf",
    quantization_config=quantization_config,
    device_map='auto'
)

############################################################################################
# Process retrieved frames with LLaVa-Next
############################################################################################

for frame in vid_uris: #retrieved_frames:
    image = Image.open(frame)
    inputs = processor(images=image, text="Describe the scene?", return_tensors="pt")
    
    with torch.no_grad():
        response = model.generate(**inputs)

    print(f"🤖 Frame {frame} Response:", response)

