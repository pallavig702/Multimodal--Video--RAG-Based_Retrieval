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
vid_uris = frame_uris("human sitting on a wheel chair", max_distance=1.55)
print(vid_uris)
     

#####################################################################
#LLM Setup
#####################################################################

api_key='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'
#userdata = {'OPENAI_API_KEY': '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'}
userdata = {'OPENAI_API_KEY': api_key}
pi_key = userdata.get('OPENAI_API_KEY')
#pi_key = os.getenv('OPENAI_API_KEY')

# Instantiate the LLM
gpt4o = ChatOpenAI(model="gpt-4o", temperature = 0.0, api_key=api_key)

# Instantiate the Output Parser
parser = StrOutputParser()

# Define the Prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are document retrieval assistant that neatly synthesizes and explains the text and images provided by the user from the query {query}"),
        (
            "user",
            [
                {
                    "type": "text",
                    "text": "{texts}"
                },
                {
                    "type": "image_url",
                    "image_url": {'url': "data:image/jpeg;base64,{image_data_1}"}
                },
                {
                    "type": "text",
                    "text": "This is a frame from a video, refer to it as a video:"
                },
                {
                    "type": "image_url",
                    "image_url": {'url': "data:image/jpeg;base64,{image_data_2}"}
                },

            ],
        ),
    ]
)

chain = prompt | gpt4o | parser



########################################################################
#Prompt Setup
#The below function will take our query, run them through our new retrieval functions, and format our prompt input, which is expecting a dictionary like:
'''
{
  "query": "the user query",
  "texts": "the retrieved texts",
  "image_data_1": "The retrieved image, base64 encoded",
  "image_data_2": "The retrieved frame, base64 encoded",
}
Note that for the sake of token consumption, context window, and cost we'll only be passing in two images (the image and a single relevant frame) and the text to the model.
'''
########################################################################
def format_prompt_inputs(user_query):

    frame = frame_uris(user_query, max_distance=1.55)[0]
    image = image_uris(user_query, max_distance=1.5)[0]
    text = text_uris(user_query, max_distance=1.3)

    inputs = {}

    # save the user query
    inputs['query'] = user_query

    # Insert Text
    inputs['texts'] = text

    # Encode the first image
    with open(image, 'rb') as image_file:
        image_data_1 = image_file.read()
    inputs['image_data_1'] = base64.b64encode(image_data_1).decode('utf-8')

    # Encode the Frame
    with open(frame, 'rb') as image_file:
        image_data_2 = image_file.read()
    inputs['image_data_2'] = base64.b64encode(image_data_2).decode('utf-8')

    return frame, image, inputs


#Full Multimodal RAG
#Image, Video, Audio, and Text retrieval and LLM processing

query = "Human sitting on the wheel chair"
frame, image, inputs = format_prompt_inputs(query)
response = chain.invoke(inputs)


display(Markdown("## Video\n"))
video = f"RagBasedVideoAnalysis/StockVideos-CC0-frames/{frame.split('/')[1]}.mp4"
display(Video(video, embed=True, width=300))
display(Markdown("---"))

display(Markdown("## AI Response\n"))
display(Markdown(response))
display(Markdown("---"))
