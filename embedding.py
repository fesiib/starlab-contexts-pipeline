import os

from pathlib import Path

import cv2
import json
import csv
import re
import webvtt
import whisper

import numpy as np
import pandas as pd

from yt_dlp import YoutubeDL

from helpers.bert import bert_embed_text
from helpers.sklearn import reduce_dim, standardize

import matplotlib.pyplot as plt

DATABASE = Path("static/database")
RESULTS = Path("static/results")
HOWTO = Path("howto100m")

class Video:
    video_id = ""
    video_link = ""
    processed = False
    video_path = ""
    subtitle_path = ""
    audio_path = ""
    metadata = {}

    def __init__(self, video_id, video_link):
        self.video_id = video_id
        self.video_link = video_link
        self.processed = False

    def process(self):
        options = {
            'format': 'bv[height<=?480][ext=mp4]+ba[ext=mp3]/best',
            #'format': 'bestvideo[height<=480][ext=mp4]+bestaudio[ext=mp3]/best',
            'outtmpl': os.path.join(DATABASE, '%(id)s.%(ext)s'),
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': {'en'},  # Download English subtitles
            'subtitlesformat': '/vtt/g',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'mp3',
                'preferredquality': '192',
            }],
            'keepvideo': True,
            'skip_download': False,
        }

        with YoutubeDL(options) as ydl:
            info = ydl.extract_info(self.video_link, download=False)
            self.metadata = ydl.sanitize_info(info)
            video_title = self.metadata.get('id')
            video_path = os.path.join(DATABASE, f'{video_title}.mp4')
            if not os.path.exists(video_path):
                ydl.download([self.video_link])
                print(f"Video '{video_title}' downloaded successfully.")
            else:
                print(f"Video '{video_title}' already exists in the directory.")

            self.video_path = video_path
            self.subtitle_path = os.path.join(DATABASE, f'{video_title}.en.vtt')
            self.audio_path = os.path.join(DATABASE, f'{video_title}.mp3')
            self.extract_transcript()
            self.processed = True
            return self.metadata

    def extract_frames(self):
        video_cap = cv2.VideoCapture(self.video_path)
        
        frames = []
        
        while (True):
            res, frame = video_cap.read()
            if (res == False):
                break

            frames.append(frame)
        
        video_cap.release()

        return frames


    def extract_transcript_from_audio(self):
        output_path = self.audio_path.replace(".mp3", ".alt.json")
        raw_transcript = {}
        if os.path.exists(output_path):
            with open(output_path, 'r') as f:
                raw_transcript = json.load(f)
        else:
            model = whisper.load_model("small.en")
            raw_transcript = model.transcribe(self.audio_path)
            with open(output_path, 'w') as f:
                json.dump(raw_transcript, f, indent=2)

        transcript = []
        for segment in raw_transcript["segments"]:
            transcript.append({
                "start": segment["start"],
                "finish": segment["end"],
                "text": segment["text"],
            })
        return transcript

    def extract_transcript(self):
        if not os.path.exists(self.subtitle_path):
            print(f"Subtitles file '{self.subtitle_path}' does not exist.")
            if not os.path.exists(self.audio_path):
                print(f"Audio file '{self.audio_path}' does not exist.")
                return []
            transcript = self.extract_transcript_from_audio(self.audio_path)

            return transcript

        subtitles = webvtt.read(self.subtitle_path)

        transcript = []
        for caption in subtitles:
            lines = caption.text.strip("\n ").split("\n")
            if len(transcript) == 0:
                transcript.append({
                    "start": caption.start,
                    "finish": caption.end,
                    "text": "\n".join(lines),
                })
                continue
            last_caption = transcript[len(transcript) - 1]

            new_text = ""
            for line in lines:
                if line.startswith(last_caption["text"], 0):
                    new_line = line[len(last_caption["text"]):-1].strip()
                    if len(new_line) > 0:
                        new_text += new_line + "\n"
                elif len(line) > 0:
                    new_text += line + "\n"
            new_text = new_text.strip("\n ")
            if len(new_text) == 0:
                transcript[len(transcript) - 1]["finish"] = caption.end
            else:
                transcript.append({
                    "start": caption.start,
                    "finish": caption.end,
                    "text": new_text,
                })
        return transcript

def format_rgba(color):
    return f"rgba({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)}, 1)"

def process_howto100m():
    video_ids_path = HOWTO / "HowTo100M_v1.csv"
    task_ids_path = HOWTO / "task_ids.csv"

    videos_info = []
    with open(video_ids_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            videos_info.append({
                "video_id": row["video_id"],
                "cat1": row["category_1"],
                "cat2": row["category_2"],
                "rank": row["rank"],
                "task_id": row["task_id"],
            })
    
    task_id_titles = {}
    with open(task_ids_path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter='\t')
        for row in csv_reader:
            task_id_titles[row["id"]] = row["task_name"]
    
    for video_info in videos_info:
        video_info["task_name"] = task_id_titles[video_info["task_id"]]

    return videos_info

def process_videos(n=100):
    ### read csv file of HowTo100M dataset
    videos = []
    
    videos_info = process_howto100m()

    ### sample 100 videos with category_1 = "Food and Entertaining" and category_2 = "Recipes" and task_id == 11967
    for video_info in videos_info:
        if video_info["cat1"] == "Food and Entertaining" and video_info["cat2"] == "Recipes" and video_info["task_id"] == "11967":
            video = Video(video_info["video_id"], f"https://www.youtube.com/watch?v={video_info['video_id']}")
            try:
                video.process()
                videos.append(video)
            except Exception as e:
                print(f"Failed to process video '{video_info['video_id']}': {e}")
                continue
        if len(videos) >= n:
            break
    print(f"Number of videos: {len(videos)}")
    return videos

def draw_embeddings(embeddings, contents, labels, colors, figure_name="embedding"):
    # do sparse dimensionality reduction on embeddings using PCA
    ### preprocess embeddings (e.g., normalize, standardize, etc.)
    embeddings = np.array(embeddings)
    embeddings = standardize(embeddings)
    ### apply PCA to reduce dimensions
    reduced_embeddings = reduce_dim(embeddings, n_components=2)
    ### visuliaze embeddings

    data = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
    data['content'] = [content for content in contents]
    data['label'] = [label for label in labels]
    data['color'] = [color for color in colors]

    fig, ax = plt.subplots()
    data.plot.scatter(x='x', y='y', ax=ax, color=data['color'])
    for i, txt in enumerate(data['label']):
        ax.annotate(txt, (data['x'].iloc[i], data['y'].iloc[i]))


    ### save the plot
    plt.savefig(RESULTS / f"{figure_name}.png")

    formatted_data = []
    for idx, row in data.iterrows():
        formatted_data.append({
            "id": row['label'],
            "x": row['x'],
            "y": row['y'],
            "color": format_rgba(row['color']),
            "data": {
                "longtext": row["content"],
            },
        })

    with open(RESULTS / f"{figure_name}.json", 'w') as f:
        json.dump(formatted_data, f, indent=2)

def json_dump_video_links(videos):
    json_dump = []
    for idx, video in enumerate(videos):
        json_dump.append(video.video_link)
    
    print(json.dumps(json_dump, indent=2))
    return

def generate_embeddings_simplest():
    videos = process_videos(100)

    ### output info about the videos

    for idx, video in enumerate(videos):
        print(f"Video ID={idx}: {video.video_id}")
        ### print title, description, tags, etc.
        if not video.processed:
            continue
        print(f"\tTitle: {video.metadata['title']}")
        print(f"Description: {video.metadata['tags']}; {video.metadata['categories']}")
        transcript = video.extract_transcript()
        print("Transcript:")
        for seg in transcript:
            print(f"{seg['start']} - {seg['finish']}: {seg['text']}")
        print("----------------------------------------------------------------")
        print()

    embeddings = []
    contents = []
    labels = []
    colors = []

    for video in videos:
        transcript = video.extract_transcript()
        transcript_str = "\n".join([f"{seg['text']} " for seg in transcript])
        embedding = bert_embed_text(transcript_str)
        embeddings.append(embedding)
        contents.append(transcript_str)
        labels.append(video.video_id)
        colors.append(np.random.rand(3,))

    draw_embeddings(embeddings, contents, labels, colors, "embedding")

def generate_embeddings_simplest_per_step():
    filepath = "_data/all_variability_assignments.json"
    videos = {}
    with open(filepath, 'r') as f:
        videos = json.load(f)

    embeddings = []
    contents = []
    labels = []
    colors = []
    for video_id, video in videos.items():
        ### random video_color
        video_color = np.random.rand(3,)
        for step_id, step in video.items():
            embedding = bert_embed_text(step["content"])
            embeddings.append(embedding)
            contents.append(step["content"])
            labels.append(f"{video_id}-{step_id}")
            colors.append(video_color)

    draw_embeddings(embeddings, contents, labels, colors, "embedding_per_step")

def main():

    
    #generate_embeddings_simplest(videos)
    generate_embeddings_simplest_per_step()

if __name__ == "__main__":
    main()
