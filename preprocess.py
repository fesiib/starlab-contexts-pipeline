import os
import base64
import random
import json

from PIL import Image
from openai import OpenAI

from backend import process_video

from pydantic_models.subgoal import TaskGraph, VideoSegmentation
from pydantic_models.info_schemas import MetaInformationSchema, SubgoalInformationSchema, InformationRelations

from pydantic_models.info_schemas import AlignmentsSchema, AlignmentClassificationSchema, AlignmentHooksSchema

PATH = "step_data/"
TASK_ID  = "custom"
META_TITLE = "$meta$"

TASK_DESCRIPTIONS = {
    "custom2": "How to make Pasta Carbonara?",
    "custom3": "How to make Pasta Carbonara?",
    "custom": "How to bake a muffin?",
    "11967": "How to make a paper airplane?",
}

LIBRARY = {
    "custom3": [
        "https://www.youtube.com/watch?v=75p4UHRIMcU",
        "https://www.youtube.com/watch?v=dzyXBU3dIys",
        "https://www.youtube.com/watch?v=D_2DBLAt57c",
        "https://www.youtube.com/watch?v=3AAdKl1UYZs",
        "https://www.youtube.com/watch?v=qoHnwOHLiMk",
        "https://www.youtube.com/watch?v=NqFi90p38N8",
    ],
    "custom2": [
        "https://www.youtube.com/watch?v=75p4UHRIMcU",
        "https://www.youtube.com/watch?v=dzyXBU3dIys",
        "https://www.youtube.com/watch?v=D_2DBLAt57c",
        # "https://www.youtube.com/watch?v=3AAdKl1UYZs",
        # "https://www.youtube.com/watch?v=qoHnwOHLiMk",
        # "https://www.youtube.com/watch?v=NqFi90p38N8",
    ],
    "11967": [
        "https://www.youtube.com/watch?v=yJQShkjNn08",
        "https://www.youtube.com/watch?v=yweUoYP1v_o",
        "https://www.youtube.com/watch?v=Ehntsffsx08",
        "https://www.youtube.com/watch?v=tdk9_Xs_CC0",
        "https://www.youtube.com/watch?v=dkhy4vn9HcY",
        "https://www.youtube.com/watch?v=QECo58lV-bE",
        "https://www.youtube.com/watch?v=SMh2sjuEwxM",
        "https://www.youtube.com/watch?v=DaEzhwLFPi8",
        "https://www.youtube.com/watch?v=J_5scvrv0LU",
        "https://www.youtube.com/watch?v=umbBEHlpTfo",
        "https://www.youtube.com/watch?v=pq_INi_4IBI",
        "https://www.youtube.com/watch?v=pYOQutHfCDo",
    ],
    "custom": [
        # "https://www.youtube.com/shorts/B-XGIGS4Ipw", # short
        # "https://www.youtube.com/shorts/fWp5z_YM07Q", # short
        "https://www.youtube.com/watch?v=aEFvNsBDCWs", # has verbal
        "https://www.youtube.com/watch?v=gN-orgrgvU8", # has verbal
        "https://www.youtube.com/watch?v=cZ2KJPGVwNU", # has verbal
    ]
}

def encode_image(image_path):
    with open(image_path, "rb") as image:
        return base64.b64encode(image.read()).decode("utf-8")

def json_remove_lists(json_obj):
    ### turn all lists into dictionaries with keys as indices `index_0`
    if isinstance(json_obj, list):
        json_obj = {f"index_{i}": json_remove_lists(item) for i, item in enumerate(json_obj)}
    elif isinstance(json_obj, dict):
        for key, value in json_obj.items():
            json_obj[key] = json_remove_lists(value)
    return json_obj

def json_to_markdown(json_obj, indent=0):
    json_obj = json_remove_lists(json_obj)
    
    markdown = ""
    indent_str = "  " * indent
    
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            markdown += f"{indent_str}- **{key}**:\n"
            markdown += json_to_markdown(value, indent + 1)
    else:
        markdown += f"{indent_str}- {json_obj}\n"
    
    return markdown

API_KEY = os.getenv('OPENAI_API_KEY')   
print(API_KEY)
client = OpenAI(
    api_key=API_KEY,
)

SEED = 13774
TEMPERATURE = 0.6
MODEL_NAME = 'gpt-4o-2024-08-06'

def get_response(messages, response_format="json_object", retries=1):
    
    generated_text = ""
    finish_reason = ""
    usages = []
    while True:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            seed=SEED,
            temperature=TEMPERATURE,
            response_format={
                "type": response_format,
            },

        )
        generated_text += response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        usages.append(response.usage)

        if finish_reason != "length":
            break
        messages.append({"role": "assistant", "content": response.choices[0].message.content})

    # print(f"Finish Reason: {finish_reason}")
    # print(f"Usages: {usages}")
    # print(f"Generated Text: {generated_text}")

    if response_format == "json_object":
        try:
            obj = json.loads(generated_text)
            keys = list(obj.keys())
            if len(keys) == 1:
                return obj[keys[0]]
            else:
                return obj
        except json.JSONDecodeError:
            if retries > 0:
                return get_response(messages, response_format, retries - 1)

    return generated_text


def get_response_pydantic(messages, response_format):
    print(json.dumps(messages, indent=2))

    completion = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=messages,
        seed=SEED,
        temperature=TEMPERATURE,
        response_format=response_format,
    )

    response = completion.choices[0].message
    if (response.refusal):
        print("REFUSED: ", response.refusal)
        return None
    
    json_response = response.parsed.get_dict()

    print(json.dumps(json_response, indent=2))
    return json_response

def save_data(task_id, videos=None, steps=None, information_relations=None):
    if videos is None:
        videos = []
    if steps is None:
        steps = []
    if information_relations is None:
        information_relations = []
    ### save all the video objects
    save_dict = []

    for video in videos:
        save_dict.append(video.to_dict())
    
    if os.path.exists(f"{PATH}{task_id}") is False:
        os.mkdir(f"{PATH}{task_id}")

    with open(f"{PATH}{task_id}/video_data.json", "w") as file:
        json.dump(save_dict, file, indent=2)

    ### save all the step objects
    with open(f"{PATH}{task_id}/step_data.json", "w") as file:
        json.dump(steps, file, indent=2)

    ### save all the information relations
    with open(f"{PATH}{task_id}/information_relations.json", "w") as file:
        json.dump(information_relations, file, indent=2)

def str_to_float(str_time):
    return sum(x * float(t) for x, t in zip([3600, 60, 1], str_time.split(":")))

def float_to_str(float_time):
    return str(int(float_time // 3600)) + ":" + str(int((float_time % 3600) // 60)) + ":" + str(int(float_time % 60))

def generate_custom_steps(subtitles):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and defining comprehensive subgoals generalizable across different how-to videos about the same task."},
        {"role": "user", "content": "Summarizes the subtitles of a video into important steps in the procedural task (steps should be based on meaningful intermediate stages of the process). You must use the subtitles to generate the steps. Return a JSON list with the following structure: [{'start': float, 'finish': float, 'title': string, 'text': string}]"},
        {"role": "user", "content": json_to_markdown(subtitles)}
    ]
    return get_response(messages)

def define_common_steps(videos):
    if (len(videos) == 0):
        return []
    
    if (len(videos) == 1):
        messages = [
            {"role": "system", "content": "You are a helpful assistant specializing in analyzing and refining subgoals that are generalizable across different how-to videos for the same task. Your goal is to ensure that the subgoals are broad enough to encompass diverse content yet specific enough to capture all critical procedural steps."},
            {"role": "user", "content": "Based on the the how-to video, identify and define subgoals that are generalizable across different videos covering the same task. Ensure that the subgoals encompass all the procedural information in the video."},
            {"role": "user", "content": f"Narration:\n```{json.dumps(videos[0])}```"}
        ]
    else: 
        ## intial subgoals
        messages = [
            {"role": "system", "content": "You are a helpful assistant specializing in analyzing and refining subgoals that are generalizable across different how-to videos for the same task. Your goal is to ensure that the subgoals are broad enough to encompass diverse content yet specific enough to capture all critical procedural steps."},
            {"role": "user", "content": "Based on two how-to videos about the same task, identify and define subgoals that are generalizable for both videos. Ensure that the subgoals encompass all the procedural information in the video."},
            {"role": "user", "content": "\n".join((f"Narration {idx + 1}:\n```{json.dumps(video)}```") for idx, video in enumerate(videos[:2]))}
        ]
    
    subgoals = get_response_pydantic(messages, TaskGraph)

    ## refine subgoals
    if len(videos) < 3:
        return subgoals
    
    for video in videos[2:]:
        messages = [
            {"role": "system", "content": "You are a helpful assistant specializing in analyzing and refining subgoals that are generalizable across different how-to videos for the same task. Your goal is to ensure that the subgoals are broad enough to encompass diverse content yet specific enough to capture all critical procedural steps."},
            {"role": "user", "content": "Given the the how-to video and the initial definitions of subgoals, refine the set of subgoals (i.e., add/remove/change subgoal definitions). Ensure that new subgoals are `equivalent` to previous set, but at the same time, comprehensively cover all procedural content in the current video. Make sure the subgoals are at the right level of abstraction: specific enough to classify diverse content effectively, but not so broad that they lose their utility."},
            {"role": "user", "content": f"Narration:\n```{json.dumps(video)}```"},
            {"role": "user", "content": f"Subgoals:\n```{json.dumps(subgoals)}```"}
        ]
        subgoals = get_response_pydantic(messages, TaskGraph)
    
    return subgoals

def generate_common_steps(subtitles, subgoals):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and segmenting content in how-to videos according to given subgoals."},
        {"role": "user", "content": "Given the subtitles of the how-to video and subgoals, segment the video according to subgoals. If some subtitles do not belong to any subgoal, define a custom subgoals & label them with `(custom)` tag. Make sure to preserve the order of the subtitles in the video."},
        {"role": "user", "content": f"Subtitles:\n```{json.dumps(subtitles)}```"},
        {"role": "user", "content": f"Subgoals:\n```{json.dumps(subgoals)}```"}
    ]

    common_steps = get_response_pydantic(messages, VideoSegmentation)
    return common_steps

def get_meta_information_relations(summary1, summary2):
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant specializing in analyzing and comparing content across different how-to videos."},
    #     {"role": "user", "content": f"Given the procedural information from two how-to videos, your task is to analyze and compare their content. Break down the similarities and differences into the following categories:\n\n- **Consistent**: Identify and concisely summarize the content that is identical or highly similar in both Source 1 and Source 2.\n- **Complementary**: Identify and concisely summarize the content that is present in Source 2 but missing from Source 1, effectively adding to or expanding upon Source 1's information.\n- **Contradictory**: Identify and concisely summarize any content that is directly conflicting or different between Source 1 and Source 2.\n\nFor each category, clearly reference the relevant parts of both sources.After categorizing the content, provide a reasoning and implications for each of the identified relations.\n\n- **References Information**: Where possible, refer to other relevant parts of the same sources that may explain the differences or similarities identified.\n- **Cross-Source Connections**: If relevant, refer to other videos or content that may help clarify the relationship between the sources or resolve any contradictions. Avoid speculation; instead, base your reasoning on explicit connections within or across the sources."},
    #     {"role": "user", "content": f"Source 1:\n```{json.dumps(summary1)}```"},
    #     {"role": "user", "content": f"Source 2:\n```{json.dumps(summary2)}```"},
    # ]
    # response = get_response_pydantic(messages, InformationRelations)
    messages = [
        {
            "role": "system",
            "content": "You are a detailed-oriented assistant specializing in analyzing and comparing procedural content across different how-to videos."
        },
        {
            "role": "user",
            "content": "Given the procedural information from two videos about the same task, identify all the new content in the new video compared to the previous video. Folow the schema provided below to structure your response."
        },
        {"role": "user", "content": f"New Video:\n```{json.dumps(summary1)}```"},
        {"role": "user", "content": f"Previous Video:\n```{json.dumps(summary2)}```"},
    ]
    response = get_response_pydantic(messages, AlignmentsSchema)
    return response

def get_subgoal_information_relations(title, context1, context2, summary1, summary2):
    # messages = [
    #     {"role": "system", "content": "You are a helpful assistant specializing in analyzing and comparing content across different how-to videos."},
    #     {"role": "user", "content": f"Given the information from two how-to videos regarding '{title}', your task is to analyze and compare their content. You are also given the overall contexts of the videos. Break down the similarities and differences into the following categories:\n\n- **Consistent**: Identify and concisely summarize the content that is identical or highly similar in both Source 1 and Source 2.\n- **Complementary**: Identify and concisely summarize the content that is present in Source 2 but missing from Source 1, effectively adding to or expanding upon Source 1's information.\n- **Contradictory**: Identify and concisely summarize any content that is directly conflicting or different between Source 1 and Source 2.\n\nFor each category, clearly reference the relevant parts of the sources. After categorizing the content, provide a reasoning and implications for each of the identified relations.\n\n- **References Information**: Where possible, refer to other relevant parts of the same sources or contexts that may explain the differences or similarities identified.\n- **Cross-Source Connections**: If relevant, refer to other videos or content that may help clarify the relationship between the sources or resolve any contradictions. Avoid speculation; instead, base your reasoning on explicit connections within or across the sources."},
    #     {"role": "user", "content": f"Context 1:\n```{json.dumps(context1)}```"},
    #     {"role": "user", "content": f"Source 1:\n```{json.dumps(summary1)}```"},
    #     {"role": "user", "content": f"Context 2:\n```{json.dumps(context2)}```"},
    #     {"role": "user", "content": f"Source 2:\n```{json.dumps(summary2)}```"},
    # ]
    # response = get_response_pydantic(messages, InformationRelations)

    messages = [
        {
            "role": "system",
            "content": "You are a detailed-oriented assistant specializing in analyzing and comparing procedural content across different how-to videos."
        },
        {
            "role": "user",
            "content": "Given the procedural information from two videos about the same task, identify all the new content in the new video compared to the previous video. Folow the schema provided below to structure your response."
        },
        {"role": "user", "content": f"New Video:\n```{json.dumps(summary1)}```"},
        {"role": "user", "content": f"Previous Video:\n```{json.dumps(summary2)}```"},
    ]
    response = get_response_pydantic(messages, AlignmentsSchema)
    return response

def get_meta_summary(title, source):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and summarizing procedural information from how-to videos. Your summaries are concise, focused, and avoid unnecessary details, ensuring that the essential information is captured without redundancy."},
        {"role": "user", "content": f"Given the narration of the how-to video, extract and summarize the information according to the provided schema."},
        {"role": "user", "content": f"Narration:\n```{source}```"},
    ]

    response = get_response_pydantic(messages, MetaInformationSchema)

    return {
        "title": title,
        **response,
    }

def get_subgoal_summary(title, source, context, process):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and summarizing procedural information from how-to videos. Your summaries are concise, focused, and ensure that essential information is captured without redundancy."},
        {"role": "user", "content": f"Given the narration of a specific step from a how-to video and the context (i.e., results of previous subgoals as well as the summary of the overall procedure), extract and summarize the unique procedural information according to the provided schema."},
        {"role": "user","content": f"Context:\n```{json.dumps(context)}\n{json.dumps(process)}```"},
        {"role": "user","content": f"Narration:\n```{json.dumps(source)}```"},
    ]

    response = get_response_pydantic(messages, SubgoalInformationSchema)

    return {
        "title": title,
        **response,
    }

def get_information_relation_classification(relation, title, new_meta, prev_meta, new_subgoal, prev_subgoal):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and classifying the information in terms of comparison and alignment between different how-to videos."},
    ]
    
    new_video = None
    prev_video = None

    if title == META_TITLE:
        new_video = new_meta
        prev_video = prev_meta
    else:
        new_video = {
            "overall_context": new_meta,
            f"{title}": new_subgoal,
        }
        prev_video = {
            "overall_context": prev_meta,
            f"{title}": prev_subgoal,
        }
    messages.extend([
        {"role": "user", "content": "Given the new information from a new video compared to the previous video as well as the original context from both videos, analyze and classify the new information into following categories:\n- **Additional Information**: The new information is considered as `additional information` if the methods and the setting are fundamentally the same (i.e., same instructions, rationale, or tips, subgoal, context, materials, outcome) , but the new video provides additional information that is not present in the previous video.\n- **Alternative Method**: The new information is considered as `alternative method` if the methods are different (i.e., different instructions, rationale, or tips), but the setting are fundamentally the same (i.e., same subgoal, context, materials, outcome).\n- **Alternative Setting**: The new information is considered as `alternative setting` if the settings are different (i.e., different subgoal, context, materials, outcome), but the methods are fundamentally the same.\n- **Alternative Example**: The new information is considered as `alternative example` if the new video provides a different settings and methods that are not present in the previous video (i.e., different instructions, rationale, or tips, subgoal, context, materials, outcome).\n."},
        {"role": "user", "content": f"Information:\n```{json.dumps(relation)}```"},
        {"role": "user", "content": f"New Video:\n```{json.dumps(new_video)}```"},
        {"role": "user", "content": f"Previous Video:\n```{json.dumps(prev_video)}```"},
    ])

    response = get_response_pydantic(messages, AlignmentClassificationSchema)
    return response

def get_hooks(classification, alignments):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and identifying connections between different pieces of information and generating metacognitive prompts to help people learn about procedural knowledge."},
        {"role": "user", "content": f"Given the information sourced from different how-to videos, cluster them based on their helpfulness to the learners as {classification} and organize them under metacognitive prompts specific to {classification}."},
        {"role": "user", "content": f"```{json.dumps(alignments)}```"},
    ]

    response = get_response_pydantic(messages, AlignmentHooksSchema)
    return response

class Video:
    video_link = ""
    video_id = None
    ### list of frames in base64 {"idx": 0, "image": "", caption: ""}
    frames = []
    ### {"start": 0, "finish": 0, "text": ""}
    subtitles = []
    ### {"start": 0, "finish": 0, "title": "", "text": ""}
    custom_steps = []
    ### {"start": 0, "finish": 0, "title": "", "text": ""}
    common_steps = []

    meta_summary = None
    subgoal_summaries = []


    def __init__(self, video_link):
        self.video_link = video_link
        self.video_id = video_link.split("/")[-1]
        self.subtitles = []
        self.frames = []
        self.custom_steps = []
        self.common_steps = []
        self.meta_summary = None
        self.subgoal_summaries = []

    def process(self):
        self.process_video()
        self.process_subtitles()

    def process_video(self):
        video_title, video_frame_paths, subtitles = process_video(self.video_link)
        self.video_id = video_title
        self.frames = []
        
        for idx, frame_path in enumerate(video_frame_paths):
            self.frames.append({
                "idx": idx,
                "path": frame_path,
            })
        
        self.subtitles = []    
        for subtitle in subtitles:
            self.subtitles.append({
                "start": str_to_float(subtitle["start"]),
                "finish": str_to_float(subtitle["finish"]),
                "text": subtitle["text"]
            })

    def process_subtitles(self):
        ## TODO: reevaluate if custom steps are needed
        # self.custom_steps = generate_custom_steps(self.subtitles)
        self.custom_steps = []
        for subtitle in self.subtitles:
            self.custom_steps.append({
                "start": subtitle["start"],
                "finish": subtitle["finish"],
                # "title": "",
                "text": subtitle["text"]
            })
        ### sort the steps by start time
        self.custom_steps = sorted(self.custom_steps, key=lambda x: x["start"])

    def get_overlapping_steps(self, segments):
        # segments = [(start, finish)] 
        if len(self.common_steps) == 0:
            return []
        steps = []
        for step in self.common_steps:
            ## check if it overlaps with any of the segments
            for start, finish in segments:
                if max(start, step["start"]) < min(finish, step["finish"]):
                    steps.append(step)
                    break
        return steps
    
    def get_common_step(self, timestamp):
        if len(self.common_steps) == 0:
            return None
        for step in self.common_steps:
            if step["start"] <= timestamp <= step["finish"]:
                return step
        for step in self.common_steps:
            if timestamp <= step["finish"]:
                return step
        return self.common_steps[-1]
    
    def get_full_narration(self):
        return "\n".join([subtitle["text"] for subtitle in self.subtitles])
    
    def find_subgoal_summary(self, title):
        for summary in self.subgoal_summaries:
            if summary["title"] == title:
                return summary
        return None

    def to_dict(self):
        return {
            "video_id": self.video_id,
            "video_link": self.video_link,
            "frames": self.frames,
            "subtitles": self.subtitles,
            "custom_steps": self.custom_steps,
            "common_steps": self.common_steps,
            "meta_summary": self.meta_summary,
            "subgoal_summaries": self.subgoal_summaries,
        }

    ### TODO: Save frames to disk & retrieve based on filename
    def from_dict(self, 
        video_link=None, video_id=None, subtitles=None,
        frames=None, custom_steps=None, common_steps=None,
        meta_summary=None, subgoal_summaries=None
    ):
        if video_link is not None:
            self.video_link = video_link
        if video_id is not None:
            self.video_id = video_id
        if subtitles is not None:
            self.subtitles = subtitles
        if frames is not None:
            self.frames = frames
        if custom_steps is not None:
            self.custom_steps = custom_steps
        if common_steps is not None:
            self.common_steps = common_steps
        if meta_summary is not None:
            self.meta_summary = meta_summary
        if subgoal_summaries is not None:
            self.subgoal_summaries = subgoal_summaries
        
def pre_process_videos(video_links):
    videos = []
    for video_link in video_links:
        video = Video(video_link)
        try:
            video.process()
            videos.append(video)
        except Exception as e:
            print(f"Error processing video: {video_link}")
            print(e)
            continue
    return videos

### UNDERSTANDING GAPS
class UnderstandingGap:
    videos = []
    steps = []

    def __init__(self, videos, steps=[]):
        self.videos = videos
        self.steps = steps

    def process_videos(self):
        if len(self.steps) == 0:
            ### Define common steps across all videos
            narrations = []
            for video in self.videos:
                narration = "\n".join([subtitle["text"] for subtitle in video.subtitles])
                narrations.append(narration)
            self.steps = define_common_steps(narrations)

        ### Split into common_steps within each video
        for video in self.videos:
            if len(video.common_steps) == 0:
                video.common_steps = generate_common_steps(video.custom_steps, self.steps)

    @staticmethod
    def generate_understanding_gaps(previous, next, previous_links, prompt):
        if len(next) == 0:
            return []
        messages = [
            {"role": "system", "content": "You are a helpful expert who knows what kind of understanding gaps learners have about specific tasks. Make sure to be as brief & to the point as possible, avoid verbose sentences. You respond in JSON format."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": "Follow this JSON format: [{'gap_title': string, 'gap_description': string, 'keywords': list, 'explanation': string, 'video_id': string, 'start': float, 'finish', float}]"},
            {"role": "user", "content": f"Here are the narrations that the learner have already listened & understood:\n{json.dumps(previous)}"},
            {"role": "user", "content": f"Here are the narrations that the learner have not seen yet:\n{json.dumps(next)}"},
            {"role": "user", "content": f"Here are the previous understanding gaps that were identified:\n{json.dumps(previous_links)}"}
        ]

        gaps = get_response(messages)
        ### check if gaps satisfy the format and if not, return an empty list
        if not isinstance(gaps, list):
            return []
        for gap in gaps:
            if not all(key in gap for key in ["gap_title", "gap_description", "keywords", "explanation", "video_id", "start", "finish"]):
                return []
        return gaps

    def transform_watch_history(self, watch_history):
        if len(watch_history) == 0:
            return [], None

        ## combine watch_history of each video under the same video_id
        last_watch = watch_history[-1]
        watch_history_per_video = {}
        for watch in watch_history[:-1]:
            video_id = watch["video_id"]
            if video_id not in watch_history_per_video:
                watch_history_per_video[video_id] = []
            watch_history_per_video[video_id].append(watch)

        previous_with_titles = []
        last_step = None
        for video in self.videos:
            if video.video_id == last_watch["video_id"]:
                last_step_raw = video.get_common_step(last_watch["finish"])
                last_step = {
                    "title": last_step_raw["title"],
                    "video_id": video.video_id,
                    "start": last_step_raw["start"],
                    "finish": last_step_raw["finish"],
                    "text": last_step_raw["text"]
                }

            if video.video_id not in watch_history_per_video:
                continue
            
            steps = video.get_overlapping_steps([(watch["start"], watch["finish"]) for watch in watch_history_per_video[video.video_id]])
            
            if len(steps) > 0:
                for step in steps:
                    previous_with_titles.append({
                        "video_id": video.video_id,
                        "start": step["start"],
                        "finish": step["finish"],
                        "title": step["title"],
                        "text": step["text"]
                    })

        return previous_with_titles, last_step

    def get_global_links(self, previous_with_titles, previous_links=[]):
        previous = []
        covered_video_ids = set()
        for watch in previous_with_titles:
            previous.append({
                "video_id": watch["video_id"],
                "start": watch["start"],
                "finish": watch["finish"],
                "text": watch["text"]
            })
            covered_video_ids.add(watch["video_id"])        

        next = []
        for video in self.videos:
            if video.video_id in covered_video_ids:
                continue
            for step in video.common_steps:
                next.append({
                    "video_id": video.video_id,
                    "start": step["start"],
                    "finish": step["finish"],
                    "text": step["text"],
                })

        prompt = f"""Given (1) the user's watch history (2) the unseen videos, and (3) previous understanding gaps, identify new or old understanding gaps the user can have about the procedure (e.g., order, presence/absence of specific steps, etc) and how they can resolve it by watching the unseen videos. Provide (1) a short title, (2) a short description, (3) few representative keywords, (4) a short explanation how the gap is resolved in the video, and (5) start&finish of the video. Also, make sure to include the understanding gaps that have been resolved yet"""
        
        gaps = self.generate_understanding_gaps(previous, next, previous_links, prompt)
        return gaps

    def get_local_links(self, previous_with_titles, last_step, previous_links=[]):
        previous = []
        covered_video_ids = set()
        for watch in previous_with_titles:
            if watch["title"] != last_step["title"]:
                continue
            previous.append({
                "video_id": watch["video_id"],
                "start": watch["start"],
                "finish": watch["finish"],
                "text": watch["text"]
            })
            covered_video_ids.add(watch["video_id"])

        next = []
        for video in self.videos:
            if video.video_id in covered_video_ids:
                continue
            ## find the step that matches the current step
            for step in video.common_steps:
                if step["title"] == last_step["title"]:
                    next.append({
                        "video_id": video.video_id,
                        "start": step["start"],
                        "finish": step["finish"],
                        "text": step["text"]
                    })
                    break

        prompt = f"""Given (1) the user's watch history (2) the unseen videos, and (3) previous understanding gaps, identify new or old understanding gaps the user can have about the step ({last_step["title"]}) and how they can resolve it by watching the unseen videos. Provide (1) a short title, (2) a short description, (3) few representative keywords, (4) a short explanation how the gap is resolved in the video, and (5) start&finish of the video."""

        gaps = self.generate_understanding_gaps(previous, next, previous_links, prompt)
        return gaps


    def to_dict(self):
        return {
            "videos": [video.to_dict() for video in self.videos],
            "steps": self.steps
        }

def fake_watch_history(videos):
    watch_history = []
    rand_video_idxs = [
        random.randint(0, len(videos) - 1) for i in range(3)
    ]

    for idx in rand_video_idxs:
        video = videos[idx]
        steps = video.common_steps
        rand_watched_idxs = [
            random.randint(0, len(steps) - 1) for i in range(2)
        ]
        for step_idx in rand_watched_idxs:
            step = steps[step_idx]
            watch_history.append({
                "video_id": video.video_id,
                "start": step["start"],
                "finish": step["finish"],
                "title": step["title"],
                "text": step["text"]
            })
    current_step = watch_history.pop()
    return watch_history, current_step

def generate_links(ug, watch_history, link_types = [], previous_links = {}):
    # with open("first_request.json", "r") as f:
    #     data = json.load(f)
    #     old_links = data["links"]
    #     ### do random sampling from old_links & return
    #     links = {}
    #     for link_type in link_types:
    #         links[link_type] = random.sample(old_links[link_type], min(2, len(old_links[link_type])))
    #     return links
    
    if len(watch_history) <= 1:
        ## read `first_request.json` and return the links
        with open("first_request.json", "r") as f:
            data = json.load(f)
            old_links = data["links"]
            old_watch_history = data["watch_history"]
            if (len(watch_history) == 1):
                watch = watch_history[0]
                old_watch = old_watch_history[0]
                same = True
                for key in old_watch.keys():
                    if key not in watch or watch[key] != old_watch[key]:
                        same = False
                        break
                if same is True:
                    return old_links

    previous_with_titles, last_step = ug.transform_watch_history(watch_history)
    links = {}
    if "global" in link_types:
        previous_global = previous_links.get("global", [])
        links["global"] = ug.get_global_links(previous_with_titles, previous_global)
        for link in links["global"]:
            link["label"] = "-Procedure-"
    
    if "local" in link_types and last_step is not None:
        previous_local = previous_links.get("local", [])
        previous_other = [link for link in previous_local if link["label"] != last_step["title"]]
        previous_local = [link for link in previous_local if link["label"] == last_step["title"]]
        links["local"] = ug.get_local_links(previous_with_titles, last_step, previous_local)
        for link in links["local"]:
            link["label"] = f"{last_step['title']}"
        links["local"] = previous_other + links["local"]

    ### save last request
    open("last_request.json", "w").write(json.dumps({
        "watch_history": watch_history,
        "link_types": link_types,
        "previous_with_titles": previous_with_titles,
        "last_step": last_step,
        "previous_links": previous_links,
        "links": links,
    }, indent=2))

    ### Do this on the frontend
    # if last_step is not None:
    #     ## generate local links
    #     ## links that current_video resolves
    #     links_self = []
    #     for link_type in links.keys():
    #         links_self += [link for link in links[link_type] if link["video_id"] == last_step["video_id"]]
            
    #         links[link_type] = [link for link in links[link_type] if link["video_id"] != last_step["video_id"]]
        
    #         links[link_type] = sorted(links[link_type], key=lambda x: x["start"])
        
    #     links_self = sorted(links_self, key=lambda x: x["start"])
    #     links["self"] = links_self
    return links

def setup_ug(task_id):
    if task_id not in LIBRARY:
        return None

    # get the video data
    videos = []
    steps = []
    video_data_path = f"{PATH}{task_id}/video_data.json"
    step_data_path = f"{PATH}{task_id}/step_data.json"

    if os.path.exists(video_data_path):
        with open(video_data_path, "r") as file:
            video_data = json.load(file)
            videos = []
            for data in video_data:
                if data["video_link"] not in LIBRARY[task_id]:
                    continue
                video = Video(data["video_link"])
                video.from_dict(**data)
                videos.append(video)
    if len(videos) == 0 or len(videos[0].subtitles) == 0:
        videos = pre_process_videos(LIBRARY[task_id])
        save_data(task_id, videos=videos)

    if os.path.exists(step_data_path):
        with open(step_data_path, "r") as file:
            steps = json.load(file)
        ug = UnderstandingGap(videos, steps)
    else:
        ug = UnderstandingGap(videos)
        ug.process_videos()
        save_data(task_id, videos=ug.videos, steps=ug.steps)
    return ug

### DYNAMIC SUMMARIES

class DynamicSummary:
    task = ""
    videos = []
    steps = []

    information_relations = []

    def __init__(self, task, videos, steps=[]):
        self.task = task
        self.videos = videos
        self.steps = steps

    def process_videos(self):
        if len(self.steps) == 0:
            ### Define common steps across all videos
            narrations = []
            for video in self.videos:
                narration = "\n".join([subtitle["text"] for subtitle in video.subtitles])
                narrations.append(narration)
            common_steps = define_common_steps(narrations)
            for step in common_steps:
                self.steps.append({
                    "title": step["title"],
                    "definition": step["definition"],
                    "dependencies": step["dependencies"] if "dependencies" in step else [],
                    "explanation": step["explanation"] if "explanation" in step else "",
                })

        ### Split into common_steps within each video
        for video in self.videos:
            if len(video.common_steps) == 0:
                cur_common_steps = generate_common_steps(video.custom_steps, self.steps)
                for step in cur_common_steps:
                    video.common_steps.append({
                        "title": step["title"],
                        "start": step["start"],
                        "finish": step["finish"],
                        "text": step["text"],
                        "explanation": step["explanation"] if "explanation" in step else "",
                    })
            
            ## Summarize (for each video) (1) problem, (2) method, (3) outcome
            if video.meta_summary is None:
                video.meta_summary = get_meta_summary(META_TITLE, video.get_full_narration())

            ## Summarize (for each subgoal): (1) context, (2) tools, (3) instructions, (4) explanations, (5) supplementary info, (6) outcome
            if len(video.subgoal_summaries) == 0:
                for main_step_def in self.steps:
                    relevant_narrations = []
                    parent_schemas = []
                    for parent_summary in video.subgoal_summaries:
                        if parent_summary["title"] in main_step_def["dependencies"]:
                            parent_schemas.append(parent_summary)
                    
                    for step in video.common_steps:
                        if step["title"] == main_step_def["title"]:
                            relevant_narrations.append(step)
                    
                    video.subgoal_summaries.append(get_subgoal_summary(main_step_def["title"], relevant_narrations, parent_schemas, video.meta_summary))
        
        if len(self.videos) >= 2 and len(self.information_relations) == 0:
            self.generate_information_relations()

        # ## TODO: Classify the information relations
        # self.classify_information_relations()

    def generate_information_relations(self):
        self.information_relations = []

        for v1_idx, video1 in enumerate(self.videos):
            for v2_idx, video2 in enumerate(self.videos):
                if v1_idx == v2_idx:
                    continue
                ### between meta
                meta_relations = get_meta_information_relations(video1.meta_summary, video2.meta_summary)
                self.information_relations.append({
                    **meta_relations,
                    "title": META_TITLE,
                    "new_video": video1.video_id,
                    "prev_video": video2.video_id,
                })
                ### between subgoals
                for step in self.steps:
                    summary1 = video1.find_subgoal_summary(step["title"])
                    if summary1 is None:
                        continue    
                    summary2 = video2.find_subgoal_summary(step["title"])
                    if summary2 is None:
                        continue
                    subgoal_relations = get_subgoal_information_relations(step["title"], video1.meta_summary, video2.meta_summary, summary1, summary2)
                    self.information_relations.append({
                        **subgoal_relations,
                        "title": step["title"],
                        "new_video": video1.video_id,
                        "prev_video": video2.video_id,
                    })
    
    def classify_information_relations(self):
        for video_pair in self.information_relations:
            title = video_pair["title"]
            relations = video_pair["alignments"]
            new_video = video_pair["new_video"]
            prev_video = video_pair["prev_video"]
            for video in self.videos:
                if new_video == video.video_id:
                    new_video = video
                if prev_video == video.video_id:
                    prev_video = video
            
            new_meta = None
            prev_meta = None
            new_subgoal = None
            prev_subgoal = None

            if title == META_TITLE:
                new_meta = new_video.meta_summary
                prev_meta = prev_video.meta_summary
            else:
                new_meta = new_video.meta_summary
                prev_meta = prev_video.meta_summary
                new_subgoal = new_video.find_subgoal_summary(title)
                prev_subgoal = prev_video.find_subgoal_summary(title)
            for index, relation in enumerate(relations):
                if "classification" in relation:
                    continue
                classification = get_information_relation_classification(relation, title, new_meta, prev_meta, new_subgoal, prev_subgoal)
                relations[index] = {
                    **relation,
                    **classification
                }

    def generate_hooks(self, original_alignments):
        ### for the set of alignments, group them under catchy hooks!
        # assign unique ids to each alignment
        alignments_per_class_and_title = {}
        for index, alignment in enumerate(original_alignments):
            alignment["id"] = f"a-{index}"
            classification = alignment["classification"]
            title = alignment["title"]
            if classification not in alignments_per_class_and_title:
                alignments_per_class_and_title[classification] = {}
            if title not in alignments_per_class_and_title[classification]:
                alignments_per_class_and_title[classification][title] = []
            alignments_per_class_and_title[classification][title].append(alignment)
        hooks_per_class_and_title = {}
        for classification, alignments_per_title in alignments_per_class_and_title.items():
            hooks_per_class_and_title[classification] = {}
            for title, alignments in alignments_per_title.items():
                hooks_per_class_and_title[classification][title] = self.__generate_hooks(classification, title, alignments)
        return hooks_per_class_and_title
    
    def __generate_hooks(self, classification, title, alignments):
        hooks_dict = get_hooks(classification, alignments)
        hooks = []
        covered_alignment_ids = []
        for hook in hooks_dict["hooks"]:
            alignment_ids = hook["alignment_ids"]
            covered_alignment_ids += alignment_ids
            hook["alignments"] = []
            for alignment_id in alignment_ids:
                for alignment in alignments:
                    if alignment["id"] == alignment_id:
                        hook["alignments"].append(alignment)
                        break
            del hook["alignment_ids"]
            hooks.append(hook)

        uncovered_alignments = []
        for alignment in alignments:
            if alignment["id"] not in covered_alignment_ids:
                uncovered_alignments.append(alignment)
        
        if len(uncovered_alignments) > 0:
            hooks.extend(self.__generate_hooks(classification, title, uncovered_alignments))
        return hooks

    @staticmethod
    def generate_dynamic_summary(new_content, prompt, previous_summary=None):
        messages = [
            {"role": "system", "content": "You are a helpful expert who can summarize the procedural content embedded in the video about the task: {self.task}. Return a response in a JSON format."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": f"Here is the new content that the learner have learned:\n{json.dumps(new_content)}."},
        ]
        if previous_summary is not None:
            messages.append({"role": "user", "content": f"Here is the previous summary:\n{json.dumps(previous_summary)}"})

        summary = get_response(messages)
        return summary


    def progressive_approach(self):
        prompt_first = "Given the new content, provide a concise summary of the procedural knowledge that user learned!"

        prompt_rest = "Given the new content, combine the new content into the previous summary and return it!"
        
        #prompt_first = "Given the new content that the learner have learned, provide a structured summary, where each piece of procedural information is presented as: 'instruction' - (what is being instructed); 'justification' - (why is it being instructed); 'verbal_presentation' - (how is it presnted in the video); and 'video_id' - (reference to video);"
        #prompt_rest = "Given the new content that the learner have learned, aggregate the new content into the previous summary. Return a structured summary, where each piece of procedural information is presented as: 'instruction' - (what is being instructed); 'justification' - (why is it being instructed); 'verbal_presentation' - (how is it presnted in the video); and 'video_id' - (reference to video);"

        summaries = []

        for video in self.videos:
            if len(summaries) == 0:
                summary = self.generate_dynamic_summary({
                    "video_id": video.video_id,
                    "narration": video.get_full_narration(),
                }, prompt_first)
            else:
                summary = self.generate_dynamic_summary({
                    "video_id": video.video_id,
                    "narration": video.get_full_narration(),  
                }, prompt_rest, summaries[-1])
            print(summary)
            summaries.append(summary)

        return summaries

def setup_ds(task_id, coarse=False):
    if task_id not in LIBRARY:
        return None

    # get the video data
    videos = []
    steps = []
    information_relations = []
    video_data_path = f"{PATH}{task_id}/video_data.json"
    step_data_path = f"{PATH}{task_id}/step_data.json"
    information_relations_path = f"{PATH}{task_id}/information_relations.json"

    if os.path.exists(video_data_path):
        with open(video_data_path, "r") as file:
            video_data = json.load(file)
            videos = []
            for data in video_data:
                if data["video_link"] not in LIBRARY[task_id]:
                    continue
                video = Video(data["video_link"])
                video.from_dict(**data)
                videos.append(video)
    if len(videos) == 0 or len(videos[0].subtitles) == 0:
        videos = pre_process_videos(LIBRARY[task_id])

    task_desc = TASK_DESCRIPTIONS[task_id]

    if os.path.exists(step_data_path):
        with open(step_data_path, "r") as file:
            steps = json.load(file)
        ds = DynamicSummary(task_desc, videos, steps)
    else:
        ds = DynamicSummary(task_desc, videos)

    if os.path.exists(information_relations_path):
        with open(information_relations_path, "r") as file:
            information_relations = json.load(file)
            ds.information_relations = information_relations
    else:
        ds.information_relations = []

    ds.process_videos()



    ds.classify_information_relations()

    save_data(task_id, videos=ds.videos, steps=ds.steps, information_relations=ds.information_relations)
    return ds

def print_relations_table(ds, video_id):
    table_prevs = []
    table_unseen = []
    for relation in ds.information_relations:
        if relation["new_video"] == video_id:
            table_prevs.append({
                "title": relation["title"],
                "alignments": relation["alignments"],
                "video_id": relation["prev_video"]
            })
        elif relation["prev_video"] == video_id:
            table_unseen.append({
                "title": relation["title"],
                "alignments": relation["alignments"],
                "video_id": relation["new_video"]
            })

    ### print the tables in markdown format where rows are videos and columns are: title, alignments
    for idx, table in enumerate([table_prevs, table_unseen]):
        if idx == 0:
            print("### Table current video vs previous videos")
        else:
            print("### Table current video vs unseen videos")
        print("Video | Title | Alignments |")
        print("| --- | --- | --- |")
        for row in table:
            alignments = "<br>".join([f"<br>**{d_id} -->**<br>" + f"<br>".join([f"-**{key}**: {value}" for key, value in d.items()]) for d_id, d in enumerate(row["alignments"])])
            print(f"| {row['video_id']} | {row['title']} | {alignments} |")
        print("")

def print_video_summaries(video):
    ### print meta_summary and subgoal_summaries
    print(f"### Video {video.video_id}")
    print("#### Meta Summary")
    for key in video.meta_summary:
        if key == "title":
            continue
        print(f"- **{key}**: {video.meta_summary[key]}")
    print("\n\n")
    for subgoal_summary in video.subgoal_summaries:
        print(f"#### Subgoal Summary: {subgoal_summary['title']}")
        for key in subgoal_summary:
            if key == "title":
                continue
            print(f"- **{key}**: {subgoal_summary[key]}")
        print("\n\n")

def print_hooks(hooks_per_class_and_title):
    for classification, hooks_per_title in hooks_per_class_and_title.items():
        print(f"# Classification: {classification}")
        for title, hooks in hooks_per_title.items():
            print(f"## Title: {title}")
            for hook in hooks:
                print(f"### Hook: {hook['title']}")
                print(f"#### Helpfulness: {hook['helpfulness']}")
                print(f"#### Relevant Alignments")
                for alignment in hook["alignments"]:
                    print(f"Alignment: {alignment['id']}: {alignment['description']}\n\n")

                    # for key in alignment:
                    #     if key == "id" or key == "description":
                    #         continue
                    #     print(f"- **{key}**: {alignment[key]}")
                    # print("\n\n")
                print("\n\n")
            print("\n\n")
        print("\n\n")

def main():
    # ds = setup_ds(TASK_ID)

    # summaries = ds.progressive_approach()
    # with open("summaries.json", "w") as file:
    #     json.dump(summaries, file, indent=2)

    # with open("summaries.md", "w") as file:
    #     file.write(json_to_markdown(summaries))

    ds = setup_ds("custom2")

    # for video in ds.videos:
    #     print_video_summaries(video)

    video_id = ds.videos[0].video_id
    # print_relations_table(ds, video_id)

    ### generate hooks for video_id wrt all information relations
    alignments = []
    
    for video_pair in ds.information_relations:
        if video_pair["prev_video"] == video_id:
            for alignment in video_pair["alignments"]:
                alignments.append({
                    **alignment,
                    "title": video_pair["title"],
                })
        
    hooks_per_class_and_title = ds.generate_hooks(alignments)

    print_hooks(hooks_per_class_and_title)

    # narrations = []
    # for video in ds.videos:
    #     narrations.append({
    #         "video_id": video.video_id,
    #         "text": video.get_full_narration(),
    #     })

    # open("narrations.json", "w").write(json.dumps(narrations, indent=2))

if __name__ == "__main__":
    main()