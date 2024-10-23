from helpers import get_response_pydantic, extend_contents, extend_subgoals
from helpers import encode_image, get_response_pydantic_with_message

from pydantic_models.segmentation import TaskGraph, get_segmentation_schema, StepsSchema, AggStepsSchema, SubgoalSchema, AllProceduralInformationSchema, TranscriptAssignmentsSchema, get_segmentation_schema_v4


def define_common_subgoals_v2(contents, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and extracting subgoals for task `{task}`. You are given a transcript of a how-to video and asked to define a task graph that consists of subgoals of the demonstrated procedure and dependencies between the subgoals. Ensure that the subgoals are (1) based on meaningful intermediate stages of the procedure, (2) broad enough to encompass diverse ways to complete the task, and (3) specific enough to capture all critical procedural steps.".format(task=task)},
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Contents:\n"
            }] + extend_contents(contents),
        },
    ]
    
    response = get_response_pydantic(messages, TaskGraph)
    subgoals = response["subgoals"]
    return subgoals

def align_common_subgoals_v2(common_subgoals_per_video, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and comparing procedural content across different how-to videos about task `{task}`. Given the task graphs (i.e., subgoals and dependencies between them) from multiple how-to videos about the same task, combine them into a single task graph. Where necessary, merge/split subgoals, and generate a unified, comprehensive set of subgoals that is applicable to all the videos. Ensure that the subgoals are based on meaningful intermediate stages of the procedure.".format(task=task)},
    ]
    for video_idx, (_, common_subgoals) in enumerate(common_subgoals_per_video.items()):
        messages.append({"role": "user", "content": f"## Video {video_idx + 1}:\n{extend_subgoals(common_subgoals)}"})
    
    response = get_response_pydantic(messages, TaskGraph)
    subgoals = response["subgoals"]
    return subgoals

def generate_common_subgoals_v3(contents, subgoals, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing and segmenting how-to videos according to task graph for task `{task}`. Given the contents of the how-to video and subgoals, segment the video according to subgoals.".format(task=task)},
    ]

    messages.append({
        "role": "user",
        "content": f"## Subgoals:\n{extend_subgoals(subgoals)}"
    })

    messages.append({
        "role": "user",
        "content": [{
            "type": "text",
            "text": f"Contents:\n"
        }] + extend_contents(contents),
    })

    titles = [subgoal["title"] for subgoal in subgoals] + ["Custom"]
    SegmentationSchema = get_segmentation_schema(titles)

    response = get_response_pydantic(messages, SegmentationSchema)
    segmentation = response["segments"]

    for segment in segmentation:
        quotes = " ".join(segment["quotes"])
        for content in contents:
            if content["text"] in quotes or quotes in content["text"]:
                content["title"] = segment["title"]
                content["explanation"] = segment["explanation"]

    common_subgoals = []
    for content in contents:
        if "title" not in content:
            content["title"] = "Custom"
            content["explanation"] = "Custom subgoal"
        if (len(common_subgoals) > 0):
            if common_subgoals[-1]["title"] == content["title"] or content["title"] == "Custom":
                common_subgoals[-1]["finish"] = content["finish"]
                common_subgoals[-1]["text"] += " " + content["text"]
                common_subgoals[-1]["frame_paths"] = common_subgoals[-1]["frame_paths"] + content["frame_paths"]
                common_subgoals[-1]["content_ids"].append(content["id"])
                continue

        common_subgoals.append({
            "title": content["title"],
            "explanation": content["explanation"],
            "start": content["start"],
            "finish": content["finish"],
            "text": content["text"],
            "frame_paths": content["frame_paths"],
            "content_ids": [content["id"]]
        })
    return common_subgoals

def assign_transcripts_v4(contents, subgoals, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing tutorial video content. Given a narration of a tutorial video for a task `{task}` and a set of steps, analyze each sentence and find the steps it is talking about. You can specify multiple steps per sentence or leave it empty if it does not belong to any of the steps. Additionally, specify relevance of the sentence to the task at hand.".format(task=task)},
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Steps:\n" + "\n".join(subgoals)
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Contents:\n"
            }] + extend_contents(contents, include_ids=True),
        },
    ]

    
    total_assignments = []
    for i in range(0, len(contents), 20):
        message = {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Assign steps to the sentences between {i + 1} and {min(i + 20, len(contents))}:\n"
            }]
        }
        response = get_response_pydantic(messages + [message], TranscriptAssignmentsSchema)
        total_assignments += response["assignments"]

    segments = []
    for index, content in enumerate(contents):
        assignment = None
        for a in total_assignments:
            if a["index"] == index + 1:
                assignment = a
                break

        if assignment is None:
            print("ERROR: Assignment not found for index", index + 1)
            continue
        title = assignment["steps"]
        segments.append({
            "start": content["start"],
            "finish": content["finish"],
            "title": title,
            "text": content["text"],
            "frame_paths": content["frame_paths"],
            "content_ids": [content["id"]],
            "relevance": assignment["relevance"],
        })

    return segments

def segment_video_v4(contents, subgoals, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing tutorial video content. Given a narration of a tutorial video for a task `{task}` and a set of steps, segment the video according to the provided steps. For each segment, specify start and end transcript indices. Map the segment to `None` if it does not map to any of the segments.".format(task=task)},
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Contents:\n"
            }] + extend_contents(contents, include_ids=True),
        },
    ]

    SegmentationSchema = get_segmentation_schema_v4(subgoals + ["None"])

    response = get_response_pydantic(messages, SegmentationSchema)

    def get_new_segment(start, finish):
        if start > finish:
            print("ERROR: Start index greater than finish index")
            return None
        new_segment = {
            "start": 0,
            "finish": 0,
            "text": "",
            "frame_paths": [],
            "content_ids": [],
        }
        for i in range(start, finish + 1):
            if i < 0 or i > len(contents)-1:
                print("ERROR: Index out of bounds")
                continue
            content = contents[i]
            if i == start:
                new_segment["start"] = content["start"]
            else:
                new_segment["text"] += " "
            if i == finish:
                new_segment["finish"] = content["finish"]
            new_segment["text"] += content["text"]
            new_segment["frame_paths"] += content["frame_paths"]
            new_segment["content_ids"].append(content["id"])
        return new_segment

    segments = []
    last_index = -1
    for segment in response["segments"]:
        start = segment["start_index"] - 1
        finish = segment["end_index"] - 1
        if start < last_index:
            print("ERROR: Overlapping segments")
        if last_index < start - 1:
            print("ERROR: Missing segment")
            new_segment = get_new_segment(last_index + 1, start - 1)
            if new_segment is not None:
                segments.append({
                    "title": "None",
                    **new_segment,
                })
        new_segment = get_new_segment(start, finish)
        if new_segment is not None:
            segments.append({
                "title": segment["step"],
                **new_segment,
            })
        last_index = finish
    return segments

def define_common_subgoals_v4(contents, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing tutorial video content. Given a narration of a tutorial video for a task `{task}`, analyze it and generate a comprehensive list of steps presented in the video. Focus on the essence of the steps and avoid including unnecessary details. Ensure that the steps are clear, concise, and cover all the critical procedural information.".format(task=task)},
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Contents:\n"
            }] + extend_contents(contents),
        },
    ]
    
    response = get_response_pydantic(messages, StepsSchema)
    subgoals = response["steps"]
    
    # segments = assign_transcripts_v4(contents, subgoals, task)
    segments = segment_video_v4(contents, subgoals, task)
    return subgoals, segments

def align_common_subgoals_v4(sequence1, sequence2, task):
    sequence1_str = "\n".join(sequence1)
    sequence2_str = "\n".join(sequence2)
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing procedural content across different how-to videos about task `{task}`. Given two lists of steps from two tutorial videos about the task, aggregate them into a single list of steps. Combine similar steps or steps that have the same overall goal. Focus on the essence of the steps and avoid including unnecessary details. Make sure to include all the steps from both videos and specify which aggregated step they belong to.".format(task=task)},
        {"role": "user", "content": f"## Video 1:\n{sequence1_str}"},
        {"role": "user", "content": f"## Video 2:\n{sequence2_str}"}
    ]
    
    response = get_response_pydantic(messages, AggStepsSchema)
    if len(response["assignments_1"]) != len(sequence1) or len(response["assignments_2"]) != len(sequence2):
        print("ERROR: Length of assignments_1 does not match the length of sequence1")

    subgoals = []
    for agg_step in response["agg_steps"]:
        subgoals.append({
            "aggregated": agg_step,
            "original_list_1": [],
            "original_list_2": []
        })
        
    for assignments, original_list in [
        ("assignments_1", "original_list_1"),
        ("assignments_2", "original_list_2")
    ]:
        for a in response[assignments]:
            found = 0
            for subgoal in subgoals:
                if a["agg_step"] == subgoal["aggregated"]:
                    subgoal[original_list].append(a["original_step"])
                    found += 1
            if found == 0:
                print("ERROR: Original step from sequence not found in agg_steps")
            if found > 1:
                print("ERROR: Original step from sequence found in multiple agg_steps")
    return subgoals

def summarize_common_subgoals_v4(subgoals, task):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing tutorial content. You are given a set of steps in the task `{task}`. Extract the common goal each of the steps are accomplishing and provide a single COMPREHENSIVE subgoal.".format(task=task)},
        {
            "role": "user",
            "content": "## Steps:\n" + "\n".join(subgoals)
        }
    ]
    
    response = get_response_pydantic(messages, SubgoalSchema)
    return response

def extract_all_procedural_info_v5(contents, task, include_image=False):
    # extract_all_procedural_info_v5_explicit(contents, task, include_image)
    extract_all_procedural_info_v5_implicit(contents, task, include_image)

def extract_all_procedural_info_v5_implicit(contents, task, include_image=False):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing tutorial video content. You are given a narration of a video about the task `{task}` and asked to extract all the provided procedural information relevant to the task from each sentence.".format(task=task)},
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Contents:\n"
            }] + extend_contents(contents, include_image),
        },
    ]

    all_pieces = []

    for content in contents:
        cur_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Extract procedural information from the following contents:\n"
                },
                {
                    "type": "text",
                    "text": f"{content['text']}\n"
                }
            ]
        }
        if include_image:
            for frame_path in content["frame_paths"]:
                frame_base64 = encode_image(frame_path)
                cur_message["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}
                })
        messages.append(cur_message)
        response, message = get_response_pydantic_with_message(messages, AllProceduralInformationSchema)
        all_pieces.append({
            "id": content["id"],
            "pieces": response["all"],
        })
        messages.append({
            "role": "assistant",
            "content": message
        })
    return all_pieces

def extract_all_procedural_info_v5_explicit(contents, task, include_image=False):
    messages = [
        {"role": "system", "content": "You are a helpful assistant specializing in analyzing tutorial video content. You are given a narration of a video about the task `{task}` and asked to extract all the provided procedural information relevant to the task from each sentence.".format(task=task)},
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"Contents:\n"
            }] + extend_contents(contents),
        },
    ]

    all_pieces = []

    for content in contents:
        cur_message = {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Extract procedural information from the following contents:\n"
                },
                {
                    "type": "text",
                    "text": f"{content['text']}\n"
                }
            ]
        }
        if include_image:
            for frame_path in content["frame_paths"]:
                frame_base64 = encode_image(frame_path)
                cur_message["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{frame_base64}"}
                })
        messages.append(cur_message)
        response, message = get_response_pydantic_with_message(messages, AllProceduralInformationSchema)
        all_pieces.append({
            "id": content["id"],
            "pieces": response["all"],
        })
        messages.append({
            "role": "assistant",
            "content": message
        })
    return all_pieces