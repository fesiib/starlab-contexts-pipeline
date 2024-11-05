from pydantic_models.comparison import AlignmentsPerAspectSchema, transform_alignments

from helpers import get_response_pydantic, get_response_pydantic_with_message, extend_contents

INCLUDE_IMAGES = False

def get_subgoal_alignments_v4(vid1, vid2, contents1, contents2, subgoal, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant specializing in analyzing and comparing procedural content across different how-to videos about the task `{task}`. Given contents from two videos for subgoal `{subgoal}`, compare the information from each video and identify new contents presented in each video. Identify contents one-by-one focusing on one specific point/aspect/relation at a time, avoid combining multiple procedural details together. If needed, refer to each video by their ID (e.g., `{vid1}`).".format(task=task, subgoal=subgoal, vid1=vid1)
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## `{vid1}`:"
            }] + extend_contents(contents1, include_images=INCLUDE_IMAGES),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## `{vid2}`:"
            }] + extend_contents(contents2, include_images=INCLUDE_IMAGES),
        },
    ]

    response = get_response_pydantic(messages, AlignmentsPerAspectSchema)
    
    alignments_1, alignments_2 = transform_alignments(response, vid1, vid2)
    return alignments_1, alignments_2

# def get_steps_alignments_v4(vid1, vid2, steps1, steps2, task):
#     messages = [
#         {
#             "role": "system",
#             "content": [{
#                 "type": "text",
#                 "text": "You are a helpful assistant specializing in analyzing and comparing procedural content across different how-to videos about the task `{task}`. Given the sequence of steps performed in each video, compare the steps and their order from each video and identify new `additional` and `alternative` steps presented in each video. If needed, refer to each video by their ID (e.g., `{vid1}`).".format(task=task, vid1=vid1)
#             }],
#         },
#         {
#             "role": "user",
#             "content": [{
#                 "type": "text",
#                 "text": f"## `{vid1}``:`\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps1)])
#             }]
#         },
#         {
#             "role": "user",
#             "content": [{
#                 "type": "text",
#                 "text": f"## `{vid2}`:`\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps2)])
#             }]
#         },
#     ]

#     response = get_response_pydantic(messages, AlignmentsPerAspectSchema)
#     alignments_1, alignments_2 = transform_alignments(response, vid1, vid2)
#     return alignments_1, alignments_2

def get_transcript_alignments_v3(vid1, vid2, contents1, contents2, task, tries):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant specializing in analyzing and comparing procedural content about task `{task}`. Given contents from the two videos, analyze and compare the information from each video and provide a comprehensive list of new supplementary and contradictory contents presented in each video. For each piece of content, focus on one specific point at a time, avoid combining multiple details. If needed, refer to each video by their ID (e.g., `{vid1}`).".format(task=task, vid1=vid1)
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## `{vid1}`:\n"
            }] + extend_contents(contents1, include_images=INCLUDE_IMAGES),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## `{vid2}`:\n"
            }] + extend_contents(contents2, include_images=INCLUDE_IMAGES),
        },
    ]

    alignments_1 = []
    alignments_2 = []

    while tries > 0:
        tries -= 1
        response, message = get_response_pydantic_with_message(messages, AlignmentsPerAspectSchema)

        alignments_1, alignments_2 = transform_alignments(response, vid1, vid2)

        if len(alignments_1) + len(alignments_2) > 0:
            break
    
        messages.append({
            "role": "assistant",
            "content": message
        })
        messages.append({
            "role": "user",
            "content": [{
                "type": "text",
                "text": "Provide additional supplementary and contradictory contents presented in the current video only. Be specific, and focus on one point at a time."
            }]
        })

    return alignments_1, alignments_2
