from pydantic_models.comparison import AlignmentsSchema4, AlignmentsSchema3, AlignmentsSchema2, AlignmentsSchema

from helpers import get_response_pydantic, get_response_pydantic_with_message, extend_contents


### TODO: May need to add images and try again!!
def get_subgoal_alignments_v2(contents1, contents2, subgoal, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant specializing in analyzing and comparing procedural content across different how-to videos about task `{task}`. Given contents from the current and previous videos for the subgoal `{subgoal}`, analyze and compare the information from each video and provide a comprehensive list of new supplementary and contradictory subgoal contents presented in the current video. For each piece of content, focus on one specific point at a time, avoid combining multiple details.".format(task=task, subgoal=subgoal)
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Previous Video:`\n"
            }] + extend_contents(contents2, False),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Current Video:`\n"
            }] + extend_contents(contents1, False),
        }
    ]

    response = get_response_pydantic(messages, AlignmentsSchema)
    return response["alignments"]

def get_meta_alignments_v2(contents1, contents2, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant specializing in analyzing and comparing procedural content across different how-to videos about task `{task}`. Given contents from the current and previous videos, analyze and compare the information from each video and provide a comprehensive list of new supplementary and contradictory contents presented in the current video. For each piece of content, focus on one specific point at a time, avoid combining multiple details.".format(task=task)
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Previous Video:`\n"
            }] + extend_contents(contents2, False),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Current Video:`\n"
            }] + extend_contents(contents1, False),
        }
    ]

    response = get_response_pydantic(messages, AlignmentsSchema)
    return response["alignments"]


### TODO: May need to add images and try again!!
def get_subgoal_alignments_v3(contents1, contents2, subgoal, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant specializing in analyzing and comparing procedural content across different how-to videos about task `{task}`. Given contents from the current and previous videos for the subgoal `{subgoal}`, analyze and compare the information from each video and provide a comprehensive list of new supplementary and contradictory subgoal contents presented in the current video. For each piece of content, focus on one specific point at a time, avoid combining multiple details.".format(task=task, subgoal=subgoal)
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Previous Video:`\n"
            }] + extend_contents(contents2, False),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Current Video:`\n"
            }] + extend_contents(contents1, False),
        }
    ]

    response = get_response_pydantic(messages, AlignmentsSchema2)
    alignments = []
    for alignment in response["supplementary_information"]:
        alignment["classification"] = "supplementary_" + alignment["classification"]
        alignments.append(alignment)
    for alignment in response["contradictory_information"]:
        alignment["classification"] = "contradictory_" + alignment["classification"]
        alignments.append(alignment)

    return alignments

def get_meta_alignments_v3(contents1, contents2, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant specializing in analyzing and comparing procedural content across different how-to videos about task `{task}`. Given contents from the current and previous videos, analyze and compare the information from each video and provide a comprehensive list of new supplementary and contradictory contents presented in the current video. For each piece of content, focus on one specific point at a time, avoid combining multiple details.".format(task=task)
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Previous Video:`\n"
            }] + extend_contents(contents2, False),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Current Video:`\n"
            }] + extend_contents(contents1, False),
        }
    ]

    response = get_response_pydantic(messages, AlignmentsSchema2)
    alignments = []
    for alignment in response["supplementary_information"]:
        alignment["classification"] = "supplementary_" + alignment["classification"]
        alignments.append(alignment)
    for alignment in response["contradictory_information"]:
        alignment["classification"] = "contradictory_" + alignment["classification"]
        alignments.append(alignment)

    return alignments

def get_transcript_alignments_v3(contents1, contents2, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant specializing in analyzing and comparing procedural content about task `{task}`. Given contents from the two videos, analyze and compare the information from each video and provide a comprehensive list of new supplementary and contradictory contents presented in the current video only. For each piece of content, focus on one specific point at a time, avoid combining multiple details.".format(task=task)
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Previous Video:`\n"
            }] + extend_contents(contents2, False),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Current Video:`\n"
            }] + extend_contents(contents1, False),
        },
    ]

    alignments = []

    tries = 5
    while tries > 0:
        tries -= 1
        response, message = get_response_pydantic_with_message(messages, AlignmentsSchema2)

        messages.append({
            "role": "assistant",
            "content": message
        })
        messages.append({
            "role": "user",
            "content": [{
                "type": "text",
                "text": "Provide additional supplementary and contradictory contents presented in the current video only. Do not repeat yourself, be specific, and focus on one point at a time."
            }]
        })
        
        for alignment in response["supplementary_information"]:
            alignment["classification"] = "supplementary_" + alignment["classification"]
            alignments.append(alignment)
            found_any = True
        for alignment in response["contradictory_information"]:
            alignment["classification"] = "contradictory_" + alignment["classification"]
            alignments.append(alignment)
            found_any = True
        if found_any is False or response["more_information_exist"] is False:
            break

    return alignments


def get_overall_alignments_v4(contents1, contents2, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant specializing in analyzing and comparing procedural content across different how-to videos about task `{task}`. Given contents from the current and previous videos, analyze and compare the information from each video and provide a comprehensive list of supplementary and contradictory contents present only in the current video with respect to previous video. For each piece of content, focus on one specific point (i.e., detail, aspect) at a time, avoid combining multiple details together.".format(task=task)
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Previous Video:`\n"
            }] + extend_contents(contents2, False),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Current Video:`\n"
            }] + extend_contents(contents1, False),
        }
    ]

    response = get_response_pydantic(messages, AlignmentsSchema3)
    alignments = []
    for alignment in response["supplementary_information"]:
        alignment["classification"] = "supplementary_" + alignment["classification"]
        alignments.append(alignment)
    for alignment in response["contradictory_information"]:
        alignment["classification"] = "contradictory_" + alignment["classification"]
        alignments.append(alignment)

    return alignments


### TODO: May need to add images and try again!!
def get_subgoal_alignments_v4(contents1, contents2, subgoal, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant specializing in analyzing and comparing procedural content across different how-to videos about the task `{task}`. Given contents from two videos for subgoal `{subgoal}`, compare the information from each video and identify new `additional` and `alternative` contents presented in each video. Identify contents one-by-one focusing on one specific point/aspect at a time, avoid combining multiple procedural details together.".format(task=task, subgoal=subgoal)
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Video 1:`\n"
            }] + extend_contents(contents1, include_images=True),
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Video 2:`\n"
            }] + extend_contents(contents2, include_images=True),
        },
    ]

    response = get_response_pydantic(messages, AlignmentsSchema4)
    new_contents_in_1 = response["new_contents_in_1"]
    new_contents_in_2 = response["new_contents_in_2"]
    return new_contents_in_1, new_contents_in_2

def get_steps_alignments_v4(steps1, steps2, task):
    messages = [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a helpful assistant specializing in analyzing and comparing procedural content across different how-to videos about the task `{task}`. Given the sequence of steps performed in each video, compare the steps and their order from each video and identify new `additional` and `alternative` steps presented in each video.".format(task=task)
            }],
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Video 1:`\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps1)])
            }]
        },
        {
            "role": "user",
            "content": [{
                "type": "text",
                "text": f"## Video 2:`\n" + "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps2)])
            }]
        },
    ]

    response = get_response_pydantic(messages, AlignmentsSchema4)
    new_contents_in_1 = response["new_contents_in_1"]
    new_contents_in_2 = response["new_contents_in_2"]
    return new_contents_in_1, new_contents_in_2



# def get_meta_alignments_v3(contents1, contents2, task):
#     messages = [
#         {
#             "role": "system",
#             "content": [{
#                 "type": "text",
#                 "text": "You are a helpful assistant specializing in analyzing and comparing procedural content across different how-to videos about task `{task}`. Given contents from the current and previous videos, analyze and compare the information from each video and provide a comprehensive list of new supplementary and contradictory contents presented in the current video. For each piece of content, focus on one specific point at a time, avoid combining multiple details.".format(task=task)
#             }],
#         },
#         {
#             "role": "user",
#             "content": [{
#                 "type": "text",
#                 "text": f"## Previous Video:`\n"
#             }] + extend_contents(contents2, False),
#         },
#         {
#             "role": "user",
#             "content": [{
#                 "type": "text",
#                 "text": f"## Current Video:`\n"
#             }] + extend_contents(contents1, False),
#         }
#     ]

#     response = get_response_pydantic(messages, AlignmentsSchema2)
#     alignments = []
#     for alignment in response["supplementary_information"]:
#         alignment["classification"] = "supplementary_" + alignment["classification"]
#         alignments.append(alignment)
#     for alignment in response["contradictory_information"]:
#         alignment["classification"] = "contradictory_" + alignment["classification"]
#         alignments.append(alignment)

#     return alignments