from pydantic_models.rag import InformationListSchema
from prompts import tutorials_to_str

SYSTEM_PROMPT_RAG = """You are a helpful assistant that can retrieve information from a library of tutorials for a procedural task. You only consider the provided information (no external knowledge) when answering the query."""

### e.g., query: "Given a tutorial, retrieve all missing, but relevant explanations for the tutorial."
USER_PROMPT_FULL_TUTORIAL = """
Given available tutorials for the task `{task}` and a tutorial that the user is currently in, please respond to the query.

Available tutorials:
{tutorials}

Current tutorial:
{cur_tutorial}

Query:
{query}
"""

def get_rag_response_full_tutorial_request(task, tutorials, tutorial, query, gen_model):
    
    tutorials_str = tutorials_to_str(tutorials)
    cur_tutorial_str = tutorial["content"]

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT_RAG,
        },
        {
            "role": "user",
            "content": USER_PROMPT_FULL_TUTORIAL.format(task=task, tutorials=tutorials_str, cur_tutorial=cur_tutorial_str, query=query),
        },
    ]
    return {
        "messages": messages,
        "model": gen_model,
        "response_format": InformationListSchema,
    }

### e.g., query: "Given a tutorial with the highlighted segment, retrieve top-{N} missing, but relevant explanations for the segment"
USER_PROMPT_TUTORIAL_SEGMENT = """
Given available tutorials for the task `{task}`, a tutorial that the user is currently in, and the specific segment that the user is interested in, please respond to the query.

Available tutorials:
{tutorials}

Current tutorial:
{cur_tutorial}

Current segment:
{cur_segment}

Query:
{query}
"""

def get_rag_response_tutorial_segment_request(task, tutorials, tutorial, segment, query, gen_model):
    
    tutorials_str = tutorials_to_str(tutorials)
    cur_tutorial_str = tutorial["content"]
    cur_segment_str = segment["content"]

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT_RAG,
        },
        {
            "role": "user",
            "content": USER_PROMPT_TUTORIAL_SEGMENT.format(task=task, tutorials=tutorials_str, cur_tutorial=cur_tutorial_str, cur_segment=cur_segment_str, query=query),
        },
    ]
    
    return {
        "messages": messages,
        "model": gen_model,
        "response_format": InformationListSchema,
    }

def get_rag_response_request(task, tutorials, tutorial, segment, query, gen_model):
    if segment is None:
        return get_rag_response_full_tutorial_request(task, tutorials, tutorial, query, gen_model)
    else:
        return get_rag_response_tutorial_segment_request(task, tutorials, tutorial, segment, query, gen_model)

def get_rag_response_response(response, **kwargs):
    return response["information_list"]