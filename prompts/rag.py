from helpers import get_response_pydantic
from pydantic_models.rag import InformationListSchema

SYSTEM_PROMPT_RAG = """You are a helpful assistant that can retrieve information from a library of tutorials for a procedural task. You only consider the provided information (no external knowledge) when answering the query."""

### e.g., query: "Given a tutorial, retrieve all missing, but relevant explanations for the tutorial."
USER_PROMPT_FULL_TUTORIAL = """
Here is available tutorials for the task `{task}` (formatted as [idx]: [title] [content]):
<library>
{library}
</library>

You are given a tutorial. Please answer the query `{query}`.
<tutorial>
{tutorial}
</tutorial>
"""

def documents_to_context(documents):
    text = ""
    for idx, document in enumerate(documents):
        text += f"[{idx+1}] {document['title']}\n```\n{document['content']}\n```\n"
    return text

def get_rag_response_full_tutorial(task, documents, tutorial, query):
    
    library_str = documents_to_context(documents)
    tutorial_str = documents_to_context([tutorial])

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT_RAG,
        },
        {
            "role": "user",
            "content": USER_PROMPT_FULL_TUTORIAL.format(task=task, library=library_str, tutorial=tutorial_str, query=query),
        },
    ]
    response = get_response_pydantic(messages, InformationListSchema)
    return response["information_list"]

### e.g., query: "Given a tutorial with the highlighted segment, retrieve top-{N} missing, but relevant explanations for the segment"
USER_PROMPT_TUTORIAL_SEGMENT = """
Here is available tutorials for the task `{task}` (formatted as [idx]: [title] [content]):
<library>
{library}
</library>

You are given a tutorial and its highlighted segment. Please answer the query `{query}`.

<tutorial>
{tutorial}
</tutorial>

<highlighted_segment>
{segment}
</highlighted_segment>
"""

def get_rag_response_tutorial_segment(task, documents, tutorial, segment, query):
    
    library_str = documents_to_context(documents)
    tutorial_str = documents_to_context([tutorial])

    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT_RAG,
        },
        {
            "role": "user",
            "content": USER_PROMPT_TUTORIAL_SEGMENT.format(task=task, library=library_str, tutorial=tutorial_str, segment=segment, query=query),
        },
    ]
    response = get_response_pydantic(messages, InformationListSchema)
    return response["information_list"]