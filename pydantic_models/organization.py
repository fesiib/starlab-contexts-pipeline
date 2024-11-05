from pydantic import BaseModel, Field

class SummarizedAlignmentSchema2(BaseModel):
    title: str = Field(..., title="A 1 to 5 words title that concisely describes all the new contents.")
    description: str = Field(..., title="a brief, specific, and clear summary of descriptions of the new contents.")
    reasoning: str = Field(..., title="a brief summary of explanations of why the contents are included in the video and not others.")
    comparison: str = Field(..., title="a brief summary of explanations of why the new contents are different from or not included in the other tutorials. Refer to the tutorials with their ID.")

class GroupSchema(BaseModel):
    title: str = Field(..., title="A 1 to 5 words title that concisely describes/summarizes all the contents in this group.")
    description: str = Field(..., title="a brief, specific, and clear summary of why the user should be interested in the contents in this group.")
    comparison: str = Field(..., title="a brief summary of the comparisons of the contents in this group with current tutorial. Refer to the tutorials with their ID.")

class ContentGroupMappingSchema(BaseModel):
    group: str = Field(..., title="the title of the group.")
    content_index: int = Field(..., title="the index of the content in original list of contents.")

class GroupsSchema(BaseModel):
    groups: list[GroupSchema] = Field(..., title="a list of groups of interesting procedural contents.")
    assignments: list[ContentGroupMappingSchema] = Field(..., title="a list of assignments mapping each content to a group.")