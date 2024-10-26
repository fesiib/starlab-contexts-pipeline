from pydantic import BaseModel, Field

class SummarizedAlignmentSchema(BaseModel):
    title: str = Field(..., title="the 5-word title of the new content in the current video.")
    description: str = Field(..., title="a brief, specific, and clear description of the new procedural content. Focus on one specific point at a time, avoid combining multiple details.")
    reasoning: str = Field(..., title="briefly explain why this specific content is in the video. Why is this included in the video?")
    comparison: str = Field(..., title="if applicable, explain why this content is different from or not included in the other video.")

class NotableInformationSchema(BaseModel):
    title: str = Field(..., title="the concise title of the notable information that is present in the current video.")
    description: str = Field(..., title="the description of the notable information.")

class HookSchema(BaseModel):
    title: str = Field(..., title="the title of the hook in a conversational manner. It should be interesting and engaging, but short!")
    description: str = Field(..., title="the elaboration on the hook. It should look like continuation of the title.")

class SummarizedAlignmentSchema2(BaseModel):
    title: str = Field(..., title="A 1 to 5 words title that concisely describes the new content.")
    description: str = Field(..., title="a brief, specific, and clear description of the new content.")
    reasoning: str = Field(..., title="a brief explanation of why this specific content is included in the video.")
    comparison: str = Field(..., title="if applicable, a brief explanation of why the new content is different from or not included in the other video(s). Do not mention the other video(s) by name.")

class GroupSchema(BaseModel):
    title: str = Field(..., title="A 1 to 5 words title that concisely describes all the contents in this group.")
    description: str = Field(..., title="a brief, specific, and clear description of the contents in this group.")
    comparison: str = Field(..., title="a brief summary of the comparisons of the contents in this group and current video.")

class ContentGroupMappingSchema(BaseModel):
    group: str = Field(..., title="the title of the group.")
    content_index: int = Field(..., title="the index of the content in original list of contents.")

class GroupsSchema(BaseModel):
    groups: list[GroupSchema] = Field(..., title="a list of groups of interesting procedural contents.")
    assignments: list[ContentGroupMappingSchema] = Field(..., title="a list of assignments mapping each content to a group.")
