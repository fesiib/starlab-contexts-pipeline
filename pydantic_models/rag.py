from pydantic import BaseModel, Field, ConfigDict

class InformationSchema(BaseModel):
    model_config = ConfigDict(extra='forbid')
    source_doc_idx: int = Field(..., title="The index of the source document in the library.")
    content: str = Field(..., title="The content of the information.")
    raw_context: str = Field(..., title="A short verbatim tutorial excerpt (maximum 10-20 words) that contains the information and contextualizes it (i.e., some context before and after the information). Denote the start and end of the excerpt with `[...]`.")

class InformationListSchema(BaseModel):
    model_config = ConfigDict(extra='forbid')
    information_list: list[InformationSchema] = Field(..., title="The list of information pieces retrieved from the library in the order of relevance.")