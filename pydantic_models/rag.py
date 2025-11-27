from pydantic import BaseModel, Field, ConfigDict

class InformationSchema(BaseModel):
    model_config = ConfigDict(extra='forbid')
    content: str = Field(..., title="The content of the information piece.")
    type: str = Field(..., title="The type of the information piece.")
    source_piece_id: str = Field(..., title="The id of the source information piece.")
    # source_doc_idx: int = Field(..., title="The index of the source tutorial in the library.")
    # source_piece_idx: int = Field(..., title="The index of the information piece in the source tutorial.")
    
    # raw_context: str = Field(..., title="A short verbatim tutorial excerpt (maximum 10-20 words) that contains the information and contextualizes it (i.e., some context before and after the information). Denote the start and end of the excerpt with `[...]`.")

class InformationListSchema(BaseModel):
    model_config = ConfigDict(extra='forbid')
    information_list: list[InformationSchema] = Field(..., title="The list of information pieces retrieved from the library in the order of relevance.")