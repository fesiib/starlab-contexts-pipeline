from pydantic import BaseModel, Field

from typing import Literal


## V4
class AlignmentsBase(BaseModel): 
    source_tutorial_id: str = Field(..., title="the id of the tutorial video that contains the new content.")
    
    title: str = Field(..., title="A 1 to 5 words title that concisely describes the new content.")
    description: str = Field(..., title="a brief, specific, and clear description of the new content.")
    reasoning: str = Field(..., title="a brief explanation of why this specific content is included in the source tutorial and not others.")
    comparison: str = Field(..., title="a brief explanation of why the new content is different from or not included in the other tutorial. Refer to the tutorials with their ID.")
    importance: int = Field(..., title="the score of the new content in terms of its importance for successful completion of the task/subgoal. Give a score from 1 to 5, where 1 is the least important (e.g., mentioning other tasks or unrelated information) and 5 is the most important (e.g., providing main tools or materials).")

class AlignmentsPerRelationSchema(BaseModel):
    additional: list[AlignmentsBase] = Field(..., title="the list of new contents that are considered supplementary to the other video.")
    alternative: list[AlignmentsBase] = Field(..., title="the list of new contents that are considered contradictory or different compared to the other video.")

class AlignmentsPerAspectSchema(BaseModel):
    materials: AlignmentsPerRelationSchema = Field(..., title="the new contents related to materials and ingreidents used in the tutorial to complete the subgoal.")
    tools: AlignmentsPerRelationSchema = Field(..., title="the new contents related to tools or equipments used in the tutorial to complete the subgoal.")
    outcomes: AlignmentsPerRelationSchema = Field(..., title="the new contents related to outcomes or results of completing the subgoal.")

    instructions: AlignmentsPerRelationSchema = Field(..., title="the new contents related to instructions presented in the tutorial for completing the subgoal.")
    explanations: AlignmentsPerRelationSchema = Field(..., title="the new contents related to justifications and reasons presented in the tutorial for performing the steps/instructions for completing the subgoal.")
    tips: AlignmentsPerRelationSchema = Field(..., title="the new contents related to tips presented in the tutorial that can help in completing the subgoal easier, faster, or more efficiently.")
    warnings: AlignmentsPerRelationSchema = Field(..., title="the new contents related to warnings presented in the tutorial that can help avoid mistakes when completing the subgoal.")
    other: AlignmentsPerRelationSchema = Field(..., title="the new contents related to other procedural aspects.")

# class MappingSchema(BaseModel):
#     tutorial_1_

# class GeneralAlignmentSchema(BaseModel):
#     new_contents: 


def transform_alignments(response, vid1, vid2):
    alignments_1 = []
    alignments_2 = []
    for aspect in response:
        for relation in response[aspect]:
            for alignment in response[aspect][relation]:
                cur_alignment = {
                    **alignment,
                    "aspect": aspect,
                    "relation": relation,
                }
                del cur_alignment["source_tutorial_id"]
                if alignment['source_tutorial_id'] == vid1:
                    alignments_1.append(cur_alignment)
                if alignment['source_tutorial_id'] == vid2:
                    alignments_2.append(cur_alignment)
    
    return alignments_1, alignments_2