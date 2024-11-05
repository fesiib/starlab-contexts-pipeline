from pydantic import BaseModel, Field

class StepSummarySchema(BaseModel):
    # "materials", "outcome", "setting", "instructions", "explanation", "tips", "tools"

    # "materials", "outcome", "setting", "instructions", "explanation", "tips", "tools"

    ### context
    # subgoal: str = Field(..., title="The detailed description of the subgoal/stage based on the narration.")
    # subgoal_content_ids: list[int] = Field(..., title="A list of narration ids that mention the subgoal.")
    materials: list[str] = Field(..., title="A comprehensive list of materials or ingredients used in the tutorial for the step(s) along with their visual descriptions.")
    materials_content_ids: list[int] = Field(..., title="A list of narration ids that mention the materials.")
    outcomes: list[str] = Field(..., title="A comprehensive list of the outcomes or results of the step(s) along with their visual descriptions.")
    outcomes_content_ids: list[int] = Field(..., title="A list of narration ids that mention the outcome.")
    tools: list[str] = Field(..., title="A comprehensive list of tools or equipments used in the tutorial for the step(s) along with their visual descriptions.")
    tools_content_ids: list[int] = Field(..., title="A list of narration ids that mention the tools.")
    # setting: str = Field(..., title="The context in which the step is performed, including the location, time, and other relevant details.")
    # setting_content_ids: list[int] = Field(..., title="A list of narration ids that mention the setting.")
    
    ### method
    instructions: str = Field(..., title="The instructions presented in the tutorial for the step(s).")
    instructions_content_ids: list[int] = Field(..., title="A list of narration ids that mention the instructions.")
    
    explanations: str = Field(..., title="The justifications and reasons presented in the tutorial for performing the step.")
    explanations_content_ids: list[int] = Field(..., title="A list of narration ids that mention the explanations.")
    
    tips: str = Field(..., title="The tips presented in the tutorial that can help in performing the step(s) easier, faster, or more efficiently.")
    tips_content_ids: list[int] = Field(..., title="A list of narration ids that mention the tips.")
    warnings: str = Field(..., title="The warnings presented in the tutorial that can help avoid mistakes when performing the step(s).")
    warnings_content_ids: list[int] = Field(..., title="A list of narration ids that mention the warnings.")

class SubgoalSummarySchema(BaseModel):
    # "materials", "outcome", "setting", "instructions", "explanation", "tips", "tools"

    ### context
    # subgoal: str = Field(..., title="The detailed description of the subgoal/stage based on the narration.")
    # subgoal_content_ids: list[int] = Field(..., title="A list of narration ids that mention the subgoal.")
    materials: list[str] = Field(..., title="A comprehensive list of materials or ingredients used in the tutorial to complete the subgoal/stage along with their visual descriptions.")
    materials_content_ids: list[int] = Field(..., title="A list of narration ids that mention the materials.")
    outcomes: list[str] = Field(..., title="A comprehensive list of the outcomes or results of completing the subgoal/stage along with their visual descriptions.")
    outcomes_content_ids: list[int] = Field(..., title="A list of narration ids that mention the outcome.")
    tools: list[str] = Field(..., title="A comprehensive list of tools or equipments used in the tutorial to complete the the subgoal/stage along with their visual descriptions.")
    tools_content_ids: list[int] = Field(..., title="A list of narration ids that mention the tools.")
    # setting: str = Field(..., title="The context in which the step is performed, including the location, time, and other relevant details.")
    # setting_content_ids: list[int] = Field(..., title="A list of narration ids that mention the setting.")
    
    ### method
    instructions: str = Field(..., title="The instructions presented in the tutorial for completing the subgoal/stage.")
    instructions_content_ids: list[int] = Field(..., title="A list of narration ids that mention the instructions.")
    
    explanations: str = Field(..., title="The justifications and reasons presented in the tutorial for performing the steps/instructions for completing the subgoal/stage.")
    explanations_content_ids: list[int] = Field(..., title="A list of narration ids that mention the explanations.")
    
    tips: str = Field(..., title="The tips presented in the tutorial that can help in completing the subgoal/stage easier, faster, or more efficiently.")
    tips_content_ids: list[int] = Field(..., title="A list of narration ids that mention the tips.")
    warnings: str = Field(..., title="The warnings presented in the tutorial that can help avoid mistakes when completing the subgoal/stage.")
    warnings_content_ids: list[int] = Field(..., title="A list of narration ids that mention the warnings.")
    

