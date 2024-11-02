from pydantic import BaseModel, Field

class StepSummarySchema(BaseModel):
    # "materials", "outcome", "setting", "instructions", "explanation", "tips", "tools"

    ### context
    subgoal: str = Field(..., title="The subgoal that the step is contributing to.")
    subgoal_content_ids: list[int] = Field(..., title="A list of narration ids that mention the subgoal.")
    materials: list[str] = Field(..., title="A list of ingredients and materials required to complete the step along with their (visual) descriptions.")
    materials_content_ids: list[int] = Field(..., title="A list of narration ids that mention the materials.")
    outcome: list[str] = Field(..., title="A list of concise (visual) descriptions of what is achieved or created upon completing the step.")
    outcome_content_ids: list[int] = Field(..., title="A list of narration ids that mention the outcome.")
    # setting: str = Field(..., title="The context in which the step is performed, including the location, time, and other relevant details.")
    # setting_content_ids: list[int] = Field(..., title="A list of narration ids that mention the setting.")
    
    ### method
    instructions: str = Field(..., title="The instructions provided for completing the step.")
    instructions_content_ids: list[int] = Field(..., title="A list of narration ids that mention the instructions.")
    
    explanation: str = Field(..., title="The reasons why the instruction was performed or consequences of the instruction.")
    explanation_content_ids: list[int] = Field(..., title="A list of narration ids that mention the explanation.")
    
    tips: str = Field(..., title="The tips that can help in completing the step easier, faster, or more efficient. Warnings that can help avoid mistakes.")
    tips_content_ids: list[int] = Field(..., title="A list of narration ids that mention the tips/warnings.")

    tools: list[str] = Field(..., title="The tools or equipment required to complete the step along with their descriptions.")
    tools_content_ids: list[int] = Field(..., title="A list of narration ids that mention the tools.")

class SubgoalSummarySchema(BaseModel):
    # "materials", "outcome", "setting", "instructions", "explanation", "tips", "tools"

    ### context
    subgoal: str = Field(..., title="The detailed description of the subgoal/stage based on the narration.")
    subgoal_content_ids: list[int] = Field(..., title="A list of narration ids that mention the subgoal.")
    materials: list[str] = Field(..., title="A list of ingredients and materials required to complete the subgoal/stage along with their (visual) descriptions.")
    materials_content_ids: list[int] = Field(..., title="A list of narration ids that mention the materials.")
    outcome: list[str] = Field(..., title="A list of concise (visual) descriptions of what is achieved or created upon completing the subgoal/stage.")
    outcome_content_ids: list[int] = Field(..., title="A list of narration ids that mention the outcome.")
    # setting: str = Field(..., title="The context in which the step is performed, including the location, time, and other relevant details.")
    # setting_content_ids: list[int] = Field(..., title="A list of narration ids that mention the setting.")
    
    ### method
    instructions: str = Field(..., title="The instructions provided for completing the subgoal/stage.")
    instructions_content_ids: list[int] = Field(..., title="A list of narration ids that mention the instructions.")
    
    explanation: str = Field(..., title="The reasons why the instruction was performed or consequences of the instruction.")
    explanation_content_ids: list[int] = Field(..., title="A list of narration ids that mention the explanation.")
    
    tips: str = Field(..., title="The tips that can help in completing the subgoal/stage easier, faster, or more efficient. Warnings that can help avoid mistakes.")
    tips_content_ids: list[int] = Field(..., title="A list of narration ids that mention the tips/warnings.")

    tools: list[str] = Field(..., title="The tools or equipment required to complete the the subgoal/stage along with their descriptions.")
    tools_content_ids: list[int] = Field(..., title="A list of narration ids that mention the tools.")