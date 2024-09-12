from pydantic import BaseModel, Field, validator

class Subgoal(BaseModel):
    title: str = Field(..., title="A concise name/title of the subgoal")
    definition: str = Field(..., title="The detailed description of the subgoal specifying the information it should cover in different videos.")
    dependencies: list[str] = Field(..., title="The list of subgoals that this subgoal depends on.")
    explanation: str = Field(..., title="The justification on why this subgoal is defined and specific examples of the information it can cover in different videos.")

    def get_dict(self):
        return {
            "title": self.title,
            "definition": self.definition,
            "dependencies": self.dependencies,
            "explanation": self.explanation,
        }

class TaskGraph(BaseModel):
    subgoals: list[Subgoal] = Field(..., title="The list of definitions of subgoals of the procedure with their dependencies")
    def get_dict(self):
        return [subgoal.get_dict() for subgoal in self.subgoals]
    
class VideoSegment(BaseModel):
    start: float = Field(..., title="The start time of the segment in seconds")
    finish: float = Field(..., title="The finish time of the segment in seconds")
    title: str = Field(..., title="The title of subgoal that the segment belongs")
    text: str = Field(..., title="The segment's combined subtitle texts")
    explanation: str = Field(..., title="The jusitifcation on why this segment is assigned to a particular subgoal")

    def get_dict(self):
        return {
            "start": self.start,
            "finish": self.finish,
            "title": self.title,
            "text": self.text,
            "explanation": self.explanation,
        }
    
class VideoSegmentation(BaseModel):
    segments: list[VideoSegment] = Field(..., title="The list of segments of the video")

    def get_dict(self):
        return [segment.get_dict() for segment in self.segments]
    