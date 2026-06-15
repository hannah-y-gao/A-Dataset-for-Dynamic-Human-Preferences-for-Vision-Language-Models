from pydantic import BaseModel
from typing import Literal, List

from utils.constants import *

class Answer(BaseModel):
    answer: Literal["a", "b", "c", "d"]

class Choices(BaseModel):
    a: str
    b: str
    c: str
    d: str

ISSUES_VALUES = tuple(str(k) for k in revision_issues.keys())
class Issue(BaseModel):
    analysis: str
    issue: Literal["wrong_scene_description", 
                   "poor_preference_construction", 
                   "no_correct_choices", 
                   "multiple_correct_choices", 
                   "low_image_dependency",
                   "non_actionable",
                   "none"]
    
class Prompt(BaseModel): # output format of generate_prompts.py
    scene: str
    preference: str
    question: str
    choices: Choices

class PromptMetadata(BaseModel): # metadata about each prompt
    timestamp: str
    img: str
    model: str
    category: int
    uuid: str

class PromptWrapper(BaseModel): # this is the format we will store prompts in
    metadata: PromptMetadata
    prompt: Prompt

class PromptRevision(BaseModel):
    proposed_fix: str
    revised_prompt: Prompt
    
class RevisionEntry(BaseModel): # output format of generate_revised_prompts.py
    issue: Issue
    prompt_revision: PromptRevision

class RevisionSession(BaseModel): # all entries in 1 session are for a single image
    img: str
    model: str
    category: int
    uuid: str 
    original_prompt: Prompt
    revised_prompts: List[RevisionEntry]
    
class BoundingBoxMappingClassification(BaseModel):
    A: str
    B: str 
    C: str
    D: str
    E: str 
    
class BoundingBoxMappingRevision(BaseModel):
    analysis: str
    classification: BoundingBoxMappingClassification
    
