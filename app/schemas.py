"""Structured output shapes that Claude fills in."""
from typing import List

from pydantic import BaseModel


class Scene(BaseModel):
    start_seconds: float
    end_seconds: float
    title: str
    description: str


class Moment(BaseModel):
    time_seconds: float
    description: str


class StepNote(BaseModel):
    start_seconds: float
    end_seconds: float
    action: str
    details: str


class SegmentNotes(BaseModel):
    summary: str
    steps: List[StepNote]
    scenes: List[Scene]
    key_moments: List[Moment]
    people: List[str]
    objects_and_places: List[str]
    on_screen_text: List[str]
    audio_notes: str


class Chapter(BaseModel):
    start_seconds: float
    end_seconds: float
    title: str
    summary: str


class Quote(BaseModel):
    time_seconds: float
    speaker: str
    quote: str


class Step(BaseModel):
    number: int
    start_seconds: float
    end_seconds: float
    title: str
    instructions: str
    tools_or_materials: List[str]
    tips_or_warnings: List[str]


class VideoReport(BaseModel):
    title: str
    one_line_summary: str
    detailed_summary: str
    genre_and_tone: str
    chapters: List[Chapter]
    step_by_step: List[Step]
    main_topics: List[str]
    people_and_characters: List[str]
    key_takeaways: List[str]
    notable_quotes: List[Quote]
    content_notes: List[str]
