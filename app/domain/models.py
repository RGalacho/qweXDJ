"""
Domain models for the local playlist generation system
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class Track(BaseModel):
    """Model representing a music track"""
    id: Optional[int] = None
    artist: str = Field(..., description="Artist name")
    title: str = Field(..., description="Song title")
    duration: int = Field(..., description="Duration in seconds")
    path: str = Field(..., description="File path to the audio file")

    class Config:
        arbitrary_types_allowed = True

class Playlist(BaseModel):
    """Model representing a playlist"""
    id: Optional[int] = None
    name: str = Field(..., description="Playlist name")
    description: Optional[str] = Field(None, description="Playlist description")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    tracks: List[Track] = Field(default_factory=list)
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="User preferences for this playlist")

    class Config:
        arbitrary_types_allowed = True

class GenerationRequest(BaseModel):
    """Model representing a playlist generation request"""
    prompt: str = Field(..., description="Natural language prompt for playlist generation")
    genre: Optional[str] = Field(None, description="Specific genre to focus on")
    mood: Optional[str] = Field(None, description="Mood or feeling")
    activity: Optional[str] = Field(None, description="Activity context (workout, study, etc.)")
    duration: Optional[int] = Field(None, description="Target playlist duration in minutes")
    limit: int = Field(default=10, ge=1, le=100, description="Number of tracks to generate")
    m3u_save: bool = Field(False, description="Save M3U playlist in repository")
    user_preferences: Optional[Dict[str, Any]] = Field(None, description="Advanced user preferences")

class ContinuationRequest(BaseModel):
    """Model representing a playlist continuation request"""
    tracks: List[str] = Field(..., description="Last 5 songs played")
    limit: int = Field(default=5, ge=1, le=10, description="Number of tracks to generate")

class Config:
    _extra = {
        "example": {
            "prompt": "Create a playlist for working with focus music",
            "mood": "focus",
            "activity": "work",
            "limit": 15
        }
    }

class GenerationResponse(BaseModel):
    """Model representing a playlist generation response"""
    playlist: Playlist = Field(..., description="Generated playlist")
    processing_time: float = Field(..., description="Time taken to process the request in seconds")
    model_used: str = Field(..., description="AI model used for generation")
    confidence_score: float = Field(..., description="Confidence score of the generation (0-1)")

class ContinuationResponse(BaseModel):
    """Model representing a playlist continuation response"""
    tracks: List[str] = Field(..., description="Next 5 songs continuation")
    processing_time: float = Field(..., description="Time taken to process the request in seconds")
    model_used: str = Field(..., description="AI model used for generation")
    confidence_score: float = Field(..., description="Confidence score of the generation (0-1)")

class SystemStatus(BaseModel):
    """Model representing system status"""
    is_healthy: bool = Field(..., description="Whether the system is healthy")
    model_loaded: bool = Field(..., description="Whether AI model is loaded")
    available_memory: int = Field(..., description="Available memory in MB")
    cpu_usage: float = Field(..., description="CPU usage percentage")
