"""
Configuration management for the local playlist generation system
"""
import os
from dataclasses import dataclass, field

@dataclass(frozen=True)
class AIModelConfig:
    """Configuration for AI model usage"""
    model_name: str = "Qwen/Qwen2.5-3B-Instruct"
    model_type: str = "qwen2"  # "llama", "mistral", etc.
    model_path: str = "./qwen2.5-3B-Instruct"
    gguf_file: str = ""
    max_context_length: int = 8192
    temperature: float = 0.7
    max_new_tokens: int = 1024
    top_p: float = 0.9
    repetition_penalty: float = 1.15


@dataclass(frozen=True)
class ChromaDBConfig:
    """Configuration for Chroma DB usage"""
    collection_path: str = "/home/rgalacho/Música"
    db_path:         str = "./chroma_data"
    db_name:         str = "audio_collection"

@dataclass(frozen=True)
class AudioSettings:
    """Configuration settings for audio processing"""
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 512
    n_mels: int = 128

@dataclass
class Settings:
    """Main configuration settings"""
    # System settings
    app_name: str = "qweXDJ"
    debug: bool = False

    # ChromaDB settings
    chromadb: ChromaDBConfig = field(default_factory=ChromaDBConfig)

    # M3U repository
    playlist_repository: str = "./playlists"

    # Model settings  
    model: AIModelConfig = field(default_factory=AIModelConfig)
    
    # Audio settings
    audio: AudioSettings = field(default_factory=AudioSettings)
    
    # API settings
    host: str = "0.0.0.0"
    port: int = 8000

# Load settings from environment or use defaults
settings = Settings()

# Validate settings
try:
    # Ensure required directories exist
    os.makedirs(os.path.dirname(settings.model.model_path), exist_ok=True)
    os.makedirs(os.path.dirname(settings.chromadb.db_path), exist_ok=True)
    os.makedirs(os.path.dirname(settings.playlist_repository), exist_ok=True)
except Exception as e:
    print(f"Warning: Could not create directory: {e}")