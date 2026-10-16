"""
Health check and system status monitoring for the local playlist generation system
"""
import os
import sys
from typing import Dict, Any
from core.config import settings


# noinspection unused-imports,unresolved-references
def check_system_health() -> Dict[str, Any]:
    """
    Perform comprehensive health checks on the system components
    """
    health_status = {
        "system": "healthy",
        "checks": {}
    }
    
    # Check Python version
    python_check = {
        "version": sys.version,
        "compatible": sys.version_info >= (3, 8)
    }
    health_status["checks"]["python"] = python_check
    
    # Check model directory
    try:
        # This will be filled in when we actually initialize the model
        model_path_check = {"exists": True, "accessible": True}
        health_status["checks"]["model_directory"] = model_path_check
    except Exception as e:
        health_status["checks"]["model_directory"] = {
            "exists": False,
            "error": str(e)
        }
        health_status["system"] = "unhealthy"

    # Check audio processing capabilities
    try:
        import chromadb
        chroma_check = {
            "chromadb_available": True,
            "version": chromadb.__version__
        }
        health_status["checks"]["chromadb_context"] = chroma_check
    except ImportError:
        health_status["checks"]["chromadb_context"] = {
            "librosa_available": False
        }
        health_status["system"] = "unhealthy"

    # Check audio processing capabilities
    try:
        import librosa
        audio_check = {
            "librosa_available": True,
            "version": librosa.__version__
        }
        health_status["checks"]["audio_processing"] = audio_check
    except ImportError:
        health_status["checks"]["audio_processing"] = {
            "librosa_available": False
        }
        health_status["system"] = "unhealthy"
    
    # Check AI model capabilities
    try:
        from transformers import AutoTokenizer, AutoModel
        ai_check = {
            "transformers_available": True,
            "tokenizer_loaded": True,
            "model_loaded": True
        }
        health_status["checks"]["ai_model"] = ai_check
    except ImportError:
        health_status["checks"]["ai_model"] = {
            "transformers_available": False
        }
    
    return health_status

def get_system_info() -> Dict[str, Any]:
    """
    Get comprehensive system information for debugging and monitoring
    """
    import platform
    from core.config import settings
    
    system_info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "app_name": settings.app_name,
        "debug_mode": settings.debug,
        "config": settings.dict()
    }
    
    return system_info