"""
Core playlist generation logic using AI model
"""
import logging
import re
from domain.models import Track, Playlist, GenerationRequest, GenerationResponse, ContinuationRequest, ContinuationResponse
from core.ai_core import AIService

logger = logging.getLogger(__name__)

TRACK_LIST_RE=r"(?P<artist>[\w ']+) - (?P<song>[\w ']+) - (?P<duration>\d+) \[(?P<path>.+)\]"

class PlaylistGenerator:
    """
    Core playlist generator that processes natural language prompts and creates playlists
    using AI model and audio analysis
    """
    def __init__(self, ai_service: AIService):
        self.ai_service = ai_service

        
    async def generate_playlist_from_prompt(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate a playlist from a natural language prompt. Is recommended to ask for a mood/style in order to refine results:
        "A calm reading session", "An energetic and upbeat playlist", "Pure rock and punk rock from hell", etc.
        
        Args:
            request: GenerationRequest with prompt and parameters
            
        Returns:
            GenerationResponse containing generated playlist
        """
        logger.info(f"generate_playlist_from_prompt - Starting playlist generation for prompt: {request.prompt}")
        
        try:
            import time
            start_time = time.time()

            ai_response = await self.ai_service.analyze_prompt_and_generate_tracks(request)

            tracks, confidence_score = PlaylistGenerator._build_tracks_from_response(ai_response)

            # Create playlist
            playlist = Playlist(
                name=f"{request.prompt}",
                description=f"AI-generated playlist based on '{request.prompt}'",
                tracks=tracks,
                user_preferences=request.user_preferences
            )

            processing_time = time.time() - start_time

            response = GenerationResponse(
                playlist=playlist,
                processing_time=processing_time,
                model_used=self.ai_service.config.model_name,
                confidence_score=confidence_score
            )

            logger.info(f"generate_playlist_from_prompt - Playlist generation completed in {response.processing_time:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"generate_playlist_from_prompt - Error generating playlist: {str(e)}")
            raise

    async def continue_playlist_from_songs(self, request: ContinuationRequest) -> ContinuationResponse:
        """
            Generate a playlist from a natural language prompt. Is recommended to ask for a mood/style in order to refine results:
            "A calm reading session", "An energetic and upbeat playlist", "Pure rock and punk rock from hell", etc.

            Args:
                request: GenerationRequest with prompt and parameters

            Returns:
                GenerationResponse containing generated playlist
            """
        logger.info(f"continue_playlist_from_songs - Continue playlist for tracks: {request.tracks}")

        try:
            import time
            start_time = time.time()

            ai_response = await self.ai_service.analyze_tracks_and_generate_continuation(request)

            tracks, confidence_score = PlaylistGenerator._build_tracks_from_response(ai_response)

            processing_time = time.time() - start_time

            response = ContinuationResponse(
                tracks=tracks,
                processing_time=processing_time,
                model_used=self.ai_service.config.model_name,
                confidence_score=confidence_score
            )

            logger.info(f"continue_playlist_from_songs - Playlist continuation completed in {response.processing_time:.2f}s")

            return response

        except Exception as e:
            logger.error(f"continue_playlist_from_songs - Error continuing playlist: {str(e)}")
            raise

    @staticmethod
    def _build_tracks_from_response(ai_response: str)-> list[Track]:
        """Parsing track list from response."""
        logger.info("_build_tracks_from_response - Parsing AI response")
        try:
            pattern = re.compile(TRACK_LIST_RE)
            track_list = list()

            for match in re.finditer(pattern, ai_response):
                track = Track(artist= match.group('artist'), title = match.group('song'),
                              duration= int(match.group('duration')),
                              path = match.group('path'))
                track_list.append(track)

            return track_list
        except Exception as e:
            logger.error(f"_build_tracks_from_response - Error parsing AI response: {str(e)}")
            raise
