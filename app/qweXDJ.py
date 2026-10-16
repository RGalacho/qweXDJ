"""
Main entry point for the local playlist generation system
"""
import logging

from flask import Flask, jsonify, request

from core.config import settings
from core.ai_core import ai_service
from domain.services.playlist_files_service import PlaylistFilesService
from domain.services.playlist_generator_service import PlaylistGenerator
from domain.models import GenerationRequest, ContinuationRequest

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialize Flask app and services
app = Flask(__name__)
playlist_files_service = PlaylistFilesService(settings.playlist_repository)
playlist_generator = PlaylistGenerator(ai_service)


@app.route('/api/generate', methods=['POST'])
async def generate_playlist_route():
    """Endpoint to run the playlist generation system.

       Example:
           {
            "prompt": "Create a steady playlist for work",
            "mood": "steady",
            "genre": "rock",
            "limit": "50"
           }
    """
    logger.info("HTTP request to generate playlist")
    
    try:

        data = request.get_json() or {}

        prompt = data.get('prompt', 'Create a relaxing playlist for meditation')
        mood = data.get('mood', '')
        genre = data.get('genre', '')
        limit = int(data.get('limit', 50))
        m3u_save = bool(data.get('m3u_save', False))

        generation_request = GenerationRequest(
            prompt=prompt,
            mood=mood,
            genre=genre,
            limit=limit,
            m3u_save=m3u_save
        )
        
        logger.info("Starting playlist generation...")

        # Generate playlist
        response = await playlist_generator.generate_playlist_from_prompt(generation_request)
        
        logger.info(f"Playlist generated successfully!")
        logger.debug(f"Playlist: {response.playlist.name}")
        logger.debug(f"Number of tracks: {len(response.playlist.tracks)}")
        logger.debug(f"Tracks: {response.playlist.tracks}")
        logger.debug(f"Processing time: {response.processing_time:.2f}s")

        if m3u_save:
            await playlist_files_service.save_playlist(response.playlist)

        return jsonify({
            "status": "success",
            "playlist": {
                "name": response.playlist.name,
                "tracks": [{ "artist": track.artist, "title": track.title } for track in response.playlist.tracks]
            },
            "processing_time": round(response.processing_time, 2)
        }), 200

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


@app.route('/api/continue', methods=['POST'])
async def continue_playlist_route():
    """Endpoint to run the playlist continuation system.

       Example:
           {
            "tracks": ["The Drones - I Don’t Ever Want To Change",
                       "The Drones - Jezebel",
                       "The Drones - Sitting On The Edge Of The Bed",
                       "Tropical Fuck Storm's - Ann",
                       "Tropical Fuck Storm's  - Chameleon Paint"
                      ],
            "limit":  "5"
           }
    """
    logger.info("HTTP request to continue playlist")

    try:

        data = request.get_json() or {}

        songs = data.get('tracks', [])
        limit = int(data.get('limit', 5))

        continuation_request = ContinuationRequest(
            tracks=songs,
            limit=limit,
        )

        logger.info("Starting playlist continuation...")

        # Generate playlist
        response = await playlist_generator.continue_playlist_from_songs(continuation_request)

        logger.info(f"Playlist incremented successfully!")
        logger.debug(f"Tracks: {response.playlist.tracks}")
        logger.debug(f"Processing time: {response.processing_time:.2f}s")
        logger.debug(f"Processing time: {response.confidence_score:.2f}s")

        return jsonify({
            "status": "success",
            "tracks": [{"artist": track.artist, "title": track.title} for track in response.playlist.tracks],
            "processing_time": round(response.processing_time, 2),
            "confidence_score": response.confidence_score
        }), 200

    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


if __name__ == "__main__":
    # Inicia el servidor de desarrollo de Flask
    app.run(debug=True, host='0.0.0.0', port=5000)
