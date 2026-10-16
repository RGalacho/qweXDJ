"""
Domain service for playlist management
"""
import os
import logging
import datetime
from pathlib import Path

from domain.models import Playlist
from core.helpers.m3u_manager import M3UManager

logger = logging.getLogger(__name__)

MAX_PLIST_NAME = 20

class PlaylistFilesService:
    """
    Domain service for playlist-related operations and business logic
    """
    
    def __init__(self, playlist_repository: str):
        self.playlist_repository = playlist_repository


    async def save_playlist(self, playlist: Playlist) -> str:
        """
        Save a m3u playlist to repository
        
        Args:
            playlist: Playlist to save
            
        Returns:
            ID of the saved playlist
        """
        try:
            logger.info("save_playlist - Saving playlist in repo")
            playlist_id = self.generate_playlist_name(playlist.name)

            playlist_file = Path(os.path.join(self.playlist_repository, playlist_id))
            M3UManager.save(playlist_file, playlist)

            logger.debug(f"save_playlist - Playlist saved with ID: {playlist_id}")
            return playlist_id
        except Exception as e:
            logger.error(f"save_playlist - Error saving playlist: {str(e)}")
            raise


    @staticmethod
    async def load_playlist(playlist_file: Path) -> Playlist:
        """
        Load a m3u playlist from repository

        Args:
            playlist_file: File path

        Returns:
            Playlist object
        """
        try:
            logger.info("save_playlist - Loading playlist")

            playlist = M3UManager.load(playlist_file)

            logger.debug(f"save_playlist - Playlist {playlist.name} loaded")
            return playlist
        except Exception as e:
            logger.error(f"save_playlist - Error loading playlist: {str(e)}")
            raise


    @staticmethod
    def generate_playlist_name(playlist_name: str)-> str:
        """
        Generate unique name for playlist

        Args:
            playlist_name: Name of the playlist
        Returns:
            Playlist name
        """
        curdate = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pname = playlist_name if playlist_name and len(playlist_name) < MAX_PLIST_NAME else "playlist"

        return f"{pname}-{curdate}.m3u"
