"""
MusicBrainz client for searching albums and fetching metadata.
"""

import time
import requests
import logging

from urllib.parse import urlencode
from typing import Optional, Dict, Any, overload


class MusicBrainzClient:
    """
    A client to interact with the MusicBrainz API for song search and metadata retrieval.
    
    Uses the MusicBrainz REST API to search for songs and retrieve metadata.
    """
    def __init__(self, base_url: str = "https://musicbrainz.org/ws/2"):
        """
        Initialize the MusicBrainz client.
        
        Args:
            base_url (str): Base URL for the MusicBrainz API
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.base_url = base_url
        self.session = requests.Session()
        # Set a user agent as required by MusicBrainz API policies
        self.session.headers.update({
            'User-Agent': 'tinyDJ/1.0 ( https://github.com/yourusername/tinyDJ )'
        })
        # Rate limiting: minimum 100ms between requests
        self.last_request_time = 0


    def _rate_limit(self):
        """
        Enforce rate limiting to comply with MusicBrainz API policies.
        """
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < 0.1:  # 100ms minimum
            sleep_time = 0.1 - time_since_last_request
            time.sleep(sleep_time)
        self.last_request_time = time.time()


    # noinspection overloads
    @overload
    def search_song(self, filename: str) -> Dict[str,str] | None:
        """
        Search for an album by artist and album name and return its MBID.

        Args:
            filename (str): Artist name

        Returns:
            :param filename:
            :param artist:
            :param album:
            :param song:
            :return:
        """
        self.logger.info("MBrainzClient.search_song - Init")

        try:
            # Rate limiting
            self._rate_limit()

            # API endpoint for release search
            url = f"{self.base_url}/recording/"

            # Parameters for the request
            params = {
                'query': filename,
                'fmt': 'json',
                'limit': 1
            }

            self.logger.info(f"Searching for filename: {filename} ({url}?{urlencode(params)}/{params})")
            return self._get_api_response(url, params)

        except Exception as e:
            self.logger.error(f"Error during song search: {e}")
            return None



    def search_song(self, artist: str, album: str, title: str) -> Dict[str,str] | None:
        """
        Search for an album by artist and album name and return its MBID.

        Args:
            artist (str): Artist name
            album (str): Album name
            song (str): Song name

        Returns:
            :param artist:
            :param album:
            :param title:
            :return:
        """
        self.logger.info("MBrainzClient.search_song - Init")

        try:
            # Rate limiting
            self._rate_limit()

            # Build the search query
            query = f"release:{album} AND artist:{artist} AND title:{title}"

            # API endpoint for release search
            url = f"{self.base_url}/recording/"

            # Parameters for the request
            params = {
                'query': query,
                'fmt': 'json',
                'limit': 1
            }

            self.logger.info(f"Searching for song: {title} by {artist} ({url}?{urlencode(params)} [{params}])")
            return self._get_api_response(url, params)

        except Exception as e:
            self.logger.error(f"Error during song search: {e}")
            return None


    def _get_api_response(self, url: str, params: dict[str, str | int]) -> Dict[str,str] | None:
        response = self.session.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if 'recordings' in data and len(data['recordings']) > 0:
                recordings = data.get('recordings',[])
                rec = recordings[0]
                title = rec.get("title", "Unknown")
                artist = rec.get("artist-credit", [{}])[0].get("name", "Unknown")
                date = rec.get("first-release-date", "Unknown")
                year = date.split("-")[0] if date else "Unknown"
                tags = [tag.get("name") for tag in rec.get("tags", [])]
                genre = ", ".join(tags) if tags else "Unspecified"

                return {"artist": artist, "title": title, "genre": genre, "year": year}
            else:
                self.logger.warning("No recordings found for the search query")
                return None
        else:
            self.logger.error(f"HTTP {response.status_code} when querying")
            return None
