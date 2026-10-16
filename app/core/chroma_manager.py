"""
Core songs context logic using chromaDB
"""
import os
import chromadb

from typing import Dict, List
from pathlib import Path
from datetime import datetime

from mutagen import File as GenericMetadata
import logging

from core.config import ChromaDBConfig
from core.helpers.musicbrainz_client import MusicBrainzClient
from core.helpers import librosa_analyzer

logger = logging.getLogger(__name__)


class ChromaService:
    @staticmethod
    def _classify_energy_level(bpm: int) -> str:
        """
        Assigns a descriptive label strictly based on the BPM.
        """
        if bpm < 80:
            return "Slow and Relaxed"
        elif 80 <= bpm <= 115:
            return "Moderate and Casual"
        elif 115 < bpm <= 135:
            return "Lively and Steady"
        else:
            return "Fast and Energetic (Upbeat)"


    def __init__(self, config: ChromaDBConfig = None):
        logger.info(f"chroma_service.__init__ - Init")
        self.config = config or ChromaDBConfig()
        self.chroma_client = chromadb.PersistentClient(path=self.config.db_path)
        self.collection = self.chroma_client.get_or_create_collection(name=self.config.db_name)
        # TODO: Maybe instance in methods in which it's used (create, update...)
        self.musicbrainz_client = MusicBrainzClient()


    def scrap_and_process_collection(self):
        logger.info("scrap_and_process_collection - Scrapping collection")
        collection_path = Path(self.config.collection_path)
        file_extensions = {".mp3", ".flac", ".wav", ".m4a", ".ogg"}
        cnt = 1 + self.collection.count()

        for audio_file in collection_path.rglob("*"):
            if audio_file.suffix.lower() in file_extensions:
                logger.info(f"scrap_and_process_collection - [{cnt}] Analyzing file: {audio_file.stem}")

                registry_file = False
                create_date = datetime.timestamp(datetime.now())
                modify_date = datetime.timestamp(datetime.now())

                exist_rec = self._find_in_metadata(audio_file, collection_path)

                if len(exist_rec['documents']) > 0:
                    logger.info(f"scrap_and_process_collection - Found entry ")
                    create_date = float(exist_rec['metadatas'][0].get('create'))
                    mdate = exist_rec['metadatas'][0].get('modify')
                    file_mdate = os.path.getmtime(audio_file)

                    if mdate <= file_mdate:
                        logger.info(f"scrap_and_process_collection - Modification verified, updating collection information ")
                        registry_file = True
                else:
                    logger.info(f"scrap_and_process_collection - Entry not found, updating collection information ")
                    registry_file = True

                if registry_file:
                    audio_meta = self._extract_audio_metadata(audio_file, collection_path)
                    self._add_to_colection(audio_meta, cnt, create_date, modify_date)
                    cnt += 1


    def _add_to_colection(self, audio_meta: dict[str, str], cnt: int,
                          cdate: float | int, mdate: float | int):
        text_doc = (
            f"Artist: {audio_meta.get('artist')} | Title: {audio_meta.get('title')} | "
            f"Duration in secs: {audio_meta.get('duration')} | Genre: {audio_meta.get('genre')} | "
            f"Tone: {audio_meta.get('tone')} | "
            f"Rhythm: {audio_meta.get('rhythm_tag')} ({audio_meta.get('bpm')} BPM) | Year: {audio_meta.get('year')} | "
            f"Path: {audio_meta.get('path')}"
        )

        metadata_db = {
            "artist": audio_meta.get('artist'),
            "title": audio_meta.get('title'),
            "duration": audio_meta.get('duration'),
            "tone": audio_meta.get('tone'),
            "bpm": str(audio_meta.get('bpm')),
            "path": audio_meta.get('path'),
            "create": cdate,
            "modify": mdate
        }

        self.collection.add(
            documents=[text_doc],
            ids=[f"title_{cnt}"],
            metadatas=[metadata_db]
        )


    def _extract_audio_metadata(self, audio_file: Path, collection_path: Path) -> Dict[str, str]:
        artist, title, genre, rhythm_tag, year = "Unknown", "Unknown", "Unspecified", "Unspecified", "Unknown"
        tone, bpm = librosa_analyzer.process_audio_librosa(audio_file)

        metadata = GenericMetadata(audio_file.resolve(), easy=True)
        duration = int(metadata.info.length)

        if metadata:
            # noinspection argument-list
            meta_mbz = self.musicbrainz_client.search_song(metadata.get('artist')[0] if metadata.get('artist') else '',
                                                           metadata.get('album')[0] if metadata.get('album') else '',
                                                           metadata.get('title')[0] if metadata.get('title') else '')
        else:
            meta_mbz = self.musicbrainz_client.search_song(audio_file.stem)

        if meta_mbz:
            artist = meta_mbz.get('artist')
            title = meta_mbz.get('title')
            genre = meta_mbz.get('genre')
            year = meta_mbz.get('year')
        else:
            artist = metadata.get('artist')[0] if metadata and metadata.get('artist') else "Unknown"
            title = metadata.get('title')[0] if metadata and metadata.get('title') else "Unknown"
            genre = metadata.get('album')[0] if metadata and metadata.get('album') else "Unspecified"
            year = metadata.get('date')[0] if metadata and metadata.get('date') else "Unknown"

        rhythm_tag = ChromaService._classify_energy_level(bpm)

        return {"artist": artist, "title": title, "duration": duration, "bpm": str(bpm), "genre": genre,
                "path": str(audio_file.relative_to(collection_path)), "rhythm_tag": rhythm_tag, "tone": tone, "year": year}


    def _find_in_metadata(self, audio_file: Path, collection_path: Path)-> chromadb.GetResult:
        file_path = str(audio_file.relative_to(collection_path))
        exist_metadata = self.collection.get(limit=1,
                                             where={"path": {"$eq": file_path}})

        return exist_metadata


    def get_local_audio_context(self, prompt: str, max_results: int = 5) -> List[str]:
        logger.info("get_local_audio_context - Searching collection")
        results = self.collection.query(
            query_texts=[prompt],
            n_results=max_results
        )
        return results['documents'][0]


    def get_local_audio_continuation_context(self, tracks: List[str], max_results=50) -> List[str]:
        logger.info("get_local_audio_continuation_context - Searching collection")
        seed_songs_text = ("Select songs that harmonize with the following sequence:\n"
                           '\n'.join(tracks)
        )
        results = self.collection.query(
            query_texts=[seed_songs_text],
            n_results=max_results
        )
        return results['documents'][0]

# Global service instance
chroma_service = ChromaService()