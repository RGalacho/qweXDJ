from pathlib import Path
from domain.models import Track, Playlist

class M3UManager:
    @classmethod
    def save(cls, filename: Path, playlist: Playlist):
        """Save/overwrite a M3U file from `playlist`

        :param filename:
        :param playlist:
        """
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"#EXTM3U\n#PLAYLIST:{filename.stem}")
            for track in playlist.tracks:
                f.write(f"#EXTINF:{track.duration},{track.artist} - {track.title}\n")
                f.write(f"{track.path}\n")


    @classmethod
    def load(cls, filename: Path)-> Playlist:
        """Load and existing M3U file and return a Playlist object."""
        manager = cls()
        playlist = Playlist()

        with (open(filename, "r", encoding="utf-8") as f):
            lines = f.readlines()

            i = 1 if lines and lines[0].startswith("#EXTM3U") else 0
            while i < len(lines):
                line = lines[i].strip()
                if line.startswith("#EXTINF:"):
                    try:
                        duration_part, info = line[8:].split(",", 1)
                        duration = int(duration_part)
                    except ValueError:
                        duration = -1
                        info = line[8:]

                    if " - " in info:
                        artist, title = info.split(" - ", 1)
                    else:
                        artist, title = "Unknown", info

                    path = lines[i + 1].strip() if i + 1 < len(lines) else ""
                    playlist.tracks.append(Track(artist=artist, title=title, duration=duration, path=path))
                    i += 2
                else:
                    i += 1

        return playlist
