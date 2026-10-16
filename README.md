# QweXDJ - Local Playlist Generator

A smart, AI-powered local music playlist generator that creates personalized playlists based on natural language prompts using your local audio collection.

## Description

QweXDJ is a desktop application that leverages AI to analyze your local music library and generate playlists based on natural language prompts. It integrates with ChromaDB for semantic search of your audio collection, and uses librosa for audio feature extraction (BPM, pitch) to ensure accurate playlist matching.

## Key Features

- **Natural Language Processing**: Generate playlists using simple English prompts ("Create a playlist for working focus music", "Songs for a rainy walk")
- **Local Audio Collection Analysis**: Automatically scans and indexes your local music library
- **AI-Driven Playlist Generation**: Uses Qwen-2.5 model via transformers for intelligent track selection
- **Semantic Search**: Leverages ChromaDB to find relevant tracks based on prompt context
- **Audio Feature Extraction**: Analyzes BPM and tonal information using librosa
- **MusicBrainz Integration**: Retrieves enriched metadata from MusicBrainz
- **Flexible Output**: Generates playlists in M3U format for easy playback

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional but recommended for faster processing)
- Git
- pip

## Installation

1. Clone the repository:
```bash
git clone https://github.com/RGalacho/qweXDJ.git
cd qweXDJ
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Download Qwen2.5 model (https://huggingface.co/Qwen/Qwen2.5-7B-Instruct):
```bash
hf download "Qwen/Qwen2.5-7B-Instruct" --local-dir ./qwen2.5-7B-Instruct
```

## Setup

1. Create a local audio collection directory:
```bash
mkdir ~/Music/qwexDJcollection
```

2. Place your audio files in the collection directory (supporting MP3, FLAC, WAV, M4A, OGG formats)

3. Configure the application. The only option needed to run is `collection_path` and can be modified in `core/config.py`:
```python
class ChromaDBConfig:
    """Configuration for Chroma DB usage"""
    collection_path: str = "/home/user/Music"
```

## Build
```commandline
poetry build
```

## Run Flask app
```commandline
poetry run flask --app app/qweXDJ.py run &
```

## Usage

1. **Start the indexing process**: Before first run, build metadata DB.
```bash
source .venv_poetry/bin/activate

python app/collection_scrapper.py
```

2. **Generate a playlist**:
Run flask app.
Request to http://127.0.0.1:5000/api/generate:
```json
{
  "prompt": "Create a steady playlist for work",
  "mood": "steady",
  "genre": "rock",
  "limit": "50"
}
```

3. **Generate playlist and save to M3U format**:
Run flask app.
Request to http://127.0.0.1:5000/api/generate:
```json
{
  "prompt": "Songs for a rainy evening",
  "mood": "relaxed",
  "limit": "100",
  "m3u_save": true
}
```

## System Architecture

The system follows a modular architecture:

```
app/
├── domain/                 # Data models and schemas
│   └── models.py           # Track, Playlist, GenerationRequest, GenerationResponse
├── core/                   # Core application logic
│   ├── ai_core.py          # AI Service with Qwen2.5 integration
│   ├── chroma_manager.py   # ChromaDB service for collection indexing
│   ├── config.py           # Configuration management
│   └── helpers/
│       ├── musicbrainz_client.py  # MusicBrainz API integration
│       └── librosa_analyzer.py    # Audio feature extraction with librosa
├── collection_scrapper.py         # ChromaDB RAG music collection maintenace
└── qweXDJ.py                      # Main application entry point
tests/                             # Tests modules
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [Transformers](https://github.com/huggingface/transformers) for AI model integration
- [ChromaDB](https://github.com/chroma-core/chroma) for semantic search
- [librosa](https://github.com/librosa/librosa) for audio analysis
- [MusicBrainz](https://musicbrainz.org/) for metadata retrieval

