"""
Helper to analyze audio pitch and tempo with librosa.
"""
import librosa
import numpy

__NOTES__ = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def process_audio_librosa(audio_file_path):
    """
    Analyzes the local audio file with librosa to estimate the
    main pitch (musical key) and tempo (BPM).
    """
    try:
        # Load 30s for fast analysis
        y, sr = librosa.load(str(audio_file_path), duration=30.0)

        # Split harmonics components
        y_harmonic, _ = librosa.effects.hpss(y)

        chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)
        mean_chroma = numpy.mean(chromagram, axis=1)
        estimated_tone = __NOTES__[numpy.argmax(mean_chroma)]

        # Estimar el ritmo (BPM)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        # Asegurarnos de extraer el valor numérico escalar del tempo
        estimated_bpm = round(float(tempo[0] if isinstance(tempo, numpy.ndarray) else tempo), 1)

        return estimated_tone, estimated_bpm
    except Exception as e:
        print(f"Error processing audio with librosa in {audio_file_path.name}: {e}")
        return "Unknown", 0.0
