"""Shrutlekh: confidence-aware speech-to-text.

Shrutlekh (Hindi: "dictation") transcribes speech with Whisper and marks every
word with how sure the model was, so a person only has to check the words that
are likely to be wrong.
"""

__version__ = "1.0.0"

from .confidence import ConfidenceThresholds, classify
from .models import Segment, Transcript, Word

__all__ = ["ConfidenceThresholds", "Segment", "Transcript", "Word", "classify", "__version__"]
