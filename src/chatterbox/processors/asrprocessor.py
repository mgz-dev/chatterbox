import re
import logging
from typing import Tuple, NamedTuple, Union
import unicodedata
import whisper
from jiwer import wer
from faster_whisper import WhisperModel as FasterWhisperModel
import torch

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class ASRResult(NamedTuple):
    path: str
    transcription: str
    score: float
    error: str = ""


class ASRVerifier:
    """
    Wraps OpenAI Whisper or faster-whisper for transcription + reference comparison.
    Uses Word Error Rate via jiwer for a word-level accuracy score.
    """
    # TODO: consider testing and migrating to https://github.com/ggml-org/whisper.cpp
    def __init__(
        self,
        model_name: str,
        use_faster_whisper: bool = False,
        device: Union[str, torch.device] = "cpu",
        compare_threshold: float = 0.95,
    ):
        self.device = device
        self.use_faster = use_faster_whisper
        self.threshold = compare_threshold

        self.model_id = model_name
        logger.debug(
            "ASRVerifier init: model=%s faster=%s device=%s",
            self.model_id,
            self.use_faster,
            self.device,
        )

        # Normalize device
        if isinstance(device, torch.device):
            torch_dev = device
            fw_dev = device.type
        else:
            torch_dev = torch.device(device)
            fw_dev = device.split(":", 1)[0]
        
        if fw_dev == "mps":
            fw_dev = "cpu"

        if self.use_faster:
            compute_type = "float16" if fw_dev == "cuda" else "float32"
            self.model = FasterWhisperModel(
                self.model_id,
                device=fw_dev,
                compute_type=compute_type,
            )
        else:
            # whisper.load_model accepts either torch.device or str
            self.model = whisper.load_model(self.model_id, device=torch_dev)

        # compile punctuation/space regexes
        self._dash_re = re.compile(r"[–—-]")
        self._space_re = re.compile(r"\s+")

    def _normalize_text(self, text: str) -> str:
        """
        strip every Unicode punctuation mark
        remove diacritics (e.g. café → cafe)
        collapse whitespace
        """
        # 1) Unicode NFKD & strip diacritics
        text = unicodedata.normalize("NFKD", text)
        text = "".join(ch for ch in text if not unicodedata.combining(ch))

        # 2) convert dashes to spaces
        text = self._dash_re.sub(" ", text)

        # 3) remove *all* punctuation (category "P")
        text = "".join(ch for ch in text if not unicodedata.category(ch).startswith("P"))

        # 4) collapse whitespace & lowercase
        text = self._space_re.sub(" ", text).lower().strip()
        return text

    def transcribe(self, audio_path: str) -> str:
        logger.debug("Transcribing %s", audio_path)
        if self.use_faster:
            segments, _ = self.model.transcribe(audio_path)
            raw = "".join(seg.text for seg in segments)
        else:
            result = self.model.transcribe(audio_path)
            raw = result["text"]
        return raw.strip()

    def transcribe_stream(self, audio_path: str):
        """Yield transcript text segments as soon as they are available."""
        if self.use_faster:
            segments, _ = self.model.transcribe(audio_path)
            for seg in segments:
                yield seg.text  # yields one segment at a time
        else:
            # OpenAI Whisper does not support true streaming out of the box
            result = self.model.transcribe(audio_path)
            yield result["text"]  # Yields all at once

    def verify(self, audio_path: str, reference: str) -> ASRResult:
        """
        Transcribe and compute score = 1 - WER(reference, hypothesis).
        """
        try:
            hyp = self.transcribe(audio_path)
            norm_hyp = self._normalize_text(hyp)
            norm_ref = self._normalize_text(reference)
            error = wer(norm_ref, norm_hyp)  # e.g. 0.12 = 12% WER
            score = max(0.0, 1.0 - error)
            logger.debug("WER=%.3f -> score=%.3f for %s", error, score, audio_path)
            return ASRResult(path=audio_path, transcription=hyp, score=score)
        except Exception as e:
            logger.error("ASR failed for %s: %s", audio_path, e)
            return ASRResult(path=audio_path, transcription="", score=0.0, error=str(e))

    def is_acceptable(self, score: float) -> bool:
        return score >= self.threshold

    def batch_verify(self, items: list[Tuple[str, str]]) -> list[ASRResult]:
        return [self.verify(p, r) for p, r in items]
