import re
from typing import List
import html
import unicodedata
from nltk.tokenize import sent_tokenize


class TextPreprocessor:
    """
    Cleans and chunks text for TTS/ASR pipelines:
      0. Additional cleaning (unicode, HTML, control-chars, typography, punctuation)
      1. Normalize whitespace
      2. Expand letter-period sequences
      3. Remove inline reference numbers
      4. Split into sentences
      5. Smart-split long sentences
      6. Group sentences into < max_chunk_chars (with tolerance)
    """

    # Precompile once
    _WS_RE = re.compile(r"\s{2,}")
    _LETTER_SEQ_RE = re.compile(r"\b(?:[A-Za-z]\.){2,}")
    _INLINE_REF_RE = re.compile(r'([.!?"\'”’)\]])(\d+)(?=\s|$)')
    _HTML_TAG_RE = re.compile(r"<[^>]+>")
    _CONTROL_RE = re.compile(r"[\x00-\x1F]+")
    _ELLIPSIS_RE = re.compile(r"\.{2,}")
    _REPEAT_PUNC_RE = re.compile(r"([!?]){2,}")
    _CLAUSE_BOUNDARY_RE = re.compile(r"[,:;](?=\s)")

    def __init__(self, max_chunk_chars: int = 400, overflow_tolerence: float = 0.2):
        """
        :param max_chunk_chars: target max length of each chunk
        :param overflow_tolerence: the amount of overflow to allow to prevent small chunks
        """
        self.max_chunk = max_chunk_chars
        self.overflow_tolerence = overflow_tolerence

    def unicode_normalize(self, text: str) -> str:
        """Normalize to Unicode NFC form (composed characters)."""
        return unicodedata.normalize("NFC", text)

    def strip_html(self, text: str) -> str:
        """Remove HTML tags and unescape any entities (e.g. &amp; -> &)."""
        without_tags = self._HTML_TAG_RE.sub("", text)
        return html.unescape(without_tags)

    def remove_control_chars(self, text: str) -> str:
        """Strip ASCII control chars (0-31) by replacing with a space."""
        return self._CONTROL_RE.sub(" ", text)

    def normalize_typography(self, text: str) -> str:
        """Convert smart quotes / dashes to plain ASCII equivalents."""
        return (
            text.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
            .replace("—", " - ")
            .replace("–", " - ")
        )

    def collapse_punctuation(self, text: str) -> str:
        """Collapse sequences of dots into a single ellipsis and repeated !/?."""
        text = self._ELLIPSIS_RE.sub("…", text)  # "..." -> "…"
        text = self._REPEAT_PUNC_RE.sub(r"\1", text)  # "!!!?" -> "!"
        return text

    def normalize_whitespace(self, text: str) -> str:
        """Collapse multiple spaces into one; strip ends."""
        return self._WS_RE.sub(" ", text.strip())

    def expand_initials(self, text: str) -> str:
        """Turn 'A.B.C.' into 'A B C'."""

        def _repl(m):
            core = m.group(0).rstrip(".")
            return " ".join(core.split("."))

        return self._LETTER_SEQ_RE.sub(_repl, text)

    def remove_inline_refs(self, text: str) -> str:
        """Strip trailing reference numbers after punctuation."""
        return self._INLINE_REF_RE.sub(r"\1", text)

    def split_sentences(self, text: str) -> List[str]:
        """NLTK Punkt tokenizer."""
        return sent_tokenize(text)

    def _split_long(self, s: str) -> List[str]:
        """
        Split `s` into pieces < max_chunk, preferring to break
        at commas/semicolons/colons. Falls back to last space
        before the limit, or hard-cut if no space.
        """
        max_len = self.max_chunk
        s = s.strip()

        # short-circuit
        if len(s) <= max_len:
            return [s]

        pieces = []
        start = 0
        length = len(s)

        while start < length:
            # if the rest fits, take it whole
            if length - start <= max_len:
                pieces.append(s[start:].strip())
                break

            # candidate window
            end = start + max_len
            segment = s[start:end]

            # look for all clause boundaries in the window
            # .end() gives index *after* the punctuation
            boundaries = [m.end() for m in self._CLAUSE_BOUNDARY_RE.finditer(segment)]

            if boundaries:
                # pick the last clause boundary within the window
                split_at = start + boundaries[-1]
            else:
                # no clause boundary: break at last space if possible
                space_idx = s.rfind(" ", start, end)
                if space_idx != -1:
                    split_at = space_idx
                else:
                    # absolute fallback: hard cut
                    split_at = end

            # extract & trim
            part = s[start:split_at].strip()
            pieces.append(part)

            # advance (skip any whitespace so we don't start with a space)
            start = split_at
            while start < length and s[start].isspace():
                start += 1

        return pieces

    def group_chunks(self, sents: List[str]) -> List[str]:
        """
        1) Break too-long sentences on word boundaries (via _split_long)
        2) Greedy-pack into <= 100+overflow of max_chunk
        3) Merge any tiny chunk (<overflow of max_chunk) into its neighbor if possible
        """
        # 1) pre-split all sentences into pieces
        pieces: List[str] = []
        for s in sents:
            s = s.strip()
            if s:
                pieces.extend(self._split_long(s))

        # thresholds
        max_limit = int(
            self.max_chunk * (1 + self.overflow_tolerence)
        )  # allow up to  overflow
        min_merge = int(
            self.max_chunk * self.overflow_tolerence
        )  # any chunk shorter than overflow of max_chunk

        # 2) greedy pack with overflow tolerance
        chunks: List[str] = []
        current: List[str] = []
        cur_len = 0

        for p in pieces:
            sep = 1 if current else 0
            if cur_len + sep + len(p) <= max_limit:
                current.append(p)
                cur_len += sep + len(p)
            else:
                chunks.append(" ".join(current))
                current, cur_len = [p], len(p)

        if current:
            chunks.append(" ".join(current))

        # 3) post-pack smoothing: merge tiny chunks into previous neighbor when possible
        i = 1
        while i < len(chunks):
            small = len(chunks[i]) < min_merge
            # try to merge into previous if it doesn’t bust the overflow limit
            if small:
                prev_len = len(chunks[i - 1])
                sep = 1  # because we’ll add a space
                if prev_len + sep + len(chunks[i]) <= max_limit:
                    # merge
                    chunks[i - 1] = chunks[i - 1] + " " + chunks[i]
                    del chunks[i]
                    # stay on same i to check next chunk against this expanded one
                    continue
            i += 1

        return chunks

    def process(self, text: str) -> List[str]:
        """
        End-to-end pipeline.
        """
        text = self.unicode_normalize(text)
        text = self.strip_html(text)
        text = self.remove_control_chars(text)
        text = self.normalize_typography(text)
        text = self.collapse_punctuation(text)
        text = self.normalize_whitespace(text)
        text = self.expand_initials(text)
        text = self.remove_inline_refs(text)
        sents = self.split_sentences(text)

        return self.group_chunks(sents)
