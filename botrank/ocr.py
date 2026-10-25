"""OCR engines.

Two offline engines are supported:

* **PaddleOCR** (``paddle``) — most accurate, heavier install
  (~500 MB with paddlepaddle), the recommended default.
* **RapidOCR** (``rapid``) — lightweight onnxruntime build (~15 MB of model
  weights, downloaded on first use), good fallback when PaddleOCR cannot be
  installed.

Both return the same normalised :class:`~botrank.parser.Box` list, and both
expose :meth:`extract` which runs the scoreboard parser on a screenshot.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Union

from .parser import Box, ParsedMatch, parse_scoreboard


def _box_from_points(points, text: str, score: float) -> Box:
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    return Box(min(xs), min(ys), max(xs), max(ys), str(text).strip(),
               float(score))


def _unpack_line(line: Sequence) -> Optional[tuple]:
    """Normalise the several line shapes used by PaddleOCR/RapidOCR."""
    if not line:
        return None
    points = line[0]
    rest = list(line[1:])
    text, score = None, 1.0
    if rest and isinstance(rest[0], (tuple, list)):
        text = str(rest[0][0])
        if len(rest[0]) > 1:
            score = float(rest[0][1])
    elif rest:
        text = str(rest[0])
        if len(rest) > 1:
            try:
                score = float(rest[1])
            except (TypeError, ValueError):
                score = 1.0
    if text is None or not text.strip():
        return None
    return points, text, score


def _flatten_paddle(result) -> List[Sequence]:
    """Handle old (``[[line, …]]``), new (``[line, …]``) and dict results."""
    if result is None:
        return []
    if isinstance(result, dict):                       # 3.x predict() style
        polygons = (result.get("dt_polys") or result.get("rec_polys")
                    or result.get("boxes"))
        texts = result.get("rec_texts")
        scores = result.get("rec_scores")
        if polygons is not None and texts is not None:
            if scores is None:
                scores = [1.0] * len(texts)
            return [[list(p), t, s] for p, t, s in zip(polygons, texts, scores)]
        return []
    if isinstance(result, (list, tuple)):
        if len(result) == 0:
            return []
        first = result[0]
        if isinstance(first, dict):
            lines: List[Sequence] = []
            for item in result:
                lines.extend(_flatten_paddle(item))
            return lines
        if (isinstance(first, (list, tuple)) and len(first) > 1
                and isinstance(first[0], (list, tuple))
                and len(first[0]) == 2
                and isinstance(first[0][0], (int, float))):
            return list(result)                        # already a line list
        return _flatten_paddle(first)                  # nested per image
    return []


def boxes_from_paddle(result) -> List[Box]:
    boxes: List[Box] = []
    for line in _flatten_paddle(result):
        unpacked = _unpack_line(line)
        if unpacked:
            points, text, score = unpacked
            boxes.append(_box_from_points(points, text, score))
    return boxes


def boxes_from_rapid(result) -> List[Box]:
    """RapidOCR returns ``([box, text, score], …), elapse``."""
    if not result:
        return []
    lines = result[0] if isinstance(result, tuple) else result
    boxes: List[Box] = []
    for line in lines or []:
        unpacked = _unpack_line(line)
        if unpacked:
            points, text, score = unpacked
            boxes.append(_box_from_points(points, text, score))
    return boxes


class BaseEngine:
    """Common behaviour: open the image, run OCR, parse the scoreboard."""
    name = "base"

    def ocr_boxes(self, image_path: Union[str, Path]) -> List[Box]:  # pragma: no cover
        raise NotImplementedError

    def extract(self, image_path: Union[str, Path], champ_db) -> ParsedMatch:
        from PIL import Image
        image = Image.open(str(image_path))
        image.load()
        width, height = image.size
        boxes = self.ocr_boxes(image_path)
        return parse_scoreboard(boxes, width, height, champ_db, image)


class PaddleEngine(BaseEngine):
    """PaddleOCR engine (Chinese+English by default)."""
    name = "paddle"

    def __init__(self, lang: str = "ch"):
        self.lang = lang
        self._ocr = None

    def _ensure(self):
        if self._ocr is None:
            from paddleocr import PaddleOCR
            try:
                self._ocr = PaddleOCR(use_angle_cls=False, lang=self.lang,
                                      show_log=False)
            except TypeError:                        # 3.x signature change
                self._ocr = PaddleOCR(lang=self.lang, show_log=False)
        return self._ocr

    def ocr_boxes(self, image_path: Union[str, Path]) -> List[Box]:
        ocr = self._ensure()
        try:
            result = ocr.ocr(str(image_path))
        except AttributeError:
            result = ocr.predict(str(image_path))
        if result is None:
            return []
        # predict() may return a generator / list of dicts
        if not isinstance(result, (list, tuple, dict)):
            result = list(result)
        return boxes_from_paddle(result)


class RapidEngine(BaseEngine):
    """RapidOCR (onnxruntime) engine — light and fast."""
    name = "rapid"

    def __init__(self, lang: str = "ch"):
        self.lang = lang
        self._engine = None

    def _ensure(self):
        if self._engine is None:
            from rapidocr_onnxruntime import RapidOCR
            try:
                self._engine = RapidOCR(lang=self.lang)
            except TypeError:
                self._engine = RapidOCR()
        return self._engine

    def ocr_boxes(self, image_path: Union[str, Path]) -> List[Box]:
        engine = self._ensure()
        result, _elapse = engine(str(image_path))
        return boxes_from_rapid(result)


def create_engine(kind: str, lang: str = "ch") -> BaseEngine:
    """Factory used by the UI (``kind`` is ``paddle`` or ``rapid``)."""
    if kind == "paddle":
        return PaddleEngine(lang=lang)
    if kind == "rapid":
        return RapidEngine(lang=lang)
    raise ValueError("Unknown OCR engine: %r" % kind)
