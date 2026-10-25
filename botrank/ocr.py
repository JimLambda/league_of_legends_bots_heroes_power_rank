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
    """Build a :class:`Box` from a polygon.

    PaddleOCR / RapidOCR usually return the polygon as 4 ``[x, y]`` points,
    but some builds flatten it to a single 8-element ``[x1,y1,…,x4,y4]`` list.
    Both shapes are accepted.
    """
    try:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
    except (TypeError, IndexError):                 # flat [x1,y1,…,x4,y4]
        flat = [float(v) for v in points]
        pts = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
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


def _is_point(pt) -> bool:
    """A polygon vertex ``[x, y]`` (also accepts numpy arrays)."""
    try:
        return (len(pt) >= 2
                and isinstance(float(pt[0]), float)
                and isinstance(float(pt[1]), float))
    except (TypeError, ValueError, IndexError):
        return False


def _is_line(candidate) -> bool:
    """A PaddleOCR line is ``[polygon, text]`` or ``[polygon, (text, score)]``.

    ``polygon`` is a list of >=3 points; the second element is text-like
    (a string, or a ``(string, score)`` tuple) — *not* yet another line.
    """
    if not isinstance(candidate, (list, tuple)) or len(candidate) < 2:
        return False
    poly = candidate[0]
    if not isinstance(poly, (list, tuple)) or len(poly) < 3:
        return False
    if not _is_point(poly[0]):
        return False
    second = candidate[1]
    if isinstance(second, (list, tuple)):          # (text, score) tuple
        return isinstance(second[0], str)
    return isinstance(second, str)


def _flatten_paddle(result) -> List[Sequence]:
    """Normalise every PaddleOCR output shape to a flat list of lines.

    PaddleOCR 2.8 ``ocr()`` returns ``[[line, …]]`` (one element per image);
    3.x ``predict()`` may return dicts.  Each line is
    ``[polygon, (text, score)]`` (or ``[polygon, text]``).
    """
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
        if not result:
            return []
        if _is_line(result):                          # the result itself is one line
            return [list(result)]
        if _is_line(result[0]):                       # a list of lines
            return list(result)
        lines: List[Sequence] = []                    # nested per-image wrapper
        for item in result:
            lines.extend(_flatten_paddle(item))
        return lines
    return []


def boxes_from_paddle(result) -> List[Box]:
    boxes: List[Box] = []
    for line in _flatten_paddle(result):
        unpacked = _unpack_line(line)
        if not unpacked:
            continue
        points, text, score = unpacked
        try:
            boxes.append(_box_from_points(points, text, score))
        except (TypeError, IndexError, ValueError):
            continue
    return boxes


def boxes_from_rapid(result) -> List[Box]:
    """RapidOCR returns ``([box, text, score], …), elapse``."""
    if not result:
        return []
    lines = result[0] if isinstance(result, tuple) else result
    boxes: List[Box] = []
    for line in lines or []:
        unpacked = _unpack_line(line)
        if not unpacked:
            continue
        points, text, score = unpacked
        try:
            boxes.append(_box_from_points(points, text, score))
        except (TypeError, IndexError, ValueError):
            continue
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
