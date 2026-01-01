# ahe_whisper/punctuator_xlmroberta.py
# -*- coding: utf-8 -*-
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import onnxruntime as ort
from sentencepiece import SentencePieceProcessor

from ahe_whisper.logger import LOGGER
from ahe_whisper.model_manager import ensure_model_available

try:
    from omegaconf import OmegaConf  # type: ignore[import]
except Exception:  # pragma: no cover
    OmegaConf = None  # type: ignore[assignment]

try:
    import yaml  # type: ignore[import]
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


class XlmRobertaPunctuator:
    def __init__(
        self,
        onnx_path: Path,
        sp_path: Path,
        config_path: Path,
        providers: Optional[Sequence[str]] = None,
    ) -> None:
        self.onnx_path = Path(onnx_path)
        self.sp_path = Path(sp_path)
        self.config_path = Path(config_path)

        self.tokenizer = SentencePieceProcessor(str(self.sp_path))

        provider_list = list(providers) if providers else ["CPUExecutionProvider"]
        session_options = ort.SessionOptions()
        self.session = ort.InferenceSession(
            str(self.onnx_path),
            sess_options=session_options,
            providers=provider_list,
        )

        cfg = self._load_config(self.config_path)
        self.pre_labels: List[str] = list(cfg["pre_labels"])
        self.post_labels: List[str] = list(cfg["post_labels"])
        self.null_token: str = str(cfg.get("null_token", "<NULL>"))
        self.acronym_token: str = str(cfg.get("acronym_token", "<ACRONYM>"))
        self.max_length: int = int(cfg.get("max_length", 512))

        outputs = self.session.get_outputs()
        self.output_names: List[str] = [o.name for o in outputs]
        if len(self.output_names) < 4:
            LOGGER.warning(
                "[PUNCT] Unexpected number of outputs from punctuator ONNX graph: %d",
                len(self.output_names),
            )

    def _load_config(self, path: Path) -> Dict[str, Any]:
        if OmegaConf is not None:
            cfg = OmegaConf.load(str(path))  # type: ignore[operator]
            try:
                from omegaconf import OmegaConf as _OC  # type: ignore[import]
                data = _OC.to_container(cfg, resolve=True)  # type: ignore[attr-defined]
            except Exception:
                data = dict(cfg)
            if not isinstance(data, dict):
                raise TypeError("Punctuator config is not a mapping")
            return data  # type: ignore[return-value]

        if yaml is None:
            raise RuntimeError(
                "Neither omegaconf nor pyyaml is available to parse punctuator config.yaml"
            )

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)  # type: ignore[call-arg]
        if not isinstance(data, dict):
            raise TypeError("Punctuator config.yaml did not parse to a mapping")
        return data  # type: ignore[return-value]

    @classmethod
    def from_project_root(
        cls,
        project_root: Optional[Path],
        providers: Optional[Sequence[str]] = None,
    ) -> "XlmRobertaPunctuator":
        root = Path(project_root) if project_root is not None else Path.cwd()
        model_path = ensure_model_available("punctuator", root)
        model_path = Path(model_path)
        model_dir = model_path.parent

        sp_path = model_dir / "sp.model"
        cfg_path = model_dir / "config.yaml"

        if not sp_path.is_file():
            raise FileNotFoundError(f"SentencePiece model not found: {sp_path}")
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")

        LOGGER.info(
            "[PUNCT] Using punctuator model at %s (sp=%s, cfg=%s)",
            model_path,
            sp_path,
            cfg_path,
        )

        return cls(model_path, sp_path, cfg_path, providers)

    def _cleanup_punct(self, text: str) -> str:
        """Lightweight cleanup to avoid obviously broken punctuation sequences.
        """
        # Remove leading sentence-final punctuation (and surrounding whitespace)
        text = re.sub(r'^[\s\u3000]*[。．\.\?？!！、，]+', '', text)

        # Collapse repeated identical punctuation (e.g., "。。", "！！", "？？", "、、")
        text = re.sub(r'([。．\.\?？!！、，])(?:\s*\1)+', r'\1', text)

        # Normalise mixed ASCII/Fullwidth question marks like "?？ ？" -> "？"
        text = re.sub(r'[\?？]+', '？', text)

        # Collapse mixed terminal punctuation patterns like "？。" / "？ 。" → "？"
        text = re.sub(r'[?？]\s*[。．\.]', '？', text)

        # And "。？" / "．？" / ".？" → "？" （疑問符を優先）
        text = re.sub(r'[。．\.]\s*[?？]', '？', text)

        return text

    def punctuate_text(self, text: str) -> str:
        if not text:
            return text

        ids: List[int] = (
            [int(self.tokenizer.bos_id())]
            + list(self.tokenizer.EncodeAsIds(text))
            + [int(self.tokenizer.eos_id())]
        )

        if len(ids) > self.max_length:
            logging.getLogger(__name__).warning(
                "[PUNCT] sequence length %d exceeds max_length=%d; truncating",
                len(ids),
                self.max_length,
            )
            keep = max(self.max_length - 2, 1)
            middle = ids[1:-1][:keep]
            ids = [ids[0]] + middle + [ids[-1]]

        input_ids = np.asarray([ids], dtype=np.int64)

        outputs = self.session.run(None, {"input_ids": input_ids})
        if len(outputs) < 4:
            LOGGER.error(
                "[PUNCT] Expected 4 outputs from ONNX graph but got %d; returning original text",
                len(outputs),
            )
            return text

        pre_preds = outputs[0][0]
        post_preds = outputs[1][0]
        cap_preds = outputs[2][0]
        sbd_preds = outputs[3][0]  # noqa: F841  # unused, we do not re-segment

        chars: List[str] = []
        token_count = len(ids)

        for token_idx in range(1, token_count - 1):
            token_id = int(ids[token_idx])
            token_piece = self.tokenizer.IdToPiece(token_id)

            if token_piece.startswith("▁") and chars:
                chars.append(" ")

            pre_label = self.pre_labels[int(pre_preds[token_idx])]
            post_label = self.post_labels[int(post_preds[token_idx])]

            if pre_label != self.null_token and token_idx > 1:
                chars.append(pre_label)

            char_start = 1 if token_piece.startswith("▁") else 0
            caps_for_token = cap_preds[token_idx]

            for char_pos, ch in enumerate(token_piece[char_start:], start=char_start):
                try:
                    if caps_for_token[char_pos]:
                        ch = ch.upper()
                except Exception:
                    pass
                chars.append(ch)
                if post_label == self.acronym_token:
                    chars.append(".")

            if post_label != self.null_token and post_label != self.acronym_token:
                chars.append(post_label)

        raw_text = "".join(chars)
        return self._cleanup_punct(raw_text)        

    def punctuate_segments(
        self,
        segments: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        new_segments: List[Dict[str, Any]] = []
        for seg in segments:
            text = seg.get("text") or ""
            try:
                punct_text = self.punctuate_text(text)
            except Exception as e:  # pragma: no cover
                LOGGER.error("[PUNCT] Failed to punctuate segment: %s", e, exc_info=True)
                punct_text = text

            new_seg = dict(seg)
            new_seg["text"] = punct_text
            new_segments.append(new_seg)
        return new_segments


_PUNCTUATOR_CACHE: Dict[str, XlmRobertaPunctuator] = {}


def get_punctuator(project_root: Optional[Path]) -> Optional[XlmRobertaPunctuator]:
    key = str(project_root) if project_root is not None else ""
    if key in _PUNCTUATOR_CACHE:
        return _PUNCTUATOR_CACHE[key]

    try:
        punctuator = XlmRobertaPunctuator.from_project_root(project_root)
    except Exception as e:  # pragma: no cover
        LOGGER.error("[PUNCT] Initialization failed: %s", e, exc_info=True)
        return None

    _PUNCTUATOR_CACHE[key] = punctuator
    return punctuator
