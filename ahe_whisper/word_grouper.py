# -*- coding: utf-8 -*-
import traceback
from typing import List, Dict, Any
from threading import Lock
import logging

LOGGER = logging.getLogger("ahe_whisper_worker")

try:
    from sudachipy import Dictionary, Tokenizer
except ImportError:
    Dictionary = None
    Tokenizer = None

_sudachi_tokenizer = None
_sudachi_lock = Lock()

def _get_text(w: Dict[str, Any]) -> str:
    text = w.get('text', w.get('word'))
    return "" if text is None else str(text)

def _get_tokenizer():
    global _sudachi_tokenizer
    if _sudachi_tokenizer is None:
        with _sudachi_lock:
            if _sudachi_tokenizer is None:
                if Dictionary is None:
                    raise ImportError("SudachiPy is not installed but is required for word grouping.")
                _sudachi_tokenizer = Dictionary().create()
    return _sudachi_tokenizer

def group_words_sudachi(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    修正版: 単語のグループ化を行うが、すべての単語を確実に保持する
    """
    if Tokenizer is None or not words:
        return words

    try:
        tokenizer = _get_tokenizer()
        mode = Tokenizer.SplitMode.C
    except (ImportError, Exception) as e:
        LOGGER.error(f"Failed to initialize SudachiPy tokenizer, returning original words. Error: {e}")
        return words

    # 正規化: text/wordと時間情報を持つ単語のみ
    norm_words = [w for w in words if _get_text(w) and w.get('start') is not None and w.get('end') is not None]
    
    if len(words) != len(norm_words):
        LOGGER.debug(f"Dropped {len(words) - len(norm_words)} word entries lacking text or timestamps.")
    
    # 単語が少ない場合は元のまま返す
    if len(norm_words) < 10:
        LOGGER.warning(f"Too few words ({len(norm_words)}) for grouping, returning original")
        return norm_words

    # テキスト全体を構築
    full_text = "".join(_get_text(w) for w in norm_words).replace(" ", "")
    
    try:
        morphemes = list(tokenizer.tokenize(full_text, mode))
    except Exception as e:
        LOGGER.error(f"SudachiPy tokenization failed, returning original words. Error: {e}")
        return norm_words

    grouped_words = []
    word_idx = 0
    unmatched_start = 0  # マッチしなかった単語の開始位置
    
    for m_idx, m in enumerate(morphemes):
        if word_idx >= len(norm_words):
            break

        m_text = m.surface()
        current_word_group = []
        temp_text = ""
        
        # この形態素に対応する単語を収集
        while word_idx < len(norm_words):
            word_to_add = norm_words[word_idx]
            word_text_stripped = _get_text(word_to_add).strip()
            
            if not m_text.startswith(temp_text + word_text_stripped):
                break

            temp_text += word_text_stripped
            current_word_group.append(word_to_add)
            word_idx += 1
            
            if temp_text == m_text:
                break

        # グループが作成できた場合
        if current_word_group:
            # マッチしなかった単語があれば先に追加
            if unmatched_start < word_idx - len(current_word_group):
                for i in range(unmatched_start, word_idx - len(current_word_group)):
                    grouped_words.append(norm_words[i])
                LOGGER.debug(f"Added {word_idx - len(current_word_group) - unmatched_start} unmatched words")
            
            # グループ化された単語を追加
            start_time = current_word_group[0]['start']
            end_time = current_word_group[-1]['end']
            avg_prob = sum(w.get('prob', 0.8) for w in current_word_group) / len(current_word_group)
            
            grouped_words.append({
                "text": m_text,
                "start": start_time,
                "end": end_time,
                "prob": avg_prob,
                "words": current_word_group
            })
            
            unmatched_start = word_idx
    
    # 残りの単語をすべて追加
    if word_idx < len(norm_words):
        LOGGER.warning(f"Adding {len(norm_words) - word_idx} remaining words that couldn't be grouped")
        grouped_words.extend(norm_words[word_idx:])
    
    # グループ化で単語が大幅に減少した場合は警告して元の単語を返す
    if len(grouped_words) < len(norm_words) * 0.5:
        LOGGER.error(f"Grouping reduced words too much: {len(norm_words)} -> {len(grouped_words)}. Returning original.")
        return norm_words
    
    LOGGER.info(f"Word grouping: {len(norm_words)} -> {len(grouped_words)} words")
    return grouped_words if grouped_words else norm_words
