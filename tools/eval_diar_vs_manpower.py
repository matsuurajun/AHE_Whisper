# -*- coding: utf-8 -*-
import argparse
import collections
import itertools
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

Token = Tuple[str, str]  # (word, speaker_id)


LINE_PATTERN = re.compile(
    r"^\[(?P<ts>[^]]+)\]\s+(?P<spk>SPEAKER_\d+):\s*(?P<text>.*)$"
)


def load_tokens(path: Path) -> List[Token]:
    tokens: List[Token] = []
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            m = LINE_PATTERN.match(line)
            if not m:
                continue
            spk = m.group("spk")
            text = m.group("text")
            for w in text.split():
                tokens.append((w, spk))
    return tokens


def needleman_wunsch_align(
    ref: Sequence[Token],
    hyp: Sequence[Token],
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_penalty: int = -1,
) -> List[Tuple[Optional[int], Optional[int]]]:
    n = len(ref)
    m = len(hyp)

    # dp[i][j] = best score aligning ref[:i] and hyp[:j]
    dp: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]
    tb: List[List[int]] = [[0] * (m + 1) for _ in range(n + 1)]  # 0=diag,1=up,2=left

    for i in range(1, n + 1):
        dp[i][0] = dp[i - 1][0] + gap_penalty
        tb[i][0] = 1
    for j in range(1, m + 1):
        dp[0][j] = dp[0][j - 1] + gap_penalty
        tb[0][j] = 2

    def score(w1: str, w2: str) -> int:
        return match_score if w1 == w2 else mismatch_score

    for i in range(1, n + 1):
        wi = ref[i - 1][0]
        for j in range(1, m + 1):
            wj = hyp[j - 1][0]
            s_diag = dp[i - 1][j - 1] + score(wi, wj)
            s_up = dp[i - 1][j] + gap_penalty
            s_left = dp[i][j - 1] + gap_penalty

            if s_diag >= s_up and s_diag >= s_left:
                dp[i][j] = s_diag
                tb[i][j] = 0
            elif s_up >= s_left:
                dp[i][j] = s_up
                tb[i][j] = 1
            else:
                dp[i][j] = s_left
                tb[i][j] = 2

    # backtrack
    i, j = n, m
    alignment: List[Tuple[Optional[int], Optional[int]]] = []
    while i > 0 or j > 0:
        d = tb[i][j]
        if d == 0:
            alignment.append((i - 1, j - 1))
            i -= 1
            j -= 1
        elif d == 1:
            alignment.append((i - 1, None))
            i -= 1
        else:
            alignment.append((None, j - 1))
            j -= 1

    alignment.reverse()
    return alignment


def build_best_mapping(
    alignment: Sequence[Tuple[Optional[int], Optional[int]]],
    ref: Sequence[Token],
    hyp: Sequence[Token],
) -> Tuple[float, Dict[str, str], int, collections.Counter]:
    ref_spks = sorted({spk for _, spk in ref})
    hyp_spks = sorted({spk for _, spk in hyp})

    # デバッグ用
    print(f"[EVAL] human speakers : {ref_spks}")
    print(f"[EVAL] system speakers: {hyp_spks}")

    if not ref_spks or not hyp_spks:
        return 0.0, {}, 0, collections.Counter()

    def evaluate(mapping: Dict[str, str]) -> Tuple[float, int, collections.Counter]:
        correct = 0
        total = 0
        confusion: collections.Counter = collections.Counter()
        for ri, hi in alignment:
            if ri is None or hi is None:
                continue
            rw, rs = ref[ri]
            hw, hs = hyp[hi]
            if rw != hw:
                # ASR が違う部分は話者評価からは除外
                continue
            true_spk = rs
            pred_spk = mapping.get(hs, "<UNK>")
            confusion[(true_spk, pred_spk)] += 1
            total += 1
            if pred_spk == true_spk:
                correct += 1
        acc = float(correct) / float(total) if total else 0.0
        return acc, total, confusion

    best_acc = -math.inf
    best_mapping: Dict[str, str] = {}
    best_total = 0
    best_confusion: collections.Counter = collections.Counter()

    # 1) hyp <= ref のときは「1対1マッピング」のみ (permutations)
    if len(hyp_spks) <= len(ref_spks):
        for perm in itertools.permutations(ref_spks, len(hyp_spks)):
            mapping = dict(zip(hyp_spks, perm))
            acc, total, confusion = evaluate(mapping)
            if total == 0:
                continue
            if acc > best_acc:
                best_acc = acc
                best_mapping = mapping
                best_total = total
                best_confusion = confusion
    else:
        # 2) hyp > ref のときは「多対1」も許可 (product)
        for choices in itertools.product(ref_spks, repeat=len(hyp_spks)):
            mapping = dict(zip(hyp_spks, choices))
            acc, total, confusion = evaluate(mapping)
            if total == 0:
                continue
            if acc > best_acc:
                best_acc = acc
                best_mapping = mapping
                best_total = total
                best_confusion = confusion

    if best_acc == -math.inf:
        return 0.0, {}, 0, collections.Counter()

    return best_acc, best_mapping, best_total, best_confusion


def print_report(
    ref_tokens: Sequence[Token],
    hyp_tokens: Sequence[Token],
    alignment: Sequence[Tuple[Optional[int], Optional[int]]],
    acc: float,
    mapping: Dict[str, str],
    total_eval: int,
    confusion: collections.Counter,
) -> None:
    # 基本統計（単語アライン）
    aligned_pairs = 0
    word_matches = 0
    for ri, hi in alignment:
        if ri is None or hi is None:
            continue
        aligned_pairs += 1
        if ref_tokens[ri][0] == hyp_tokens[hi][0]:
            word_matches += 1

    word_acc = float(word_matches) / float(aligned_pairs) if aligned_pairs else 0.0

    print("=== Diarization Evaluation (manpower vs system) ===")
    print(f"- Human tokens      : {len(ref_tokens)}")
    print(f"- System tokens     : {len(hyp_tokens)}")
    print(f"- Aligned positions : {aligned_pairs}")
    print(f"- Word match ratio  : {word_acc:.4f}")
    print()
    print("Best speaker mapping (system → human):")
    for sys_spk, ref_spk in mapping.items():
        print(f"  {sys_spk}  →  {ref_spk}")
    print()
    print(f"Overall speaker-label accuracy : {acc:.4f}  (on {total_eval} matched tokens)")
    print()

    # 話者別精度
    per_true: Dict[str, collections.Counter] = {}
    for (true_spk, pred_spk), cnt in confusion.items():
        per_true.setdefault(true_spk, collections.Counter())
        per_true[true_spk][pred_spk] += cnt

    print("Per-speaker accuracy (true human speaker):")
    for true_spk in sorted(per_true.keys()):
        ctr = per_true[true_spk]
        total = sum(ctr.values())
        correct = ctr[true_spk]
        acc_s = float(correct) / float(total) if total else 0.0
        print(f"  {true_spk}: {correct:4d}/{total:4d}  ({acc_s:.4f})")
        for pred_spk, c in ctr.most_common():
            print(f"    predicted {pred_spk:10s}: {c:4d}")
        print()

    # 混同行列フル表示
    all_ref = sorted({t[1] for t in ref_tokens})
    all_pred = sorted({t[1] for t in ref_tokens} | {m for _, m in confusion.keys()})

    print("Confusion matrix (true × predicted, counts):")
    header = "true\\pred".ljust(12) + "".join(p.ljust(12) for p in all_pred)
    print(header)
    for tr in all_ref:
        row = [tr.ljust(12)]
        for pr in all_pred:
            row.append(str(confusion.get((tr, pr), 0)).ljust(12))
        print("".join(row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate diarization (system txt) against manpower-checked txt."
    )
    parser.add_argument(
        "--human",
        type=Path,
        required=True,
        help="Manpower-checked txt file (ground truth speakers).",
    )
    parser.add_argument(
        "--system",
        type=Path,
        required=True,
        help="System-generated txt file (AHE-Whisper output).",
    )
    args = parser.parse_args()

    human_tokens = load_tokens(args.human)
    system_tokens = load_tokens(args.system)

    if not human_tokens:
        raise SystemExit(f"no tokens parsed from human file: {args.human}")
    if not system_tokens:
        raise SystemExit(f"no tokens parsed from system file: {args.system}")

    alignment = needleman_wunsch_align(human_tokens, system_tokens)
    acc, mapping, total_eval, confusion = build_best_mapping(
        alignment, human_tokens, system_tokens
    )
    print_report(human_tokens, system_tokens, alignment, acc, mapping, total_eval, confusion)


if __name__ == "__main__":
    main()
