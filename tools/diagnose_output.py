#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
診断スクリプト: AHE-Whisperの出力を詳細に分析
"""

import json
from pathlib import Path

def analyze_output(raw_output_path: str):
    """出力データを詳細に分析"""
    
    with open(raw_output_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print("=" * 60)
    print("AHE-Whisper 出力分析レポート")
    print("=" * 60)
    
    # 基本情報
    duration = data.get('duration_sec', 0)
    print(f"\n【基本情報】")
    print(f"音声長: {duration:.2f}秒 ({duration/60:.1f}分)")
    
    # メトリクス
    metrics = data.get('metrics', {})
    print(f"\n【メトリクス】")
    print(f"ASRカバレッジ: {metrics.get('asr.coverage_ratio', 0)*100:.2f}%")
    print(f"VAD音声比率: {metrics.get('vad.speech_ratio', 0)*100:.2f}%")
    print(f"検出話者数: {metrics.get('diarizer.num_speakers_found', 0)}")
    print(f"単語数: {metrics.get('asr.word_count', 0)}")
    
    # 単語の詳細分析
    words = data.get('words', [])
    if words:
        print(f"\n【単語データ】")
        print(f"総単語数: {len(words)}")
        
        # 時間範囲
        valid_words = [w for w in words if w.get('start') is not None and w.get('end') is not None]
        if valid_words:
            first_word = valid_words[0]
            last_word = valid_words[-1]
            
            first_text = first_word.get("text") or first_word.get("word", "")
            last_text = last_word.get("text") or last_word.get("word", "")
            
            print(f"最初の単語: {first_text} ({first_word['start']:.2f}秒)")
            print(f"最後の単語: {last_text} ({last_word['end']:.2f}秒)")
            
            # カバレッジ計算
            duration = data.get('duration_sec', last_word['end'])
            word_coverage = (last_word['end'] - first_word['start']) / duration * 100 if duration > 0 else 0
            print(f"単語カバレッジ: {word_coverage:.2f}%")

    # セグメントの詳細分析
    segments = data.get('speaker_segments', [])
    print(f"\n【話者セグメント】")
    print(f"総セグメント数: {len(segments)}")
    
    if segments:
        # 話者別の集計
        speaker_times = {}
        for seg in segments:
            speaker = seg['speaker']
            duration_seg = seg['end'] - seg['start']
            if speaker not in speaker_times:
                speaker_times[speaker] = {'duration': 0, 'count': 0}
            speaker_times[speaker]['duration'] += duration_seg
            speaker_times[speaker]['count'] += 1
        
        print(f"\n話者別統計:")
        for speaker, stats in sorted(speaker_times.items()):
            pct = stats['duration'] / duration * 100
            print(f"  {speaker}: {stats['duration']:.1f}秒 ({pct:.1f}%), {stats['count']}セグメント")
        
        # セグメントカバレッジ
        total_seg_duration = sum(seg['end'] - seg['start'] for seg in segments)
        seg_coverage = total_seg_duration / duration * 100
        print(f"\nセグメントカバレッジ: {seg_coverage:.2f}%")
        
        # 最初と最後のセグメント
        print(f"\n最初のセグメント: {segments[0]['start']:.2f}-{segments[0]['end']:.2f}秒")
        print(f"最後のセグメント: {segments[-1]['start']:.2f}-{segments[-1]['end']:.2f}秒")
        
        # ギャップ分析
        print(f"\n【ギャップ分析】")
        if len(segments) > 1:
            gaps = []
            for i in range(1, len(segments)):
                gap = segments[i]['start'] - segments[i-1]['end']
                if gap > 0.1:  # 0.1秒以上のギャップ
                    gaps.append((segments[i-1]['end'], segments[i]['start'], gap))
            
            if gaps:
                print(f"検出されたギャップ: {len(gaps)}個")
                for end, start, gap in sorted(gaps, key=lambda x: x[2], reverse=True)[:5]:
                    print(f"  {end:.2f}-{start:.2f}秒: {gap:.2f}秒のギャップ")
            else:
                print("大きなギャップなし")
    
    # 問題の診断
    print(f"\n【診断結果】")
    
    problems = []
    
    # セグメントカバレッジが低い
    if segments:
        seg_end = segments[-1]['end']
        if seg_end < duration * 0.9:
            problems.append(f"セグメントが音声の90%未満しかカバーしていない ({seg_end:.1f}/{duration:.1f}秒)")
    
    # 単語数とセグメント内容の不一致
    if words and segments:
        seg_word_count = sum(len(seg['text'].split()) for seg in segments)
        if seg_word_count < len(words) * 0.5:
            problems.append(f"セグメント内の単語数が全体の50%未満 ({seg_word_count}/{len(words)})")
    
    # フォールバック使用
    if data.get('is_fallback', False):
        problems.append("フォールバックモードが使用された")
    
    if problems:
        print("⚠️ 検出された問題:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("✅ 大きな問題は検出されませんでした")

if __name__ == "__main__":
    import sys, os

    if len(sys.argv) < 2:
        print("使い方: python diagnose_output.py <出力JSONファイルのパス>")
        sys.exit(1)

    json_path = sys.argv[1]
    if not os.path.exists(json_path):
        print(f"❌ ファイルが存在しません: {json_path}")
        sys.exit(1)

    analyze_output(json_path)
