#!/usr/bin/env python3
"""
Podcast Chapter Generator
Identifies topic shifts when the host begins speaking using pyannote diarization + Gemini AI.
"""

import os
import re
import sys
import argparse
import warnings
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

import torch
from dotenv import load_dotenv
from tqdm import tqdm

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*weights_only.*")

load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
GEMINI_MODEL = "gemini-3-flash-preview"
INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def seconds_to_chapter_ts(seconds: float) -> str:
    """Convert float seconds → YouTube chapter timestamp.
    Under 1 hour: MM:SS  (e.g. 07:42)
    1 hour or more: HH:MM:SS (e.g. 01:07:42)
    """
    total = int(seconds)
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def seconds_to_timestamp(seconds: float) -> str:
    """Convert float seconds → [HH:MM:SS.mmm] for timeline display."""
    td = timedelta(seconds=seconds)
    total_ms = int(td.total_seconds() * 1000)
    h = total_ms // 3600000
    m = (total_ms % 3600000) // 60000
    s = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"[{h:02d}:{m:02d}:{s:02d}.{ms:03d}]"


def parse_srt_timestamp(ts: str) -> float:
    """Convert SRT timestamp string → float seconds."""
    ts = ts.strip().replace(",", ".")
    parts = ts.split(":")
    h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s


def parse_vtt_timestamp(ts: str) -> float:
    """Convert VTT timestamp string → float seconds."""
    ts = ts.strip()
    parts = ts.split(":")
    if len(parts) == 2:
        m, s = float(parts[0]), float(parts[1])
        return m * 60 + s
    h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
    return h * 3600 + m * 60 + s


# ---------------------------------------------------------------------------
# Audio conversion
# ---------------------------------------------------------------------------

def ensure_wav(audio_path: Path) -> Path:
    """Convert MP3 → WAV if necessary. Returns path to WAV file."""
    if audio_path.suffix.lower() == ".wav":
        return audio_path
    wav_path = OUTPUT_DIR / (audio_path.stem + "_converted.wav")
    if wav_path.exists():
        print(f"[*] WAV already exists, skipping conversion: {wav_path.name}")
        return wav_path
    print(f"[*] Converting {audio_path.name} → WAV ...")
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(str(audio_path))
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(str(wav_path), format="wav")
        print(f"[+] WAV saved to {wav_path}")
        return wav_path
    except Exception as e:
        print(f"[!] pydub conversion failed ({e}), trying ffmpeg-python ...")
        import ffmpeg
        (
            ffmpeg
            .input(str(audio_path))
            .output(str(wav_path), ar=16000, ac=1)
            .overwrite_output()
            .run(quiet=True)
        )
        print(f"[+] WAV saved to {wav_path}")
        return wav_path


# ---------------------------------------------------------------------------
# Speaker Diarization
# ---------------------------------------------------------------------------

def run_diarization(wav_path: Path, hf_token: str) -> list[dict]:
    """
    Run pyannote speaker diarization. Returns list of segments:
    [{"speaker": "SPEAKER_00", "start": 12.3, "end": 45.6}, ...]
    """
    print("[*] Loading pyannote diarization pipeline ...")
    from pyannote.audio import Pipeline

    # huggingface_hub reads HF_TOKEN from the environment automatically (set by load_dotenv)
    os.environ["HF_TOKEN"] = hf_token

    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_label = f"CUDA — {torch.cuda.get_device_name(0)}"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_label = "Apple Silicon MPS"
    else:
        device = torch.device("cpu")
        device_label = "CPU (no GPU available — this will be slow)"
    print(f"[*] Using device: {device_label}")

    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
    pipeline = pipeline.to(device)

    print("[*] Running diarization — this may take a few minutes ...")
    diarization = pipeline(str(wav_path))

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end,
        })

    print(f"[+] Diarization complete: {len(segments)} segments found.")

    # Unload model and free GPU/CPU memory immediately
    del pipeline
    del diarization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    print("[+] Model unloaded, memory freed.")

    return segments


# ---------------------------------------------------------------------------
# Host Identification
# ---------------------------------------------------------------------------

def identify_host(segments: list[dict], manual_override: str | None = None) -> str:
    """
    Identify the host speaker ID.
    Logic: speaker with the most segments (or earliest start as tiebreaker).
    A manual override always wins.
    """
    if manual_override:
        print(f"[*] Manual host override: {manual_override}")
        return manual_override

    count = defaultdict(int)
    first_seen = {}
    for seg in segments:
        spk = seg["speaker"]
        count[spk] += 1
        if spk not in first_seen:
            first_seen[spk] = seg["start"]

    # Sort by segment count descending, then earliest first appearance ascending
    ranked = sorted(count.keys(), key=lambda s: (-count[s], first_seen[s]))
    host = ranked[0]
    print(f"[+] Auto-detected host: {host} ({count[host]} segments)")
    for spk in ranked:
        print(f"    {spk}: {count[spk]} segments, first at {seconds_to_chapter_ts(first_seen[spk])}")
    return host


# ---------------------------------------------------------------------------
# Subtitle Parsing
# ---------------------------------------------------------------------------

def parse_subtitles(sub_path: Path) -> list[dict]:
    """
    Parse .srt or .vtt subtitle file.
    Returns list of: {"start": float, "end": float, "text": str}
    """
    ext = sub_path.suffix.lower()
    entries = []

    if ext == ".srt":
        content = sub_path.read_text(encoding="utf-8", errors="replace")
        # SRT block pattern
        pattern = re.compile(
            r"\d+\s*\n"
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n"
            r"([\s\S]*?)(?=\n\n|\Z)",
            re.MULTILINE,
        )
        for m in pattern.finditer(content):
            start = parse_srt_timestamp(m.group(1))
            end = parse_srt_timestamp(m.group(2))
            text = m.group(3).strip().replace("\n", " ")
            # Strip HTML tags that may appear in SRT
            text = re.sub(r"<[^>]+>", "", text)
            if text:
                entries.append({"start": start, "end": end, "text": text})

    elif ext == ".vtt":
        content = sub_path.read_text(encoding="utf-8", errors="replace")
        # Skip WEBVTT header
        lines = content.splitlines()
        i = 0
        while i < len(lines) and not "-->" in lines[i]:
            i += 1
        while i < len(lines):
            line = lines[i].strip()
            if "-->" in line:
                parts = re.split(r"\s+-->\s+", line)
                start = parse_vtt_timestamp(parts[0].split()[-1])
                end = parse_vtt_timestamp(parts[1].split()[0])
                i += 1
                text_lines = []
                while i < len(lines) and lines[i].strip():
                    text_lines.append(lines[i].strip())
                    i += 1
                text = " ".join(text_lines)
                text = re.sub(r"<[^>]+>", "", text)
                if text:
                    entries.append({"start": start, "end": end, "text": text})
            i += 1
    else:
        raise ValueError(f"Unsupported subtitle format: {ext}. Use .srt or .vtt")

    print(f"[+] Parsed {len(entries)} subtitle entries from {sub_path.name}")
    return entries


# ---------------------------------------------------------------------------
# Master Timeline Builder
# ---------------------------------------------------------------------------

def assign_speaker_to_subtitle(sub: dict, segments: list[dict]) -> str:
    """
    Find which speaker is active for most of the subtitle's duration.
    Uses overlap calculation against diarization segments.
    """
    sub_start = sub["start"]
    sub_end = sub["end"]
    overlap = defaultdict(float)

    for seg in segments:
        o_start = max(sub_start, seg["start"])
        o_end = min(sub_end, seg["end"])
        if o_end > o_start:
            overlap[seg["speaker"]] += o_end - o_start

    if not overlap:
        return "UNKNOWN"
    return max(overlap, key=overlap.get)


def build_master_timeline(subtitles: list[dict], segments: list[dict]) -> list[dict]:
    """
    Merge subtitle entries with speaker diarization.
    Returns list of enriched entries with "speaker" field added.
    """
    print("[*] Building master timeline ...")
    timeline = []
    for sub in tqdm(subtitles, desc="Merging speaker labels"):
        speaker = assign_speaker_to_subtitle(sub, segments)
        timeline.append({**sub, "speaker": speaker})
    return timeline


def format_timeline_for_gemini(timeline: list[dict]) -> str:
    """
    Format the master timeline as a clean text block for the Gemini prompt.
    Format: [HH:MM:SS.mmm] [Speaker_XX]: "text..."
    """
    lines = []
    for entry in timeline:
        ts = seconds_to_timestamp(entry["start"])
        spk = entry["speaker"]
        text = entry["text"]
        lines.append(f'{ts} [{spk}]: "{text}"')
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gemini Chapter Generation
# ---------------------------------------------------------------------------

def generate_chapters_with_gemini(
    timeline_text: str,
    host_id: str,
    api_key: str,
) -> str:
    """
    Send the master timeline to gemini-3-flash-preview and request YouTube chapters.
    """
    print(f"[*] Sending timeline to Gemini ({GEMINI_MODEL}) ...")
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL)

    prompt = f"""You are an expert podcast editor. Below is a full podcast transcript with speaker labels and precise timestamps.

THE HOST of this podcast is: {host_id}

IMPORTANT RULE — THE ">>" MARKER:
In the transcript, the symbol ">>" at the start of a line's text marks the exact moment that speaker begins talking (a speaker turn change). This marker comes from YouTube's auto-generated captions.
Example:
  [00:03:22.159] [SPEAKER_02]: ">> So what I wanted to ask next is..."
The timestamp [00:03:22.159] is the EXACT moment the host starts speaking.

YOUR TASK:
1. Analyze the transcript and produce a detailed chapter list — aim for roughly 1 chapter every 3–5 minutes.
2. A chapter may ONLY start on a line where the Host ({host_id}) has ">>" at the start of their text — meaning they have just begun speaking.
3. Use the EXACT timestamp from that ">>" line as the chapter timestamp.
4. Cover both major topic shifts and clear sub-topics — be granular.
5. The first chapter always starts at 00:00 (or 00:00:00 if the episode is over 1 hour).
6. Write chapter titles in THE SAME LANGUAGE as the transcript. Keep titles short and concise (5–7 words max) but descriptive enough to tell the listener what the segment is about.
7. Return ONLY the chapter list in exactly this format (one per line):
   - Under 1 hour:  MM:SS Chapter Title  (e.g. 07:42 Title)
   - 1 hour or more: HH:MM:SS Chapter Title  (e.g. 01:07:42 Title)

Do NOT add any explanation, preamble, or extra text. Output the chapter list only.

TRANSCRIPT:
{timeline_text}
"""

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.9,
            max_output_tokens=65536,
        ),
    )
    return response.text.strip()


# ---------------------------------------------------------------------------
# Output Formatter
# ---------------------------------------------------------------------------

def parse_gemini_chapters(raw: str) -> list[tuple[str, str]]:
    """
    Parse Gemini's chapter output into (formatted_timestamp, title) tuples.
    Accepts MM:SS or HH:MM:SS from Gemini, outputs the correct YouTube format
    (MM:SS under 1h, HH:MM:SS at 1h+).
    """
    chapters = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^(\d{1,2}:\d{2}(?::\d{2})?)\s+(.+)$", line)
        if m:
            ts = m.group(1)
            title = m.group(2).strip()
            parts = ts.split(":")
            if len(parts) == 2:
                total_seconds = int(parts[0]) * 60 + int(parts[1])
            else:
                total_seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            formatted = seconds_to_chapter_ts(total_seconds)
            chapters.append((formatted, title))
    return chapters


def write_chapters_file(
    chapters: list[tuple[str, str]],
    audio_stem: str,
    host_id: str,
    raw_gemini: str,
) -> Path:
    """
    Write the final YouTube-style chapter file to output/.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    out_path = OUTPUT_DIR / f"{audio_stem}_chapters.txt"

    lines = [
        f"# YouTube Chapters — {audio_stem}",
        f"# Host: {host_id}",
        f"# Generated by Podcast Chapter Generator",
        "",
    ]
    for ts, title in chapters:
        lines.append(f"{ts} {title}")

    lines += [
        "",
        "# --- Raw Gemini Output (for reference) ---",
        raw_gemini,
    ]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[+] Chapters written to: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def auto_detect_inputs() -> tuple[Path, Path]:
    """
    Scan the input/ folder and return the first audio file and first subtitle file found.
    Exits with a helpful message if none are found.
    """
    audio_exts = {".mp3", ".wav"}
    sub_exts = {".srt", ".vtt"}

    audio_files = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in audio_exts]
    sub_files = [f for f in INPUT_DIR.iterdir() if f.suffix.lower() in sub_exts]

    if not audio_files:
        print(f"[!] No audio file (.mp3 or .wav) found in {INPUT_DIR}/")
        print("    Drop your audio file there and re-run.")
        sys.exit(1)
    if not sub_files:
        print(f"[!] No subtitle file (.srt or .vtt) found in {INPUT_DIR}/")
        print("    Drop your subtitle file there and re-run.")
        sys.exit(1)

    if len(audio_files) > 1:
        print(f"[!] Multiple audio files found in {INPUT_DIR}/:")
        for f in audio_files:
            print(f"    {f.name}")
        print("    Remove all but one, or use --audio to specify.")
        sys.exit(1)
    if len(sub_files) > 1:
        print(f"[!] Multiple subtitle files found in {INPUT_DIR}/:")
        for f in sub_files:
            print(f"    {f.name}")
        print("    Remove all but one, or use --subs to specify.")
        sys.exit(1)

    return audio_files[0], sub_files[0]


def main():
    parser = argparse.ArgumentParser(
        description="Podcast Chapter Generator — identifies topic shifts when the host speaks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python podcast_chapter_generator.py
  python podcast_chapter_generator.py --host SPEAKER_01
  python podcast_chapter_generator.py --audio input/ep.mp3 --subs input/ep.srt
  python podcast_chapter_generator.py --skip-diarization output/diarization.json
        """,
    )
    parser.add_argument("--audio", default=None, help="Path to audio file (.wav or .mp3). Auto-detected from input/ if omitted.")
    parser.add_argument("--subs", default=None, help="Path to subtitle file (.srt or .vtt). Auto-detected from input/ if omitted.")
    parser.add_argument("--host", default=None, help="Manually specify the host Speaker ID (e.g. SPEAKER_01)")
    parser.add_argument("--save-diarization", default=None, metavar="FILE",
                        help="Save diarization segments to a JSON file for reuse")
    parser.add_argument("--skip-diarization", default=None, metavar="FILE",
                        help="Skip diarization and load segments from a previously saved JSON file")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True)
    INPUT_DIR.mkdir(exist_ok=True)

    # --- Resolve inputs: CLI args take priority, otherwise auto-detect ---
    if args.audio and args.subs:
        audio_path = Path(args.audio)
        sub_path = Path(args.subs)
        if not audio_path.exists():
            print(f"[!] Audio file not found: {audio_path}")
            sys.exit(1)
        if not sub_path.exists():
            print(f"[!] Subtitle file not found: {sub_path}")
            sys.exit(1)
    else:
        audio_path, sub_path = auto_detect_inputs()
        print(f"[+] Auto-detected audio:    {audio_path.name}")
        print(f"[+] Auto-detected subtitles: {sub_path.name}")
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_gemini_api_key_here":
        print("[!] GEMINI_API_KEY not set in .env — please add your key.")
        sys.exit(1)
    if not HF_TOKEN or HF_TOKEN == "your_huggingface_token_here":
        print("[!] HF_TOKEN not set in .env — required for pyannote.audio.")
        sys.exit(1)

    print("=" * 60)
    print("  Podcast Chapter Generator")
    print("=" * 60)

    # --- Step 1: Convert audio if needed ---
    wav_path = ensure_wav(audio_path)

    # --- Step 2: Diarization ---
    if args.skip_diarization:
        import json
        diar_path = Path(args.skip_diarization)
        print(f"[*] Loading diarization from {diar_path} ...")
        with open(diar_path, "r") as f:
            segments = json.load(f)
        print(f"[+] Loaded {len(segments)} segments.")
    else:
        segments = run_diarization(wav_path, HF_TOKEN)
        # Always auto-save diarization so re-runs can skip the GPU step
        import json
        diar_out = Path(args.save_diarization) if args.save_diarization else OUTPUT_DIR / f"{audio_path.stem}_diarization.json"
        with open(diar_out, "w") as f:
            json.dump(segments, f, indent=2)
        print(f"[+] Diarization saved to {diar_out}")
        print(f"    Re-run faster with: --skip-diarization \"{diar_out}\"")

    # --- Step 3: Identify host ---
    host_id = identify_host(segments, manual_override=args.host)

    # --- Step 4: Parse subtitles ---
    subtitles = parse_subtitles(sub_path)

    # --- Step 5: Build master timeline ---
    timeline = build_master_timeline(subtitles, segments)

    # --- Step 6: Format for Gemini ---
    timeline_text = format_timeline_for_gemini(timeline)

    # Save master timeline for reference
    timeline_out = OUTPUT_DIR / f"{audio_path.stem}_master_timeline.txt"
    timeline_out.write_text(timeline_text, encoding="utf-8")
    print(f"[+] Master timeline saved to: {timeline_out}")

    # --- Step 7: Generate chapters via Gemini ---
    raw_chapters = generate_chapters_with_gemini(timeline_text, host_id, GEMINI_API_KEY)
    print("\n[Gemini Raw Output]\n" + raw_chapters)

    # --- Step 8: Parse and write output ---
    chapters = parse_gemini_chapters(raw_chapters)
    if not chapters:
        print("[!] Could not parse any chapters from Gemini output. Check raw output above.")
        # Write raw output anyway
        fallback = OUTPUT_DIR / f"{audio_path.stem}_chapters_raw.txt"
        fallback.write_text(raw_chapters, encoding="utf-8")
        print(f"[*] Raw output saved to: {fallback}")
        sys.exit(1)

    out_file = write_chapters_file(chapters, audio_path.stem, host_id, raw_chapters)

    # --- Print final result ---
    print("\n" + "=" * 60)
    print("  FINAL CHAPTERS")
    print("=" * 60)
    for ts, title in chapters:
        print(f"  {ts}  {title}")
    print("=" * 60)
    print(f"\n[done] Output: {out_file.resolve()}")


if __name__ == "__main__":
    main()
