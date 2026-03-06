# Podcast Chapter Generator

Automatically generates YouTube-style chapters from a podcast episode — no manual timestamping needed.

Drop in the audio file and the subtitle file, run one command, and get a ready-to-paste chapter list in seconds.

## What This Is For

If you publish a podcast on YouTube, YouTube requires you to manually write chapters in the video description in the format:

```
00:00:00 Introduction
00:05:30 Guest Background
00:18:00 Main Topic
...
```

Doing this by hand for a 1–2 hour episode means scrubbing through the entire recording, noting every time the host changes topic, and writing accurate timestamps for each one. For a weekly podcast this can take 30–60 minutes per episode.

This tool automates the entire process:
- It listens to who is speaking at every moment using AI speaker diarization
- It figures out which speaker is the host
- It merges that speaker data with your subtitle file to build a full labelled transcript
- It sends that transcript to Gemini, which reads it and writes the chapter list for you — only opening a new chapter when the **host** speaks (not the guest), matching your podcast's real editorial structure

The output is a plain `.txt` file you can copy-paste directly into YouTube.

---

## How It Works

1. Converts audio to WAV (16kHz mono) if needed — skips if already done
2. Runs **pyannote.audio** speaker diarization on GPU to label every spoken segment by speaker ID
3. Auto-detects the **Host** (speaker with most segments) — or you override manually with `--host`
4. Parses the `.srt` / `.vtt` subtitle file
5. Builds a **Master Timeline** — every subtitle line is tagged with its speaker
6. Sends the full timeline to **Gemini** with a prompt instructing it to only open chapters when the host speaks
7. Saves the chapter list to `output/` — ready to paste into YouTube

---

## What You Need

| Requirement | Where to get it |
|---|---|
| Python 3.10+ | [python.org](https://python.org) |
| NVIDIA GPU (recommended) | RTX 3090 or similar — CPU works but is slow |
| Gemini API Key | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) |
| Hugging Face Token | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |
| Audio file | `.mp3` or `.wav` of the episode |
| Subtitle file | `.srt` or `.vtt` — YouTube auto-generated subtitles work great |

---

## What Goes in Each Folder

```
ultimate-youtube-chapter-generator/
├── input/                  ← Put your .mp3 and .srt here before running
├── output/                 ← All generated files appear here after running
├── .env                    ← Your API keys (never commit this file)
├── podcast_chapter_generator.py
├── requirements.txt
└── README.md
```

### input/
Drop exactly **one audio file** and **one subtitle file** here. The script picks them up automatically — no renaming required.

```
input/
  ├── My Podcast Episode 42.mp3
  └── My Podcast Episode 42.srt
```

### output/
After each run, the following files are saved here:

| File | Description |
|---|---|
| `*_chapters.txt` | Final YouTube-style chapter list — copy this into YouTube |
| `*_master_timeline.txt` | Full transcript with every line tagged by speaker |
| `*_diarization.json` | Saved speaker diarization — reuse with `--skip-diarization` to skip the GPU step on re-runs |

---

## Setup

### 1. Install dependencies

```bash
# GPU (NVIDIA — do this first)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Everything else
pip install -r requirements.txt
```

### 2. Add your API keys to `.env`

```
GEMINI_API_KEY=your_key_here
HF_TOKEN=your_huggingface_token_here
```

### 3. Accept pyannote model terms (one-time, required)

While logged into the Hugging Face account that owns your token, visit both pages and click "Agree and access repository":
- [huggingface.co/pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [huggingface.co/pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)

---

## Usage

### Basic — drop files in `input/` and run

```bash
python podcast_chapter_generator.py
```

### The host was detected wrong — override it

After the first run, the log shows all detected speakers and their segment counts. If the wrong one was picked:

```bash
python podcast_chapter_generator.py --host SPEAKER_00
```

### Re-run without repeating the slow GPU diarization step

Diarization is automatically saved to `output/*_diarization.json` after every run. Use it on re-runs:

```bash
python podcast_chapter_generator.py --host SPEAKER_00 --skip-diarization "output/episode_diarization.json"
```

### Specify files manually (if you have multiple in input/)

```bash
python podcast_chapter_generator.py --audio input/episode.mp3 --subs input/episode.srt
```

---

## Example Output

```
00:00:00 Wstęp
00:01:37 Gość i relacja ze sztuką
00:08:03 Historia ludowa a historia zwykłych ludzi
00:12:56 Czym jest przednówek
00:18:48 Jak wyglądała przeciętna wieś
00:26:31 Relacje między poddanymi a panami
00:37:36 Mit a rzeczywistość pańszczyzny
00:46:31 Opcje wyjścia z poddaństwa
01:05:51 Ludzkie historie
01:20:58 Przemoc i dyscyplina
01:31:46 Codzienność medyczna na wsi
01:43:52 Zakończenie
```

---

## Tips

- **Diarization only needs to run once per episode.** It auto-saves to `output/`. Use `--skip-diarization` on every re-run to go straight to the Gemini step.
- **Wrong host detected?** Check the terminal output after diarization — it lists all speaker IDs with segment counts. The host usually speaks first or has the most short segments (questions). Use `--host SPEAKER_XX` to correct it.
- **YouTube auto-generated subtitles** downloaded as `.srt` work perfectly as input.
- An RTX 3090 processes a 1h45m episode in roughly **8–12 minutes**. CPU takes 45–90 minutes for the same.
