import os
import torch
from pyannote.audio import Pipeline
import whisper
from speechbrain.pretrained import EncoderClassifier

# Ensure we're using workspace environment variables
print("🔧 Verifying environment setup...")
print(f"HUGGINGFACE_HUB_CACHE: {os.environ.get('HUGGINGFACE_HUB_CACHE', 'NOT SET')}")
print(f"TORCH_HOME: {os.environ.get('TORCH_HOME', 'NOT SET')}")
print(f"XDG_CACHE_HOME: {os.environ.get('XDG_CACHE_HOME', 'NOT SET')}")

# Define the target directory for models
models_dir = "/workspace/models"
os.makedirs(models_dir, exist_ok=True)

# Create subdirectories for organization
os.makedirs(os.path.join(models_dir, "pyannote"), exist_ok=True)
os.makedirs(os.path.join(models_dir, "whisper"), exist_ok=True)
os.makedirs(os.path.join(models_dir, "speechbrain"), exist_ok=True)

print(f"📁 Models will be downloaded to: {models_dir}")

# --- 1. Download Pyannote Diarization Model ---
print("\n🎤 Downloading Pyannote diarization model (pyannote/speaker-diarization-3.1)...")
try:
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if not hf_token:
        print("⚠️  WARNING: HUGGINGFACE_TOKEN environment variable not set.")
        print("   Attempting to download without it, but it may fail if you haven't accepted the user agreement.")
        print("   Visit: https://huggingface.co/pyannote/speaker-diarization-3.1")
        print("   Then set: export HUGGINGFACE_TOKEN=your_token_here")

    # PyAnnote will use HUGGINGFACE_HUB_CACHE from our environment
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token,
        cache_dir=os.path.join(models_dir, "pyannote")
    )
    print("✅ Pyannote model downloaded successfully.")
except Exception as e:
    print(f"❌ ERROR downloading Pyannote model: {e}")
    print("   Please ensure you have accepted the user agreement and set your HUGGINGFACE_TOKEN.")

# --- 2. Download Whisper Transcription Model ---
print("\n🗣️  Downloading Whisper transcription model (openai/whisper-large-v3)...")
try:
    # Whisper will use TORCH_HOME from our environment, but we can also specify download_root
    whisper_model = whisper.load_model("large-v3", download_root=os.path.join(models_dir, "whisper"))
    print("✅ Whisper model downloaded successfully.")
except Exception as e:
    print(f"❌ ERROR downloading Whisper model: {e}")

# --- 3. Download SpeechBrain Speaker Embedding Model ---
print("\n🧠 Downloading SpeechBrain speaker embedding model (speechbrain/spkrec-ecapa-voxceleb)...")
try:
    # SpeechBrain uses a `savedir` parameter
    speaker_encoder = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=os.path.join(models_dir, "speechbrain", "spkrec-ecapa-voxceleb")
    )
    print("✅ SpeechBrain model downloaded successfully.")
except Exception as e:
    print(f"❌ ERROR downloading SpeechBrain model: {e}")

print("\n🎉 Model download script finished!")
print(f"📁 Models are cached in: {models_dir}")
print("📋 To check what was downloaded:")
print(f"   ls -la {models_dir}")
print(f"   du -sh {models_dir}/*")
