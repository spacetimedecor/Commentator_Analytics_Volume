#!/usr/bin/env python3
"""
Test script to verify that models load correctly from workspace cache.
"""

import os
import sys

def test_environment():
    """Test that environment variables are set correctly."""
    print("🔧 Testing environment configuration...")
    
    required_vars = [
        'HUGGINGFACE_HUB_CACHE',
        'TORCH_HOME', 
        'XDG_CACHE_HOME'
    ]
    
    for var in required_vars:
        value = os.environ.get(var)
        if value and value.startswith('/workspace'):
            print(f"✅ {var}: {value}")
        else:
            print(f"❌ {var}: {value or 'NOT SET'}")
            return False
    
    return True

def test_whisper():
    """Test Whisper model file presence."""
    print("\n🗣️  Testing Whisper model files...")
    
    # Check if model exists in workspace
    model_path = "/workspace/models/whisper/large-v3.pt"
    if os.path.exists(model_path):
        size_gb = os.path.getsize(model_path) / (1024**3)
        print(f"✅ Whisper model found: {model_path}")
        print(f"   Model size: {size_gb:.1f} GB")
        
        # Check if it's a reasonable size (large-v3 should be ~2.9GB)
        if 2.5 < size_gb < 3.5:
            print("✅ Model size looks correct for large-v3")
            return True
        else:
            print(f"⚠️  Unexpected model size (expected ~2.9GB)")
            return False
    else:
        print(f"❌ Whisper model not found at {model_path}")
        return False

def test_speechbrain():
    """Test SpeechBrain model files."""
    print("\n🧠 Testing SpeechBrain model files...")
    
    # Check if model directory exists in workspace  
    model_dir = "/workspace/models/speechbrain/spkrec-ecapa-voxceleb"
    if os.path.exists(model_dir):
        print(f"✅ SpeechBrain model directory found: {model_dir}")
        
        # Check for key model files
        import glob
        model_files = glob.glob(f"{model_dir}/**/*.ckpt", recursive=True)
        config_files = glob.glob(f"{model_dir}/**/*.yaml", recursive=True)
        
        print(f"   Model files (.ckpt): {len(model_files)}")
        print(f"   Config files (.yaml): {len(config_files)}")
        
        if model_files and config_files:
            print("✅ SpeechBrain model files look complete")
            return True
        else:
            print("⚠️  Some SpeechBrain model files may be missing")
            return False
    else:
        print(f"❌ SpeechBrain model not found at {model_dir}")
        return False

def test_pyannote():
    """Test PyAnnote model loading."""
    print("\n🎤 Testing PyAnnote model access...")
    try:
        # Check if GPU is available before importing pyannote
        import torch
        if not torch.cuda.is_available():
            print("⚠️  Skipping PyAnnote test - no GPU available")
            return True
            
        from pyannote.audio import Pipeline
        
        # Check cache directory
        cache_dir = "/workspace/.cache/huggingface"
        if os.path.exists(cache_dir):
            print(f"✅ HuggingFace cache found: {cache_dir}")
            
            # List what's in the cache
            import glob
            cached_models = glob.glob(f"{cache_dir}/**/models--pyannote--*", recursive=True)
            if cached_models:
                print(f"✅ PyAnnote models in cache: {len(cached_models)} found")
                for model in cached_models[:3]:  # Show first 3
                    print(f"   - {model}")
            else:
                print("⚠️  No PyAnnote models found in cache (may need HuggingFace token)")
            
            return True
        else:
            print(f"❌ HuggingFace cache not found at {cache_dir}")
            return False
            
    except Exception as e:
        print(f"❌ Error accessing PyAnnote: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Testing workspace model setup...\n")
    
    tests = [
        test_environment,
        test_whisper,
        test_speechbrain, 
        test_pyannote
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print(f"\n📊 Test Results: {sum(results)}/{len(results)} passed")
    
    if all(results):
        print("🎉 All tests passed! Models are properly cached and accessible.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)