# 🚀 Commentator Analytics v2.0 Implementation Plan

## 📋 System Overview

This implementation plan outlines the complete rebuild of the commentator analytics system using:
- **Apache Airflow** for workflow orchestration
- **RunPod API** for dynamic GPU instance management
- **FastAPI Agent Pattern** for reliable task execution
- **Single persistent volume** for all data and models
- **Purpose-built container images** for each processing step
- **Vector database** (Chroma) for efficient speaker similarity search
- **Diarization-first pipeline** with open-source models
- **Hybrid speaker identification** (averaging + consensus)

## 🏗️ Architecture Overview

```
Single Pod FastAPI Agent Pipeline Flow:
YouTube URL → [Create Pod + Agent] → 
HTTP /execute: Audio Download → 
HTTP /execute: Diarization → 
HTTP /execute: Transcription → 
HTTP /execute: Speaker ID → 
HTTP /execute: Results Storage → [Terminate Pod]
```

### Core Components:
1. **FastAPI Agent System** (HTTP command execution in pod)
2. **Single Persistent Volume** (shared across all commands)
3. **Airflow-RunPod Orchestration System**
4. **Multi-Tool Container Image** (all models + FastAPI agent)
5. **Vector Database Speaker Management**
6. **Hybrid Speaker Identification Pipeline**

---

## 📁 Current Project Structure

```
new_approach/
├── airflow/                        # Airflow orchestration
│   ├── dags/                       # Workflow definitions
│   │   ├── commentator_analytics_pipeline.py
│   │   └── test_runpod_dag.py
│   ├── plugins/                    # Custom operators & hooks
│   │   ├── operators/
│   │   │   ├── runpod_operators.py
│   │   └── hooks/
│   │       ├── runpod_hook.py
│   └── docker/                     # Airflow containers
│       ├── Dockerfile.airflow
│       ├── docker-compose.yml
│       └── requirements-airflow.txt
├── runpod/                         # RunPod processing container
│   ├── scripts/                    # Processing scripts
│   │   ├── audio_download.py
│   │   └── test_script.py
│   ├── utils/                      # Shared utilities
│   │   └── job_utils.py
│   ├── Dockerfile                  # Container definition
│   ├── pod_agent.py               # FastAPI agent
│   └── requirements.txt
├── config/                         # Configuration files
│   ├── runpod_config.py
│   └── __init__.py
├── dev/                            # Development & testing
│   ├── scripts/                    # Build, test, deploy tools
│   │   ├── build-and-push.sh
│   │   ├── cleanup_pods.py
│   │   ├── integration_test.py
│   │   ├── setup.py
│   │   └── test_local.sh
│   └── tests/                      # Test suite
│       ├── test_syntax.py
│       └── __init__.py
├── README.md                       # Documentation
└── .gitignore                      # Git ignore rules
```

---

## 🔧 Phase 1: Foundation Infrastructure ✅

### 1.1 Airflow-RunPod Integration ✅

#### **Custom Operators Implemented:**

##### **RunCommandViaAgentOperator** ✅
```python
class RunCommandViaAgentOperator(BaseOperator):
    """Executes commands via FastAPI agent on persistent RunPod instance"""
    
    def __init__(self, 
                 command: str,
                 job_env: Dict[str, str] = None,
                 pod_url: str = None,
                 timeout: int = 3600,
                 **kwargs):
        super().__init__(**kwargs)
        self.command = command
        self.job_env = job_env or {}
        self.pod_url = pod_url
        self.timeout = timeout
    
    def execute(self, context):
        # Get pod URL from XCom if not provided
        pod_url = self.pod_url
        if not pod_url:
            pod_url = context['task_instance'].xcom_pull(task_ids='create_pod', key='pod_url')
        
        if not pod_url:
            raise Exception("No pod URL provided or found in XCom")
        
        # Build full command with environment variables
        env_setup = " && ".join([f"export {k}='{v}'" for k, v in self.job_env.items()])
        full_command = f"{env_setup} && {self.command}" if env_setup else self.command
        
        # Execute command via FastAPI agent
        result = self.runpod_hook.execute_command_via_agent(
            pod_url=pod_url,
            command=full_command,
            timeout=self.timeout
        )
        
        if result.get('exitCode') != 0:
            raise Exception(f"Command failed with exit code {result['exitCode']}. Stderr: {result['stderr']}")
        
        return {"status": "completed", "result": result, "pod_url": pod_url}
```

### 1.2 RunPod Hook Implementation ✅

#### **RunPodHook** ✅
```python
class RunPodHook(BaseHook):
    """Manages RunPod API interactions"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.runpod.ai/v2"
    
    def create_pod(self, template_id: str, volumes: List[str]) -> str:
        """Create new pod instance"""
        # Implementation in runpod_hook.py
        pass
    
    def execute_command_via_agent(self, pod_url: str, command: str, timeout: int = 3600) -> Dict:
        """Execute command via FastAPI agent"""
        # Implementation in runpod_hook.py
        pass
    
    def terminate_pod(self, pod_id: str) -> bool:
        """Terminate pod instance"""
        # Implementation in runpod_hook.py
        pass
```

### 1.3 Configuration Management ✅

#### **RunPod Configuration** ✅
```python
# config/runpod_config.py
RUNPOD_CONFIG = {
    "api_key": os.getenv("RUNPOD_API_KEY"),
    "default_gpu_type": "RTX 2000 Ada",
    "default_volume_id": os.getenv("RUNPOD_VOLUME_ID"),
    "container_image": "spacetimedecor/multi-tool:latest",
    "timeouts": {
        "startup": 300,      # 5 minutes
        "task": 3600,        # 1 hour  
        "shutdown": 60       # 1 minute
    }
}

# Workspace Configuration
WORKSPACE_CONFIG = {
    "base_path": "/volume/workspace",
    "directories": {
        "audio": "/volume/workspace/audio",
        "segments": "/volume/workspace/segments", 
        "transcripts": "/volume/workspace/transcripts",
        "results": "/volume/workspace/results",
        "models": "/volume/workspace/models",
        "logs": "/volume/workspace/logs",
        "temp": "/volume/workspace/temp"
    },
    "cleanup_after_job": ["audio", "segments", "transcripts", "results", "temp"]
}

# Model Configuration
MODELS_CONFIG = {
    "audio_download": {
        "primary_method": "pytubefix",
        "fallback_method": "yt-dlp",
        "output_format": "wav",
        "sample_rate": 16000,
        "channels": 1  # mono
    },
    "diarization": {
        "model_name": "pyannote/speaker-diarization-3.1",
        "requires_auth": True,
        "gpu_memory": "8GB"
    },
    "transcription": {
        "model_name": "openai/whisper-large-v3", 
        "gpu_memory": "6GB"
    },
    "speaker_embedding": {
        "model_name": "speechbrain/spkrec-ecapa-voxceleb",
        "gpu_memory": "4GB"
    }
}
```

---

## 🎯 Phase 2: Audio Processing Pipeline ✅

### 2.1 Audio Download Implementation ✅

#### **Audio Download Script** ✅
```python
# runpod/scripts/audio_download.py
import os
import logging
from pathlib import Path
from pytubefix import YouTube
import yt_dlp
import subprocess

def download_with_pytubefix(url: str, output_dir: str, filename_prefix: str = None) -> str:
    """Download audio using pytubefix (primary method)"""
    try:
        yt = YouTube(url)
        
        # Get audio stream
        audio_stream = yt.streams.filter(only_audio=True, file_extension='mp4').first()
        
        if not audio_stream:
            raise Exception("No audio stream found")
        
        # Download
        filename = f"{filename_prefix}_{yt.title}" if filename_prefix else yt.title
        output_path = audio_stream.download(output_path=output_dir, filename=f"{filename}.mp4")
        
        # Convert to WAV
        wav_path = Path(output_dir) / f"{filename}.wav"
        subprocess.run([
            'ffmpeg', '-i', output_path, '-acodec', 'pcm_s16le', 
            '-ar', '16000', '-ac', '1', str(wav_path)
        ], check=True)
        
        # Remove original MP4
        os.remove(output_path)
        
        return str(wav_path)
        
    except Exception as e:
        raise Exception(f"pytubefix download failed: {e}")

def download_with_ytdlp(url: str, output_dir: str, filename_prefix: str = None) -> str:
    """Download audio using yt-dlp (fallback method)"""
    try:
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': f'{output_dir}/{filename_prefix}_%(title)s.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
            
        # Find the downloaded file
        for file in Path(output_dir).glob("*.wav"):
            if filename_prefix in file.name:
                return str(file)
        
        raise Exception("Downloaded file not found")
        
    except Exception as e:
        raise Exception(f"yt-dlp download failed: {e}")

def main():
    """Main audio download function"""
    # Implementation with both methods and error handling
    pass
```

### 2.2 FastAPI Agent Implementation ✅

#### **FastAPI Agent** ✅
```python
# runpod/pod_agent.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import subprocess
import logging
import asyncio
from typing import Dict, Any

app = FastAPI(title="RunPod Command Agent", version="1.0.0")

class CommandRequest(BaseModel):
    command: str
    timeout: int = 3600

class CommandResponse(BaseModel):
    stdout: str
    stderr: str
    exitCode: int

@app.post("/execute", response_model=CommandResponse)
async def execute_command(request: CommandRequest):
    """Execute command and return results"""
    logger.info(f"Received command: {request.command}")
    
    try:
        process = await asyncio.create_subprocess_shell(
            request.command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=request.timeout
            )
            
            return CommandResponse(
                stdout=stdout.decode('utf-8') if stdout else "",
                stderr=stderr.decode('utf-8') if stderr else "",
                exitCode=process.returncode
            )
            
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise HTTPException(
                status_code=504, 
                detail=f"Command execution timed out after {request.timeout} seconds"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "agent": "running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
```

---

## 🔄 Phase 3: Next Steps (Planned)

### 3.1 Diarization Pipeline 🚧

#### **Diarization Script** (To Implement)
```python
# runpod/scripts/diarization.py
import os
import json
import torch
from pathlib import Path
from pyannote.audio import Pipeline
from utils.job_utils import setup_logging, get_job_config

def main():
    """Main diarization function"""
    try:
        config = get_job_config()
        logger = setup_logging("diarization", config["batch_id"])
        
        # Load pyannote model
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=os.environ.get("HUGGINGFACE_TOKEN")
        )
        
        # Process audio files
        audio_files = list(Path(config["input_dir"]).glob("*.wav"))
        
        for audio_file in audio_files:
            # Run diarization
            diarization = pipeline(str(audio_file))
            
            # Extract segments
            segments = []
            for segment, _, speaker in diarization.itertracks(yield_label=True):
                segments.append({
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": speaker,
                    "duration": segment.end - segment.start
                })
            
            # Save results
            output_file = Path(config["output_dir"]) / f"{audio_file.stem}_segments.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "audio_file": audio_file.name,
                    "segments": segments
                }, f, indent=2)
        
        logger.info(f"Processed {len(audio_files)} files")
        
    except Exception as e:
        logger.error(f"Diarization failed: {e}")
        raise

if __name__ == "__main__":
    main()
```

### 3.2 Transcription Pipeline 🚧

#### **Transcription Script** (To Implement)
```python
# runpod/scripts/transcription.py
import os
import json
import torch
import whisper
from pathlib import Path
from utils.job_utils import setup_logging, get_job_config

def main():
    """Main transcription function"""
    try:
        config = get_job_config()
        logger = setup_logging("transcription", config["batch_id"])
        
        # Load Whisper model
        model = whisper.load_model("large-v3")
        
        # Process segment files
        segment_files = list(Path(config["input_dir"]).glob("*_segments.json"))
        
        for segment_file in segment_files:
            with open(segment_file, 'r') as f:
                segments_data = json.load(f)
            
            # Transcribe each segment
            transcribed_segments = []
            for segment in segments_data["segments"]:
                # Extract segment audio
                segment_audio = extract_audio_segment(
                    segments_data["audio_file"], 
                    segment["start"], 
                    segment["end"]
                )
                
                # Transcribe
                result = model.transcribe(segment_audio)
                
                transcribed_segments.append({
                    **segment,
                    "text": result["text"],
                    "language": result["language"]
                })
            
            # Save results
            output_file = Path(config["output_dir"]) / f"{segment_file.stem}_transcribed.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "audio_file": segments_data["audio_file"],
                    "segments": transcribed_segments
                }, f, indent=2)
        
        logger.info(f"Transcribed {len(segment_files)} files")
        
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        raise

def extract_audio_segment(audio_file: str, start: float, end: float):
    """Extract audio segment for transcription"""
    # Implementation for audio segment extraction
    pass

if __name__ == "__main__":
    main()
```

### 3.3 Speaker Identification Pipeline 🚧

#### **Speaker ID Script** (To Implement)
```python
# runpod/scripts/speaker_identification.py
import os
import json
import torch
from pathlib import Path
from speechbrain.pretrained import EncoderClassifier
from utils.job_utils import setup_logging, get_job_config

def main():
    """Main speaker identification function"""
    try:
        config = get_job_config()
        logger = setup_logging("speaker_identification", config["batch_id"])
        
        # Load ECAPA model
        encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )
        
        # Load vector database
        # vector_db = load_vector_database(config["vector_db_path"])
        
        # Process transcribed segments
        transcribed_files = list(Path(config["input_dir"]).glob("*_transcribed.json"))
        
        for transcribed_file in transcribed_files:
            with open(transcribed_file, 'r') as f:
                transcripts_data = json.load(f)
            
            identified_segments = []
            for segment in transcripts_data["segments"]:
                # Generate embedding
                embedding = generate_speaker_embedding(segment)
                
                # Identify speaker
                speaker_result = identify_speaker(embedding)
                
                identified_segments.append({
                    **segment,
                    "identified_speaker": speaker_result["speaker"],
                    "confidence": speaker_result["confidence"]
                })
            
            # Save results
            output_file = Path(config["output_dir"]) / f"{transcribed_file.stem}_identified.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "audio_file": transcripts_data["audio_file"],
                    "segments": identified_segments
                }, f, indent=2)
        
        logger.info(f"Identified speakers in {len(transcribed_files)} files")
        
    except Exception as e:
        logger.error(f"Speaker identification failed: {e}")
        raise

def generate_speaker_embedding(segment):
    """Generate speaker embedding for segment"""
    # Implementation for speaker embedding generation
    pass

def identify_speaker(embedding):
    """Identify speaker using vector database"""
    # Implementation for speaker identification
    pass

if __name__ == "__main__":
    main()
```

---

## 📊 Phase 4: Complete Pipeline Integration 🚧

### 4.1 Enhanced Main Pipeline DAG

#### **Enhanced commentator_analytics_pipeline.py** (To Implement)
```python
from airflow import DAG
from datetime import datetime, timedelta
from plugins.operators.runpod_operators import (
    CreateRunPodOperator, 
    TerminateRunPodOperator,
    RunCommandViaAgentOperator
)

default_args = {
    'owner': 'commentator-analytics',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

dag = DAG(
    'commentator_analytics_pipeline',
    default_args=default_args,
    description='Complete commentator analytics processing pipeline',
    schedule_interval=None,
    catchup=False,
    tags=['commentator', 'analytics', 'gpu']
)

# Create GPU pod
create_pod = CreateRunPodOperator(
    task_id='create_analytics_pod',
    image_name='spacetimedecor/multi-tool:latest',
    volume_id='{{ var.value.RUNPOD_VOLUME_ID }}',
    gpu_type='RTX 2000 Ada',
    dag=dag
)

# Download audio
download_audio = RunCommandViaAgentOperator(
    task_id='download_audio',
    command='python /app/scripts/audio_download.py',
    job_env={
        'JOB_TYPE': 'audio_download',
        'VIDEO_URLS': '{{ dag_run.conf.get("video_urls") }}',
        'OUTPUT_DIR': '/volume/workspace/audio',
        'BATCH_ID': '{{ ds_nodash }}',
        'MAX_DOWNLOADS': '{{ dag_run.conf.get("max_downloads", 5) }}'
    },
    dag=dag
)

# Diarization
diarize_audio = RunCommandViaAgentOperator(
    task_id='diarize_audio',
    command='python /app/scripts/diarization.py',
    job_env={
        'JOB_TYPE': 'diarization',
        'INPUT_DIR': '/volume/workspace/audio',
        'OUTPUT_DIR': '/volume/workspace/segments',
        'BATCH_ID': '{{ ds_nodash }}',
        'MODEL_CACHE_DIR': '/volume/workspace/models',
        'HUGGINGFACE_TOKEN': '{{ var.value.HUGGINGFACE_TOKEN }}'
    },
    dag=dag
)

# Transcription
transcribe_segments = RunCommandViaAgentOperator(
    task_id='transcribe_segments',
    command='python /app/scripts/transcription.py',
    job_env={
        'JOB_TYPE': 'transcription',
        'INPUT_DIR': '/volume/workspace/segments',
        'OUTPUT_DIR': '/volume/workspace/transcripts',
        'BATCH_ID': '{{ ds_nodash }}',
        'MODEL_CACHE_DIR': '/volume/workspace/models'
    },
    dag=dag
)

# Speaker identification
identify_speakers = RunCommandViaAgentOperator(
    task_id='identify_speakers',
    command='python /app/scripts/speaker_identification.py',
    job_env={
        'JOB_TYPE': 'speaker_identification',
        'INPUT_DIR': '/volume/workspace/transcripts',
        'OUTPUT_DIR': '/volume/workspace/results',
        'BATCH_ID': '{{ ds_nodash }}',
        'MODEL_CACHE_DIR': '/volume/workspace/models',
        'VECTOR_DB_PATH': '/volume/workspace/vector_db'
    },
    dag=dag
)

# Cleanup pod
terminate_pod = TerminateRunPodOperator(
    task_id='terminate_pod',
    pod_id='{{ ti.xcom_pull(task_ids="create_analytics_pod", key="pod_id") }}',
    trigger_rule='all_done',
    dag=dag
)

# Pipeline flow
create_pod >> download_audio >> diarize_audio >> transcribe_segments >> identify_speakers >> terminate_pod
```

---

## 🎯 Success Metrics & KPIs

### Performance Metrics
- **Audio Download**: Target >95% success rate with pytubefix
- **Diarization Accuracy**: Target <12% DER (Pyannote v3.1)
- **Transcription Accuracy**: Target >95% WER (Whisper Large v3)
- **Speaker Identification**: Target >90% accuracy (ECAPA-TDNN + Vector DB)
- **Processing Speed**: Target 2-3x real-time (sequential processing)

### Operational Metrics
- **Pipeline Reliability**: Target >99% success rate
- **GPU Utilization**: Target >85% during processing
- **Cost Efficiency**: Target <$0.05 per video minute
- **Processing Latency**: Target <30 minutes for 1-hour video

---

## 🔧 Implementation Status

### ✅ **Phase 1 Complete: Foundation**
- [x] Airflow-RunPod integration
- [x] Custom operators and hooks
- [x] Configuration management
- [x] FastAPI agent system
- [x] Audio download pipeline
- [x] Container infrastructure

### 🚧 **Phase 2 In Progress: Model Integration**
- [ ] Diarization script implementation
- [ ] Transcription script implementation  
- [ ] Speaker identification script
- [ ] Vector database integration
- [ ] Model caching system

### 📋 **Phase 3 Planned: Complete Pipeline**
- [ ] End-to-end pipeline testing
- [ ] Performance optimization
- [ ] Cost optimization
- [ ] Monitoring and alerting
- [ ] Documentation completion

### 🎯 **Phase 4 Future: Advanced Features**
- [ ] Speaker enrollment pipeline
- [ ] Validation workflows
- [ ] Batch processing optimization
- [ ] Real-time processing capabilities

---

## 🚨 Known Issues & Solutions

### Current Issues:
1. **YouTube Download Reliability**: Resolved with pytubefix as primary method
2. **XCom Data Sharing**: Fixed with flexible task ID lookup
3. **Container Path References**: Fixed with new directory structure
4. **Model Loading**: Needs implementation for GPU models

### Technical Debt:
1. **Error Handling**: Needs comprehensive error handling in all scripts
2. **Resource Management**: Needs GPU memory optimization
3. **Cost Monitoring**: Needs real-time cost tracking
4. **Scaling**: Needs horizontal scaling considerations

---

## 📚 Reference Links

### Documentation:
- [Apache Airflow Documentation](https://airflow.apache.org/)
- [RunPod API Documentation](https://docs.runpod.io/)
- [Pyannote Audio Documentation](https://github.com/pyannote/pyannote-audio)
- [OpenAI Whisper Documentation](https://github.com/openai/whisper)

### Models:
- [Pyannote Speaker Diarization](https://huggingface.co/pyannote/speaker-diarization-3.1)
- [Whisper Large v3](https://huggingface.co/openai/whisper-large-v3)
- [ECAPA-TDNN](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb)

This implementation plan serves as a comprehensive roadmap for the commentator analytics system, documenting both completed work and future development plans.