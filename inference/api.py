"""
FastAPI service for real-time speech-to-text transcription.
Provides WebSocket endpoint for streaming audio and REST endpoint for file upload.
"""

import io
import tempfile
from pathlib import Path
from typing import List, Sequence

import torch
import torchaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from training.dataset import get_tokenizer
from inference.transcribe_model import TranscribeModel

# Configuration
MODEL_PATH = "/workspaces/Speech_To_Text_Model/models/test37/model_latest.pth"
SAMPLE_RATE = 16000
DEVICE = "cpu"

# Global model and tokenizer (loaded once at startup)
model = None
tokenizer = None
blank_token = None


def strip_repeats_and_blanks(sequence: Sequence[int], blank_token: int) -> List[int]:
    """Remove repeated tokens and blank tokens from CTC output."""
    decoded: List[int] = []
    prev = None
    for token in sequence:
        if token == blank_token or token == prev:
            prev = token
            continue
        decoded.append(token)
        prev = token
    return decoded


def transcribe_audio(waveform: torch.Tensor) -> str:
    """Transcribe a waveform tensor and return the text."""
    global model, tokenizer, blank_token
    
    # Ensure mono and batch dimensions
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.dim() == 2 and waveform.shape[0] > 1:
        # Multi-channel, convert to mono
        waveform = waveform.mean(dim=0, keepdim=True)
    
    waveform = waveform.to(DEVICE)
    
    with torch.no_grad():
        log_probs, _ = model(waveform)
    
    predicted_ids = torch.argmax(log_probs, dim=-1)[0].cpu().tolist()
    predicted_ids = strip_repeats_and_blanks(predicted_ids, blank_token)
    return tokenizer.decode(predicted_ids).strip()


def load_audio_from_bytes(audio_bytes: bytes, original_sr: int = None) -> torch.Tensor:
    """Load audio from bytes and resample if necessary."""
    # Write to temp file for torchaudio to load
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(audio_bytes)
        temp_path = f.name
    
    try:
        waveform, sr = torchaudio.load(temp_path)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)(waveform)
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0)
        else:
            waveform = waveform.squeeze(0)
        return waveform
    finally:
        Path(temp_path).unlink(missing_ok=True)


def pcm_to_tensor(pcm_data: bytes, sample_rate: int = 48000) -> torch.Tensor:
    """Convert raw PCM float32 audio data to tensor and resample."""
    import numpy as np
    # PCM data from browser is float32
    audio_array = np.frombuffer(pcm_data, dtype=np.float32)
    waveform = torch.from_numpy(audio_array.copy())
    
    if sample_rate != SAMPLE_RATE:
        waveform = waveform.unsqueeze(0)
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=SAMPLE_RATE)(waveform)
        waveform = waveform.squeeze(0)
    
    return waveform


# Create FastAPI app
app = FastAPI(title="Speech-to-Text API", version="1.0.0")

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Load model and tokenizer on startup."""
    global model, tokenizer, blank_token
    
    print("Loading model and tokenizer...")
    tokenizer = get_tokenizer()
    blank_token = tokenizer.token_to_id("☐")
    
    if Path(MODEL_PATH).exists():
        model = TranscribeModel.load(MODEL_PATH, map_location=DEVICE)
        model.eval()
        print(f"Model loaded from {MODEL_PATH}")
    else:
        print(f"WARNING: Model not found at {MODEL_PATH}")


@app.get("/")
async def root():
    """Serve the demo HTML page."""
    return FileResponse("demo.html")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": DEVICE
    }


@app.post("/transcribe")
async def transcribe_file(file: UploadFile = File(...)):
    """Transcribe an uploaded audio file."""
    if model is None:
        return {"error": "Model not loaded"}
    
    audio_bytes = await file.read()
    waveform = load_audio_from_bytes(audio_bytes)
    transcription = transcribe_audio(waveform)
    
    return {"transcription": transcription}


class AudioBuffer:
    """Buffer for accumulating audio chunks for streaming transcription."""
    def __init__(self, chunk_duration: float = 2.0):
        self.chunks: List[torch.Tensor] = []
        self.chunk_duration = chunk_duration
        self.sample_rate = SAMPLE_RATE
        self.min_samples = int(chunk_duration * SAMPLE_RATE)
    
    def add_chunk(self, chunk: torch.Tensor):
        self.chunks.append(chunk)
    
    def get_accumulated_audio(self) -> torch.Tensor:
        if not self.chunks:
            return torch.tensor([])
        return torch.cat(self.chunks)
    
    def total_samples(self) -> int:
        return sum(c.shape[0] for c in self.chunks)
    
    def should_transcribe(self) -> bool:
        return self.total_samples() >= self.min_samples
    
    def clear(self):
        self.chunks = []


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """
    WebSocket endpoint for real-time audio streaming transcription.
    
    Protocol:
    - Client sends JSON message: {"type": "config", "sample_rate": 48000}
    - Client sends binary audio data (PCM float32)
    - Server sends JSON responses: {"type": "transcription", "text": "..."}
    """
    await websocket.accept()
    
    if model is None:
        await websocket.send_json({"type": "error", "message": "Model not loaded"})
        await websocket.close()
        return
    
    audio_buffer = AudioBuffer(chunk_duration=2.0)
    client_sample_rate = 48000  # Default browser sample rate
    full_transcription = ""
    
    try:
        while True:
            message = await websocket.receive()
            
            if "text" in message:
                # JSON config message
                import json
                data = json.loads(message["text"])
                if data.get("type") == "config":
                    client_sample_rate = data.get("sample_rate", 48000)
                    await websocket.send_json({"type": "config_ack", "sample_rate": client_sample_rate})
                elif data.get("type") == "end":
                    # Process any remaining audio
                    if audio_buffer.total_samples() > 0:
                        waveform = audio_buffer.get_accumulated_audio()
                        transcription = transcribe_audio(waveform)
                        full_transcription += " " + transcription
                        await websocket.send_json({
                            "type": "transcription",
                            "text": transcription,
                            "full_text": full_transcription.strip(),
                            "final": True
                        })
                    break
                    
            elif "bytes" in message:
                # Binary audio data
                audio_data = message["bytes"]
                chunk = pcm_to_tensor(audio_data, client_sample_rate)
                audio_buffer.add_chunk(chunk)
                
                # Transcribe when we have enough audio
                if audio_buffer.should_transcribe():
                    waveform = audio_buffer.get_accumulated_audio()
                    transcription = transcribe_audio(waveform)
                    full_transcription += " " + transcription
                    
                    await websocket.send_json({
                        "type": "transcription",
                        "text": transcription,
                        "full_text": full_transcription.strip(),
                        "final": False
                    })
                    
                    audio_buffer.clear()
                    
    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await websocket.send_json({"type": "error", "message": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
