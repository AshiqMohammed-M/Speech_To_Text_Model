from pathlib import Path
from typing import List, Sequence

import torch
import torchaudio

from training.dataset import get_tokenizer
from inference.transcribe_model import TranscribeModel


def strip_repeats_and_blanks(sequence: Sequence[int], blank_token: int) -> List[int]:
    decoded: List[int] = []
    prev = None
    for token in sequence:
        if token == blank_token or token == prev:
            prev = token
            continue
        decoded.append(token)
        prev = token
    return decoded


def test_transcribe_audio(
    audio_path: str,
    model_path: str,
    sample_rate: int = 16000,
    device: str = "cpu",
) -> str:
    tokenizer = get_tokenizer()
    blank_token = tokenizer.token_to_id("☐")

    model = TranscribeModel.load(model_path, map_location=device)
    model.eval()
    print(f"Model loaded from {model_path}")

    waveform, sr = torchaudio.load(audio_path)
    print(f'Waveform shape: {waveform.shape}, Sample rate: {sr}')
    if sr != sample_rate:
        waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sample_rate)(
            waveform
        )
    print(f'Waveform shape after resampling: {waveform.shape}, Sample rate: {sample_rate}')
    # Convert [channels, time] to mono [time], then to batch [1, time].
    waveform = waveform.mean(dim=0)
    waveform = waveform.unsqueeze(0).to(device)
    print(f'Waveform shape after mono and batch: {waveform.shape}')
    with torch.no_grad():
        log_probs, _ = model(waveform)

    predicted_ids = torch.argmax(log_probs, dim=-1)[0].cpu().tolist()
    predicted_ids = strip_repeats_and_blanks(predicted_ids, blank_token)
    return tokenizer.decode(predicted_ids).strip()


if __name__ == "__main__":
    MODEL_PATH = "/workspaces/Speech_To_Text_Model/models/test37/model_latest.pth"
    AUDIO_PATH = "/workspaces/Speech_To_Text_Model/inputs/harvard.wav"
    if MODEL_PATH.startswith("<INSERT"):
        raise ValueError("Please set MODEL_PATH before running this script.")

    if not Path(AUDIO_PATH).exists():
        raise FileNotFoundError(f"Audio file not found: {AUDIO_PATH}")

    transcription = test_transcribe_audio(AUDIO_PATH, MODEL_PATH)
    print("Transcription:", transcription)