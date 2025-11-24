import os
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataset, get_tokenizer
from transcribe_model import TranscribeModel

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
torch.autograd.set_detect_anomaly(True)

vq_initial_loss_weight = 10
vq_warmup_steps = 1000
vq_final_loss_weight = 0.5
num_epochs = 1000
starting_steps = 0
num_examples = None
model_id = "test37"
num_batch_repeats = 1
log_every = 20
checkpoint_every = 200
preview_rows = 3

BATCH_SIZE = 64
LEARNING_RATE = 0.005


def run_loss_function(log_probs, target, blank_tokens):
    loss_function = nn.CTCLoss(blank=blank_tokens)
    input_lengths = tuple(log_probs.shape[1] for _ in range(log_probs.shape[0]))
    target_lengths = (target != blank_tokens).sum(dim=1)
    target_lengths = tuple(t.item() for t in target_lengths)
    input_seq_first = log_probs.permute(1, 0, 2)
    loss = loss_function(input_seq_first, target, input_lengths, target_lengths)
    return loss

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

def greedy_decode(log_probs: torch.Tensor, blank_token: int) -> List[List[int]]:
    predictions = torch.argmax(log_probs, dim=-1).cpu().tolist()
    return [strip_repeats_and_blanks(seq, blank_token) for seq in predictions]

def truncate_text(text: str, max_len: int) -> str:
    text = text.replace("\n", " ").strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"

def render_table(rows: Iterable[Tuple[str, str, str]], max_widths=(8, 40, 40)) -> str:
    headers = ("Example", "Model Output", "Ground Truth")
    widths = [max(len(h), w) for h, w in zip(headers, max_widths)]

    def format_row(columns: Sequence[str]) -> str:
        padded = [
            truncate_text(col, width).ljust(width) for col, width in zip(columns, widths)
        ]
        return "| " + " | ".join(padded) + " |"

    border = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    lines = [border, format_row(headers), border]
    for row in rows:
        lines.append(format_row(row))
    lines.append(border)
    return "\n".join(lines)

def preview_batch_predictions(
    log_probs: torch.Tensor,
    references: Sequence[str],
    tokenizer,
    blank_token: int,
    limit: int,
) -> str:
    decoded_ids = greedy_decode(log_probs, blank_token)
    rows = []
    for idx in range(min(limit, len(decoded_ids))):
        pred_ids = decoded_ids[idx]
        prediction = tokenizer.decode(pred_ids) if pred_ids else ""
        rows.append((f"#{idx}", prediction, references[idx]))
    return render_table(rows)

def main():
    log_dir = Path("runs") / "speech_to_text" / model_id
    if log_dir.exists():
        shutil.rmtree(log_dir)
    writer = SummaryWriter(log_dir.as_posix())

    tokenizer = get_tokenizer()
    blank_token = tokenizer.token_to_id("☐")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    checkpoint_dir = Path("models") / model_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest_path = checkpoint_dir / "model_latest.pth"
    best_path = checkpoint_dir / "model_best.pth"
    best_running_loss = float("inf")

    if latest_path.exists():
        print(f"Loading model from {latest_path}")
        model = TranscribeModel.load(latest_path.as_posix(), map_location=device).to(
            device
        )
    else:
        model = TranscribeModel(
            num_codebooks=2,
            codebook_size=32,
            embedding_dim=16,
            vocab_size=len(tokenizer.get_vocab()),
            strides=[6, 6, 6],
            initial_mean_pooling_kernal_size=4,
            num_transformer_layers=2,
            max_seq_length=400,
        ).to(device)

    num_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Number of trainable parameters: {num_trainable_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    dataloader = get_dataset(
        batch_size=BATCH_SIZE, num_examples=num_examples, num_workers=1
    )

    ctc_losses = []
    vq_losses = []
    num_batches = len(dataloader)
    steps = starting_steps

    try:
        for epoch in range(num_epochs):
            for idx, batch in enumerate(dataloader):
                for _ in range(num_batch_repeats):
                    audio = batch["audio"]
                    target = batch["input_ids"]

                    if target.shape[1] > audio.shape[1]:
                        print(
                            "Padding audio, target is longer than audio. Audio Shape: ",
                            audio.shape,
                            " Target Shape: ",
                            target.shape,
                        )
                        audio = torch.nn.functional.pad(
                            audio, (0, 0, 0, target.shape[1] - audio.shape[1])
                        )
                        print("After padding: ", audio.shape)

                    audio = audio.to(device)
                    target = target.to(device)

                    optimizer.zero_grad(set_to_none=True)
                    output, vq_loss = model(audio)
                    ctc_loss = run_loss_function(output, target, blank_token)

                    vq_loss_weight = max(
                        vq_final_loss_weight,
                        vq_initial_loss_weight
                        - (vq_initial_loss_weight - vq_final_loss_weight)
                        * (steps / max(1, vq_warmup_steps)),
                    )
                    total_loss = ctc_loss + vq_loss_weight * vq_loss
                    if torch.isinf(total_loss) or torch.isnan(total_loss):
                        print(
                            "Loss is invalid, skipping step", audio.shape, target.shape
                        )
                        continue
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()

                    ctc_losses.append(ctc_loss.item())
                    if vq_loss is not None:
                        vq_losses.append(vq_loss.item())
                    steps += 1

                    if steps % log_every == 0 and ctc_losses:
                        avg_ctc_loss = sum(ctc_losses) / len(ctc_losses)
                        avg_vq_loss = (
                            sum(vq_losses) / len(vq_losses) if vq_losses else 0.0
                        )
                        avg_loss = avg_ctc_loss + vq_loss_weight * avg_vq_loss
                        print(
                            f"[Epoch {epoch + 1}] Step {steps}/{num_epochs * num_batches}, "
                            f"Batch {idx + 1}/{num_batches}, "
                            f"CTC {avg_ctc_loss:.4f}, VQ {avg_vq_loss:.4f}, "
                            f"Weight {vq_loss_weight:.4f}, Total {avg_loss:.4f}"
                        )
                        writer.add_scalar("train/ctc_loss", avg_ctc_loss, steps)
                        writer.add_scalar("train/vq_loss", avg_vq_loss, steps)
                        writer.add_scalar("train/total_loss", avg_loss, steps)
                        try:
                            table = preview_batch_predictions(
                                output.detach(),
                                batch["text"],
                                tokenizer,
                                blank_token,
                                preview_rows,
                            )
                            print(table)
                        except Exception as decode_error:
                            print(f"Failed to render preview table: {decode_error}")
                        ctc_losses = []
                        vq_losses = []

                    if steps % checkpoint_every == 0:
                        model.save(latest_path.as_posix())
                        if avg_loss < best_running_loss:
                            best_running_loss = avg_loss
                            model.save(best_path.as_posix())
    except KeyboardInterrupt:
        print("Interrupted training; saving latest checkpoint before exiting.")
        model.save(latest_path.as_posix())
    finally:
        writer.close()


if __name__ == "__main__":
    main()

