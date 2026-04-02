from typing import Optional, Union

import torch.nn as nn
import torch
from rvq import ResidualVectorQuantizer
from self_attention import Transformer
from downsampling import DownsamplingNetwork

class TranscribeModel(nn.Module):
    def __init__(
        self,
        num_codebooks: int,
        codebook_size: int,
        embedding_dim: int,
        vocab_size: int,
        strides: list[int],
        initial_mean_pooling_kernal_size: int,
        num_transformer_layers: int,
        max_seq_length: int = 2000,
    ):
        super().__init__()
        self.options = {
            "num_codebooks": num_codebooks,
            "codebook_size": codebook_size,
            "embedding_dim": embedding_dim,
            "vocab_size": vocab_size,
            "strides": strides,
            "initial_mean_pooling_kernal_size": initial_mean_pooling_kernal_size,
            "num_transformer_layers": num_transformer_layers,
            "max_seq_length": max_seq_length,
        }
        self.downsampling_network = DownsamplingNetwork(
            embedding_dim=embedding_dim,
            hidden_dim=embedding_dim // 2,
            strides=strides,
            initial_mean_pooling_kernal_size=initial_mean_pooling_kernal_size,
        )
        self.pre_rvq_transformer = Transformer(
            embedding_dim,
            num_layers=num_transformer_layers,
            max_seq_length=max_seq_length,
        )
        self.rvq = ResidualVectorQuantizer(num_codebooks, codebook_size, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)  # add channel dim for Conv1d based encoder
        x = self.downsampling_network(x)
        x = self.pre_rvq_transformer(x)
        x, loss = self.rvq(x)
        x = self.output_layer(x)
        x = torch.log_softmax(x, dim=-1)
        return x, loss

    def save(self, path: str):
        print("Saving model to", path)
        torch.save({"model": self.state_dict(), "options": self.options}, path)

    @classmethod
    def load(
        cls,
        path: str,
        map_location: Optional[Union[str, torch.device]] = None,
    ):
        print("Loading model from", path)
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(**checkpoint["options"])
        model.load_state_dict(checkpoint["model"])
        return model

if __name__ == "__main__":
    model = TranscribeModel(
        num_codebooks=3,
        codebook_size=64,
        embedding_dim=64,
        vocab_size=100,
        strides=[6, 6, 8, 4, 2],
        initial_mean_pooling_kernal_size=4,
        num_transformer_layers=2,
        max_seq_length=2000,
    )
    x = torch.randn(2, 23768)
    logits, aux_loss = model(x)
    print(logits.shape, aux_loss.item())