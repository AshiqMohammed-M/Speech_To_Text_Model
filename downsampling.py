import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class ResidualDownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, kernel_size=4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding="same")
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: (batch_size, in_channels, seq_len)

        output = self.conv1(x)
        output = self.bn1(output)
        output = self.relu(output) + x # Residual Connection
        output = self.conv2(output)
        return output
    
class DownsamplingNetwork(nn.Module):
    def __init__(
            self,
            embedding_dim=120,
            hidden_dim=64,
            in_channels=1,
            initial_mean_pooling_kernal_size=2,
            strides=[6,6,8,4,2],
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        self.mean_pooling = nn.MaxPool1d(kernel_size=initial_mean_pooling_kernal_size)

        for i in range(len(strides)):
            self.layers.append(
                ResidualDownSampleBlock(
                    hidden_dim if i > 0 else in_channels,
                    hidden_dim,
                    strides[i],
                    kernel_size=8,
                )
            )
        self.final_conv = nn.Conv1d(
            hidden_dim, embedding_dim, kernel_size=4, padding="same"
        )

    def forward(self, x):
        x = self.mean_pooling(x)
        for layer in self.layers:
            x = layer(x)

        x = self.final_conv(x)
        x = x.transpose(1, 2)
        return x
    

if __name__=="__main__":
    # block = ResidualDownSampleBlock(1, 64, 2)
    # x = torch.randn(2, 1, 100)
    # print(block(x).shape)
    pass