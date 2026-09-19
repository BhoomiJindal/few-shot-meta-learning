import torch.nn as nn


def conv_block(in_channels, out_channels):
    """
    One block of the standard Conv-4 backbone used across few-shot learning
    literature: 3x3 convolution -> batch norm -> ReLU -> 2x2 max pool.

    Batch norm stabilizes training by normalizing activations within each
    batch. ReLU introduces non-linearity (without it, stacking convolutions
    would collapse into one big linear operation, no more powerful than a
    single layer). Max pooling halves the spatial size, which is why 4 of
    these blocks shrink 84x84 down to roughly 5x5 by the end.
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=2),
    )


class Conv4Backbone(nn.Module):
    """
    Standard 4-layer convolutional embedding network used across few-shot
    learning literature (Vinyals et al. 2016; Snell et al. 2017; Finn et al.
    2017). Maps an input image of shape (3, 84, 84) to a flat embedding
    vector, used as the shared feature extractor for both MAML and
    Prototypical Networks in this project.
    """

    def __init__(self, in_channels=3, hidden_channels=64):
        super().__init__()
        self.block1 = conv_block(in_channels, hidden_channels)
        self.block2 = conv_block(hidden_channels, hidden_channels)
        self.block3 = conv_block(hidden_channels, hidden_channels)
        self.block4 = conv_block(hidden_channels, hidden_channels)

    def forward(self, x):
        x = self.block1(x)   # (batch, 64, 42, 42)
        x = self.block2(x)   # (batch, 64, 21, 21)
        x = self.block3(x)   # (batch, 64, 10, 10)
        x = self.block4(x)   # (batch, 64, 5, 5)
        x = x.view(x.size(0), -1)  # flatten to (batch, 64*5*5) = (batch, 1600)
        return x

    @property
    def output_dim(self):
        return 64 * 5 * 5  # 1600