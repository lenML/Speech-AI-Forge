import typing as tp
import torch
import torch.nn as nn


class ConvLayer(nn.Module):
    def __init__(self, 
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int,
                 stride:int,
                 activation:str="GELU",
                 dropout_rate:float=0.0,
                 ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size-stride)//2,
        )
        self.drop = nn.Dropout(dropout_rate)
        self.norm = nn.LayerNorm(out_channels)
        self.activ = getattr(nn, activation)()

    def forward(self, x:torch.Tensor):
        """
        Args:
            x: (b, t, c)
        Return:
            x: (b, t, c)
        """
        x = x.transpose(2, 1)
        x = self.conv(x)
        x = x.transpose(2, 1)
        x = self.drop(x)
        x = self.norm(x)
        x = self.activ(x)
        return x


class ResidualConvLayer(nn.Module):
    def __init__(self,
                 hidden_channels:int,
                 n_layers:int=2,
                 kernel_size:int=5,
                 activation:str="GELU",
                 dropout_rate:float=0.0,
                 ):
        super().__init__()
        layers = [
            ConvLayer(hidden_channels, hidden_channels, kernel_size, 1, activation, dropout_rate)
            for _ in range(n_layers)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x:torch.Tensor):
        """
        Args:
            x: (b, t, c)
        Returns:
            x: (b, t, c)
        """
        return x + self.layers(x)


class ResidualConvBlock(nn.Module):
    def __init__(self, 
                 in_channels:int,
                 hidden_channels:int,
                 out_channels:int,
                 n_layers:int=2,
                 n_blocks:int=5,
                 middle_layer:tp.Optional[nn.Module]=None,
                 kernel_size:int=5,
                 activation:str="GELU",
                 dropout_rate:float=0.0,
                 ):
        super().__init__()
        self.in_proj = nn.Conv1d(
            in_channels,
            hidden_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size-1)//2,
        ) if in_channels != hidden_channels else nn.Identity()

        self.conv1 = nn.Sequential(*[
            ResidualConvLayer(hidden_channels, n_layers, kernel_size, activation, dropout_rate)
            for _ in range(n_blocks)
        ])

        if middle_layer is None:
            self.middle_layer = nn.Identity()
        elif isinstance(middle_layer, nn.Module):
            self.middle_layer = middle_layer
        else:
            raise TypeError("unknown middle layer type:{}".format(type(middle_layer)))
        
        self.conv2 = nn.Sequential(*[
            ResidualConvLayer(hidden_channels, n_layers, kernel_size, activation, dropout_rate)
            for _ in range(n_blocks)
        ])

        self.out_proj = nn.Conv1d(
            hidden_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=(kernel_size-1)//2,
        ) if out_channels != hidden_channels else nn.Identity()

    def forward(self, x:torch.Tensor, **middle_layer_kwargs):
        """
        Args:
            x: (b, t1, c)
        Return:
            x: (b, t2, c)
        """
        x = self.in_proj(x.transpose(2, 1)).transpose(2, 1)
        x = self.conv1(x)
        if isinstance(self.middle_layer, nn.MaxPool1d) or isinstance(self.middle_layer, nn.Conv1d):
            x = self.middle_layer(x.transpose(2, 1)).transpose(2, 1)
        elif isinstance(self.middle_layer, nn.Identity):
            x = self.middle_layer(x)
        else:
            # incase of phoneme-pooling layer
            x = self.middle_layer(x, **middle_layer_kwargs)
        x = self.conv2(x)
        x = self.out_proj(x.transpose(2, 1)).transpose(2, 1)
        return x


class MelReduceEncoder(nn.Module):
    def __init__(self, 
                 in_channels:int,
                 out_channels:int,
                 hidden_channels:int=384,
                 reduction_rate:int=4,
                 n_layers:int=2,
                 n_blocks:int=5,
                 kernel_size:int=3,
                 activation:str="GELU",
                 dropout:float=0.0,
                 ):
        super().__init__()
        self.reduction_rate = reduction_rate
        middle_conv = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=reduction_rate,
            stride=reduction_rate,
            padding=0
        )
        self.encoder = ResidualConvBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            n_layers=n_layers,
            n_blocks=n_blocks,
            middle_layer=middle_conv,
            kernel_size=kernel_size,
            activation=activation,
            dropout_rate=dropout
        )
    
    def forward(self, x:torch.Tensor):
        return self.encoder(x)

