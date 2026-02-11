import torch
import torch.nn as nn
import torch.nn.functional as F

class EEGEncoder(nn.Module):
    """
    EEG Encoder to map raw EEG signals to Whisper's embedding space.
    
    Architecture:
    1. 1D Convolutional Neural Network (CNN) for feature extraction.
    2. Optional Transformer Encoder or LSTM for temporal modeling.
    3. Adapter Layer to project to Whisper's embedding dimension.
    """
    def __init__(
        self, 
        input_channels=64, 
        output_dim=512,  # Whisper 'base' model hidden size
        hidden_dim=256,
        kernel_size=3,
        stride=1,
        num_layers=4,
        dropout=0.1
    ):
        super(EEGEncoder, self).__init__()
        
        self.input_channels = input_channels
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # 1. Feature Extraction (CNN)
        layers = []
        in_dim = input_channels
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_dim, hidden_dim, kernel_size=kernel_size, stride=stride, padding=kernel_size//2))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
            
        self.cnn_encoder = nn.Sequential(*layers)
        
        # 2. Temporal Modeling (Simple Transformer Encoder)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. Adapter Layer (Projection to Whisper dim)
        self.adapter = nn.Linear(hidden_dim, output_dim)
        
        # Optional: Learnable scaling for adaptive normalization (mentioned in methodology 2.3)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """
        Args:
            x: Input EEG tensor of shape (Batch, Time, Channels) from dataset
            
        Returns:
            Embeddings of shape (Batch, Time, Output_Dim)
        """
        # Dataset returns (B, T, C), but CNN expects (B, C, T)
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)
        
        # CNN expects (Batch, Channels, Time)
        x = self.cnn_encoder(x)
        
        # Transformer expects (Batch, Time, Channels/Features)
        x = x.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
        
        x = self.transformer_encoder(x)
        
        # Project to output dimension
        x = self.adapter(x)
        
        # Apply adaptive scaling
        x = x * self.scale + self.bias
        
        return x

if __name__ == "__main__":
    # Test the model
    batch_size = 2
    channels = 64
    time_steps = 1000
    
    model = EEGEncoder(input_channels=channels, output_dim=512)
    input_tensor = torch.randn(batch_size, channels, time_steps)
    output = model(input_tensor)
    
    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
