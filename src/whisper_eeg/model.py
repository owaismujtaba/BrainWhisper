"""
EEG-to-Text model using Whisper decoder with EEG encoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import WhisperForConditionalGeneration, WhisperModel, WhisperConfig

class MultiScaleConv1d(nn.Module):
    """
    Priority 4: Multi-Scale Feature Extraction
    Applies parallel convolutions with different kernel sizes to capture 
    features at different frequency/temporal scales.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        
        # Split output channels among 3 branches
        self.out_channels = out_channels
        
        # Priority: Weighted distribution (30% High, 30% Mid, 40% Low)
        # Low frequencies (Delta/Theta) carry significant linguistic info in EEG
        self.branch1_channels = int(out_channels * 0.30)
        self.branch2_channels = int(out_channels * 0.30)
        self.branch3_channels = out_channels - (self.branch1_channels + self.branch2_channels)
        
        # Branch 1: Small kernel (High Frequency / Local details)
        self.branch1 = nn.Conv1d(in_channels, self.branch1_channels, kernel_size=3, padding=1)
        
        # Branch 2: Medium kernel (Mid Frequency / Alpha-Beta rhythms)
        # padding = (kernel - 1) / 2 = 7
        self.branch2 = nn.Conv1d(in_channels, self.branch2_channels, kernel_size=15, padding=7)
        
        # Branch 3: Large kernel (Low Frequency / Delta-Theta rhythms)
        # padding = 15
        self.branch3 = nn.Conv1d(in_channels, self.branch3_channels, kernel_size=31, padding=15)
        
    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        # Concatenate along channel dimension
        return torch.cat([b1, b2, b3], dim=1)


class EEGEncoder(nn.Module):
    """
    Encoder for EEG signals.
    Includes subsampling to reduce sequence length to match Whisper's expectations.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 768,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Convolutional Subsampling
        # Project and downsample: 2000 -> 1000
        # Kernel 3, Stride 2, Padding 1 preserves (L+2*1-3)/2 + 1 = (L-1)/2 + 1 approx L/2
        # Priority 3: Multi-Stage Subsampling
        # Instead of one aggressive stride-2 conv, we use a block that processes features first
        self.conv_subsample = nn.Sequential(
            # Layer 1: Multi-Scale Feature Extraction (replaces simple stride-1 conv)
            MultiScaleConv1d(input_dim, hidden_dim),
            nn.GELU(),
            # Layer 2: Downsampling (Stride 2)
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        
        # BN/LN
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
        self.positional_encoding = PositionalEncoding(hidden_dim, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        
        # Priority 2: Adapter Module for Domain Alignment
        # Gated MLP (concatenation-based variant similar to SwiGLU ideas)
        self.adapter_gate = nn.Linear(hidden_dim, hidden_dim)
        self.adapter_feat = nn.Linear(hidden_dim, hidden_dim)
        self.adapter_out = nn.Linear(hidden_dim, hidden_dim)
        self.adapter_ln = nn.LayerNorm(hidden_dim)
        
        # Output Normalization & Reach
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_scale = nn.Parameter(torch.tensor(1.0))
    
    def forward(self, eeg, mask=None):
        # eeg: (batch, seq_len, input_dim)
        # Conv1d expects (batch, channels, seq_len)
        x = eeg.transpose(1, 2)
        x = self.conv_subsample(x)
        x = x.transpose(1, 2) # Back to (batch, seq_len', hidden)
        
        # Apply LayerNorm
        x = self.layer_norm(x)
        
        x = self.positional_encoding(x)
        
        # Handle mask: Exact resizing to match feature map
        if mask is not None:
             # mask is (batch, seq_len_in) like (B, 2000)
             # x is (batch, seq_len_out, hidden) like (B, 1000, 768)
             
             # We need mask to be (batch, seq_len_out)
             # Use interpolate: needs (batch, channels, time)
             mask_float = mask.unsqueeze(1).float() # (B, 1, L_in)
             
             target_len = x.size(1)
             
             # Interpolate to exact output size using Linear + Threshold (more robust boundaries)
             mask_downsampled = torch.nn.functional.interpolate(
                 mask_float, 
                 size=target_len, 
                 mode='linear',
                 align_corners=False
             )
             
             mask = (mask_downsampled > 0.5).squeeze(1).long() # (B, L_out)
             
             src_key_padding_mask = (mask == 0)
        else:
            src_key_padding_mask = None
            
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        
        # Apply Adapter (Gated MLP)
        gate = torch.sigmoid(self.adapter_gate(x))
        feat = self.adapter_feat(x)
        x = self.adapter_out(gate * feat)
        x = self.adapter_ln(x)
        
        # Priority 6: Adaptive Normalization & Statistics Matching
        # Final safeguard to ensure features match Whisper's expected distribution (Unit Variance)
        # We add a learnable scalar to allow the model to scale the entire feature map if needed
        x = self.output_norm(x) * self.output_scale
        
        # Return both features and the downsampled mask (for the decoder)
        return x, mask


class EEGWhisperModel(nn.Module):
    # ... __init__ is unchanged ...
    
    def __init__(
        self,
        whisper_model_name: str = "openai/whisper-tiny",
        eeg_input_dim: int = 512,
        encoder_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        freeze_decoder: bool = False,
    ):
        super().__init__()
        
        print(f"Loading Whisper model: {whisper_model_name}")
        self.whisper = WhisperForConditionalGeneration.from_pretrained(whisper_model_name)
        
        if freeze_decoder:
            print("Freezing Whisper Decoder parameters (except Cross-Attention)...")
            
            # 1. Freeze everything first
            for param in self.whisper.parameters():
                param.requires_grad = False
                
            # 2. Unfreeze Cross-Attention layers
            # These are the layers that attend to our EEG Encoder outputs
            # We need them to adapt to the new modality
            for name, param in self.whisper.named_parameters():
                if "encoder_attn" in name:
                    param.requires_grad = True
                    # print(f"  Unfrozen: {name}") # Optional debug output
        
        self.whisper_hidden_size = self.whisper.config.d_model
        
        self.eeg_encoder = EEGEncoder(
            input_dim=eeg_input_dim,
            hidden_dim=self.whisper_hidden_size,
            num_layers=encoder_layers,
            num_heads=num_heads, 
            dropout=dropout,
        )
        
        # CTC Head for auxiliary loss
        # Projects encoder features (hidden_dim) to vocabulary size
        self.ctc_head = nn.Linear(self.whisper_hidden_size, self.whisper.config.vocab_size)
        
    def forward(self, eeg, labels=None, eeg_mask=None):
        encoder_hidden_states, downsampled_mask = self.eeg_encoder(eeg, mask=eeg_mask)
        
        outputs = self.whisper(
            encoder_outputs=(encoder_hidden_states,),
            attention_mask=downsampled_mask, # Pass mask for encoder outputs
            labels=labels,
        )
        
        # Compute CTC Logits
        ctc_logits = self.ctc_head(encoder_hidden_states) # (Batch, Seq, Vocab)
        # outputs.ctc_logits = ctc_logits # DEPRECATED: Failed with DataParallel
        
        return outputs, ctc_logits, downsampled_mask

    def generate(self, eeg, eeg_mask=None, max_length=25, **kwargs):
        encoder_hidden_states, downsampled_mask = self.eeg_encoder(eeg, mask=eeg_mask)
        
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden_states)
        
        # Prepare Generation Config
        generation_config = self.whisper.generation_config
        
        # Update with kwargs
        gen_kwargs = kwargs.copy()
        
        # Handle timestamp/language/task overrides
        if 'return_timestamps' not in gen_kwargs:
            gen_kwargs['return_timestamps'] = False
            
        if 'language' not in gen_kwargs:
            gen_kwargs['language'] = "en"
        if 'task' not in gen_kwargs:
            gen_kwargs['task'] = "transcribe"
            
        if 'forced_decoder_ids' not in gen_kwargs:
            gen_kwargs['forced_decoder_ids'] = None
            
        # FORCE ENGLISH: 
        # If language is specified, we should ensure forced_decoder_ids reflect that.
        # But specifically, if forced_decoder_ids is None, we can create them for English.
        if gen_kwargs['forced_decoder_ids'] is None:
             # Use the processor/tokenizer logic if available, or manually set
             # <|startoftranscript|><|en|><|transcribe|>
             # For Whisper, the model usually has a method or we can use the tokenizer.
             # Ideally: processor.get_decoder_prompt_ids(language="en", task="transcribe")
             # But we don't have the processor here, only the tokenizer might be outside.
             # However, the model configuration has lang/task.
             
             # Actually, simpler: pass language='en' and task='transcribe' to generate,
             # and let the model build forced_decoder_ids internally IF we don't set them.
             # So ensure they are NOT None if we want to override, OR ensure lang/task are set.
             pass 

        if 'repetition_penalty' not in gen_kwargs:
            gen_kwargs['repetition_penalty'] = 1.2
        if 'no_repeat_ngram_size' not in gen_kwargs:
            gen_kwargs['no_repeat_ngram_size'] = 3

        # Explicitly disable implied processor creation to avoid conflicts
        if 'suppress_tokens' not in gen_kwargs:
            gen_kwargs['suppress_tokens'] = None
        if 'begin_suppress_tokens' not in gen_kwargs:
            gen_kwargs['begin_suppress_tokens'] = None
            
        generated_ids = self.whisper.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=downsampled_mask, # Pass mask here too
            max_length=max_length,
            # Force language arguments on the generate call
            language=gen_kwargs.get('language', 'en'),
            task=gen_kwargs.get('task', 'transcribe'),
            **{k: v for k, v in gen_kwargs.items() if k not in ['language', 'task']}
        )
        
        return generated_ids


class PositionalEncoding(nn.Module):
    """Standard positional encoding"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        import math
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Safer handling for odd dimensions
        if d_model % 2 == 0:
             pe[:, 1::2] = torch.cos(position * div_term)
        else:
             # Take exactly the number of terms needed (d_model // 2)
             pe[:, 1::2] = torch.cos(position * div_term[:d_model//2])
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


