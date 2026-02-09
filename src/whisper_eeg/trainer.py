"""
Training logic for EEG-to-Text model
"""
import torch
import torch.nn as nn
from torch.optim import AdamW

from tqdm import tqdm
import json
from pathlib import Path
from .config import Config

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        config: Config,
        tokenizer=None,
    ):
        self.config = config
        self.device = config.training.device
        
        # CPU/GPU logic
        if self.device == 'cuda' and not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            self.device = 'cpu'
            
        self.model = model.to(self.device)
        
        # Multi-GPU support
        self.is_parallel = False
        if self.device == 'cuda' and torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            self.model = nn.DataParallel(self.model)
            self.is_parallel = True
            
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = config.training.num_epochs
        self.tokenizer = tokenizer
        self.use_whisper = config.model.use_whisper
        
        # Optimizer and scheduler
        # Filter frozen parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
        from transformers import get_linear_schedule_with_warmup
        
        # Scheduler with Warmup
        # 10% warmup is a safe default for Transformers
        total_steps = len(self.train_loader) * self.num_epochs
        warmup_steps = int(0.1 * total_steps)
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # Legacy Loss function (only for custom model)
        self.criterion = nn.CrossEntropyLoss(reduction='mean', label_smoothing=0.1)
        
        self.checkpoint_dir = Path(config.training.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_wers': [], # Store Word Error Rates if possible, or just loss
        }
        
        self.best_val_loss = float('inf')
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            ctc_loss = 0.0
            eeg = batch['eeg'].to(self.device)
            eeg_mask = batch.get('eeg_mask').to(self.device) if 'eeg_mask' in batch else None
            
            if self.use_whisper:
                # Whisper Model
                labels = batch['labels'].to(self.device)
                
                # Forward pass - returns (outputs, ctc_logits, downsampled_mask)
                outputs, ctc_logits, downsampled_mask = self.model(eeg, labels=labels, eeg_mask=eeg_mask)
                
                # Manual loss computation with Label Smoothing
                logits = outputs.logits
                loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
                
                # Reshape for loss
                loss_whisper = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                # CTC Auxiliary Loss
                ctc_loss = 0.0
                if ctc_logits is not None and self.config.training.ctc_weight > 0:
                    # CTCLoss expects (Input_Seq_Len, Batch, Vocab) (if batch_first=False)
                    # or (Batch, Input_Seq_Len, Vocab) (if batch_first=True)
                    # PyTorch default CTCLoss is batch_first=False usually, let's check or permute
                     
                    ctc_logits = ctc_logits.log_softmax(dim=2)
                    
                    # Prepare lengths
                    # Output Mask length (downsampled_mask matches encoder output length)
                    if downsampled_mask is not None:
                        input_lengths = downsampled_mask.sum(dim=1).long() # Length after conv subsampling
                    else:
                        input_lengths = torch.full((eeg.size(0),), ctc_logits.size(1), device=self.device, dtype=torch.long)
                        
                    # Prepare targets for CTC
                    # Remove padding (-100) and 0s
                    ctc_targets = []
                    target_lengths = []
                    for label in labels:
                        # Filter out -100 and pad tokens
                        mask = (label != -100) & (label != self.tokenizer.pad_token_id)
                        filtered = label[mask]
                        ctc_targets.append(filtered)
                        target_lengths.append(len(filtered))
                    
                    ctc_targets = torch.cat(ctc_targets)
                    target_lengths = torch.tensor(target_lengths, device=self.device, dtype=torch.long)
                    
                    ctc_criterion = nn.CTCLoss(blank=self.tokenizer.pad_token_id, zero_infinity=True)
                    
                    # Permute for CTCLoss (Time, Batch, Vocab)
                    ctc_logits = ctc_logits.permute(1, 0, 2)
                    
                    ctc_loss = ctc_criterion(ctc_logits, ctc_targets, input_lengths, target_lengths)
                    
                loss = loss_whisper + self.config.training.ctc_weight * ctc_loss
                
            else:
                # Legacy Custom Model
                text = batch['text'].to(self.device) # Legacy field
                text_mask = batch['text_mask'].to(self.device)
                
                text_input = text[:, :-1]
                text_target = text[:, 1:]
                text_mask_input = text_mask[:, :-1]
                
                logits = self.model(eeg, text_input, eeg_mask, text_mask_input)
                loss = self.criterion(
                    logits.reshape(-1, self.model.module.vocab_size if self.is_parallel else self.model.vocab_size),
                    text_target.reshape(-1)
                )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item(), 'ctc': ctc_loss.item() if isinstance(ctc_loss, torch.Tensor) else ctc_loss})
        
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        # Metrics
        import jiwer
        predictions = []
        references = []
        
        pbar = tqdm(self.val_loader, desc='Validating')
        for i, batch in enumerate(pbar):
            eeg = batch['eeg'].to(self.device)
            eeg_mask = batch.get('eeg_mask').to(self.device) if 'eeg_mask' in batch else None
            
            if self.use_whisper:
                labels = batch['labels'].to(self.device)
                
                # 1. Compute Loss
                outputs, ctc_logits, _ = self.model(eeg, labels=labels, eeg_mask=eeg_mask)
                loss = outputs.loss.mean()
                
                # 2. Generate for Metrics (batched)
                # Note: DataParallel wrapper handles generate if call forward/generate correctly
                # But typically we access module.generate
                model_to_call = self.model.module if self.is_parallel else self.model
                
                
                # Use max length from config
                max_gen_len = getattr(self.config.model, 'max_generation_length', 100)
                generated_ids = model_to_call.generate(
                    eeg, 
                    eeg_mask=eeg_mask, 
                    max_length=max_gen_len,
                    repetition_penalty=1.2,
                    no_repeat_ngram_size=3
                )
                decoded_preds = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                
                # Decode references (labels)
                # Replace -100 with pad token id to decode correctly
                labels[labels == -100] = self.tokenizer.pad_token_id
                decoded_refs = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                # --- VISUALIZE CTC PREDICTIONS (DEBUG) ---
                if i == 0 and ctc_logits is not None:
                     # Take argmax of first sample in batch
                     ctc_pred_ids = torch.argmax(ctc_logits[0], dim=-1)
                     # Simple greedy decode: just decode all tokens (including blanks/repeats) to see raw output
                     # Ideally we should collapse repeats, but raw is useful to see "what is generated"
                     ctc_text_raw = self.tokenizer.decode(ctc_pred_ids)
                     print(f"\n[DEBUG] Sample CTC (Raw): {ctc_text_raw[:200]}...") # truncate
                     print(f"[DEBUG] Reference: {decoded_refs[0][:200]}...")
                # -----------------------------------------
                
                predictions.extend(decoded_preds)
                references.extend(decoded_refs)
                
            else:
                text = batch['text'].to(self.device)
                text_mask = batch['text_mask'].to(self.device)
                
                text_input = text[:, :-1]
                text_target = text[:, 1:]
                text_mask_input = text_mask[:, :-1]
                
                logits = self.model(eeg, text_input, eeg_mask, text_mask_input)
                loss = self.criterion(
                    logits.reshape(-1, self.model.module.vocab_size if self.is_parallel else self.model.vocab_size),
                    text_target.reshape(-1)
                )
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(self.val_loader)
        
        # Calculate WER/CER
        wer = 0.0
        cer = 0.0
        if predictions and references:
            # Create transformation to normalize text
            # Jiwer 4.0+ simplifies this, but to be safe we can use simple transforms or just raw
            # Let's try to match standard normalization
            transformation = jiwer.Compose([
                jiwer.ToLowerCase(),
                jiwer.RemovePunctuation(),
                jiwer.RemoveMultipleSpaces(),
                jiwer.Strip(),
            ]) 
            
            try:
                # jiwer.wer/cer can take truth and hypothesis directly
                # We can apply transformation manually if needed, or pass it to some versions
                # For safety with 4.0, let's just apply transform to lists
                
                refs_norm = [transformation(r) for r in references]
                preds_norm = [transformation(p) for p in predictions]
                
                wer = jiwer.wer(refs_norm, preds_norm)
                cer = jiwer.cer(refs_norm, preds_norm)
                
                # Debug: Print first 3 comparisons
                print("\n--- WER Debug ---")
                for i in range(min(3, len(refs_norm))):
                    print(f"Ref:  {refs_norm[i]}")
                    print(f"Pred: {preds_norm[i]}")
                    print("-" * 20)
                
                # Check for empty predictions
                empty_preds = sum(1 for p in preds_norm if not p.strip())
                if empty_preds > 0:
                    print(f"WARNING: {empty_preds}/{len(preds_norm)} predictions were empty!")
                    
            except Exception as e:
                print(f"Error calculating metrics: {e}")
        
        return avg_loss, wer, cer
    
    def train(self):
        print(f"Starting training for {self.num_epochs} epochs")
        print(f"Device: {self.device}")
        
        for epoch in range(self.num_epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"{'='*50}")
            
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            print(f"Train Loss: {train_loss:.4f}")
            
            if self.val_loader is not None:
                val_loss, val_wer, val_cer = self.validate()
                self.history['val_loss'].append(val_loss)
                self.history['val_wers'].append(val_wer)
                
                print(f"Val Loss: {val_loss:.4f} | WER: {val_wer:.4f} | CER: {val_cer:.4f}")
                
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(epoch, is_best=True)
            
            self.scheduler.step()
            
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(epoch)
            
            self.save_history()
    
    def save_checkpoint(self, epoch, is_best=False):
        # Unwrap model if DataParallel to save clean state_dict
        model_state = self.model.module.state_dict() if self.is_parallel else self.model.state_dict()
        
        checkpoint = {
            'epoch': epoch,
            'model_state': model_state,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config.model.__dict__ # Save all model params (layers, dims, etc.)
        }
        
        if is_best:
            path = self.checkpoint_dir / 'best_model.pt'
            print(f"Saving best model to {path}")
        else:
            path = self.checkpoint_dir / f'checkpoint_epoch_{epoch+1:03d}.pt'
        
        # Retry logic for saving
        max_retries = 3
        for i in range(max_retries):
            try:
                torch.save(checkpoint, path)
                break
            except Exception as e:
                print(f"Save failed (attempt {i+1}/{max_retries}): {e}")
                if i == max_retries - 1:
                    print("CRITICAL: Failed to save checkpoint after multiple attempts.")
                else:
                    import time
                    time.sleep(1) # Wait a bit before retrying
    
    def save_history(self):
        history_file = self.checkpoint_dir / 'history.json'
        with open(history_file, 'w') as f:
            json.dump(self.history, f, indent=2)
