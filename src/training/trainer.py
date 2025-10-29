"""
Training pipeline for MoE cryptocurrency prediction model with LoRA optimization.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import json
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# LoRA imports
from peft import LoraConfig, get_peft_model, TaskType

import sys
sys.path.append(str(Path(__file__).parent.parent))
from models.moe_model import MoECryptoPredictor, MoELoss
from data.preprocessor import CryptoDataPreprocessor
from utils.device_manager import setup_device_from_config


class CryptoDataset(Dataset):
    """
    Dataset class for multi-timeframe cryptocurrency data.
    """
    
    def __init__(self, 
                 data_dict: Dict[str, Dict[str, np.ndarray]],
                 timeframes: List[str]):
        """
        Initialize dataset.
        
        Args:
            data_dict: Dictionary mapping timeframes to processed data
            timeframes: List of timeframe identifiers
        """
        self.data_dict = data_dict
        self.timeframes = timeframes
        
        # Find common length (minimum across all timeframes)
        lengths = []
        for tf in timeframes:
            if tf in data_dict and 'X' in data_dict[tf]:
                lengths.append(len(data_dict[tf]['X']))
        
        if not lengths:
            raise ValueError("No valid data found")
        
        self.length = min(lengths)
        print(f"Dataset length: {self.length} (limited by shortest timeframe)")
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Get item by index.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (inputs, targets)
        """
        inputs = {}
        targets = {}
        
        for tf in self.timeframes:
            if tf in self.data_dict and idx < len(self.data_dict[tf]['X']):
                # Input features
                inputs[tf] = torch.FloatTensor(self.data_dict[tf]['X'][idx])
                
                # Targets (price_change, direction, volatility, magnitude, percentile)
                y = self.data_dict[tf]['y'][idx]
                if tf not in targets:  # Use first timeframe as primary target
                    targets['price_change'] = torch.FloatTensor([y[0]])
                    targets['direction'] = torch.LongTensor([y[1]])
                    targets['volatility'] = torch.FloatTensor([y[2]])
                    targets['magnitude'] = torch.FloatTensor([y[3]])
                    targets['percentile'] = torch.FloatTensor([y[4]])
        
        return inputs, targets


class MoETrainer:
    """
    Trainer class for MoE cryptocurrency prediction model.
    """
    
    def __init__(self,
                 model: MoECryptoPredictor,
                 train_dataset: CryptoDataset,
                 val_dataset: CryptoDataset,
                 config: Dict[str, Any]):
        """
        Initialize trainer.
        
        Args:
            model: MoE model to train
            train_dataset: Training dataset
            val_dataset: Validation dataset
            config: Training configuration
        """
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Device setup with automatic detection
        self.device, self.device_config = setup_device_from_config({'training': config})
        self.model.to(self.device)
        
        # Data loaders with device-optimized settings
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.device_config['batch_size'],
            shuffle=True,
            num_workers=self.device_config['num_workers'],
            pin_memory=self.device_config['pin_memory']
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.device_config['batch_size'],
            shuffle=False,
            num_workers=self.device_config['num_workers'],
            pin_memory=self.device_config['pin_memory']
        )
        
        # Loss function with new consistency and Huber loss support
        self.criterion = MoELoss(
            price_weight=config.get('price_weight', 1.0),
            direction_weight=config.get('direction_weight', 0.5),
            consistency_weight=config.get('consistency_weight', 0.0),  # NEW: по умолчанию выключено
            volatility_weight=config.get('volatility_weight', 0.3),
            magnitude_weight=config.get('magnitude_weight', 0.2),
            percentile_weight=config.get('percentile_weight', 0.2),
            diversity_weight=config.get('diversity_weight', 0.1),
            use_huber_loss=config.get('use_huber_loss', False)  # NEW: по умолчанию MSE
        )
        
        # Gradient accumulation
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        
        # Optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Scheduler
        if config.get('scheduler') == 'cosine':
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=config['num_epochs'],
                eta_min=config.get('min_lr', 1e-6)
            )
        elif config.get('scheduler') == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=config.get('patience', 10),
                verbose=True
            )
        else:
            self.scheduler = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_price_loss': [],
            'val_price_loss': [],
            'train_direction_acc': [],
            'val_direction_acc': [],
            'learning_rate': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        
        # Output directory
        self.output_dir = Path(config.get('output_dir', './models'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Starting epoch (for resume training)
        self.start_epoch = 0
    
    def load_checkpoint(self, checkpoint_path: str) -> None:
        """
        Load checkpoint to resume training.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        print(f"Loading checkpoint from {checkpoint_path}...")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model state loaded from epoch {checkpoint['epoch']}")
        
        # Load optimizer state (with error handling for config changes)
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("✅ Optimizer state loaded")
        except (ValueError, KeyError) as e:
            print(f"⚠️  Could not load optimizer state (likely due to config change): {e}")
            print("⚠️  Optimizer will start fresh (this is OK, training will continue)")
        
        # Load scheduler state if exists
        if self.scheduler and 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("✅ Scheduler state loaded")
            except (ValueError, KeyError) as e:
                print(f"⚠️  Could not load scheduler state: {e}")
                print("⚠️  Scheduler will start fresh")
        
        # Load training history
        if 'history' in checkpoint:
            self.history = checkpoint['history']
            print(f"✅ Training history loaded ({len(self.history['train_loss'])} epochs)")
        
        # Load best validation loss
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
            print(f"✅ Best validation loss: {self.best_val_loss:.6f}")
        
        # Set starting epoch
        self.start_epoch = checkpoint['epoch'] + 1
        print(f"✅ Will resume from epoch {self.start_epoch}")
        print(f"ℹ️  Note: Training will continue but optimizer/scheduler may have reset due to config changes")
    
    def apply_lora(self, 
                   target_modules: List[str] = None,
                   r: int = 16,
                   lora_alpha: int = 32,
                   lora_dropout: float = 0.1) -> None:
        """
        Apply LoRA (Low-Rank Adaptation) to the model.
        
        Args:
            target_modules: List of module names to apply LoRA to
            r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
        """
        # First, let's find the actual module names in our model
        print("🔍 Analyzing model structure for LoRA...")
        lora_targets = []
        
        for name, module in self.model.named_modules():
            if 'linear' in name.lower() or 'proj' in name.lower():
                if isinstance(module, torch.nn.Linear):
                    lora_targets.append(name)
                    print(f"   Found Linear layer: {name}")
        
        if not lora_targets:
            print("⚠️ No suitable Linear layers found for LoRA. Skipping LoRA application.")
            return
        
        # Use found targets or provided ones
        if target_modules is None:
            target_modules = lora_targets[:10]  # Limit to first 10 to avoid too many
        
        print(f"🎯 Applying LoRA to {len(target_modules)} modules:")
        for module in target_modules:
            print(f"   - {module}")
        
        try:
            # For custom models, we'll implement a simpler LoRA-like approach
            print("⚠️ LoRA with PEFT library is not compatible with our custom MoE architecture.")
            print("🔧 Implementing manual parameter reduction instead...")
            
            # Freeze most parameters and only train a subset
            total_params = 0
            trainable_params = 0
            
            for name, param in self.model.named_parameters():
                total_params += param.numel()
                
                # Only train final layers and some expert layers
                if any(keyword in name.lower() for keyword in ['price_head', 'direction_head', 'volatility_head', 'final_', 'gating']):
                    param.requires_grad = True
                    trainable_params += param.numel()
                else:
                    param.requires_grad = False
            
            print(f"📊 Parameter reduction applied:")
            print(f"   Total parameters: {total_params:,}")
            print(f"   Trainable parameters: {trainable_params:,}")
            print(f"   Reduction ratio: {trainable_params/total_params:.2%}")
            
            # Update optimizer to only train unfrozen parameters
            self.optimizer = AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config['learning_rate'],
                weight_decay=self.config.get('weight_decay', 0.01)
            )
            
            print("✅ Parameter reduction successfully applied!")
            
        except Exception as e:
            print(f"❌ Failed to apply parameter reduction: {e}")
            print("Continuing with full model training...")
    
    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        
        total_loss = 0
        total_price_loss = 0
        total_direction_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        
        # Debug: Check first batch data range
        first_batch = True
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            targets = {k: v.to(self.device) for k, v in targets.items()}
            
            # Debug: Print first batch statistics
            if first_batch:
                print(f"\n[*] First batch diagnostics:")
                print(f"   Price change range: [{targets['price_change'].min():.6f}, {targets['price_change'].max():.6f}]")
                print(f"   Price change mean: {targets['price_change'].mean():.6f}, std: {targets['price_change'].std():.6f}")
                print(f"   Volatility range: [{targets['volatility'].min():.6f}, {targets['volatility'].max():.6f}]")
                print(f"   Direction: {targets['direction'].float().mean():.2f} (0.5 = balanced)")
                first_batch = False
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Compute loss
            loss_dict = self.criterion(outputs, targets)
            loss = loss_dict['total_loss']
            
            # Scale loss for gradient accumulation
            loss = loss / self.gradient_accumulation_steps
            
            # Debug: Print first batch loss components
            if batch_idx == 0 and hasattr(self, '_first_epoch_logged') is False:
                print(f"\n📊 Loss components (first batch):")
                for key, value in loss_dict.items():
                    if key != 'total_loss':
                        print(f"   {key}: {value.item():.2f}")
                print(f"   total_loss: {loss_dict['total_loss'].item():.2f}")
                print(f"   Loss weights: price={self.criterion.price_weight}, dir={self.criterion.direction_weight}, consistency={self.criterion.consistency_weight}, vol={self.criterion.volatility_weight}")
                print(f"   Huber Loss: {self.criterion.use_huber_loss}")
                print(f"   Gradient accumulation steps: {self.gradient_accumulation_steps}\n")
                self._first_epoch_logged = True
            
            # Backward pass
            loss.backward()
            
            # Perform optimizer step every N accumulation steps
            if (batch_idx + 1) % self.gradient_accumulation_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                # Gradient clipping
                if self.config.get('grad_clip'):
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['grad_clip']
                    )
                
                self.optimizer.step()
                self.optimizer.zero_grad()
            
            # Update metrics (use unscaled loss for metrics)
            total_loss += loss_dict['total_loss'].item()
            if 'price_loss' in loss_dict:
                total_price_loss += loss_dict['price_loss'].item()
            
            # Direction accuracy
            if 'direction_logits' in outputs and 'direction' in targets:
                pred_direction = torch.argmax(outputs['direction_logits'], dim=1)
                correct = (pred_direction == targets['direction'].squeeze()).sum().item()
                total_direction_correct += correct
            
            total_samples += targets['price_change'].size(0)
            
            # Update progress bar (show unscaled loss)
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        return {
            'train_loss': total_loss / len(self.train_loader),
            'train_price_loss': total_price_loss / len(self.train_loader),
            'train_direction_acc': total_direction_correct / total_samples
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        
        total_loss = 0
        total_price_loss = 0
        total_direction_correct = 0
        total_samples = 0
        
        all_price_preds = []
        all_price_targets = []
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                targets = {k: v.to(self.device) for k, v in targets.items()}
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Compute loss
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['total_loss']
                
                # Update metrics
                total_loss += loss.item()
                if 'price_loss' in loss_dict:
                    total_price_loss += loss_dict['price_loss'].item()
                
                # Direction accuracy
                if 'direction_logits' in outputs and 'direction' in targets:
                    pred_direction = torch.argmax(outputs['direction_logits'], dim=1)
                    correct = (pred_direction == targets['direction'].squeeze()).sum().item()
                    total_direction_correct += correct
                
                # Collect predictions for R2 calculation
                if 'price_change' in outputs:
                    all_price_preds.extend(outputs['price_change'].cpu().numpy())
                    all_price_targets.extend(targets['price_change'].cpu().numpy())
                
                total_samples += targets['price_change'].size(0)
        
        # Calculate R2 score
        r2 = r2_score(all_price_targets, all_price_preds) if all_price_preds else 0
        
        return {
            'val_loss': total_loss / len(self.val_loader),
            'val_price_loss': total_price_loss / len(self.val_loader),
            'val_direction_acc': total_direction_correct / total_samples,
            'val_r2': r2
        }
    
    def train(self) -> None:
        """
        Main training loop.
        """
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        if self.start_epoch > 0:
            print(f"Resuming training from epoch {self.start_epoch}")
        
        start_time = time.time()
        
        try:
            for epoch in range(self.start_epoch, self.config['num_epochs']):
                print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}")
                
                # Train
                train_metrics = self.train_epoch()
                
                # Validate
                val_metrics = self.validate_epoch()
                
                # Update history
                self.history['train_loss'].append(train_metrics['train_loss'])
                self.history['val_loss'].append(val_metrics['val_loss'])
                self.history['train_price_loss'].append(train_metrics['train_price_loss'])
                self.history['val_price_loss'].append(val_metrics['val_price_loss'])
                self.history['train_direction_acc'].append(train_metrics['train_direction_acc'])
                self.history['val_direction_acc'].append(val_metrics['val_direction_acc'])
                self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
                
                # Print metrics
                print(f"Train Loss: {train_metrics['train_loss']:.4f}, "
                      f"Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"Train Dir Acc: {train_metrics['train_direction_acc']:.4f}, "
                      f"Val Dir Acc: {val_metrics['val_direction_acc']:.4f}")
                print(f"Val R2: {val_metrics['val_r2']:.4f}")
                
                # Save checkpoint after every epoch
                is_best = False
                if val_metrics['val_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['val_loss']
                    self.best_model_state = self.model.state_dict().copy()
                    is_best = True
                    print("[+] New best model!")
                
                # Save checkpoint (always, but mark if best)
                self.save_checkpoint(epoch, is_best=is_best)
                
                # Save training history and plots after each epoch
                self.save_training_history()
                self.plot_training_history()
                
                # Learning rate scheduling
                if self.scheduler:
                    if isinstance(self.scheduler, ReduceLROnPlateau):
                        self.scheduler.step(val_metrics['val_loss'])
                    else:
                        self.scheduler.step()
                
                # Early stopping
                if self.config.get('early_stopping'):
                    patience = self.config.get('early_stopping_patience', 20)
                    if epoch - self.get_best_epoch() > patience:
                        print(f"Early stopping after {patience} epochs without improvement")
                        break
        
        except KeyboardInterrupt:
            print("\n\n[!] Training interrupted by user (Ctrl+C)")
            print(f"[*] Saving checkpoint at epoch {epoch+1}...")
            self.save_checkpoint(epoch, is_best=False)
            print("[*] Checkpoint saved. You can resume training later.")
            raise  # Re-raise to propagate to main
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        
        # Load best model
        if self.best_model_state:
            self.model.load_state_dict(self.best_model_state)
        
        # Save final results
        self.save_training_history()
        self.plot_training_history()
    
    def get_best_epoch(self) -> int:
        """Get epoch with best validation loss."""
        return np.argmin(self.history['val_loss'])
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        try:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'best_val_loss': self.best_val_loss,
                'history': self.history,
                'config': self.config
            }
            
            if self.scheduler:
                checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
            # Save regular checkpoint
            checkpoint_path = self.output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"✅ Saved checkpoint to: {checkpoint_path}")
            
            # Save best model
            if is_best:
                best_path = self.output_dir / "best_model.pt"
                torch.save(checkpoint, best_path)
                print(f"✅ Saved best model to: {best_path}")
            
            # Clean up old checkpoints (keep only last N)
            keep_last_n = self.config.get('keep_last_n_checkpoints', 5)
            if keep_last_n > 0:
                # Sort checkpoints by epoch number, not filename
                checkpoints = list(self.output_dir.glob("checkpoint_epoch_*.pt"))
                checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))  # Extract epoch number
                print(f"📊 Total checkpoints: {len(checkpoints)}, keeping last {keep_last_n}")
                if checkpoints:
                    print(f"   Range: {checkpoints[0].stem} -> {checkpoints[-1].stem}")
                # Keep best_model.pt and last N checkpoints
                if len(checkpoints) > keep_last_n:
                    for old_checkpoint in checkpoints[:-keep_last_n]:
                        print(f"🗑️  Deleting old checkpoint: {old_checkpoint.name}")
                        old_checkpoint.unlink()  # Delete old checkpoint
        except Exception as e:
            print(f"❌ Error saving checkpoint: {e}")
            import traceback
            traceback.print_exc()
    
    def save_training_history(self) -> None:
        """Save training history to JSON."""
        history_path = self.output_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"📊 Training history saved: {history_path.name}")
    
    def plot_training_history(self) -> None:
        """Plot training curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.history['train_loss'], label='Train')
        axes[0, 0].plot(self.history['val_loss'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Price loss
        axes[0, 1].plot(self.history['train_price_loss'], label='Train')
        axes[0, 1].plot(self.history['val_price_loss'], label='Validation')
        axes[0, 1].set_title('Price Prediction Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MSE Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Direction accuracy
        axes[1, 0].plot(self.history['train_direction_acc'], label='Train')
        axes[1, 0].plot(self.history['val_direction_acc'], label='Validation')
        axes[1, 0].set_title('Direction Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].plot(self.history['learning_rate'])
        axes[1, 1].set_title('Learning Rate')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('LR')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = self.output_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📈 Training curves saved: {plot_path.name}")


def create_trainer_config() -> Dict[str, Any]:
    """Create default training configuration."""
    return {
        'batch_size': 32,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'scheduler': 'cosine',
        'min_lr': 1e-6,
        'early_stopping': True,
        'early_stopping_patience': 20,
        'num_workers': 4,
        'price_weight': 1.0,
        'direction_weight': 0.5,
        'volatility_weight': 0.3,
        'diversity_weight': 0.1,
        'output_dir': './models',
        'keep_last_n_checkpoints': 5  # Keep only last N checkpoints (0 = keep all)
    }


def main():
    """Example training script."""
    # This would be called from a separate training script
    # with actual data loading and model initialization
    pass


if __name__ == "__main__":
    main()
