#!/usr/bin/env python3
"""
WiFi-3D-Fusion Model Training Script
===================================

Advanced training pipeline for CSI-based person detection and re-identification.
Supports continuous learning, adaptive model improvement, and real-time feedback.

Usage:
    python train_model.py --config configs/fusion.yaml --device esp32
    python train_model.py --source nexmon --continuous --auto-improve
"""

import os
import sys
import yaml
import argparse
import logging
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import time
from datetime import datetime
import json
from typing import Dict, List, Tuple, Optional
import threading
import queue
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingConfig:
    """Training configuration parameters"""
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 100
    weight_decay: float = 1e-4
    save_interval: int = 10
    validation_split: float = 0.2
    data_augmentation: bool = True
    continuous_learning: bool = False
    auto_improvement: bool = False
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

class CSIDataset(Dataset):
    """Dataset for CSI-based person detection training"""
    
    def __init__(self, data_dir: str, augmentation: bool = True):
        self.data_dir = Path(data_dir)
        self.augmentation = augmentation
        self.samples = self._load_samples()
        logger.info(f"📊 Loaded {len(self.samples)} training samples from {data_dir}")
    
    def _load_samples(self) -> List[Dict]:
        """Load CSI samples and labels from data directory"""
        samples = []
        
        # Load from CSI logs
        csi_dir = self.data_dir / "csi_logs"
        if csi_dir.exists():
            for pkl_file in csi_dir.glob("*.pkl"):
                try:
                    with open(pkl_file, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Extract CSI amplitude data
                    if isinstance(data, dict) and 'amp_fused' in data:
                        csi_data = data['amp_fused']
                        if isinstance(csi_data, np.ndarray) and len(csi_data) > 0:
                            # Generate synthetic labels for demonstration
                            # In real use, you'd have ground truth labels
                            label = self._generate_synthetic_label(csi_data)
                            
                            samples.append({
                                'csi_data': csi_data,
                                'label': label,
                                'file': str(pkl_file),
                                'timestamp': pkl_file.stem.split('_')[-1]
                            })
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {pkl_file}: {e}")
        
        # Load from recorded sequences
        reid_dir = self.data_dir.parent / "data" / "reid"
        if reid_dir.exists():
            for person_dir in reid_dir.glob("person_*"):
                try:
                    # Load person-specific CSI patterns
                    for seq_file in person_dir.glob("*.pkl"):
                        with open(seq_file, 'rb') as f:
                            seq_data = pickle.load(f)
                        
                        if 'csi_sequence' in seq_data:
                            for frame in seq_data['csi_sequence']:
                                samples.append({
                                    'csi_data': frame,
                                    'label': {'person_id': person_dir.name, 'present': True},
                                    'file': str(seq_file),
                                    'person': person_dir.name
                                })
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load {person_dir}: {e}")
        
        return samples
    
    def _generate_synthetic_label(self, csi_data: np.ndarray) -> Dict:
        """Generate synthetic labels based on CSI characteristics"""
        # Analyze CSI variance and patterns to detect movement
        variance = np.var(csi_data)
        mean_amp = np.mean(csi_data)
        
        # Simple heuristic for person detection
        person_present = variance > 0.001 and mean_amp > 0.1
        confidence = min(variance * 1000, 1.0)
        
        return {
            'person_present': person_present,
            'confidence': confidence,
            'variance': variance,
            'mean_amplitude': mean_amp
        }
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        csi_data = sample['csi_data']
        label = sample['label']
        
        # Ensure consistent input size
        if len(csi_data) < 256:
            # Pad with zeros
            padded_data = np.zeros(256)
            padded_data[:len(csi_data)] = csi_data
            csi_data = padded_data
        elif len(csi_data) > 256:
            # Truncate
            csi_data = csi_data[:256]
        
        # Data augmentation
        if self.augmentation:
            csi_data = self._augment_data(csi_data)
        
        # Convert to tensors
        x = torch.FloatTensor(csi_data).unsqueeze(0)  # Add channel dimension
        
        # Convert label to tensor
        if isinstance(label, dict):
            if 'person_present' in label:
                y = torch.FloatTensor([1.0 if label['person_present'] else 0.0])
            else:
                y = torch.FloatTensor([1.0])  # Default to person present
        else:
            y = torch.FloatTensor([float(label)])
        
        return x, y
    
    def _augment_data(self, csi_data: np.ndarray) -> np.ndarray:
        """Apply data augmentation to CSI data"""
        # Add noise
        noise_level = np.random.uniform(0, 0.01)
        csi_data = csi_data + np.random.normal(0, noise_level, csi_data.shape)
        
        # Scale amplitude
        scale_factor = np.random.uniform(0.8, 1.2)
        csi_data = csi_data * scale_factor
        
        # Phase shift simulation
        phase_shift = np.random.uniform(0, 0.1)
        csi_data = csi_data + phase_shift
        
        return csi_data

class CSIPersonDetector(nn.Module):
    """Neural network for CSI-based person detection"""
    
    def __init__(self, input_size: int = 256, hidden_size: int = 128):
        super(CSIPersonDetector, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.MaxPool1d(2),
            
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.MaxPool1d(2),
            
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.classifier(x)
        return x

class ContinuousLearner:
    """Continuous learning system for real-time model improvement"""
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.data_queue = queue.Queue(maxsize=1000)
        self.running = False
        self.learn_thread = None
        self.optimizer = optim.Adam(model.parameters(), lr=config.learning_rate * 0.1)
        self.criterion = nn.BCELoss()
        
    def start(self):
        """Start continuous learning thread"""
        self.running = True
        self.learn_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learn_thread.start()
        logger.info("🔄 Continuous learning started")
    
    def stop(self):
        """Stop continuous learning"""
        self.running = False
        if self.learn_thread:
            self.learn_thread.join()
        logger.info("⏹️ Continuous learning stopped")
    
    def add_sample(self, csi_data: np.ndarray, label: float, confidence: float = 1.0):
        """Add new training sample from real-time detection"""
        try:
            # Only learn from high-confidence samples
            if confidence > 0.7:
                self.data_queue.put((csi_data, label, confidence), block=False)
        except queue.Full:
            logger.warning("⚠️ Learning queue full, dropping sample")
    
    def _learning_loop(self):
        """Continuous learning loop"""
        batch_data = []
        
        while self.running:
            try:
                # Collect samples for mini-batch
                while len(batch_data) < 8 and self.running:
                    try:
                        sample = self.data_queue.get(timeout=1.0)
                        batch_data.append(sample)
                    except queue.Empty:
                        continue
                
                if len(batch_data) >= 4:  # Minimum batch size
                    self._update_model(batch_data)
                    batch_data = []
                
            except Exception as e:
                logger.error(f"❌ Error in continuous learning: {e}")
                time.sleep(1)
    
    def _update_model(self, batch_data: List[Tuple]):
        """Update model with new batch of data"""
        try:
            # Prepare batch
            x_batch = []
            y_batch = []
            
            for csi_data, label, confidence in batch_data:
                # Normalize CSI data
                if len(csi_data) < 256:
                    padded_data = np.zeros(256)
                    padded_data[:len(csi_data)] = csi_data
                    csi_data = padded_data
                elif len(csi_data) > 256:
                    csi_data = csi_data[:256]
                
                x_batch.append(torch.FloatTensor(csi_data).unsqueeze(0))
                y_batch.append(torch.FloatTensor([label]))
            
            x = torch.stack(x_batch).to(self.config.device)
            y = torch.stack(y_batch).to(self.config.device)
            
            # Forward pass
            self.model.train()
            self.optimizer.zero_grad()
            
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            logger.info(f"🎯 Model updated: loss={loss.item():.4f}, samples={len(batch_data)}")
            
        except Exception as e:
            logger.error(f"❌ Error updating model: {e}")

class WiFiTrainer:
    """Main training orchestrator"""
    
    def __init__(self, config_path: str, args: argparse.Namespace):
        self.args = args
        self.config = self._load_config(config_path)
        self.training_config = TrainingConfig(
            continuous_learning=args.continuous,
            auto_improvement=args.auto_improve,
            device=args.device if hasattr(args, 'device') else 'cpu'
        )
        
        # Initialize model
        self.model = CSIPersonDetector().to(self.training_config.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay
        )
        self.criterion = nn.BCELoss()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        # Continuous learning
        self.continuous_learner = None
        if self.training_config.continuous_learning:
            self.continuous_learner = ContinuousLearner(self.model, self.training_config)
        
        # Setup directories
        self.setup_directories()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"📋 Loaded config from {config_path}")
        return config
    
    def setup_directories(self):
        """Setup training directories"""
        self.model_dir = Path("env/weights")
        self.log_dir = Path("env/logs")
        self.data_dir = Path("env")
        
        for directory in [self.model_dir, self.log_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📁 Training directories ready")
    
    def train(self):
        """Main training loop"""
        logger.info(f"🚀 Starting WiFi-3D-Fusion training")
        logger.info(f"📊 Device: {self.training_config.device}")
        logger.info(f"🔧 Source: {self.config.get('source', 'unknown')}")
        logger.info(f"🔄 Continuous learning: {self.training_config.continuous_learning}")
        
        # Load dataset
        dataset = CSIDataset(
            str(self.data_dir), 
            augmentation=self.training_config.data_augmentation
        )
        
        if len(dataset) == 0:
            logger.error("❌ No training data found! Please collect some CSI data first.")
            return False
        
        # Split dataset
        train_size = int(len(dataset) * (1 - self.training_config.validation_split))
        val_size = len(dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.training_config.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.training_config.batch_size, 
            shuffle=False
        )
        
        logger.info(f"📈 Training samples: {len(train_dataset)}")
        logger.info(f"📊 Validation samples: {len(val_dataset)}")
        
        # Start continuous learning if enabled
        if self.continuous_learner:
            self.continuous_learner.start()
        
        # Training loop
        best_val_loss = float('inf')
        training_history = []
        
        for epoch in range(self.training_config.epochs):
            start_time = time.time()
            
            # Training phase
            train_loss = self._train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Record history
            epoch_time = time.time() - start_time
            training_history.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'lr': self.optimizer.param_groups[0]['lr'],
                'time': epoch_time
            })
            
            # Logging
            logger.info(
                f"📊 Epoch {epoch+1}/{self.training_config.epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.2f}% | "
                f"LR: {self.optimizer.param_groups[0]['lr']:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pth', epoch, val_loss, val_acc)
                logger.info(f"💾 New best model saved! Val loss: {val_loss:.4f}")
            
            # Regular checkpoint
            if (epoch + 1) % self.training_config.save_interval == 0:
                self.save_model(f'checkpoint_epoch_{epoch+1}.pth', epoch, val_loss, val_acc)
        
        # Final save
        self.save_model('final_model.pth', epoch, val_loss, val_acc)
        self.save_training_history(training_history)
        
        # Stop continuous learning
        if self.continuous_learner:
            self.continuous_learner.stop()
        
        logger.info(f"✅ Training completed! Best validation loss: {best_val_loss:.4f}")
        return True
    
    def _train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(self.training_config.device)
            target = target.to(self.training_config.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(self.training_config.device)
                target = target.to(self.training_config.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()
                
                # Calculate accuracy
                predicted = (output > 0.5).float()
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def save_model(self, filename: str, epoch: int, val_loss: float, val_acc: float):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'config': self.config,
            'training_config': self.training_config.__dict__,
            'timestamp': datetime.now().isoformat()
        }
        
        filepath = self.model_dir / filename
        torch.save(checkpoint, filepath)
        logger.info(f"💾 Model saved: {filepath}")
    
    def save_training_history(self, history: List[Dict]):
        """Save training history"""
        history_file = self.log_dir / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(history_file, 'w') as f:
            json.dump(history, f, indent=2)
        logger.info(f"📊 Training history saved: {history_file}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="WiFi-3D-Fusion Model Training")
    parser.add_argument(
        '--config', 
        default='configs/fusion.yaml',
        help='Configuration file path'
    )
    parser.add_argument(
        '--source',
        choices=['esp32', 'nexmon', 'dummy'],
        help='CSI data source (overrides config)'
    )
    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Training device'
    )
    parser.add_argument(
        '--continuous',
        action='store_true',
        help='Enable continuous learning during training'
    )
    parser.add_argument(
        '--auto-improve',
        action='store_true',
        help='Enable automatic model improvement based on feedback'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate'
    )
    
    args = parser.parse_args()
    
    # Update config if source is specified
    if args.source:
        logger.info(f"🔧 Using data source: {args.source}")
    
    # Check for CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        logger.warning("⚠️ CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    # Initialize trainer
    trainer = WiFiTrainer(args.config, args)
    
    # Update training config from args
    trainer.training_config.epochs = args.epochs
    trainer.training_config.batch_size = args.batch_size
    trainer.training_config.learning_rate = args.lr
    
    # Start training
    try:
        success = trainer.train()
        if success:
            logger.info("🎉 Training completed successfully!")
            
            # Show next steps
            logger.info("📋 Next steps:")
            logger.info("   1. Test your model with: python run_js_visualizer.py")
            logger.info("   2. Run real-time detection: ./run_wifi3d.sh")
            logger.info("   3. View training logs in: env/logs/")
            
        else:
            logger.error("❌ Training failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⏹️ Training interrupted by user")
        if trainer.continuous_learner:
            trainer.continuous_learner.stop()
    except Exception as e:
        logger.error(f"❌ Training error: {e}")
        if trainer.continuous_learner:
            trainer.continuous_learner.stop()
        sys.exit(1)

if __name__ == "__main__":
    main()
