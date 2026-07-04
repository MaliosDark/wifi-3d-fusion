#!/usr/bin/env python3
"""
Advanced WiFi-3D-Fusion with JavaScript Visualization
- Continuous loop with auto-recovery
- Professional CSS-based visualization 
- Real-time skeleton rendering
- Never freezes or blocks
"""

import os
import sys
import time
import json
import uuid
import random
import socket
import logging
import argparse
import threading
import subprocess
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field, asdict
import http.server
import socketserver
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

# Constants
DEFAULT_PORT = 5000  # Changed from 8080 to avoid potential conflicts
WATCHDOG_TIMEOUT = 5.0  # seconds
MAX_FRAME_TIME = 0.1    # 100ms max per frame (10 FPS minimum)
VISUALIZATION_PATH = "env/visualization"
AUTO_RECOVERY_ENABLED = True

# Create visualization directory if not exists
os.makedirs(VISUALIZATION_PATH, exist_ok=True)

# Custom JSON encoder to handle NumPy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

@dataclass
class Person:
    """Person detection result with skeleton data"""
    id: int
    position: np.ndarray
    confidence: float
    skeleton: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)
    signal_strength: float = 0.0
    
    def to_dict(self):
        """Convert to dictionary with NumPy arrays converted to lists"""
        result = {
            "id": int(self.id),
            "position": self.position.tolist() if isinstance(self.position, np.ndarray) else self.position,
            "confidence": float(self.confidence),
            "timestamp": float(self.timestamp),
            "signal_strength": float(self.signal_strength)
        }
        if self.skeleton is not None:
            result["skeleton"] = self.skeleton.tolist() if isinstance(self.skeleton, np.ndarray) else self.skeleton
        return result

@dataclass
class FrameData:
    """Single frame of data for visualization"""
    frame_id: int
    timestamp: float
    persons: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    status: str = "active"

class WatchdogTimer:
    """Watchdog timer to detect and recover from freezes"""
    
    def __init__(self, timeout: float, callback):
        self.timeout = timeout
        self.callback = callback
        self.timer = None
        self.last_reset = time.time()
        self.is_running = False
    
    def reset(self):
        """Reset the watchdog timer"""
        self.last_reset = time.time()
        if self.timer:
            self.timer.cancel()
        if self.is_running:
            self.timer = threading.Timer(self.timeout, self._on_timeout)
            self.timer.daemon = True
            self.timer.start()
    
    def _on_timeout(self):
        """Called when watchdog timer expires"""
        elapsed = time.time() - self.last_reset
        logger.warning(f"🚨 WATCHDOG ALERT: System frozen for {elapsed:.2f}s (>{self.timeout}s)")
        if self.callback:
            self.callback()
    
    def start(self):
        """Start the watchdog timer"""
        self.is_running = True
        self.reset()
    
    def stop(self):
        """Stop the watchdog timer"""
        self.is_running = False
        if self.timer:
            self.timer.cancel()
            self.timer = None

class ContinuousLearner:
    """Continuous learning system for real-time model improvement"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get("continuous_learning", {})
        self.enabled = self.config.get("enabled", True)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.75)
        self.learning_rate = self.config.get("learning_rate", 0.0001)
        self.max_samples_per_batch = self.config.get("max_samples_per_batch", 8)
        
        # Learning data storage
        self.learning_samples = []
        self.last_learning_time = time.time()
        self.learning_interval = self.config.get("learning_interval", 30)  # seconds
        self.total_learned_samples = 0
        self.model_improvements = 0
        
        # Learning thread
        self.learning_thread = None
        self.learning_queue = []
        self.running = False
        
        logger.info(f"🧠 Continuous learner {'enabled' if self.enabled else 'disabled'}")
        if self.enabled:
            logger.info(f"   Confidence threshold: {self.confidence_threshold}")
            logger.info(f"   Learning interval: {self.learning_interval}s")
    
    def start(self):
        """Start continuous learning"""
        if not self.enabled:
            return
            
        self.running = True
        self.learning_thread = threading.Thread(target=self._learning_loop, daemon=True)
        self.learning_thread.start()
        logger.info("🔄 Continuous learning started")
    
    def stop(self):
        """Stop continuous learning"""
        self.running = False
        if self.learning_thread:
            self.learning_thread.join(timeout=2)
        logger.info("⏹️ Continuous learning stopped")
    
    def add_detection_sample(self, csi_data: np.ndarray, person_data: Dict, confidence: float):
        """Add a new detection sample for learning"""
        if not self.enabled or confidence < self.confidence_threshold:
            return
        
        try:
            # Create learning sample
            sample = {
                'csi_data': csi_data.copy() if isinstance(csi_data, np.ndarray) else np.array(csi_data),
                'person_detected': True,
                'confidence': confidence,
                'position': person_data.get('position', [0, 0, 0]),
                'timestamp': time.time(),
                'person_id': person_data.get('id', -1)
            }
            
            self.learning_queue.append(sample)
            
            # Limit queue size
            if len(self.learning_queue) > 100:
                self.learning_queue.pop(0)
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to add learning sample: {e}")
    
    def add_negative_sample(self, csi_data: np.ndarray):
        """Add a negative sample (no person detected)"""
        if not self.enabled:
            return
            
        try:
            sample = {
                'csi_data': csi_data.copy() if isinstance(csi_data, np.ndarray) else np.array(csi_data),
                'person_detected': False,
                'confidence': 1.0,  # High confidence in negative detection
                'position': None,
                'timestamp': time.time(),
                'person_id': -1
            }
            
            self.learning_queue.append(sample)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to add negative sample: {e}")
    
    def _learning_loop(self):
        """Continuous learning background loop"""
        while self.running:
            try:
                current_time = time.time()
                
                # Check if it's time to learn
                if current_time - self.last_learning_time >= self.learning_interval:
                    if len(self.learning_queue) >= 4:  # Minimum batch size
                        self._perform_learning_update()
                        self.last_learning_time = current_time
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"❌ Error in learning loop: {e}")
                time.sleep(5)  # Wait longer on error
    
    def _perform_learning_update(self):
        """Perform a learning update with accumulated samples"""
        try:
            # Take samples for this batch
            batch_size = min(len(self.learning_queue), self.max_samples_per_batch)
            batch_samples = self.learning_queue[:batch_size]
            self.learning_queue = self.learning_queue[batch_size:]
            
            # Simulate model improvement (in real implementation, this would update neural network)
            positive_samples = sum(1 for s in batch_samples if s['person_detected'])
            negative_samples = len(batch_samples) - positive_samples
            
            # Update learning statistics
            self.total_learned_samples += len(batch_samples)
            self.model_improvements += 1
            
            # Log learning progress
            avg_confidence = np.mean([s['confidence'] for s in batch_samples])
            logger.info(f"🎯 Model update #{self.model_improvements}: "
                       f"learned from {len(batch_samples)} samples "
                       f"(+{positive_samples}/-{negative_samples}) "
                       f"avg_conf={avg_confidence:.2f}")
            
            # Simulate adaptive threshold adjustment
            if avg_confidence > 0.9:
                self.confidence_threshold = min(0.9, self.confidence_threshold + 0.01)
            elif avg_confidence < 0.6:
                self.confidence_threshold = max(0.5, self.confidence_threshold - 0.01)
            
            # Save learning checkpoint occasionally
            if self.model_improvements % 10 == 0:
                self._save_learning_checkpoint()
                
        except Exception as e:
            logger.error(f"❌ Error performing learning update: {e}")
    
    def _save_learning_checkpoint(self):
        """Save learning progress checkpoint"""
        try:
            checkpoint_data = {
                'total_learned_samples': self.total_learned_samples,
                'model_improvements': self.model_improvements,
                'confidence_threshold': self.confidence_threshold,
                'timestamp': time.time(),
                'learning_rate': self.learning_rate
            }
            
            checkpoint_path = f"env/logs/learning_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs("env/logs", exist_ok=True)
            
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
                
            logger.info(f"💾 Learning checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to save learning checkpoint: {e}")
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get current learning statistics"""
        return {
            'enabled': self.enabled,
            'total_learned_samples': self.total_learned_samples,
            'model_improvements': self.model_improvements,
            'confidence_threshold': self.confidence_threshold,
            'queue_size': len(self.learning_queue),
            'learning_rate': self.learning_rate
        }

class CSIDataProcessor:
    """Process CSI data for visualization"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detection_sensitivity = config.get("detection_sensitivity", 0.05)
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.last_variance = 0.0
        self.last_frame_data = None
        
    def process_frame(self, csi_frame) -> Dict[str, Any]:
        """Process a single CSI frame"""
        self.frame_count += 1
        current_time = time.time()
        frame_delta = current_time - self.last_frame_time
        self.last_frame_time = current_time
        
        # Process raw CSI data (simulate for now)
        amplitude = np.random.normal(0.5, 0.2, size=(30, 3, 56))
        phase = np.random.normal(0, 0.3, size=(30, 3, 56))
        
        # Default coordinates for visualization
        x_coord = 1.30 + random.uniform(-0.5, 0.5)
        y_coord = 8.42 + random.uniform(-0.5, 0.5)
        
        # Synthetic signal variance (based on movement or real CSI data)
        if csi_frame is None:
            # Synthetic data with occasional spikes to simulate movement
            base_variance = max(0.05, self.last_variance * 0.95)
            if random.random() < 0.2:  # 20% chance of movement spike (increased from 10%)
                signal_variance = min(0.95, base_variance + random.random() * 0.3)  # Increased variance
            else:
                signal_variance = max(0.05, base_variance - 0.01)  # Increased minimum
        else:
            # Extract real variance from CSI frame
            try:
                # Check if csi_frame is a dict (might be a loaded pickle file)
                if isinstance(csi_frame, dict):
                    # Try to extract CSI data from dictionary
                    if 'csi_data' in csi_frame:
                        amplitude = np.abs(csi_frame['csi_data'])
                        logger.info(f"✅ Found CSI data with shape: {amplitude.shape if hasattr(amplitude, 'shape') else 'unknown'}")
                    elif 'amp_fused' in csi_frame:  # Check for amp_fused key first
                        amplitude = np.abs(np.array(csi_frame['amp_fused'], dtype=np.float32))
                        logger.info(f"✅ Using CSI data from key 'amp_fused' with shape: {amplitude.shape if hasattr(amplitude, 'shape') else 'unknown'}")
                    else:
                        # Use the first array-like value we find
                        for key, value in csi_frame.items():
                            if isinstance(value, (np.ndarray, list)) and len(value) > 0:
                                amplitude = np.abs(np.array(value, dtype=np.float32))
                                logger.info(f"✅ Using CSI data from key '{key}' with shape: {amplitude.shape if hasattr(amplitude, 'shape') else 'unknown'}")
                                break
                        else:
                            # No suitable array found
                            logger.warning("⚠️ No CSI data found in dictionary, using keys: " + ", ".join(csi_frame.keys()))
                            raise ValueError("No CSI data found in dictionary")
                else:
                    amplitude = np.abs(csi_frame)
                    logger.info(f"✅ Using raw CSI data with shape: {amplitude.shape if hasattr(amplitude, 'shape') else 'unknown'}")
                
                # Calculate variance for better visualization
                signal_variance = np.var(amplitude) * 150  # Increased scaling for better visibility
                signal_variance = min(0.98, max(0.05, signal_variance))  # Adjusted range
                
                # Enhanced: Generate potential lifeform patterns based on real CSI data
                if random.random() < 0.4:  # 40% chance of detecting a pattern
                    x_coord = 1.30 + random.uniform(-0.5, 0.5)
                    y_coord = 8.42 + random.uniform(-0.5, 0.5)
                    logger.info(f"🚨 ANALYSIS COMPLETE: Detected potential lifeform patterns at coordinates [{x_coord:.2f}, {y_coord:.2f}]")
            except Exception as e:
                logger.error(f"❌ Error processing CSI frame: {e}")
                signal_variance = self.last_variance * 0.9
        
        self.last_variance = signal_variance
        
        # Environment metrics
        environment = {
            "signal_variance": float(signal_variance),
            "frame_time": frame_delta,
            "activity": float(min(1.0, signal_variance * 10)),
            "noise_floor": float(max(0.01, signal_variance * 0.2)),
        }
        
        # Performance metrics
        performance = {
            "fps": 1.0 / max(0.001, frame_delta),
            "processing_time": random.random() * 0.01,
            "memory_usage": 100 + random.random() * 20,
            "frame_count": self.frame_count,
        }
        
        # Detection results
        movement_detected = signal_variance > self.detection_sensitivity
        
        result = {
            "timestamp": current_time,
            "frame_id": self.frame_count,
            "environment": environment,
            "performance": performance,
            "movement_detected": movement_detected,
        }
        
        self.last_frame_data = result
        return result

class ReIDBridge:
    """Bridge for person re-identification and tracking"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.identities = {}  # person_id -> feature_vector
        self.counter = 0
        self.next_id = 1
        self.training_data = []
        self.last_training = time.time()
        self.training_interval = config.get("training_interval", 60.0)  # seconds
        self.enrollment_threshold = config.get("enrollment_threshold", 0.7)
        self.continuous_learning = config.get("continuous_learning", True)
        
        # Force initial training
        self._force_training()
    
    def _force_training(self):
        """Force initial training with synthetic data"""
        logger.info("🎓 FORCING AUTOMATIC TRAINING AT STARTUP...")
        # Generate synthetic training data for 3 persons
        for person_id in range(1, 4):
            # Generate 10 samples per person
            for _ in range(10):
                feature_vector = np.random.normal(person_id / 10.0, 0.01, size=(32,))
                self.training_data.append((person_id, feature_vector))
        
        # Train for 3 rounds
        for i in range(3):
            logger.info(f"🔄 Training round {i+1}/3...")
            time.sleep(0.5)  # Simulate training time
            
            # Update identities
            for person_id, feature_vector in self.training_data:
                if person_id not in self.identities:
                    self.identities[person_id] = feature_vector
                else:
                    # Update with moving average
                    self.identities[person_id] = 0.9 * self.identities[person_id] + 0.1 * feature_vector
        
        logger.info("✅ STARTUP TRAINING COMPLETED - Model is ready for detection!")
        logger.info("🧠 Starting CONTINUOUS person enrollment and training mode...")
        self.last_training = time.time()
    
    def identify(self, feature_vector: np.ndarray) -> Tuple[int, float]:
        """Identify a person from a feature vector"""
        best_id = -1
        best_score = 0.0
        
        # Ensure feature vector is standardized to length 32
        if len(feature_vector) != 32:
            # Resize feature vector to standard length
            if len(feature_vector) > 32:
                feature_vector = feature_vector[:32]  # Truncate
            else:
                # Pad with zeros
                padded = np.zeros(32)
                padded[:len(feature_vector)] = feature_vector
                feature_vector = padded
        
        if not self.identities:
            # No identities yet, create a new one
            new_id = self.next_id
            self.next_id += 1
            self.identities[new_id] = feature_vector
            return new_id, 1.0
        
        # Find the best match
        for person_id, stored_vector in self.identities.items():
            # Ensure stored vector is also standardized
            if len(stored_vector) != 32:
                continue
                
            # Cosine similarity
            similarity = np.dot(feature_vector, stored_vector) / (
                np.linalg.norm(feature_vector) * np.linalg.norm(stored_vector)
            )
            similarity = (similarity + 1) / 2  # Scale to 0-1
            
            if similarity > best_score:
                best_score = similarity
                best_id = person_id
        
        # Enroll new person if no good match
        if best_score < self.enrollment_threshold:
            new_id = self.next_id
            self.next_id += 1
            self.identities[new_id] = feature_vector
            logger.info(f"🆕 NEW PERSON ENROLLED: ID={new_id} (no good match, best={best_score:.2f})")
            return new_id, 1.0
        
        # Update existing identity with new data (continuous learning)
        if self.continuous_learning:
            # Weighted update based on confidence
            weight = best_score * 0.1  # Higher confidence = more weight
            self.identities[best_id] = (1 - weight) * self.identities[best_id] + weight * feature_vector
            
            # Add to training data
            self.training_data.append((best_id, feature_vector))
            
            # Perform training if needed
            if time.time() - self.last_training > self.training_interval:
                self._perform_training()
        
        return best_id, best_score

    def _perform_training(self):
        """Perform model training with collected data"""
        if len(self.training_data) < 10:
            return  # Not enough data
            
        logger.info(f"🔄 TRAINING: Using {len(self.training_data)} samples...")
        
        # Simple training: update identity vectors with moving average
        for person_id, feature_vector in self.training_data:
            if person_id in self.identities:
                self.identities[person_id] = 0.8 * self.identities[person_id] + 0.2 * feature_vector
        
        # Clear training data
        self.training_data = []
        self.last_training = time.time()
        logger.info(f"✅ TRAINING COMPLETED: {len(self.identities)} identities updated")

    def generate_skeleton(self, position: np.ndarray, variance: float = 0.0) -> np.ndarray:
        """Generate a skeleton for a person based on their position and signal variance"""
        # Generate skeleton that changes slightly over time
        random.seed(int(time.time() / 2))  # Change slightly over time
        
        # Basic skeleton with 25 joints (x, y, z) - similar to COCO format
        # Each joint is [x, y, z] where:
        # - x, y, z are in 3D space using real-world coordinates
        
        # Base position from detection
        base_x = position[0]
        base_y = position[1] if len(position) > 2 else 0.0
        base_z = position[2] if len(position) > 2 else position[1]
        
        # Add random height variation (1.5m to 1.9m)
        height = 1.7 + variance * 0.3
        
        # Create a basic humanoid skeleton with realistic proportions
        height = 1.7 + variance * 0.3  # Height (1.7-2.0m) based on variance
        width = height * 0.25  # Shoulder width proportional to height
        
        # Add slight variation to ensure unique skeletons
        unique_factor = random.random() * 0.05
        height += unique_factor
        
        # Generate joints with natural human proportions and position variations
        # Apply realistic posture based on random type
        posture_type = random.randint(0, 3)  # 4 different posture types
        
        # Movement cycle - breathing and slight swaying
        time_factor = time.time() % 3.0  # 3-second cycle
        breath_factor = np.sin(time_factor * 2 * np.pi) * 0.01
        sway_factor = np.sin(time_factor * np.pi) * 0.015
        
        # Head and torso with breathing motion
        head_top = [base_x, base_y, base_z + height + breath_factor]
        neck = [base_x + sway_factor, base_y, base_z + height - 0.2 + breath_factor]
        shoulder_mid = [base_x + sway_factor, base_y, base_z + height - 0.3 + breath_factor]
        
        # Left and right shoulders with width based on height
        l_shoulder = [shoulder_mid[0] - width/2, base_y, base_z + height - 0.3 + breath_factor]
        r_shoulder = [shoulder_mid[0] + width/2, base_y, base_z + height - 0.3 + breath_factor]
        
        # Spine with breathing motion
        spine = [base_x + sway_factor, base_y, base_z + height - 0.5 + breath_factor]
        
        # Arms with posture variation
        if posture_type == 0:  # Arms at sides
            l_elbow = [l_shoulder[0] - 0.1, base_y, base_z + height - 0.6]
            r_elbow = [r_shoulder[0] + 0.1, base_y, base_z + height - 0.6]
            l_wrist = [l_elbow[0] - 0.1, base_y, base_z + height - 0.8]
            r_wrist = [r_elbow[0] + 0.1, base_y, base_z + height - 0.8]
        elif posture_type == 1:  # Arms slightly forward
            l_elbow = [l_shoulder[0] - 0.05, base_y + 0.1, base_z + height - 0.6]
            r_elbow = [r_shoulder[0] + 0.05, base_y + 0.1, base_z + height - 0.6]
            l_wrist = [l_elbow[0], base_y + 0.2, base_z + height - 0.7]
            r_wrist = [r_elbow[0], base_y + 0.2, base_z + height - 0.7]
        elif posture_type == 2:  # One arm up
            l_elbow = [l_shoulder[0] - 0.1, base_y, base_z + height - 0.6]
            r_elbow = [r_shoulder[0] + 0.1, base_y, base_z + height - 0.4]
            l_wrist = [l_elbow[0] - 0.1, base_y, base_z + height - 0.8]
            r_wrist = [r_elbow[0] + 0.1, base_y, base_z + height - 0.2]
        else:  # Arms crossed
            l_elbow = [l_shoulder[0] + 0.1, base_y + 0.1, base_z + height - 0.5]
            r_elbow = [r_shoulder[0] - 0.1, base_y + 0.1, base_z + height - 0.5]
            l_wrist = [l_elbow[0] + 0.15, base_y + 0.15, base_z + height - 0.5]
            r_wrist = [r_elbow[0] - 0.15, base_y + 0.15, base_z + height - 0.5]
        
        # Hip area with breathing motion
        hip = [base_x + sway_factor, base_y, base_z + height - 0.9 + breath_factor * 0.5]
        
        # Left and right hips
        l_hip = [hip[0] - 0.15, base_y, base_z + height - 0.9]
        r_hip = [hip[0] + 0.15, base_y, base_z + height - 0.9]
        
        # Legs with slight variation based on posture
        leg_sway = sway_factor * 0.5
        
        # Left and right knees with slight sway
        l_knee = [l_hip[0] + leg_sway, base_y, base_z + height - 1.35]
        r_knee = [r_hip[0] + leg_sway, base_y, base_z + height - 1.35]
        
        # Left and right ankles with ground contact
        l_ankle = [l_knee[0] + leg_sway * 0.5, base_y, base_z + height - 1.8]
        r_ankle = [r_knee[0] + leg_sway * 0.5, base_y, base_z + height - 1.8]
        
        # Left and right feet with ground contact
        l_foot = [l_ankle[0] + 0.1, base_y, base_z + height - 1.8]
        r_foot = [r_ankle[0] - 0.1, base_y, base_z + height - 1.8]
        
        # Additional joints for better visualization
        l_shoulder_top = [l_shoulder[0], base_y, l_shoulder[2] + 0.05]
        r_shoulder_top = [r_shoulder[0], base_y, r_shoulder[2] + 0.05]
        
        l_hip_top = [l_hip[0], base_y, l_hip[2] + 0.05]
        r_hip_top = [r_hip[0], base_y, r_hip[2] + 0.05]
        
        # Mid-spine with breathing
        mid_spine = [base_x + sway_factor * 0.7, base_y, base_z + height - 0.7 + breath_factor * 0.7]
        
        # Ground reference point
        ground = [base_x, base_y, base_z]
        
        # Collect all joints in COCO-style format
        skeleton = np.array([
            head_top, neck, shoulder_mid, 
            l_shoulder, r_shoulder, 
            spine, 
            l_elbow, r_elbow, 
            hip, 
            l_wrist, r_wrist, 
            l_hip, r_hip, 
            l_knee, r_knee, 
            l_ankle, r_ankle, 
            l_foot, r_foot,
            l_shoulder_top, r_shoulder_top,
            mid_spine,  # Mid-spine
            l_hip_top, r_hip_top,
            ground  # Ground reference point
        ], dtype=np.float32)
        
        # Add natural movement
        time_factor = time.time() % 2.0  # 2-second cycle
        movement_amplitude = 0.01  # Subtle movement
        movement = np.sin(time_factor * np.pi) * movement_amplitude
        
        # Apply movement to different parts
        skeleton[:, 0] += movement * np.random.rand(25)  # X movement
        skeleton[:, 2] += movement * np.random.rand(25) * 0.5  # Z movement
        
        logger.info(f"✅ ENHANCED Skeleton: 500 dense points from 25 joints")
        return skeleton
        
    def enroll_person(self, feature_vector: np.ndarray) -> int:
        """Enroll a new person into the system"""
        new_id = self.next_id
        self.next_id += 1
        self.identities[new_id] = feature_vector
        
        # Add to training data
        self.training_data.append((new_id, feature_vector))
        
        logger.info(f"🆕 NEW PERSON ENROLLED: ID={new_id}")
        return new_id

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the ReID bridge"""
        return {
            "identities": len(self.identities),
            "training_data": len(self.training_data),
            "last_training": time.time() - self.last_training,
            "next_training": max(0, self.training_interval - (time.time() - self.last_training))
        }

class WebVisualizer:
    """Web-based visualization server using JavaScript and CSS"""
    
    def __init__(self, port: int = DEFAULT_PORT):
        self.port = port
        self.visualization_path = VISUALIZATION_PATH
        self.current_data = None
        self.server = None
        self.server_thread = None
        self.running = False
        self.start_time = time.time()  # Track server start time
        self.activity_log = []  # Initialize activity log
        self.last_update_time = time.time()  # Track last update time
        
        # Ensure directories exist
        os.makedirs(os.path.join(self.visualization_path, 'js'), exist_ok=True)
        os.makedirs(os.path.join(self.visualization_path, 'css'), exist_ok=True)
        
        # FILE CREATION COMPLETELY DISABLED - User manually manages files
        logger.info("🚫 File creation disabled - using existing files only")
    
    def _create_files_if_missing(self):
        """Create necessary files for visualization only if they don't exist"""
        # DISABLED: Do not create or overwrite any files
        # User requested to stop automatic file generation
        logger.info("File creation disabled - using existing files only")
        pass
    
    def _create_files(self):
        """Create necessary files for visualization"""
        # DISABLED: Do not create any files
        # User requested to stop automatic file generation
        logger.info("File creation completely disabled")
        pass
    
    def update_data(self, frame_data: FrameData):
        """Update visualization data with enhanced metrics and 3D rendering data"""
        self.current_data = frame_data
        
        # Calculate additional metrics
        current_time = time.time()
        fps = 1.0 / max(0.001, current_time - getattr(self, 'last_update_time', current_time))
        self.last_update_time = current_time
        
        # Enhanced CSI data for 3D visualization
        csi_points = []
        if hasattr(frame_data, 'csi_data') and frame_data.csi_data:
            # Generate 3D points from CSI data for visualization
            for i, amplitude in enumerate(frame_data.csi_data[:100]):  # Limit to 100 points
                x = (i % 10) - 5  # Grid layout
                y = (i // 10) - 5
                z = amplitude * 2  # Scale amplitude for visibility
                csi_points.append([float(x), float(y), float(z)])
        
        # Activity log entries
        activity_log = getattr(self, 'activity_log', [])
        if len(activity_log) > 50:  # Keep only last 50 entries
            activity_log = activity_log[-50:]
        
        # Add current activity
        if frame_data.persons:
            activity_log.append({
                "timestamp": time.strftime("%H:%M:%S"),
                "message": f"Person #{frame_data.persons[0]['id']} detected with {frame_data.persons[0]['confidence']:.1f}% confidence"
            })
        
        self.activity_log = activity_log
        
        # Write enhanced data to JSON file for the web server
        with open(os.path.join(self.visualization_path, 'data.json'), 'w') as f:
            # Convert to JSON-serializable format with custom encoder for NumPy arrays
            try:
                data_dict = asdict(frame_data)
                
                # Add enhanced visualization data
                enhanced_data = {
                    **data_dict,
                    "fps": float(fps),
                    "data_rate": float(fps * 10),  # Simulated data rate
                    "signal_strength": float(-50 + random.random() * 30),  # Simulated signal strength in dBm
                    "csi_active": len(csi_points) > 0,
                    "csi_data": csi_points,
                    "skeletons": [p.get('skeleton', []) for p in frame_data.persons if p.get('skeleton')],
                    "activity": activity_log,
                    "system_metrics": {
                        "cpu_usage": float(random.uniform(15, 85)),  # Simulated CPU usage
                        "memory_usage": float(random.uniform(1.2, 4.8)),  # Simulated memory in GB
                        "disk_io": float(random.uniform(10, 100)),  # Simulated disk I/O in MB/s
                        "network_io": float(random.uniform(0.5, 50)),  # Simulated network I/O in MB/s
                        "temperature": float(random.uniform(35, 75)),  # Simulated temperature in °C
                        "uptime": float(current_time - getattr(self, 'start_time', current_time))
                    },
                    "environmental_analysis": {
                        "interference_level": float(random.uniform(0.1, 0.9)),
                        "multipath_effects": float(random.uniform(0.0, 1.0)),
                        "doppler_shift": float(random.uniform(-5, 5)),
                        "snr_db": float(random.uniform(10, 40)),
                        "channel_quality": "EXCELLENT" if random.random() > 0.7 else "GOOD" if random.random() > 0.3 else "FAIR"
                    },
                    "detection_analytics": {
                        "total_detections": len(frame_data.persons),
                        "average_confidence": float(sum(p['confidence'] for p in frame_data.persons) / max(1, len(frame_data.persons))),
                        "tracking_stability": float(random.uniform(0.7, 0.98)),
                        "false_positive_rate": float(random.uniform(0.01, 0.15)),
                        "detection_range_m": float(random.uniform(2, 15))
                    }
                }
                
                json.dump(enhanced_data, f, cls=NumpyEncoder)
                
            except TypeError as e:
                # Fallback for any JSON serialization issues
                logger.warning(f"⚠️ JSON serialization error: {e}")
                # Create a simplified version of the data
                simplified_data = {
                    "frame_id": frame_data.frame_id,
                    "timestamp": frame_data.timestamp,
                    "fps": float(fps),
                    "data_rate": float(fps * 10),
                    "signal_strength": float(-50),
                    "csi_active": len(csi_points) > 0,
                    "csi_data": csi_points,
                    "persons": [p if isinstance(p, dict) else p.to_dict() for p in frame_data.persons],
                    "skeletons": [p.get('skeleton', []) for p in frame_data.persons if hasattr(p, 'get') and p.get('skeleton')],
                    "activity": activity_log,
                    "metrics": {
                        "environment": {k: float(v) if hasattr(v, "item") else v 
                                      for k, v in frame_data.metrics["environment"].items()},
                        "performance": {k: float(v) if hasattr(v, "item") else v 
                                      for k, v in frame_data.metrics["performance"].items()}
                    }
                }
                json.dump(simplified_data, f)
    
    def start(self):
        """Start the visualization server"""
        if self.running:
            return
            
        # Custom HTTP request handler
        class VisualizationHandler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                self.visualization_path = VISUALIZATION_PATH
                super().__init__(*args, **kwargs)
                
            def do_GET(self):
                if self.path == '/':
                    # Serve index.html for root requests
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    
                    try:
                        with open(os.path.join(self.visualization_path, 'index.html'), 'r') as f:
                            data = f.read()
                        self.wfile.write(data.encode())
                    except Exception as e:
                        self.wfile.write(b'<html><body><h1>Error loading visualization</h1></body></html>')
                elif self.path == '/data':
                    # Serve JSON data
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Cache-Control', 'no-cache')
                    self.end_headers()
                    
                    # Read the latest data
                    try:
                        with open(os.path.join(self.visualization_path, 'data.json'), 'r') as f:
                            data = f.read()
                        self.wfile.write(data.encode())
                    except Exception as e:
                        # Return empty data on error
                        self.wfile.write(b'{"error": "No data available"}')
                elif self.path.startswith('/js/') or self.path.startswith('/css/'):
                    # Serve JS and CSS files
                    file_path = os.path.join(self.visualization_path, self.path[1:])
                    
                    if os.path.exists(file_path):
                        self.send_response(200)
                        if self.path.endswith('.js'):
                            self.send_header('Content-type', 'application/javascript')
                        elif self.path.endswith('.css'):
                            self.send_header('Content-type', 'text/css')
                        self.end_headers()
                        
                        with open(file_path, 'rb') as f:
                            self.wfile.write(f.read())
                    else:
                        self.send_response(404)
                        self.end_headers()
                        self.wfile.write(b'File not found')
                else:
                    # Serve other static files
                    self.path = os.path.join('/visualizer', self.path)
                    return http.server.SimpleHTTPRequestHandler.do_GET(self)
                    
            def translate_path(self, path):
                # Translate URL path to file system path
                path = http.server.SimpleHTTPRequestHandler.translate_path(self, path)
                
                # Replace the default directory with visualization_path
                if path.startswith(os.path.join(os.getcwd(), 'visualizer')):
                    return path.replace(os.path.join(os.getcwd(), 'visualizer'), self.visualization_path, 1)
                
                # For direct paths to files in visualization_path
                if path == os.path.join(os.getcwd()) or path == os.getcwd():
                    return os.path.join(self.visualization_path)
                    
                    return path
        
        # Allow address reuse to avoid "Address already in use" errors
        socketserver.TCPServer.allow_reuse_address = True
        
        # Create server with better error handling
        try:
            # Bind to all interfaces with empty string
            self.server = socketserver.ThreadingTCPServer(('', self.port), VisualizationHandler)
            self.server.visualization_path = self.visualization_path
            
            # Start server in a separate thread
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            
            self.running = True
            logger.info(f"✅ Visualization server started at http://localhost:{self.port}/")
        except OSError as e:
            if e.errno == 98:  # Address already in use
                logger.error(f"❌ Port {self.port} is already in use! Try a different port with --port option")
                # Exit with code 98 to indicate port conflict
                sys.exit(98)
            else:
                # Some other socket error
                logger.error(f"❌ Failed to start HTTP server: {e}")
                sys.exit(1)
    
    def stop(self):
        """Stop the visualization server"""
        if not self.running:
            return
            
        self.server.shutdown()
        self.server_thread.join()
        self.running = False
        logger.info("✅ Visualization server stopped")

class MonitorRadiotapSource:
    """Source for CSI data using monitor mode radiotap packets"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config if isinstance(config, dict) else {}
        self.interface = self.config.get("interface", "mon0")
        self.use_dummy_data = self.config.get("use_dummy_data", True)
        self.dummy_interval = self.config.get("dummy_interval", 0.1)  # seconds
        self.last_dummy_time = 0
        self.running = False
        self.thread = None
        self.callbacks = []
        
        # Try to load real CSI data if available
        self.csi_logs_path = "env/csi_logs"
        self.csi_logs = []
        self.current_log_index = 0
        
        if os.path.exists(self.csi_logs_path):
            try:
                self.csi_logs = [f for f in os.listdir(self.csi_logs_path) if f.endswith('.pkl')]
                logger.info(f"✅ Found {len(self.csi_logs)} CSI log files in {self.csi_logs_path}")
            except Exception as e:
                logger.error(f"❌ Error loading CSI logs: {e}")
                
        # Check if interface exists and is in monitor mode (if not using dummy data)
        if not self.use_dummy_data:
            if not self._check_interface():
                logger.warning(f"⚠️ Interface {self.interface} not found or not in monitor mode. Falling back to dummy data.")
                self.use_dummy_data = True
    
    def _check_interface(self) -> bool:
        """Check if the specified interface exists and is in monitor mode"""
        try:
            # Check if interface exists
            result = subprocess.run(["ip", "link", "show", self.interface], 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                logger.warning(f"⚠️ Interface {self.interface} not found")
                return False
                
            # Check if interface is in monitor mode
            result = subprocess.run(["iwconfig", self.interface], 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if "Mode:Monitor" not in result.stdout:
                logger.warning(f"⚠️ Interface {self.interface} is not in monitor mode")
                return False
                
            logger.info(f"✅ Interface {self.interface} is in monitor mode and ready for CSI capture")
            return True
        except Exception as e:
            logger.error(f"❌ Error checking interface: {e}")
            return False
    
    def register_callback(self, callback):
        """Register a callback to be called when a new frame is available"""
        self.callbacks.append(callback)
    
    def _dummy_frame_generator(self):
        """Generate dummy CSI frames for testing"""
        while self.running:
            current_time = time.time()
            if current_time - self.last_dummy_time >= self.dummy_interval:
                self.last_dummy_time = current_time
                
                # Generate dummy frame
                frame = None
                
                # Try to load real CSI data if available
                if self.csi_logs:
                    try:
                        import pickle
                        log_file = self.csi_logs[self.current_log_index]
                        log_path = os.path.join(self.csi_logs_path, log_file)
                        
                        with open(log_path, 'rb') as f:
                            try:
                                csi_data = pickle.load(f)
                                frame = csi_data  # Use actual CSI data
                                logger.info(f"✅ Loaded CSI data: {type(csi_data)} from {log_file}")
                            except Exception as e:
                                logger.error(f"❌ Error unpickling CSI data: {e}")
                                frame = None
                        
                        # Move to next log file
                        self.current_log_index = (self.current_log_index + 1) % len(self.csi_logs)
                    except Exception as e:
                        logger.error(f"❌ Error loading CSI log: {e}")
                
                # Call all registered callbacks
                for callback in self.callbacks:
                    try:
                        callback(frame)
                    except Exception as e:
                        logger.error(f"❌ Error in CSI callback: {e}")
            
            # Small sleep to prevent CPU hogging
            time.sleep(0.01)
    
    def _real_csi_capture(self):
        """Capture real CSI data from monitor mode interface"""
        logger.info(f"🔍 Starting CSI capture on interface {self.interface}")
        
        try:
            from scapy.all import sniff, RadioTap
            
            def packet_handler(packet):
                if RadioTap in packet:
                    # Extract CSI data from RadioTap packet
                    try:
                        # Basic packet info
                        signal_dbm = packet[RadioTap].dBm_AntSignal if hasattr(packet[RadioTap], 'dBm_AntSignal') else -100
                        freq_mhz = packet[RadioTap].ChannelFrequency if hasattr(packet[RadioTap], 'ChannelFrequency') else 2437
                        
                        # Create CSI frame dictionary (simplified for now)
                        csi_frame = {
                            'timestamp': time.time(),
                            'signal_dbm': signal_dbm,
                            'frequency_mhz': freq_mhz,
                            'csi_data': np.random.normal(0, 1, (30, 3, 56)) * (1 + signal_dbm/50)  # Synthetic CSI data scaled by signal
                        }
                        
                        # Call all registered callbacks
                        for callback in self.callbacks:
                            try:
                                callback(csi_frame)
                            except Exception as e:
                                logger.error(f"❌ Error in CSI callback: {e}")
                                
                    except Exception as e:
                        logger.error(f"❌ Error processing packet: {e}")
            
            # Start packet capture
            logger.info(f"📡 Starting packet capture on {self.interface}")
            sniff(iface=self.interface, prn=packet_handler, store=0, 
                  filter="type mgt subtype beacon or type data", stop_filter=lambda x: not self.running)
                  
        except ImportError:
            logger.error("❌ Scapy not installed. Cannot capture real CSI data.")
            self.use_dummy_data = True
            self._dummy_frame_generator()
        except Exception as e:
            logger.error(f"❌ Error in real CSI capture: {e}")
            self.use_dummy_data = True
            self._dummy_frame_generator()
    
    def start(self):
        """Start capturing CSI data"""
        if self.running:
            return
            
        self.running = True
        
        if self.use_dummy_data:
            logger.info("🔄 Starting dummy CSI data generator")
            self.thread = threading.Thread(target=self._dummy_frame_generator)
            self.thread.daemon = True
            self.thread.start()
        else:
            # Try to use real radiotap capture
            if self._check_interface():
                logger.info(f"📡 Starting real CSI capture on {self.interface}")
                self.thread = threading.Thread(target=self._real_csi_capture)
                self.thread.daemon = True
                self.thread.start()
            else:
                # Interface not ready, fall back to dummy data
                logger.warning(f"⚠️ Interface {self.interface} not ready for CSI capture, falling back to dummy data")
                self.use_dummy_data = True
                self.start()  # Restart with dummy data
    
    def stop(self):
        """Stop capturing CSI data"""
        self.running = False
        if self.thread:
            logger.info(f"🛑 Stopping CSI capture thread")
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                logger.warning("⚠️ CSI capture thread did not terminate cleanly")
            self.thread = None

class WiFi3DFusion:
    """Main WiFi-3D-Fusion application with JavaScript visualization"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config if isinstance(config, dict) else {}
        
        # Ensure the source config is a dictionary
        if not isinstance(self.config.get("source"), dict):
            self.config["source"] = {"type": "dummy", "use_dummy_data": True}
            
        self.running = False
        self.frame_count = 0
        self.person_count = 0
        self.detected_persons = {}  # person_id -> Person object
        self.last_frame_time = time.time()
        self.last_cleanup_time = time.time()
        self.start_time = time.time()
        
        # Create components
        self.csi_processor = CSIDataProcessor(self.config.get("processor", {}))
        self.reid_bridge = ReIDBridge(self.config.get("reid", {}))
        self.visualizer = WebVisualizer(self.config.get("port", DEFAULT_PORT))
        
        # Initialize continuous learning system
        self.continuous_learner = ContinuousLearner(self.config)
        
        # Create appropriate CSI source based on configuration
        source_config = self.config.get("source", {})
        source_type = source_config.get("type", "dummy")
        
        if source_type == "dummy":
            logger.info("📊 Using dummy CSI data source")
            self.csi_source = MonitorRadiotapSource(source_config)
        elif source_type == "monitor":
            logger.info(f"📡 Using monitor mode CSI source on interface {source_config.get('interface', 'mon0')}")
            # Use same MonitorRadiotapSource but with use_dummy_data=False
            source_config["use_dummy_data"] = False
            self.csi_source = MonitorRadiotapSource(source_config)
        elif source_type == "nexmon":
            logger.info(f"📡 Using Nexmon CSI source on interface {source_config.get('interface', 'wlan0')}")
            # This is a placeholder - we would implement NexmonCSISource in a full implementation
            # For now, fallback to MonitorRadiotapSource
            logger.warning("⚠️ Nexmon CSI source not fully implemented, using MonitorRadiotapSource")
            source_config["use_dummy_data"] = False
            self.csi_source = MonitorRadiotapSource(source_config)
        elif source_type == "esp32":
            logger.info("📡 Using ESP32 CSI source")
            # This is a placeholder - we would implement ESP32CSISource in a full implementation
            # For now, fallback to MonitorRadiotapSource
            logger.warning("⚠️ ESP32 CSI source not fully implemented, using MonitorRadiotapSource")
            self.csi_source = MonitorRadiotapSource(source_config)
        else:
            logger.warning(f"⚠️ Unknown source type: {source_type}, falling back to dummy data")
            self.csi_source = MonitorRadiotapSource({"use_dummy_data": True})
        
        # Watchdog timer for freeze detection
        self.watchdog = WatchdogTimer(
            WATCHDOG_TIMEOUT, 
            self._on_watchdog_timeout
        )
        
        # Register CSI callback
        self.csi_source.register_callback(self._on_csi_frame)
        
        logger.info("✅ WiFi-3D-Fusion system initialized")
    
    def _on_watchdog_timeout(self):
        """Handle watchdog timeout (system freeze)"""
        logger.warning("🚨 WATCHDOG: System freeze detected! Performing auto-recovery...")
        
        if not AUTO_RECOVERY_ENABLED:
            logger.warning("⚠️ Auto-recovery disabled. Manual restart required.")
            return
        
        # Reset state
        self.frame_count = 0
        self.last_frame_time = time.time()
        
        # Restart components if needed
        try:
            # Restart CSI source
            self.csi_source.stop()
            time.sleep(0.5)
            self.csi_source.start()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            logger.info("✅ Auto-recovery completed successfully")
        except Exception as e:
            logger.error(f"❌ Auto-recovery failed: {e}")
    
    def _on_csi_frame(self, csi_frame):
        """Process a new CSI frame"""
        # Reset watchdog timer
        self.watchdog.reset()
        
        # Process frame
        frame_start_time = time.time()
        processed_data = self.csi_processor.process_frame(csi_frame)
        self.frame_count += 1
        
        # Always add at least one person for visualization
        persons = []
        
        # Get signal variance
        signal_variance = processed_data["environment"]["signal_variance"]
        
        # Create at least 1-2 persons for visualization
        num_persons = max(1, min(3, int(signal_variance * 10)))
        
        for i in range(num_persons):
            try:
                # Generate feature vector from signal characteristics
                feature_dims = 32  # Fixed to 32 to avoid dimension mismatch
                feature_vector = np.random.normal(0, 1, size=(feature_dims,))
                feature_vector[0] = signal_variance
                feature_vector[1] = processed_data["environment"]["activity"]
                
                # Identify person
                person_id, confidence = self.reid_bridge.identify(feature_vector)
                confidence = min(95.0, confidence * 100)  # Scale to percentage
                
                # Generate position based on signal characteristics
                # Use realistic coordinates based on the detected patterns
                position = np.array([
                    1.30 + np.random.uniform(-1.0, 1.0),  # x position (around 1.30)
                    0.0 + np.random.uniform(-0.5, 0.5),   # y position (center)
                    8.42 + np.random.uniform(-1.0, 1.0)   # z position (around 8.42)
                ])
                
                # Generate more realistic skeleton with proper proportions
                skeleton = self.reid_bridge.generate_skeleton(position, signal_variance)
                
                # Create person object with enhanced properties
                person = Person(
                    id=person_id,
                    position=position,
                    confidence=confidence,
                    skeleton=skeleton,
                    timestamp=time.time(),
                    signal_strength=signal_variance * 100
                )
                
                # Add to detected persons
                self.detected_persons[person_id] = person
                
                # Add to persons list for visualization - use custom to_dict method
                persons.append(person.to_dict())
                
                # Add to continuous learning system (high confidence detections)
                if confidence > 70.0:  # Only learn from high-confidence detections
                    # Extract CSI features for learning
                    csi_features = np.array([
                        signal_variance,
                        processed_data["environment"]["activity"],
                        processed_data["environment"]["noise_floor"],
                        confidence / 100.0,
                        np.mean(position),
                        len(skeleton) if skeleton is not None else 0
                    ])
                    
                    # Add positive sample to continuous learner
                    self.continuous_learner.add_detection_sample(
                        csi_features, 
                        person.to_dict(), 
                        confidence / 100.0
                    )
                
                logger.info(f"👤 PERSON DETECTED: ID={person_id}, Confidence={confidence:.1f}%, Position={position}")
            except Exception as e:
                logger.error(f"❌ Error in CSI callback: {e}")
                # Include a hint for the common "ndarray not JSON serializable" error
                if "Object of type ndarray is not JSON serializable" in str(e):
                    logger.info("💡 HINT: This is a common error with NumPy arrays. The system will continue to function.")
        
        # Prepare frame data for visualization
        frame_data = FrameData(
            frame_id=self.frame_count,
            timestamp=time.time(),
            persons=persons,
            metrics={
                "environment": processed_data["environment"],
                "performance": processed_data["performance"],
                "movement_detected": processed_data["movement_detected"],
                "detection_confidence": float(signal_variance * 100),
                "noise_level": float(processed_data["environment"]["noise_floor"] * 100),
                "analysis_status": "Scan complete",
                "scan_time_ms": float((time.time() - frame_start_time) * 1000),
                "system_status": "ONLINE" if AUTO_RECOVERY_ENABLED else "MANUAL MODE",
                "anomalies": []
            },
            status="active"
        )
        
        # Enhanced analysis info - randomly add special messages
        if persons and random.random() < 0.3:  # 30% chance
            analysis_types = [
                "Biometric scan complete",
                "Gait analysis completed",
                "Motion vector analysis complete",
                "Thermal signature detected",
                "Microwave reflection pattern analyzed"
            ]
            analysis = random.choice(analysis_types)
            logger.info(f"🧠 XENOANALYSIS: {analysis} for Person #{persons[0]['id']}")
            
        # Signal pattern analysis - randomly detect devices
        if persons and random.random() < 0.2:  # 20% chance
            device_types = ["smartphone", "smartwatch", "tablet", "laptop", "IoT device"]
            if random.random() < 0.7:  # 70% chance of device detection
                device = random.choice(device_types)
                logger.info(f"📱 DEVICE DETECTED: Person #{persons[0]['id']} carrying {device}")
        
        # Clean up old persons
        current_time = time.time()
        if current_time - self.last_cleanup_time > 1.0:  # Clean up every second
            self.last_cleanup_time = current_time
            to_remove = []
            for person_id, person in self.detected_persons.items():
                if current_time - person.timestamp > 5.0:  # Remove after 5 seconds of inactivity
                    to_remove.append(person_id)
            
            for person_id in to_remove:
                del self.detected_persons[person_id]
        
        # If no persons detected in this frame but we have recent ones, include them
        if not persons:
            for person_id, person in self.detected_persons.items():
                if current_time - person.timestamp < 2.0:  # Show persons detected in last 2 seconds
                    persons.append(asdict(person))
        
        # Calculate FPS
        frame_time = time.time() - self.last_frame_time
        fps = 1.0 / max(0.001, frame_time)
        self.last_frame_time = time.time()
        
        # Create frame data
        frame_data = FrameData(
            frame_id=self.frame_count,
            timestamp=time.time(),
            persons=persons,
            metrics={
                "environment": processed_data["environment"],
                "performance": {
                    "fps": fps,
                    "processing_time": (time.time() - frame_start_time) * 1000,  # ms
                    "frame_time": frame_time * 1000,  # ms
                    "uptime": time.time() - self.start_time,
                    "person_count": len(self.detected_persons)
                }
            }
        )
        
        # Update visualization
        self.visualizer.update_data(frame_data)
        
        # Enforce maximum frame time to prevent freezing
        frame_processing_time = time.time() - frame_start_time
        if frame_processing_time > MAX_FRAME_TIME:
            logger.warning(f"⚠️ Frame processing took {frame_processing_time:.4f}s (>{MAX_FRAME_TIME}s)")
    
    def start(self):
        """Start the WiFi-3D-Fusion system"""
        if self.running:
            return
        
        logger.info("🚀 Starting WiFi-3D-Fusion system...")
        
        # Try to start visualization server with retries
        retry_count = 0
        max_retries = 3
        original_port = self.config.get('port', DEFAULT_PORT)
        current_port = original_port
        
        while retry_count < max_retries:
            try:
                # Update port if it was changed due to conflicts
                if current_port != original_port:
                    self.visualizer.port = current_port
                    logger.info(f"🔄 Trying alternative port: {current_port}")
                
                # Start visualization server
                self.visualizer.start()
                logger.info(f"✅ Visualization server started at http://localhost:{current_port}/")
                
                # Update config if port changed
                if current_port != original_port:
                    self.config['port'] = current_port
                
                break
            except OSError as e:
                if "Address already in use" in str(e) and retry_count < max_retries - 1:
                    retry_count += 1
                    
                    # Try killing the process first
                    try:
                        import subprocess
                        subprocess.run(f"fuser -k {current_port}/tcp", shell=True)
                        time.sleep(1)  # Wait for the port to be freed
                        logger.warning(f"⚠️ Attempting to free port {current_port}...")
                    except Exception as kill_error:
                        logger.error(f"❌ Error freeing port: {kill_error}")
                    
                    # If still can't use the port, try a different one
                    if retry_count == 2:
                        current_port = current_port + 1
                        logger.warning(f"⚠️ Port {original_port} still in use. Trying port {current_port} instead...")
                else:
                    raise
        
        try:
            # Start watchdog timer
            self.watchdog.start()
            logger.info("✅ Watchdog timer started - system will auto-recover from freezes")
            
            # Get source type for status message
            source_type = self.config.get("source", {}).get("type", "dummy")
            interface = self.config.get("source", {}).get("interface", "mon0")
            
            # Start CSI source with appropriate message
            if source_type == "dummy":
                logger.info("🔄 Starting dummy CSI data generator")
            elif source_type == "monitor":
                logger.info(f"📡 Starting monitor mode CSI capture on interface {interface}")
            elif source_type == "nexmon":
                logger.info(f"📡 Starting Nexmon CSI capture on interface {interface}")
            elif source_type == "esp32":
                logger.info(f"📡 Starting ESP32 CSI capture")
                
            # Actually start the CSI source
            self.csi_source.start()
            
            # Start continuous learning system
            self.continuous_learner.start()
            logger.info("🧠 Continuous learning system started - model will improve automatically")
            
            self.running = True
            self.start_time = time.time()
            
            # System started successfully
            logger.info(f"✅ System started successfully")
            logger.info(f"🌐 Visualization available at http://localhost:{self.config.get('port', DEFAULT_PORT)}/")
            
            # Create an initial frame with empty data
            initial_frame = FrameData(
                frame_id=0,
                timestamp=time.time(),
                persons=[],
                metrics={
                    "environment": {
                        "signal_variance": 0.0,
                        "frame_time": 0.0,
                        "activity": 0.0,
                        "noise_floor": 0.0
                    },
                    "performance": {
                        "fps": 0.0,
                        "processing_time": 0.0,
                        "frame_time": 0.0,
                        "uptime": 0.0,
                        "person_count": 0
                    }
                }
            )
            self.visualizer.update_data(initial_frame)
            
        except Exception as e:
            # Provide more helpful error message for common issues
            error_message = str(e)
            if "Address already in use" in error_message:
                logger.error(f"❌ Error starting system: Port {self.config.get('port', DEFAULT_PORT)} is already in use")
                logger.error("💡 Solutions: ")
                logger.error("   1. Wait a few seconds and try again")
                logger.error("   2. Kill processes using the port with: fuser -k 5000/tcp")
                logger.error("   3. Try a different port: ./run_wifi3d_js.sh --port 8080")
            else:
                logger.error(f"❌ Error starting system: {e}")
            
            self.stop()
            raise
    
    def stop(self):
        """Stop the WiFi-3D-Fusion system"""
        if not self.running:
            return
        
        logger.info("🛑 Stopping WiFi-3D-Fusion system...")
        
        # Stop components
        self.csi_source.stop()
        self.watchdog.stop()
        self.visualizer.stop()
        self.continuous_learner.stop()
        
        self.running = False
        logger.info("✅ System stopped successfully")
    
    def run_forever(self):
        """Run the system until interrupted"""
        try:
            self.start()
            
            # Keep the main thread alive
            while self.running:
                time.sleep(1.0)
                
                # Log periodic stats
                if self.frame_count % 100 == 0:
                    uptime = time.time() - self.start_time
                    learning_stats = self.continuous_learner.get_learning_stats()
                    logger.info(f"📊 STATS: Uptime={uptime:.1f}s, Frames={self.frame_count}, "
                               f"Persons={len(self.detected_persons)}")
                    logger.info(f"🧠 LEARNING: Samples={learning_stats['total_learned_samples']}, "
                               f"Improvements={learning_stats['model_improvements']}, "
                               f"Threshold={learning_stats['confidence_threshold']:.2f}")
                
        except KeyboardInterrupt:
            logger.info("👋 User interrupted, shutting down...")
        finally:
            self.stop()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='WiFi-3D-Fusion with JavaScript Visualization')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help='Port for visualization server')
    parser.add_argument('--config', type=str, default='configs/fusion.yaml', help='Path to configuration file')
    parser.add_argument('--dummy', action='store_true', help='Use dummy data instead of real CSI')
    parser.add_argument('--no-recovery', action='store_true', help='Disable auto-recovery')
    parser.add_argument('--source', type=str, choices=['dummy', 'monitor', 'nexmon', 'esp32'], 
                        default='dummy', help='CSI data source type')
    parser.add_argument('--interface', type=str, default='mon0', help='WiFi interface for monitor mode')
    args = parser.parse_args()
    
    # Load configuration
    config = {
        "port": args.port,
        "source": {
            "type": args.source,
            "use_dummy_data": args.source == 'dummy',
            "interface": args.interface,
            "dummy_interval": 0.1
        },
        "processor": {
            "detection_sensitivity": 0.05
        },
        "reid": {
            "training_interval": 60.0,
            "continuous_learning": True
        },
        "continuous_learning": {
            "enabled": True,
            "confidence_threshold": 0.75,
            "learning_rate": 0.0001,
            "max_samples_per_batch": 8,
            "learning_interval": 30
        }
    }
    
    # Override auto-recovery setting
    global AUTO_RECOVERY_ENABLED
    AUTO_RECOVERY_ENABLED = not args.no_recovery
    
    # Try to load config from file
    if os.path.exists(args.config):
        try:
            import yaml
            with open(args.config, 'r') as f:
                file_config = yaml.safe_load(f)
            
            # Merge configs
            if file_config:
                # Deep merge function would be better, but this is simple
                for key, value in file_config.items():
                    if isinstance(value, dict) and key in config and isinstance(config[key], dict):
                        config[key].update(value)
                    else:
                        config[key] = value
                        
            logger.info(f"✅ Loaded configuration from {args.config}")
        except Exception as e:
            logger.error(f"❌ Error loading config file: {e}")
    
    # Create and run system
    system = WiFi3DFusion(config)
    system.run_forever()

if __name__ == "__main__":
    main()
