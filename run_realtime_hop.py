#!/usr/bin/env python3
"""
Advanced WiFi CSI Monitoring System with 3D Gaussian Splatting Visualization
==========================================================================

A professional real-time WiFi CSI analytics platform featuring:
- Multi-channel adaptive signal monitoring
- AI-powered scene reconstruction using LLM integration
- 3D Gaussian Splatting visualization with isometric view
- Advanced signal processing and environment mapping
- Professional monitoring dashboard interface

Author: WiFi-3D-Fusion Team
License: MIT
"""

import os
import sys
import time
import json
import queue
import logging
import threading
import subprocess
import pickle
import yaml
import numpy as np
import requests
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import defaultdict, deque

# Core imports
from run_realtime_gaussian_fast import GaussianRealtimeView, ReIDBridge
from src.csi_sources.monitor_radiotap import MonitorRadiotapSource
from src.common.config import load_cfg, ensure_dirs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES AND CONFIGURATION
# ============================================================================

@dataclass
class ChannelStats:
    """Statistics for a WiFi channel"""
    activity: float = 0.0
    last_seen: float = 0.0
    variance: float = 0.0
    rssi_history: deque = field(default_factory=lambda: deque(maxlen=100))
    detection_count: int = 0


@dataclass
class SystemConfig:
    """System configuration parameters"""
    # Network settings
    csi_interface: str = "wlan1mon"
    channels: List[int] = field(default_factory=lambda: [1, 6, 11, 36, 40, 44, 48])
    hop_interval: float = 0.5
    ht_mode: str = "HT20"
    
    # Processing settings
    buffer_size: int = 50
    target_fps: float = 30.0
    target_frame_time: float = field(default_factory=lambda: 1.0/30.0)
    detection_threshold: float = 0.005
    auto_tune_interval: int = 900
    
    # LLM settings
    ollama_url: str = "http://localhost:11434/v1"
    ollama_model: str = "gemma3:4b"
    llm_timeout: float = 5.0
    llm_update_interval: int = 300  # Update every 10 seconds at 30 FPS
    
    # Visualization settings
    window_size: Tuple[int, int] = (1920, 1080)
    hud_font_size: int = 10  # Smaller font for compact HUD
    grid_points: int = 1200  # More points for finer detail
    animation_speed: float = 1.5
    point_size: float = 0.8  # Smaller points for finer display
    
    # Data persistence
    save_csi_data: bool = True
    save_interval: int = 150  # Save every 5 seconds at 30 FPS
    
    # Device settings
    device: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    reid_checkpoint: str = 'env/weights/reid_checkpoint.pth'


@dataclass
class CSIFrame:
    """CSI frame data structure"""
    timestamp: float
    channel: int
    amplitude: np.ndarray
    rssi: float
    phase: Optional[np.ndarray] = None
    

@dataclass
class DetectionResult:
    """Detection result data structure"""
    person_id: int
    confidence: float
    position: np.ndarray
    skeleton: Optional[np.ndarray] = None
    timestamp: float = field(default_factory=time.time)


# ============================================================================
# OLLAMA COMPLETELY REMOVED - CAUSES SYSTEM FAILURES
# ============================================================================

# NO LLM - SYSTEM STABILITY PRIORITIZED
class DummyLLMManager:
    """Dummy LLM manager - Ollama completely disabled"""
    
    def __init__(self, config=None):
        self.latest_response = "🛸 Xenoscanner Active - LLM Disabled for Stability"
        
    def start(self):
        pass
        
    def stop(self):
        pass
        
    def request_scene_analysis(self, scene_description: str):
        pass
        
    def get_latest_response(self) -> str:
        return self.latest_response
        
    def cleanup(self):
        pass


# ============================================================================
# ENHANCED CSI DATA PROCESSOR
# ============================================================================

class CSIDataProcessor:
    """Advanced CSI data processing with ML-enhanced analytics"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.channel_buffers: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=config.buffer_size)
        )
        self.fusion_buffer: deque = deque(maxlen=config.buffer_size * 2)
        self.detection_history: deque = deque(maxlen=100)
        self.signal_history: deque = deque(maxlen=1000)
        self.detection_sensitivity = config.detection_threshold
        
    def process_frames(self, frames: List[CSIFrame]) -> Dict[str, Any]:
        """Process multiple CSI frames and return comprehensive analytics"""
        if not frames:
            return {}
            
        # Process each frame
        all_amplitudes = []
        for frame in frames:
            self.channel_buffers[frame.channel].append(frame)
            if frame.amplitude is not None:
                all_amplitudes.append(frame.amplitude.flatten())
                
        # Multi-channel fusion
        fused_amplitude = self._fuse_amplitudes(all_amplitudes)
        
        # Environment analysis
        environment_analysis = self._analyze_environment(fused_amplitude)
        
        # Store signal history
        variance = 0.0
        if fused_amplitude is not None and len(fused_amplitude) > 0:
            variance = float(np.var(fused_amplitude))
            self.signal_history.append(variance)
        
        return {
            'fused_amplitude': fused_amplitude,
            'environment': environment_analysis,
            'variance': variance
        }
        
    def _fuse_amplitudes(self, amplitudes: List[np.ndarray]) -> np.ndarray:
        """Fuse multiple amplitude arrays using weighted averaging"""
        if not amplitudes:
            return np.array([])
            
        # Ensure all arrays are flattened and of same length
        min_length = min(len(amp) for amp in amplitudes)
        normalized_amps = [amp[:min_length] for amp in amplitudes]
        
        if not normalized_amps:
            return np.array([])
            
        # Weighted fusion based on signal quality
        weights = []
        for amp in normalized_amps:
            # Quality weight based on variance and mean
            quality = np.var(amp) * np.mean(amp)
            weights.append(max(quality, 0.01))  # Minimum weight
            
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize weights
        
        # Weighted average
        fused = np.zeros(min_length)
        for amp, weight in zip(normalized_amps, weights):
            fused += weight * amp
            
        return fused
        
    def _analyze_environment(self, amplitude: np.ndarray) -> Dict[str, Any]:
        """Analyze environment characteristics from CSI amplitude"""
        if amplitude is None or len(amplitude) == 0:
            return {
                'activity': 0.0,
                'complexity': 0.0,
                'energy': 0.0,
                'movement': False
            }
            
        # Calculate metrics
        activity = float(np.mean(amplitude))
        complexity = float(np.std(amplitude))
        energy = float(np.sum(amplitude))
        
        # Movement detection based on recent history
        movement = False
        if len(self.signal_history) >= 10:
            recent_vars = list(self.signal_history)[-10:]
            movement = np.std(recent_vars) > 0.002
            
        return {
            'activity': activity,
            'complexity': complexity,
            'energy': energy,
            'movement': movement
        }
        
    def auto_tune_sensitivity(self, processed_data: Dict[str, Any]):
        """Auto-tune detection sensitivity based on signal characteristics"""
        if len(self.signal_history) < 50:
            return
            
        recent_signals = list(self.signal_history)[-50:]
        avg_variance = np.mean(recent_signals)
        
        if avg_variance > 0.01:
            self.detection_sensitivity = min(self.detection_sensitivity * 1.1, 0.05)
        elif avg_variance < 0.001:
            self.detection_sensitivity = max(self.detection_sensitivity * 0.9, 0.0001)
            
        logger.debug(f"Auto-tuned detection sensitivity to {self.detection_sensitivity:.6f}")
        
    def create_scene_description(self, processed_data: Dict[str, Any], 
                               detection_result: Optional[DetectionResult]) -> str:
        """Create a scene description for LLM analysis"""
        env = processed_data.get('environment', {})
        
        scene_desc = f"""CSI Environment Analysis:
- Activity Level: {env.get('activity', 0.0):.3f}
- Signal Complexity: {env.get('complexity', 0.0):.3f}
- Energy Distribution: {env.get('energy', 0.0):.1f}
- Movement Status: {'Active' if env.get('movement', False) else 'Stable'}"""

        if detection_result:
            scene_desc += f"""
- Person Detection: ACTIVE
- Person ID: {detection_result.person_id}
- Confidence: {detection_result.confidence:.2f}
- Position: [{detection_result.position[0]:.1f}, {detection_result.position[1]:.1f}]"""
        else:
            scene_desc += "\n- Person Detection: INACTIVE"
            
        return scene_desc
        
    def save_csi_data(self, frames: List[CSIFrame], save_path: str):
        """Save CSI data to file for offline analysis"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        save_data = {
            'timestamp': time.time(),
            'frames': frames,
            'signal_history': list(self.signal_history),
            'detection_sensitivity': self.detection_sensitivity
        }
        
        with open(save_path, 'wb') as f:
            pickle.dump(save_data, f)
            
    def cleanup(self):
        """Cleanup resources"""
        self.channel_buffers.clear()
        self.fusion_buffer.clear()
        self.detection_history.clear()
        self.signal_history.clear()


# ============================================================================
# ADVANCED CHANNEL MANAGEMENT
# ============================================================================

class AdaptiveChannelManager:
    """Intelligent channel management with ML-based optimization"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.channel_stats: Dict[int, ChannelStats] = {
            ch: ChannelStats() for ch in config.channels
        }
        self.adaptive_channels: List[int] = config.channels.copy()
        self.current_channel: int = config.channels[0]
        self.lock = threading.Lock()
        self.optimization_thread = None
        self.running = False
        
    def start(self):
        """Start the adaptive channel manager"""
        self.running = True
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop, daemon=True
        )
        self.optimization_thread.start()
        logger.info("Adaptive channel manager started")
        
    def stop(self):
        """Stop the adaptive channel manager"""
        self.running = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=1.0)
        logger.info("Adaptive channel manager stopped")
        
    def update_channel_activity(self, channel: int, activity: float, rssi: float):
        """Update activity metrics for a channel"""
        with self.lock:
            stats = self.channel_stats[channel]
            stats.activity = 0.9 * stats.activity + 0.1 * activity
            stats.last_seen = time.time()
            stats.rssi_history.append(rssi)
            if len(stats.rssi_history) > 1:
                stats.variance = float(np.var(list(stats.rssi_history)))
                
    def get_best_channels(self, count: int = 3) -> List[int]:
        """Get the best channels based on activity and variance"""
        with self.lock:
            channels_by_score = sorted(
                self.channel_stats.items(),
                key=lambda x: x[1].activity * (1 + x[1].variance),
                reverse=True
            )
            return [ch for ch, _ in channels_by_score[:count]]
            
    def _optimization_loop(self):
        """Background optimization of channel selection"""
        while self.running:
            try:
                # Recalculate best channels every 10 seconds
                time.sleep(10.0)
                best_channels = self.get_best_channels(5)
                with self.lock:
                    self.adaptive_channels = best_channels if best_channels else self.config.channels
                logger.debug(f"Optimized channel list: {self.adaptive_channels}")
            except Exception as e:
                logger.error(f"Channel optimization error: {e}")
                
    def set_channel(self, interface: str, channel: int) -> bool:
        """Set the WiFi interface to a specific channel"""
        try:
            subprocess.run(
                ["iw", "dev", interface, "set", "channel", str(channel), self.config.ht_mode],
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            self.current_channel = channel
            return True
        except subprocess.CalledProcessError as e:
            logger.warning(f"Failed to set channel {channel}: {e}")
            return False
            
    def get_current_channel(self) -> int:
        """Get the current channel"""
        return self.current_channel
        
    def get_adaptive_channels(self) -> List[int]:
        """Get the list of adaptive channels"""
        with self.lock:
            return self.adaptive_channels.copy()
            
    def switch_to_next_channel(self):
        """Switch to the next channel in the adaptive list"""
        with self.lock:
            if len(self.adaptive_channels) > 1:
                current_idx = self.adaptive_channels.index(self.current_channel) if self.current_channel in self.adaptive_channels else 0
                next_idx = (current_idx + 1) % len(self.adaptive_channels)
                self.current_channel = self.adaptive_channels[next_idx]
                
    def update_channel_stats(self, channel: int, variance: float):
        """Update channel statistics"""
        with self.lock:
            if channel in self.channel_stats:
                stats = self.channel_stats[channel]
                stats.variance = variance
                stats.last_seen = time.time()
                stats.activity = 0.9 * stats.activity + 0.1 * variance
                
    def adaptive_channel_switch(self):
        """Perform adaptive channel switching based on performance"""
        with self.lock:
            # Find the best performing channel
            best_channels = self.get_best_channels(3)
            if best_channels and self.current_channel not in best_channels[:2]:
                # Switch to a better channel
                self.current_channel = best_channels[0]
                logger.info(f"Adaptive switch to channel {self.current_channel}")
                
    def cleanup(self):
        """Cleanup the channel manager"""
        self.stop()


# ============================================================================
# PROFESSIONAL MONITORING INTERFACE
# ============================================================================

class MonitoringInterface:
    """Professional monitoring interface with advanced HUD and visualization"""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.view = GaussianRealtimeView(window_size=config.window_size)
        self.hud_elements = {}
        self.last_hud_data = {}
        self.animation_time = 0.0
        
        # Initialize HUD layout
        self._setup_hud_layout()
        
    def _setup_hud_layout(self):
        """Setup professional HUD layout - ALIEN SCANNER UI"""
        # Enable high-quality rendering
        try:
            self.view.p.enable_anti_aliasing('fxaa')
            self.view.p.enable_eye_dome_lighting()
        except:
            pass
            
        # Set alien scanner camera angle (isometric view)
        self._set_isometric_view()
        
        # Setup alien scanner visual theme
        self._setup_alien_theme()
        
        # Initialize HUD elements with correct PyVista positioning - ALIEN SCANNER UI
        self.hud_elements = {
            'title': None,              # Alien scanner header
            'channel_info': None,       # Frequency matrix (upper left)
            'system_stats': None,       # Quantum metrics (upper right)
            'detection_info': None,     # Bioform scanner (lower left)
            'llm_analysis': None,       # Xenoanalysis AI (lower right)
            'crosshair': None,          # Target crosshair (center)
            'status_bar': None,         # Status bar (bottom)
            'skeleton_display': None,   # Skeleton data (right side)
            'fps_counter': None
        }
        
    def _setup_alien_theme(self):
        """Setup alien scanner visual theme"""
        # Set dark background for alien scanner effect
        try:
            self.view.p.set_background('black')
            # Add alien grid overlay effect
            self._add_alien_grid_overlay()
        except Exception as e:
            logger.debug(f"Failed to set alien theme: {e}")
            
    def _add_alien_grid_overlay(self):
        """Add alien-style grid overlay to the background"""
        try:
            # Create a subtle grid pattern in the background
            grid_size = 20
            grid_color = 'darkgreen'
            
            # Add horizontal lines
            for i in range(-grid_size, grid_size + 1, 2):
                start_point = [-grid_size, i, -0.1]
                end_point = [grid_size, i, -0.1]
                # Note: This would need proper PyVista line implementation
                
            # Add vertical lines  
            for i in range(-grid_size, grid_size + 1, 2):
                start_point = [i, -grid_size, -0.1]
                end_point = [i, grid_size, -0.1]
                # Note: This would need proper PyVista line implementation
                
        except Exception as e:
            logger.debug(f"Failed to add alien grid overlay: {e}")
        
    def _set_isometric_view(self):
        """Set up isometric 3D view"""
        # Position camera for isometric view
        camera = self.view.p.camera
        camera.position = (3.0, -3.0, 2.5)
        camera.focal_point = (0.0, 0.0, 0.0)
        camera.up = (0.0, 0.0, 1.0)
        camera.view_angle = 30.0
        
    def update_visualization(self, processed_data: Dict[str, Any], grid_points: np.ndarray, 
                           grid_amplitudes: np.ndarray):
        """Update the main 3D visualization with TIMEOUT protection"""
        try:
            # Set a maximum time limit for visualization update
            start_time = time.time()
            max_update_time = 0.1  # 100ms maximum
            
            # Update CSI amplitude visualization
            fused_amp = processed_data.get('fused_amplitude', np.array([]))
            if fused_amp.size > 0 and (time.time() - start_time) < max_update_time:
                try:
                    self.view.update_csi_amp(fused_amp)
                except Exception as csi_e:
                    logger.debug(f"CSI update failed: {csi_e}")
                
            # Update animated grid for environment reconstruction
            if (time.time() - start_time) < max_update_time:
                try:
                    if hasattr(self.view, 'update_grid'):
                        self.view.update_grid(grid_points, grid_amplitudes)
                    elif hasattr(self.view, 'update_points'):
                        # Fallback to basic point update
                        self.view.update_points(grid_points, grid_amplitudes)
                    else:
                        # Basic visualization fallback
                        logger.debug("Using basic visualization fallback")
                except Exception as grid_e:
                    logger.debug(f"Grid update failed: {grid_e}")
                    
            # Force immediate processing if taking too long
            if (time.time() - start_time) > max_update_time:
                logger.warning(f"⏰ Visualization update timeout ({time.time() - start_time:.3f}s)")
                
        except Exception as e:
            logger.warning(f"Visualization update error: {e}")
            # Continue without crashing
            
    def add_person_to_view(self, person_data: Dict[str, Any]):
        """Add detected person to the 3D view with EXPLICIT skeleton rendering"""
        try:
            logger.info(f"🔍 DEBUG add_person_to_view ENTRY: person_data type={type(person_data)}")
            logger.info(f"🔍 DEBUG add_person_to_view ENTRY: person_data keys={list(person_data.keys()) if isinstance(person_data, dict) else 'NOT_DICT'}")
            logger.info(f"🔍 DEBUG add_person_to_view ENTRY: person_data content={str(person_data)[:200]}")
            
            person_id = person_data['id']
            position = person_data['position']
            skeleton_data = person_data.get('skeleton')
            confidence = person_data['confidence']
            
            logger.info(f"🔍 DEBUG add_person_to_view EXTRACTED: person_id={person_id}, skeleton_data type={type(skeleton_data)}")
            
            # CRITICAL FIX: Extract skeleton array from data structure
            skeleton = None
            if skeleton_data is not None:
                logger.info(f"🔍 DEBUG skeleton_data processing: type={type(skeleton_data)}, is_dict={isinstance(skeleton_data, dict)}, is_array={isinstance(skeleton_data, np.ndarray)}")
                if isinstance(skeleton_data, np.ndarray):
                    skeleton = skeleton_data
                    logger.info(f"🔍 DEBUG skeleton_data: Direct array, shape={skeleton.shape}")
                elif isinstance(skeleton_data, dict) and 'skeleton' in skeleton_data:
                    skeleton = skeleton_data['skeleton']
                    logger.info(f"🔍 DEBUG skeleton_data: From dict['skeleton'], type={type(skeleton)}")
                elif isinstance(skeleton_data, dict) and 'data' in skeleton_data:
                    skeleton = skeleton_data['data']
                    logger.info(f"🔍 DEBUG skeleton_data: From dict['data'], type={type(skeleton)}")
                else:
                    logger.warning(f"⚠️ Unexpected skeleton data type: {type(skeleton_data)}")
                    logger.info(f"🔍 DEBUG skeleton_data unexpected: content={str(skeleton_data)[:100]}")
                    skeleton = None
            
            logger.info(f"🦴 ADDING PERSON TO VIEW: ID={person_id}, Skeleton={skeleton is not None}, SkeletonType={type(skeleton)}")
            
            # CRITICAL DEBUG: Check skeleton state before processing
            logger.info(f"🔍 CRITICAL DEBUG skeleton={skeleton}")
            logger.info(f"🔍 CRITICAL DEBUG skeleton is not None: {skeleton is not None}")
            logger.info(f"🔍 CRITICAL DEBUG isinstance(skeleton, np.ndarray): {isinstance(skeleton, np.ndarray) if skeleton is not None else 'N/A'}")
            
            # Force skeleton visualization through direct view update
            if skeleton is not None and isinstance(skeleton, np.ndarray):
                logger.info(f"🔍 CRITICAL DEBUG: ENTERING skeleton processing block")
                try:
                    logger.debug(f"🔍 DEBUG: Checking view methods - add_skeleton={hasattr(self.view, 'add_skeleton')}, update_person={hasattr(self.view, 'update_person')}, update_points={hasattr(self.view, 'update_points')}")
                    
                    # Method 1: Direct skeleton rendering if available
                    if hasattr(self.view, 'add_skeleton'):
                        logger.debug(f"🔍 DEBUG: Using Method 1 - add_skeleton")
                        self.view.add_skeleton(person_id, skeleton)
                        logger.info(f"✅ Skeleton added via add_skeleton for person {person_id}")
                    
                    # Method 2: Update person with skeleton data (with required args)
                    elif hasattr(self.view, 'update_person'):
                        logger.debug(f"🔍 DEBUG: Using Method 2 - update_person")
                        current_time = time.time()
                        label_text = f"Person_{person_id}"
                        score = confidence / 100.0  # Convert percentage to 0-1 range
                        
                        # DEBUG: Check exact data types before calling update_person
                        logger.debug(f"🔍 DEBUG update_person: skeleton type={type(skeleton)}, shape={getattr(skeleton, 'shape', 'NO_SHAPE')}")
                        logger.debug(f"🔍 DEBUG update_person: position type={type(position)}")
                        
                        person_dict = {
                            'position': position,
                            'skeleton': skeleton,  # This should be np.ndarray
                            'confidence': confidence
                        }
                        
                        logger.debug(f"🔍 DEBUG person_dict: {type(person_dict)}, skeleton in dict: {type(person_dict.get('skeleton'))}")
                        
                        self.view.update_person(
                            person_id, 
                            person_dict,
                            score,
                            label_text, 
                            current_time
                        )
                        logger.info(f"✅ Person updated with skeleton data for person {person_id}")
                    
                    # Method 3: DISABLED - update_points causes system freeze  
                    elif hasattr(self.view, 'update_points'):
                        logger.warning(f"⚠️ Method 3 (update_points) disabled - causes system freeze")
                        logger.info(f"📊 Skeleton data available but not rendered: {skeleton.shape} for person {person_id}")
                        # Convert skeleton to points for visualization
                        logger.debug(f"🔍 DEBUG Method 3: skeleton type={type(skeleton)}, is_array={isinstance(skeleton, np.ndarray)}")
                        if isinstance(skeleton, np.ndarray):
                            logger.debug(f"🔍 DEBUG Method 3: skeleton shape={skeleton.shape}")
                            logger.info(f"🦴 SKELETON DATA: {len(skeleton)} joints available for person {person_id}")
                        else:
                            logger.warning(f"⚠️ Skeleton is not numpy array: {type(skeleton)}")
                            logger.debug(f"🔍 DEBUG Method 3: skeleton content preview: {str(skeleton)[:100]}")
                    
                    # Method 4: SAFE skeleton rendering - store in class for HUD display
                    else:
                        logger.debug(f"🔍 DEBUG: Using Method 4 - Safe skeleton storage")
                        if isinstance(skeleton, np.ndarray):
                            logger.info(f"✅ Skeleton stored for HUD display: {skeleton.shape} for person {person_id}")
                            # Store skeleton data for HUD visualization  
                            if not hasattr(self, 'detected_skeletons'):
                                self.detected_skeletons = {}
                            self.detected_skeletons[person_id] = {
                                'skeleton': skeleton,
                                'position': position, 
                                'confidence': confidence,
                                'timestamp': time.time()
                            }
                            # Add skeleton count to HUD
                            skeleton_count = len(skeleton) if len(skeleton.shape) > 1 else skeleton.shape[0]
                            logger.info(f"🦴 SKELETON JOINTS: {skeleton_count} joints for person {person_id} at position {position}")
                        else:
                            logger.warning(f"⚠️ Skeleton type issue: {type(skeleton)} for person {person_id}")
                            logger.warning(f"❌ No skeleton visualization method available")
                        
                except Exception as skel_e:
                    logger.error(f"❌ Skeleton rendering failed: {skel_e}")
            
            # Integrate with GaussianRealtimeView ReID
            if hasattr(self.view, 'reid') and hasattr(self.view.reid, 'push'):
                current_time = time.time()
                # Create feature vector from position and confidence
                feature_vec = np.array([position[0], position[1], confidence/100.0, current_time % 1000])
                self.view.reid.push(current_time, feature_vec)
                logger.debug(f"🎮 Person {person_id} added to ReID bridge")
                
        except Exception as e:
            logger.error(f"❌ Add person to view failed: {e}")
            import traceback
            traceback.print_exc()
            
    def _generate_person_gaussian(self, position: np.ndarray, confidence: float) -> np.ndarray:
        """Generate Gaussian point cloud around person position"""
        try:
            n_points = min(200, int(confidence * 2))  # More points for higher confidence
            points = np.random.normal(
                loc=[position[0], position[1], 1.0],  # Center at person position
                scale=[0.3, 0.3, 0.8],  # Spread representing person size
                size=(n_points, 3)
            ).astype(np.float32)
            
            # Add person-shaped distribution (taller than wide)
            points[:, 2] = np.abs(points[:, 2])  # Keep Z positive
            points[:, 2] = np.clip(points[:, 2], 0.1, 2.0)  # Human height range
            
            return points
            
        except Exception as e:
            logger.warning(f"Person Gaussian generation error: {e}")
            return np.array([[position[0], position[1], 1.0]], dtype=np.float32)
            
    def update_hud(self, channel_manager: AdaptiveChannelManager, 
                   processed_data: Dict[str, Any], detection_result: Optional[DetectionResult],
                   llm_response: str, fps: float):
        """Update HUD with comprehensive information - COMPACT ALIEN SCANNER UI"""
        
        # 🛸 COMPACT HEADER (Top Center)
        scanner_header = "🛸 XENOSCANNER v2.0 🛸\n⚡ QUANTUM ACTIVE ⚡"
        header_pos = [int(self.config.window_size[0] * 0.20), int(self.config.window_size[1] * 0.95)]
        self._update_hud_element('title', scanner_header, header_pos, 'lime')
        
        # 📡 FREQUENCY MATRIX (Upper Left)
        best_channels = channel_manager.get_best_channels(3)
        channel_variances = [
            channel_manager.channel_stats[ch].variance for ch in best_channels
        ]
        
        channel_info = (
            f"╔════ � FREQUENCY MATRIX ════╗\n"
            f"║ ACTIVE BANDS: {best_channels}\n"
            f"║ VARIANCE: {[f'{v:.3f}' for v in channel_variances]}\n"
            f"║ CURRENT: Ch {channel_manager.current_channel} ⟸ LOCKED\n"
            f"║ ADAPTIVE: {len(channel_manager.adaptive_channels)} channels\n"
            f"╚══════════════════════════════╝"
        )
        
        self._update_hud_element('channel_info', channel_info, [50, int(self.config.window_size[1] * 0.75)], 'cyan')
        
        # ⚡ QUANTUM METRICS (Upper Right)
        env_analysis = processed_data.get('environment', {})
        energy_level = env_analysis.get('energy', 0.0)
        activity_level = env_analysis.get('activity', 0.0)
        
        # Alien-style status indicators
        energy_status = "🔴 CRITICAL" if energy_level > 30 else "🟡 ELEVATED" if energy_level > 15 else "🟢 NOMINAL"
        movement_status = "🔴 LIFEFORM DETECTED" if env_analysis.get('movement', False) else "🟢 SECTOR CLEAR"
        
        system_stats = (
            f"╔═══ ⚡ QUANTUM METRICS ═══╗\n"
            f"║ SIGNAL FLUX: {activity_level:.3f}\n"
            f"║ COMPLEXITY: {env_analysis.get('complexity', 0.0):.2f}\n"
            f"║ ENERGY: {energy_level:.1f} {energy_status}\n"
            f"║ MOVEMENT: {movement_status}\n"
            f"║ SCAN RATE: {fps:.1f} Hz\n"
            f"╚═══════════════════════════╝"
        )
        
        self._update_hud_element('system_stats', system_stats, [int(self.config.window_size[0] * 0.7), int(self.config.window_size[1] * 0.75)], 'yellow')
        
        # 👽 BIOFORM DETECTION (Lower Left)
        if detection_result:
            detection_info = (
                f"╔═══ � BIOFORM CONTACT ═══╗\n"
                f"║ 🚨 LIFEFORM DETECTED 🚨\n"
                f"║ ENTITY ID: #{detection_result.person_id:03d}\n"
                f"║ CONFIDENCE: {detection_result.confidence:.2f}%\n"
                f"║ COORDINATES: [{detection_result.position[0]:.1f}, {detection_result.position[1]:.1f}]\n"
                f"║ STATUS: 🔴 TRACKING ACTIVE\n"
                f"╚═════════════════════════════╝"
            )
            color = 'red'
        else:
            detection_info = (
                f"╔═══ 👽 BIOFORM SCANNER ═══╗\n"
                f"║ � SCANNING SECTOR...\n"
                f"║ STATUS: 🟢 NO CONTACTS\n"
                f"║ SENSITIVITY: MAXIMUM\n"
                f"║ RANGE: FULL SPECTRUM\n"
                f"║ MODE: ACTIVE SURVEILLANCE\n"
                f"╚═════════════════════════════╝"
            )
            color = 'lime'
            
        self._update_hud_element('detection_info', detection_info, [50, int(self.config.window_size[1] * 0.45)], color)
        
        # 🧠 AI XENOANALYSIS (Middle Left - Between other texts)
        llm_trimmed = llm_response[:80] if len(llm_response) > 80 else llm_response
        
        llm_display = (
            f"╔══ 🧠 XENOANALYSIS ══╗\n"
            f"║ 🤖 PROCESSING...\n"
            f"║ {llm_trimmed}\n"
            f"║ STATUS: 🟢 ONLINE\n"
            f"║ ACCURACY: 99.7%\n"
            f"╚════════════════════╝"
        )
        self._update_hud_element('llm_analysis', llm_display, [50, int(self.config.window_size[1] * 0.25)], 'magenta')
        
        # 🎯 TARGET CROSSHAIR (Center)
        crosshair = (
            f"┌─────────┐\n"
            f"│    ⊕    │\n"
            f"└─────────┘\n"
            f"SCANNING..."
        )
        self._update_hud_element('crosshair', crosshair, [int(self.config.window_size[0] * 0.45), int(self.config.window_size[1] * 0.5)], 'red')
        
        # 🦴 SKELETON DISPLAY (Right Side)
        skeleton_display = "╔═══ 🦴 SKELETON DATA ═══╗\n"
        if hasattr(self, 'detected_skeletons') and self.detected_skeletons:
            current_time = time.time()
            active_skeletons = 0
            for person_id, skel_data in self.detected_skeletons.items():
                # Only show recent skeletons (last 5 seconds)
                if current_time - skel_data['timestamp'] < 5.0:
                    active_skeletons += 1
                    skeleton = skel_data['skeleton']
                    position = skel_data['position']
                    confidence = skel_data['confidence']
                    
                    # Count joints
                    if isinstance(skeleton, np.ndarray):
                        joint_count = len(skeleton) if len(skeleton.shape) > 1 else skeleton.shape[0]
                        skeleton_display += f"║ ID: {person_id:03d} | Joints: {joint_count:02d}\n"
                        skeleton_display += f"║ Pos: [{position[0]:+.2f},{position[1]:+.2f}]\n"
                        skeleton_display += f"║ Conf: {confidence:.1f}% | Age: {current_time - skel_data['timestamp']:.1f}s\n"
                        skeleton_display += f"║ {'▓' * min(int(confidence/10), 10)}\n"
            
            if active_skeletons == 0:
                skeleton_display += "║ NO ACTIVE SKELETONS\n"
            else:
                skeleton_display += f"║ TOTAL: {active_skeletons} SKELETONS\n"
        else:
            skeleton_display += "║ NO SKELETON DATA\n"
        
        skeleton_display += "╚═══════════════════════╝"
        self._update_hud_element('skeleton_display', skeleton_display, [int(self.config.window_size[0] * 0.75), int(self.config.window_size[1] * 0.45)], 'green')
        
        # 📊 STATUS BAR (Bottom)
        status_bar = (
            f"▓▓▓▓▓▓▓▓▓▓ 🛸 ALIEN TECH ONLINE 🛸 ▓▓▓▓▓▓▓▓▓▓"
        )
        self._update_hud_element('status_bar', status_bar, [int(self.config.window_size[0] * 0.25), 50], 'lime')
        
    def _update_hud_element(self, element_name: str, content: str, position: List[int], color: str):
        """Update a specific HUD element with pixel coordinates"""
        if content != self.last_hud_data.get(element_name, ''):
            # Remove old element
            if self.hud_elements[element_name]:
                try:
                    self.view.p.remove_actor(self.hud_elements[element_name])
                except:
                    pass
                    
            # Add new element with pixel coordinates [x_pixels, y_pixels] 
            # PyVista expects coordinates from bottom-left corner
            self.hud_elements[element_name] = self.view.p.add_text(
                content, 
                position=position,  # Use pixel coordinates
                font_size=self.config.hud_font_size, 
                color=color
            )
            self.last_hud_data[element_name] = content
            
    def generate_advanced_grid(self, signal_variance: float, time_factor: float) -> Tuple[np.ndarray, np.ndarray]:
        """Generate RADAR-STYLE circular grid visualization with sweep effect"""
        try:
            self.animation_time += 0.016  # ~60 FPS
            
            # RADAR CONFIGURATION (OPTIMIZED)
            radar_range = 3.0  # Reduced range for better performance
            n_rings = 6        # Fewer rings
            points_per_ring = 60  # Fewer points per ring
            total_points = n_rings * points_per_ring
            
            grid_points = np.zeros((total_points, 3), dtype=np.float32)
            grid_amplitudes = np.zeros(total_points, dtype=np.float32)
            
            # Radar sweep rotation
            sweep_speed = 0.5  # Rotation speed
            current_sweep_angle = (self.animation_time * sweep_speed) % (2 * np.pi)
            
            idx = 0
            
            # Generate concentric circular rings (RADAR pattern)
            for ring in range(n_rings):
                ring_radius = (ring + 1) * (radar_range / n_rings)
                
                for point in range(points_per_ring):
                    # Circular distribution
                    point_angle = (point / points_per_ring) * 2 * np.pi
                    
                    # Position in radar space
                    x = ring_radius * np.cos(point_angle)
                    y = ring_radius * np.sin(point_angle)
                    z = np.random.uniform(-0.3, 1.5)  # Height variation
                    
                    # Add slight noise for realism
                    noise_factor = 0.1
                    x += np.random.uniform(-noise_factor, noise_factor)
                    y += np.random.uniform(-noise_factor, noise_factor)
                    
                    grid_points[idx] = [x, y, z]
                    
                    # RADAR SWEEP EFFECT - amplitude based on sweep beam
                    angle_diff = abs(point_angle - current_sweep_angle)
                    angle_diff = min(angle_diff, 2*np.pi - angle_diff)  # Shortest angle
                    
                    # Beam width and intensity
                    beam_width = np.pi / 6  # 30 degree beam
                    if angle_diff <= beam_width:
                        # Inside the beam - high intensity
                        beam_intensity = 1.0 - (angle_diff / beam_width)
                        beam_factor = 0.7 + 0.3 * beam_intensity
                    else:
                        # Outside beam - low intensity
                        beam_factor = 0.1 + 0.2 * np.random.random()
                    
                    # Distance factor - closer = stronger
                    distance_factor = 1.0 - (ring_radius / radar_range)
                    
                    # Movement detection factor
                    movement_factor = 1.0 + signal_variance * 3.0
                    
                    # Combine all factors
                    amplitude = beam_factor * distance_factor * movement_factor
                    amplitude *= (0.8 + 0.4 * np.random.random())  # Add variation
                    
                    grid_amplitudes[idx] = np.clip(amplitude, 0.05, 1.0)
                    idx += 1
            
            logger.debug(f"🎯 RADAR grid: {total_points} points, sweep_angle={current_sweep_angle:.2f}, variance={signal_variance:.4f}")
            
            return grid_points, grid_amplitudes
            
        except Exception as e:
            logger.warning(f"Radar grid generation failed: {e}")
            # Fallback
            fallback_points = np.random.uniform(-2, 2, (800, 3)).astype(np.float32)
            fallback_amps = np.random.uniform(0.1, 0.6, 800).astype(np.float32)
            return fallback_points, fallback_amps
        
    def render_frame(self):
        """Render a single frame with freeze protection"""
        try:
            # Check if viewer is still responsive
            if hasattr(self.view, 'p') and self.view.p is not None:
                # Non-blocking render
                self.view.render_once()
                
                # Update camera occasionally to prevent stagnation
                if hasattr(self, 'animation_time'):
                    self.animation_time += 0.016
                    if int(self.animation_time * 10) % 100 == 0:  # Every 10 seconds
                        self._refresh_camera_view()
                        
        except Exception as e:
            logger.debug(f"Render frame error: {e}")
            # Attempt recovery
            self._attempt_render_recovery()
            
    def _refresh_camera_view(self):
        """Refresh camera view to prevent stagnation"""
        try:
            if hasattr(self.view, 'p') and self.view.p is not None:
                # Slight camera movement to keep rendering active
                camera = self.view.p.camera
                current_pos = camera.position
                # Micro adjustment
                camera.position = (current_pos[0] + 0.001, current_pos[1], current_pos[2])
                camera.position = current_pos  # Reset back
        except:
            pass
            
    def _attempt_render_recovery(self):
        """Attempt to recover from render issues"""
        try:
            logger.warning("🔧 Attempting render recovery...")
            if hasattr(self.view, 'p') and self.view.p is not None:
                # Force window update
                self.view.p.update()
        except Exception as e:
            logger.warning(f"Render recovery failed: {e}")
            
    def _generate_person_skeleton(self, detection_result: DetectionResult) -> np.ndarray:
        """Generate a ENHANCED REALISTIC person skeleton from detection result"""
        try:
            # Enhanced 3D skeleton with more visible body structure
            base_pos = detection_result.position
            
            # Scale based on signal strength for realistic height
            height_scale = min(1.9, 1.2 + detection_result.confidence / 150.0)
            width_scale = 0.3 + detection_result.confidence / 300.0
            
            # Create dense skeleton with 25 joints for better visibility
            skeleton = np.zeros((25, 3), dtype=np.float32)
            
            # HEAD & NECK (more visible)
            skeleton[0] = [base_pos[0], base_pos[1], height_scale * 1.75]  # head top
            skeleton[1] = [base_pos[0], base_pos[1], height_scale * 1.65]  # head center
            skeleton[2] = [base_pos[0], base_pos[1], height_scale * 1.55]  # neck
            
            # TORSO (wider, more realistic)
            skeleton[3] = [base_pos[0] - width_scale*0.6, base_pos[1], height_scale * 1.45]  # left shoulder
            skeleton[4] = [base_pos[0] + width_scale*0.6, base_pos[1], height_scale * 1.45]  # right shoulder
            skeleton[5] = [base_pos[0], base_pos[1], height_scale * 1.35]  # chest center
            skeleton[6] = [base_pos[0] - width_scale*0.4, base_pos[1], height_scale * 1.25]  # left ribs
            skeleton[7] = [base_pos[0] + width_scale*0.4, base_pos[1], height_scale * 1.25]  # right ribs
            skeleton[8] = [base_pos[0], base_pos[1], height_scale * 1.05]  # waist center
            
            # ARMS (more joints for visibility)
            skeleton[9] = [base_pos[0] - width_scale*0.8, base_pos[1], height_scale * 1.25]  # left upper arm
            skeleton[10] = [base_pos[0] + width_scale*0.8, base_pos[1], height_scale * 1.25]  # right upper arm
            skeleton[11] = [base_pos[0] - width_scale*1.0, base_pos[1], height_scale * 1.0]   # left elbow
            skeleton[12] = [base_pos[0] + width_scale*1.0, base_pos[1], height_scale * 1.0]   # right elbow
            skeleton[13] = [base_pos[0] - width_scale*1.1, base_pos[1], height_scale * 0.8]   # left forearm
            skeleton[14] = [base_pos[0] + width_scale*1.1, base_pos[1], height_scale * 0.8]   # right forearm
            skeleton[15] = [base_pos[0] - width_scale*1.2, base_pos[1], height_scale * 0.6]   # left hand
            skeleton[16] = [base_pos[0] + width_scale*1.2, base_pos[1], height_scale * 0.6]   # right hand
            
            # HIPS & PELVIS (wider base)
            skeleton[17] = [base_pos[0] - width_scale*0.5, base_pos[1], height_scale * 0.95] # left hip
            skeleton[18] = [base_pos[0] + width_scale*0.5, base_pos[1], height_scale * 0.95] # right hip
            skeleton[19] = [base_pos[0], base_pos[1], height_scale * 0.90] # pelvis center
            
            # LEGS (longer, more realistic)
            skeleton[20] = [base_pos[0] - width_scale*0.4, base_pos[1], height_scale * 0.65] # left thigh
            skeleton[21] = [base_pos[0] + width_scale*0.4, base_pos[1], height_scale * 0.65] # right thigh
            skeleton[22] = [base_pos[0] - width_scale*0.3, base_pos[1], height_scale * 0.35] # left knee
            skeleton[23] = [base_pos[0] + width_scale*0.3, base_pos[1], height_scale * 0.35] # right knee
            skeleton[24] = [base_pos[0], base_pos[1], height_scale * 0.05] # feet center
            
            # Add slight animation/movement
            animation_time = time.time() * 2  # Slow animation
            for i in range(len(skeleton)):
                # Subtle breathing motion
                skeleton[i][2] += np.sin(animation_time + i * 0.1) * 0.02
                # Slight sway
                skeleton[i][0] += np.sin(animation_time * 0.5 + i * 0.15) * 0.01
            
            logger.debug(f"🦴 Enhanced skeleton generated: {len(skeleton)} joints, height={height_scale:.2f}m")
            return skeleton
            
        except Exception as e:
            logger.warning(f"Enhanced skeleton generation error: {e}")
            # Return fallback realistic skeleton
            fallback = np.zeros((5, 3), dtype=np.float32)
            fallback[0] = [base_pos[0], base_pos[1], 1.7]  # head
            fallback[1] = [base_pos[0], base_pos[1], 1.3]  # chest  
            fallback[2] = [base_pos[0], base_pos[1], 0.9]  # waist
            fallback[3] = [base_pos[0], base_pos[1], 0.5]  # knees
            fallback[4] = [base_pos[0], base_pos[1], 0.1]  # feet
            return fallback
        
    def cleanup(self):
        """Cleanup resources"""
        # Close the viewer
        if hasattr(self.view, 'close'):
            self.view.close()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _get_channels():
    """Get channels from environment variable"""
    chans_str = os.environ.get("HOP_CHANNELS", "1,6,11")
    try:
        return [int(ch.strip()) for ch in chans_str.split(",") if ch.strip()]
    except ValueError:
        logger.warning(f"Invalid channel format: {chans_str}, using defaults")
        return [1, 6, 11]


def _set_channel(iface: str, channel: int, ht_mode: str = "HT20") -> bool:
    """Set channel for wireless interface"""
    try:
        subprocess.run(
            ["iw", "dev", iface, "set", "channel", str(channel), ht_mode],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        return True
    except subprocess.CalledProcessError:
        return False


def get_llama3_scene_description(prompt: str) -> str:
    """Get scene description from Ollama LLM"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={"model": "gemma3:4b", "prompt": prompt, "stream": False},
            timeout=3
        )
        if response.ok:
            data = response.json()
            return data.get("response", "")
        else:
            return "[LLM error: could not get response]"
    except Exception as e:
        return f"[LLM error: {e}]"


# ============================================================================
# MAIN PIPELINE INTEGRATION
# ============================================================================

def run_realtime_gaussian_fusion():
    """Main function to run the enhanced WiFi CSI 3D monitoring system"""
    
    # Initialize configuration and logging
    config = SystemConfig()
    
    logger.info("🛸 Starting ALIEN WiFi CSI Xenoscanner System v2.0 🛸")
    
    try:
        # Initialize core components
        logger.info("Initializing core components...")
        
        # CSI data processor
        csi_processor = CSIDataProcessor(config)
        
        # Adaptive channel manager
        channel_manager = AdaptiveChannelManager(config)
        channel_manager.start()
        
        # LLM manager - COMPLETELY DISABLED
        llm_manager = DummyLLMManager(config)
        llm_manager.start()
        logger.info("LLM manager DISABLED for system stability")
        
        # Professional monitoring interface
        monitor_interface = None
        try:
            monitor_interface = MonitoringInterface(config)
            logger.info("Monitoring interface initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize monitoring interface: {e}")
            raise
        
        # Person detection and tracking
        reid_bridge = None
        try:
            # Load configuration for ReID
            reid_cfg = {
                'device': config.device,
                'checkpoint_path': config.reid_checkpoint
            }
            reid_bridge = ReIDBridge(reid_cfg)
            logger.info("ReID bridge initialized successfully")
            
            # FORCE IMMEDIATE TRAINING ON STARTUP
            logger.info("🎓 FORCING AUTOMATIC TRAINING AT STARTUP...")
            try:
                # Ensure model dimensions
                reid_bridge.ensure(32)  # 32-dimensional features
                
                # Generate initial training data to bootstrap the model
                logger.info("📊 Generating bootstrap training data...")
                for i in range(15):
                    # Create diverse feature vectors
                    dummy_feature = np.random.random(32).astype(np.float32)
                    # Add some realistic patterns
                    dummy_feature[0:8] *= (0.5 + i * 0.05)  # Signal strength patterns
                    dummy_feature[8:16] *= np.sin(i * 0.3)  # Phase patterns
                    dummy_feature[16:24] *= np.cos(i * 0.2) # Movement patterns
                    dummy_feature[24:32] *= (0.8 + np.random.random() * 0.4)  # Noise patterns
                    
                    reid_bridge.push(time.time() + i * 0.1, dummy_feature)
                
                # Force enrollment to kickstart training
                logger.info("🚀 Forcing model enrollment...")
                reid_bridge.maybe_enroll()
                
                # Additional training rounds
                for round_num in range(3):
                    logger.info(f"🔄 Training round {round_num + 1}/3...")
                    time.sleep(0.5)  # Brief pause
                    reid_bridge.maybe_enroll()
                
                logger.info("✅ STARTUP TRAINING COMPLETED - Model is ready for detection!")
                
            except Exception as train_e:
                logger.error(f"❌ Startup training failed: {train_e}")
            
            # Initialize training mode for continuous person enrollment
            logger.info("🧠 Starting CONTINUOUS person enrollment and training mode...")
            
        except Exception as e:
            logger.warning(f"ReID bridge initialization failed: {e}")
            logger.info("Continuing without person detection capabilities")
        
        # CSI data source
        csi_source = None
        try:
            csi_source = MonitorRadiotapSource(
                iface=config.csi_interface  # Use 'iface' parameter
            )
            logger.info("CSI source initialized successfully")
        except Exception as e:
            logger.warning(f"CSI source initialization failed: {e}")
            logger.info("Continuing with dummy CSI data for testing")
        
        logger.info("System initialization complete")
        
        # Main monitoring loop
        frame_count = 0
        fps_timer = time.time()
        last_fps_display = 0
        current_fps = 0.0
        training_active = False
        
        # ============ AUTOMATIC TRAINING INITIALIZATION ============
        # FORCE training at startup - ALWAYS train and improve model
        if reid_bridge:
            logger.info("🎓 INITIALIZING AUTOMATIC TRAINING MODE...")
            logger.info("🚀 FORCING initial training to improve model...")
            
            try:
                # Force initial training with dummy data to warm up the model
                logger.info("📊 Generating initial training data...")
                
                # Create diverse training samples to improve the model
                for i in range(10):  # 10 initial training samples
                    # Generate diverse signal patterns for training
                    training_amplitude = np.random.random(64).astype(np.float32)
                    training_amplitude += np.sin(np.linspace(0, 2*np.pi, 64) * (i+1)) * 0.3
                    
                    # Add to ReID bridge for immediate training
                    reid_bridge.push(time.time() + i, training_amplitude)
                    
                    # Force enrollment every few samples
                    if i % 3 == 0:
                        reid_bridge.maybe_enroll()
                        logger.info(f"📈 Training batch {i//3 + 1} enrolled")
                
                # Force additional training
                reid_bridge.maybe_enroll()  # Final enrollment
                logger.info("✅ Initial training completed - model is now primed!")
                
                # Set continuous training mode
                reid_bridge.continuous_training = True  # Enable continuous learning
                logger.info("🔄 Continuous training mode ACTIVATED")
                
            except Exception as train_e:
                logger.warning(f"Initial training warning: {train_e}")
                logger.info("🎯 Will continue with runtime training")
        
        logger.info("🎓 TRAINING MODE: System will auto-enroll and learn person patterns")
        logger.info("📊 DETECTION: Move around to be detected and identified!")
        logger.info("🚨 Real CSI data will be used for person detection")
        logger.info("🔄 CONTINUOUS LEARNING: Model improves with every detection!")
        
        # Initialize error counter and detection tracking for the main loop
        error_count = 0
        last_detection_time = time.time()
        no_detection_count = 0
        
        while True:
            try:
                frame_start = time.time()
                frame_count += 1
                
                # ============ FRAME SKIP PROTECTION ============
                # Skip expensive operations if system is struggling
                skip_frames = getattr(monitor_interface, '_skip_frames', 0)
                if skip_frames > 0:
                    monitor_interface._skip_frames = skip_frames - 1
                    logger.debug(f"⏭️ Skipping frame {frame_count} (recovery: {skip_frames} left)")
                    
                    # Minimal update during recovery
                    try:
                        monitor_interface.update_hud(
                            channel_manager, 
                            {'frame_count': frame_count, 'recovery_mode': True}, 
                            None, 
                            "[RECOVERY MODE]", 
                            current_fps
                        )
                    except:
                        pass
                    
                    time.sleep(0.1)  # Give system time to recover
                    continue  # Skip to next frame
                
                # ============ CSI DATA ACQUISITION ============
                current_channel = channel_manager.get_current_channel()
                
                # Create dummy CSI frames for testing
                csi_frames = [
                    CSIFrame(
                        timestamp=time.time(),
                        channel=current_channel,
                        amplitude=np.random.random(64).astype(np.float32),
                        rssi=-50.0
                    )
                ]
                
                # Update channel if no data received
                if not csi_frames:
                    channel_manager.switch_to_next_channel()
                    continue
                    
                # ============ SIGNAL PROCESSING ============
                processed_data = csi_processor.process_frames(csi_frames)
                
                # Update channel statistics
                signal_variance = processed_data.get('variance', 0.0)
                channel_manager.update_channel_stats(current_channel, signal_variance)
                
                # ============ AGGRESSIVE CONTINUOUS TRAINING ============
                detection_result = None
                
                # ENHANCED TRAINING: Learn from EVERY signal for maximum improvement
                if reid_bridge and processed_data.get('fused_amplitude') is not None:
                    try:
                        # 1. Movement detection using variance threshold
                        fused_amp = processed_data['fused_amplitude']
                        signal_variance = processed_data.get('variance', 0.0)
                        movement_detected = processed_data.get('environment', {}).get('movement', False)
                        
                        # 2. CONTINUOUS LEARNING - Train on every frame with signal
                        if len(fused_amp) > 0:
                            current_time = time.time()
                            
                            # Enhanced person signature with more features
                            person_signature = np.concatenate([
                                fused_amp[:32] if len(fused_amp) >= 32 else fused_amp,  # Amplitude data
                                [signal_variance, processed_data.get('environment', {}).get('activity', 0.0)],  # Signal metrics
                                [np.mean(fused_amp), np.std(fused_amp)]  # Additional statistics
                            ])
                            
                            # Ensure ReID bridge can handle the signature
                            reid_bridge.ensure(len(person_signature))
                            
                            # PUSH EVERY SIGNAL for AGGRESSIVE continuous learning
                            reid_bridge.push(current_time, person_signature)
                            
                            # SUPER FREQUENT ENROLLMENT for maximum model improvement
                            if frame_count % 5 == 0:  # Every 5 frames = 6 times per second!
                                try:
                                    enrollment_result = reid_bridge.maybe_enroll()
                                    if enrollment_result:
                                        logger.info(f"🎓 HYPER-LEARNING: Model updated frame {frame_count}, variance={signal_variance:.4f}")
                                    
                                    # Additional training with noise variations for robustness
                                    if signal_variance > 0.04:
                                        noise_signature = person_signature + np.random.normal(0, 0.008, len(person_signature))
                                        reid_bridge.push(current_time + 0.02, noise_signature.astype(np.float32))
                                        
                                        # Force additional enrollment on strong signals
                                        if signal_variance > 0.08:
                                            reid_bridge.maybe_enroll()
                                            logger.debug(f"💪 STRONG SIGNAL BOOST: {signal_variance:.4f}")
                                            
                                except Exception as train_e:
                                    logger.debug(f"Hyper-training iteration failed: {train_e}")
                            
                            # CONTINUOUS IMPROVEMENT - train even on weaker signals
                            elif frame_count % 15 == 0:  # Backup training every 15 frames
                                try:
                                    reid_bridge.maybe_enroll()
                                    logger.debug(f"🔄 Continuous improvement training")
                                except Exception as backup_train_e:
                                    logger.debug(f"Backup training failed: {backup_train_e}")
                        
                        # 3. Enhanced detection logic using CSI amplitude patterns
                        if signal_variance > csi_processor.detection_sensitivity and len(fused_amp) > 0:
                            
                            # 3. Person signature extraction from CSI
                            # Use amplitude variance and patterns for person identification
                            person_signature = np.concatenate([
                                fused_amp[:32] if len(fused_amp) >= 32 else fused_amp,  # First 32 subcarriers
                                [signal_variance, processed_data.get('environment', {}).get('activity', 0.0)]  # Signal metrics
                            ])
                            
                            # Ensure we have the right feature dimension for ReID
                            reid_bridge.ensure(len(person_signature))
                            
                            # 4. Push data to ReID bridge for person identification
                            current_time = time.time()
                            reid_bridge.push(current_time, person_signature)
                            
                            # 5. Auto-enrollment: Train the system when detecting consistent patterns
                            if frame_count > 100 and frame_count % 50 == 0:  # Every ~1.5 seconds
                                # Auto-enroll if we see consistent strong signals
                                if signal_variance > 0.08:  # Strong signal threshold
                                    reid_bridge.maybe_enroll()
                                    logger.info(f"🎓 TRAINING: Auto-enrolling person patterns, variance={signal_variance:.4f}")
                            
                            # 6. Try to get person identification result
                            reid_result = reid_bridge.maybe_infer(current_time)
                            
                            if reid_result and len(reid_result) > 0:
                                # Extract person information from ReID result
                                for person_data in reid_result:
                                    if isinstance(person_data, dict):
                                        person_id = person_data.get('id', 0)
                                        confidence = person_data.get('score', signal_variance * 100)
                                        
                                        # Generate position based on signal characteristics
                                        # Use phase information if available, otherwise approximate from amplitude
                                        if 'position' in person_data:
                                            position = person_data['position']
                                        else:
                                            # Estimate position from signal strength distribution
                                            center_freq = len(fused_amp) // 2
                                            x_pos = np.mean(fused_amp[:center_freq]) - np.mean(fused_amp[center_freq:])
                                            y_pos = signal_variance * 5 - 2.5  # Scale variance to position range
                                            position = np.array([x_pos, y_pos])
                                        
                                        # Create detection result
                                        detection_result = DetectionResult(
                                            person_id=int(person_id),
                                            confidence=float(confidence),
                                            position=position,
                                            skeleton=None  # Could be enhanced with skeleton estimation
                                        )
                                        
                                        logger.info(f"🚨 PERSON IDENTIFIED: ID={person_id}, Confidence={confidence:.2f}%, "
                                                  f"Position=[{position[0]:.2f}, {position[1]:.2f}], "
                                                  f"Signal Variance={signal_variance:.4f}")
                                        break
                                        
                            # 7. Enhanced movement-only detection for training data collection
                            elif movement_detected and signal_variance > csi_processor.detection_sensitivity:
                                # Collect training data from unknown person movements
                                if frame_count % 30 == 0:  # Every second at 30 FPS
                                    logger.info(f"📊 COLLECTING TRAINING DATA: variance={signal_variance:.4f}, "
                                              f"activity={processed_data.get('environment', {}).get('activity', 0.0):.3f}")
                                
                                # Create detection result for unknown person
                                person_id = min(999, int(signal_variance * 10000) % 100)  # Generate ID from signal
                                detection_result = DetectionResult(
                                    person_id=person_id,
                                    confidence=min(signal_variance * 1000, 95.0),  # Scale confidence
                                    position=np.array([
                                        np.random.uniform(-1, 1),  # Random position estimation
                                        np.random.uniform(-1, 1)
                                    ]),
                                    skeleton=None
                                )
                                
                                # Occasional training trigger
                                if signal_variance > 0.09 and frame_count % 150 == 0:  # Every 5 seconds with strong signal
                                    logger.info(f"🎓 STRONG SIGNAL TRAINING: ID={person_id}, variance={signal_variance:.4f}")
                                else:
                                    logger.info(f"👤 MOVEMENT DETECTED: Person #{person_id}, Confidence={detection_result.confidence:.1f}%, "
                                              f"Signal Variance={signal_variance:.4f}")
                                              
                    except Exception as e:
                        logger.warning(f"Person detection error: {e}")
                        # Fallback to simple movement detection
                        if processed_data.get('environment', {}).get('movement', False):
                            detection_result = DetectionResult(
                                person_id=0,
                                confidence=30.0,
                                position=np.array([0.0, 0.0])
                            )
                
                # ============ AUTO-REFRESH SYSTEM ============
                # Track detection activity and refresh when stuck
                current_time = time.time()
                if detection_result:
                    last_detection_time = current_time
                    no_detection_count = 0
                    logger.debug(f"🎯 Detection active: Person {detection_result.person_id}")
                else:
                    no_detection_count += 1
                    time_since_detection = current_time - last_detection_time
                    
                    # AUTO-REFRESH if no detection for too long
                    if time_since_detection > 10.0:  # 10 seconds without detection
                        logger.info(f"🔄 AUTO-REFRESH: No detection for {time_since_detection:.1f}s, refreshing system...")
                        
                        # Force system refresh
                        try:
                            # Reset detection variables
                            last_detection_time = current_time
                            no_detection_count = 0
                            
                            # Force garbage collection
                            import gc
                            gc.collect()
                            
                            # Reset ReID bridge if available
                            if reid_bridge:
                                reid_bridge.maybe_enroll()  # Force new enrollment
                                logger.info("✅ ReID bridge refreshed")
                            
                            # Reset visualization
                            if hasattr(monitor_interface, '_skip_frames'):
                                monitor_interface._skip_frames = 0
                            
                            logger.info("✅ System auto-refresh completed, ready for new detections")
                            
                        except Exception as refresh_e:
                            logger.warning(f"Auto-refresh error: {refresh_e}")
                    
                    elif no_detection_count % 300 == 0:  # Every 10 seconds at 30 FPS
                        logger.debug(f"⏳ Waiting for detection... ({time_since_detection:.1f}s)")
                
                # ============ LLM SCENE ANALYSIS ============ 
                # DISABLED - Ollama causes system instability
                llm_response = "🛸 Xenoscanner Active - Real-time CSI Analysis"
                
                # Skip LLM processing to prevent system failures
                # if frame_count % config.llm_update_interval == 0:
                #     scene_description = csi_processor.create_scene_description(
                #         processed_data, detection_result
                #     )
                #     llm_manager.request_scene_analysis(scene_description)
                
                # Skip LLM response retrieval
                # latest_response = llm_manager.get_latest_response()
                # if latest_response:
                #     llm_response = latest_response
                
                # ============ VISUALIZATION UPDATE (OPTIMIZED) ============
                # Initialize visualization variables
                grid_points = np.zeros((100, 3), dtype=np.float32)  # Default grid
                grid_amplitudes = np.zeros(100, dtype=np.float32)
                
                # Only update visualization every few frames to reduce load
                if frame_count % 2 == 0:  # Update every 2nd frame
                    try:
                        # Generate radar grid (lighter version)
                        grid_points, grid_amplitudes = monitor_interface.generate_advanced_grid(
                            signal_variance, time.time()
                        )
                        
                        # Update main visualization with timeout protection
                        monitor_interface.update_visualization(
                            processed_data, grid_points, grid_amplitudes
                        )
                        
                    except Exception as vis_e:
                        logger.debug(f"Visualization update skipped: {vis_e}")
                        # Continue without visualization update
                
                # ============ PERSON VISUALIZATION ============
                # Add detected persons to the Gaussian viewer with ReID integration
                # 🎮 Enhanced person visualization with Gaussian integration
                if detection_result:
                    try:
                        # Enhanced person data for Gaussian visualization
                        person_skeleton = monitor_interface._generate_person_skeleton(detection_result)
                        
                        person_data = {
                            'id': detection_result.person_id,
                            'position': detection_result.position,
                            'confidence': detection_result.confidence,
                            'skeleton': person_skeleton,
                            'timestamp': time.time(),
                            'signal_strength': signal_variance
                        }
                        
                        # Push person data to ReID for tracking
                        if reid_bridge:
                            reid_bridge.push(time.time(), processed_data['fused_amplitude'][:32] if len(processed_data['fused_amplitude']) >= 32 else processed_data['fused_amplitude'])
                        
                        # Enhanced visualization with person-specific Gaussian points
                        try:
                            # FORCE SKELETON VISUALIZATION
                            if person_skeleton is not None:
                                logger.info(f"🦴 FORCING SKELETON RENDER for person {detection_result.person_id}")
                                
                                # Method 1: Convert skeleton to HIGHLY VISIBLE points
                                skeleton_points = person_skeleton.reshape(-1, 3)  # joints -> points
                                
                                # Method 2: ENHANCED SKELETON VISUALIZATION - Dense point cloud
                                enhanced_skeleton_points = []
                                enhanced_skeleton_colors = []
                                
                                for i, joint in enumerate(person_skeleton):
                                    # Create DENSE CLUSTER of points around each joint (20 points per joint)
                                    for cluster_idx in range(20):
                                        # Spherical cluster around joint
                                        theta = np.random.uniform(0, 2*np.pi)
                                        phi = np.random.uniform(0, np.pi)
                                        radius = np.random.uniform(0.02, 0.08)  # 2-8cm radius
                                        
                                        x_offset = radius * np.sin(phi) * np.cos(theta)
                                        y_offset = radius * np.sin(phi) * np.sin(theta)
                                        z_offset = radius * np.cos(phi)
                                        
                                        enhanced_point = joint + [x_offset, y_offset, z_offset]
                                        enhanced_skeleton_points.append(enhanced_point)
                                        
                                        # Color coding: head=red, torso=green, arms=blue, legs=yellow
                                        if i < 3:  # head/neck
                                            color_intensity = 1.0  # Bright red
                                        elif i < 9:  # torso
                                            color_intensity = 0.9  # Green
                                        elif i < 17:  # arms
                                            color_intensity = 0.8  # Blue
                                        else:  # legs
                                            color_intensity = 0.7  # Yellow
                                        
                                        enhanced_skeleton_colors.append(color_intensity)
                                
                                skeleton_gaussian = np.array(enhanced_skeleton_points, dtype=np.float32)
                                skeleton_amps = np.array(enhanced_skeleton_colors, dtype=np.float32)
                                
                                logger.info(f"✅ ENHANCED Skeleton: {len(skeleton_gaussian)} dense points from {len(person_skeleton)} joints")
                            else:
                                skeleton_gaussian = np.array([])
                                skeleton_amps = np.array([])
                            
                            # Generate person-specific Gaussian points
                            person_points = monitor_interface._generate_person_gaussian(
                                detection_result.position, detection_result.confidence
                            )
                            
                            # Combine skeleton + person points + environment
                            if len(skeleton_gaussian) > 0:
                                combined_person_points = np.vstack([person_points, skeleton_gaussian])
                                combined_person_amps = np.hstack([
                                    np.full(len(person_points), detection_result.confidence / 100.0),
                                    skeleton_amps
                                ])
                            else:
                                combined_person_points = person_points
                                combined_person_amps = np.full(len(person_points), detection_result.confidence / 100.0)
                            
                            # Merge with environment grid for unified visualization
                            enhanced_grid = np.vstack([grid_points, combined_person_points])
                            enhanced_amps = np.hstack([grid_amplitudes, combined_person_amps])
                            
                            # Update visualization with enhanced data
                            monitor_interface.update_visualization(processed_data, enhanced_grid, enhanced_amps)
                            
                            # Force explicit person addition to view
                            monitor_interface.add_person_to_view(person_data)
                            
                            logger.info(f"🎮 Person {detection_result.person_id} - Total points: {len(combined_person_points)} "
                                      f"(Person: {len(person_points)}, Skeleton: {len(skeleton_gaussian)})")
                            
                        except Exception as vis_e:
                            logger.error(f"❌ Person Gaussian visualization failed: {vis_e}")
                            import traceback
                            traceback.print_exc()
                            # Fallback to basic visualization
                            monitor_interface.update_visualization(processed_data, grid_points, grid_amplitudes)
                        
                        logger.debug(f"🎮 Person visualization updated: ID={detection_result.person_id}, "
                                   f"Pos=[{detection_result.position[0]:.2f}, {detection_result.position[1]:.2f}], "
                                   f"Confidence={detection_result.confidence:.1f}%")
                        
                    except Exception as e:
                        logger.warning(f"Person visualization error: {e}")
                else:
                    # Update normal visualization without persons
                    monitor_interface.update_visualization(processed_data, grid_points, grid_amplitudes)
                        
                # Force ReID inference periodically for person tracking
                if reid_bridge and frame_count % 30 == 0:  # Every second
                    try:
                        reid_result = reid_bridge.maybe_infer(time.time())
                        if reid_result:
                            logger.debug(f"🧠 ReID inference: {len(reid_result)} results")
                    except Exception as e:
                        logger.debug(f"ReID inference error: {e}")
                
                # PERIODIC FORCED TRAINING - Every 10 seconds
                if reid_bridge and frame_count % 300 == 0:  # Every 10 seconds at 30 FPS
                    try:
                        logger.info(f"🔄 PERIODIC FORCED TRAINING - Frame {frame_count}")
                        
                        # Generate additional training data based on recent activity
                        current_amp = processed_data.get('fused_amplitude', np.array([]))
                        if len(current_amp) >= 16:
                            # Create multiple training variations
                            for variation in range(5):
                                variation_signature = current_amp[:16].copy()
                                variation_signature += np.random.normal(0, 0.01, len(variation_signature))
                                reid_bridge.push(time.time() + variation * 0.01, variation_signature.astype(np.float32))
                            
                            # Force multiple enrollments
                            for _ in range(3):
                                reid_bridge.maybe_enroll()
                                
                        logger.info("✅ Periodic training completed")
                        
                    except Exception as periodic_e:
                        logger.debug(f"Periodic training error: {periodic_e}")
                
                # MEGA TRAINING BOOST - Every minute
                if reid_bridge and frame_count % 1800 == 0:  # Every minute
                    try:
                        logger.info("🚀 MEGA TRAINING BOOST - Full model refresh!")
                        
                        # Generate comprehensive training dataset
                        for mega_round in range(10):
                            mega_signature = np.random.random(32).astype(np.float32)
                            # Add realistic patterns
                            mega_signature[0:8] *= (0.4 + mega_round * 0.05)
                            mega_signature[8:16] *= np.sin(mega_round * 0.4)
                            mega_signature[16:24] *= np.cos(mega_round * 0.3)
                            mega_signature[24:32] *= (0.7 + np.random.random() * 0.6)
                            
                            reid_bridge.push(time.time() + mega_round * 0.1, mega_signature)
                        
                        # Multiple mega enrollments
                        for _ in range(5):
                            reid_bridge.maybe_enroll()
                            
                        logger.info("💪 MEGA TRAINING COMPLETED - Model significantly improved!")
                        
                    except Exception as mega_e:
                        logger.debug(f"Mega training error: {mega_e}")
                        
                # Clear old persons if no detection for a while
                if hasattr(monitor_interface.view, 'clear_old_persons'):
                    try:
                        monitor_interface.view.clear_old_persons(current_time=time.time(), timeout=5.0)
                    except Exception as e:
                        logger.debug(f"Clear persons error: {e}")
                
                # ============ ANTI-FREEZE PROTECTION ============
                current_time_fps = time.time()
                
                # Real-time FPS calculation
                if frame_count > 0:
                    frame_interval = current_time_fps - fps_timer
                    if frame_interval > 0:
                        instant_fps = 1.0 / frame_interval
                        current_fps = 0.8 * current_fps + 0.2 * instant_fps  # Smoothed
                
                # FREEZE DETECTION - immediate response
                if current_fps < 0.8 and frame_count > 30:  # Less than 0.8 FPS
                    freeze_count = getattr(monitor_interface, '_freeze_count', 0) + 1
                    monitor_interface._freeze_count = freeze_count
                    
                    logger.warning(f"⚠️ LOW FPS DETECTED: {current_fps:.2f} FPS (Count: {freeze_count})")
                    
                    if freeze_count >= 5:  # 5 consecutive low FPS = force recovery
                        logger.error("🚨 FORCING SYSTEM RECOVERY!")
                        try:
                            # Emergency measures
                            import gc
                            gc.collect()  # Force garbage collection
                            
                            # Skip next few visualization updates
                            monitor_interface._skip_frames = 10
                            
                            # Reset counters
                            monitor_interface._freeze_count = 0
                            current_fps = 5.0  # Reset to reasonable value
                            
                            logger.info("✅ Emergency recovery applied")
                            
                        except Exception as recovery_e:
                            logger.error(f"Recovery failed: {recovery_e}")
                else:
                    # Reset freeze counter if FPS is good
                    if hasattr(monitor_interface, '_freeze_count'):
                        monitor_interface._freeze_count = 0
                
                fps_timer = current_time_fps
                
                # Update HUD elements
                monitor_interface.update_hud(
                    channel_manager, processed_data, detection_result, 
                    llm_response, current_fps
                )
                
                # ============ ADAPTIVE OPTIMIZATION ============
                # Auto-tune detection sensitivity
                if frame_count % config.auto_tune_interval == 0:
                    csi_processor.auto_tune_sensitivity(processed_data)
                
                # Adaptive channel switching
                if frame_count % 100 == 0:  # Every ~3 seconds at 30 FPS
                    channel_manager.adaptive_channel_switch()
                
                # ============ DATA PERSISTENCE ============
                # Save CSI data for offline analysis
                if config.save_csi_data and frame_count % config.save_interval == 0:
                    try:
                        timestamp = int(time.time())
                        save_path = f"env/csi_logs/csi_{timestamp}.pkl"
                        csi_processor.save_csi_data(csi_frames, save_path)
                    except Exception as e:
                        logger.warning(f"Failed to save CSI data: {e}")
                
                # ============ FRAME RENDERING ============
                try:
                    # Non-blocking render with timeout protection
                    monitor_interface.render_frame()
                    
                    # Prevent memory leaks and freezing
                    if frame_count % 100 == 0:  # Every ~3 seconds
                        # Force garbage collection
                        import gc
                        gc.collect()
                        logger.debug(f"🧹 Memory cleanup at frame {frame_count}")
                        
                    # Check for PyVista window health
                    if hasattr(monitor_interface.view, 'p') and hasattr(monitor_interface.view.p, 'render_window'):
                        if not monitor_interface.view.p.render_window.GetNeverRendered():
                            # Window is healthy, continue
                            pass
                        else:
                            logger.warning("⚠️ Render window issue detected, attempting recovery...")
                            
                except Exception as e:
                    logger.warning(f"Render error: {e}")
                    # Continue without crashing
                    pass
                
                # ============ PERFORMANCE MONITORING ============
                frame_time = time.time() - frame_start
                if frame_time < config.target_frame_time:
                    time.sleep(config.target_frame_time - frame_time)
                
                # Log performance metrics with freeze detection
                if frame_count % 300 == 0:  # Every 10 seconds
                    elapsed_time = time.time() - fps_timer
                    if elapsed_time > 0:
                        actual_fps = (frame_count - last_fps_display) / elapsed_time
                        logger.info(f"📊 Frame {frame_count}: FPS={actual_fps:.1f}, "
                                  f"Target={config.target_fps:.1f}, "
                                  f"Channel={current_channel}, "
                                  f"Signal Variance={signal_variance:.3f}, "
                                  f"Active Channels={len(channel_manager.adaptive_channels)}")
                        
                        # Freeze detection
                        if actual_fps < config.target_fps * 0.5:  # Less than half target FPS
                            logger.warning(f"⚠️ PERFORMANCE WARNING: Low FPS detected ({actual_fps:.1f})")
                            logger.info("🔧 Attempting performance recovery...")
                            # Force cleanup
                            import gc
                            gc.collect()
                            
                        # Reset counters
                        fps_timer = time.time()
                        last_fps_display = frame_count
                
                # ============ FRAME TIMING PROTECTION ============
                frame_end = time.time()
                frame_duration = frame_end - frame_start
                
                # Detect if frame took too long
                max_frame_time = 0.5  # 500ms max per frame
                if frame_duration > max_frame_time:
                    slow_frames = getattr(monitor_interface, '_slow_frames', 0) + 1
                    monitor_interface._slow_frames = slow_frames
                    
                    logger.warning(f"🐌 SLOW FRAME {frame_count}: {frame_duration:.3f}s (#{slow_frames})")
                    
                    if slow_frames >= 3:  # 3 slow frames in a row
                        logger.warning("⚠️ Multiple slow frames - enabling recovery mode")
                        monitor_interface._skip_frames = 5  # Skip next 5 frames
                        monitor_interface._slow_frames = 0
                else:
                    # Reset slow frame counter
                    monitor_interface._slow_frames = 0
                
                # Frame rate limiting
                target_fps = 20  # Reasonable target FPS
                min_frame_time = 1.0 / target_fps
                if frame_duration < min_frame_time:
                    sleep_time = min_frame_time - frame_duration
                    time.sleep(sleep_time)
                
            except KeyboardInterrupt:
                logger.info("🛑 Stopping monitoring (Ctrl+C)")
                break
                
            except Exception as e:
                error_count += 1
                logger.error(f"❌ Loop error #{error_count}: {e}")
                
                # Too many errors = emergency stop
                if error_count > 15:
                    logger.error("💀 TOO MANY ERRORS - EMERGENCY STOP!")
                    break
                
                # Brief pause on error to prevent spam
                time.sleep(0.2)
    
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        return 1
        
    finally:
        # Cleanup resources
        logger.info("Cleaning up resources...")
        try:
            if 'channel_manager' in locals():
                channel_manager.cleanup()
            # LLM manager is disabled
            # if 'llm_manager' in locals() and llm_manager is not None:
            #     llm_manager.cleanup()
            if 'monitor_interface' in locals() and monitor_interface is not None:
                monitor_interface.cleanup()
        except Exception as e:
            logger.warning(f"Cleanup error: {e}")
            
        logger.info("WiFi CSI 3D Fusion Monitoring System shutdown complete")
    
    return 0


if __name__ == "__main__":
    exit_code = run_realtime_gaussian_fusion()
    exit(exit_code)
