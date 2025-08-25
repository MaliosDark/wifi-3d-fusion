# WiFi-3D-Fusion Quick Commands

## 🚀 Start Real-Time Visualization
```bash
# Basic web visualization
python run_js_visualizer.py

# With specific device source
python run_js_visualizer.py --source esp32
python run_js_visualizer.py --source nexmon
```

## 🎯 Model Training

### Quick Training (50 epochs)
```bash
./train_wifi3d.sh --quick
```

### Advanced Training with Continuous Learning
```bash
./train_wifi3d.sh --continuous --auto-improve
```

### Full Training (500 epochs, GPU)
```bash
./train_wifi3d.sh --intensive
```

### Custom Training
```bash
./train_wifi3d.sh --source esp32 --epochs 200 --device cuda --continuous
```

## 🔧 Manual Training
```bash
# Basic training
python train_model.py --config configs/fusion.yaml

# With continuous learning
python train_model.py --continuous --auto-improve --epochs 100

# GPU training with specific parameters
python train_model.py --device cuda --epochs 500 --batch-size 64 --lr 0.0005
```

## 📊 System Status

### Check Training Progress
```bash
# View training logs
tail -f env/logs/training_history_*.json

# Check model files
ls -la env/weights/

# View learning checkpoints
ls -la env/logs/learning_checkpoint_*.json
```

### Monitor Real-Time Performance
```bash
# Open web interface
firefox http://localhost:5000

# Check server logs
tail -f ~/.vscode-server/logs/wifi3d_*.log
```

## 🎮 Device Configuration

### ESP32 Setup
1. Flash ESP32 with CSI firmware
2. Configure WiFi and UDP settings
3. Set target IP to your PC
4. Use port 5566 (default)

### Nexmon Setup
```bash
# Enable monitor mode
sudo ip link set wlan0 down
sudo iw dev wlan0 set type monitor
sudo ip link set wlan0 up

# Run with Nexmon
python run_js_visualizer.py --source nexmon --interface wlan0
```

## 🧠 Features

### Continuous Learning
- **Automatic**: Model learns from high-confidence detections
- **Background**: No interruption to visualization
- **Adaptive**: Adjusts detection thresholds automatically
- **Persistent**: Saves learning checkpoints

### Real-Time Visualization
- **3D Skeletons**: Full human skeleton rendering
- **Ground Noise**: Animated circular wave patterns
- **Manual Controls**: Orbit, zoom, pan with mouse
- **Live Metrics**: FPS, detection count, confidence levels

### Professional Dashboard
- **CSI Status**: Real-time signal metrics
- **Person Tracking**: Multi-person detection and tracking
- **System Performance**: Memory, processing time
- **Activity Logs**: Real-time event logging

## 📋 Configuration Files

### Main Config: `configs/fusion.yaml`
```yaml
source: esp32                    # esp32, nexmon, dummy
continuous_learning:
  enabled: true
  confidence_threshold: 0.75
  learning_interval: 30
training:
  batch_size: 32
  epochs: 100
  device: auto
```

## 🚨 Troubleshooting

### No Data
```bash
# Test with dummy data
python run_js_visualizer.py --source dummy

# Check CSI files
ls env/csi_logs/

# Verify device connection
ping <ESP32_IP>  # For ESP32
iwconfig         # For Nexmon
```

### Training Issues
```bash
# Use CPU if GPU fails
./train_wifi3d.sh --device cpu

# Reduce memory usage
./train_wifi3d.sh --batch-size 16

# Check dependencies
python -c "import torch; print(torch.cuda.is_available())"
```

### Port Conflicts
```bash
# Kill existing processes
pkill -f run_js_visualizer.py
fuser -k 5000/tcp

# Use different port
python run_js_visualizer.py --port 5001
```
