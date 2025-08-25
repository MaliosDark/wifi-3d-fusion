# WiFi-3D-Fusion: Advanced CSI-Based Person Detection & Visualization

## 🚀 Quick Start Guide

### Method 1: Web-Based Real-Time Visualization (Recommended)
```bash
# Install dependencies
bash scripts/install_all.sh

# Start web-based real-time visualization
source .venv/bin/activate
python run_js_visualizer.py

# Open browser to http://localhost:5000
```

### Method 2: Traditional Pipeline
```bash
# ESP32-CSI UDP (default port 5566):
./scripts/run_realtime.sh --source esp32

# Or Nexmon (requires monitor-mode interface)
sudo ./scripts/run_realtime.sh --source nexmon
```

## 🎯 Model Training & Continuous Learning

### Train Your Own Detection Model
```bash
# Basic training with current configuration
python train_model.py --config configs/fusion.yaml

# Train with specific device source
python train_model.py --source esp32 --device cuda --epochs 200

# Enable continuous learning (model improves automatically)
python train_model.py --continuous --auto-improve

# Advanced training with custom parameters
python train_model.py \
    --source nexmon \
    --device cuda \
    --epochs 500 \
    --batch-size 64 \
    --lr 0.0005 \
    --continuous \
    --auto-improve
```

### Continuous Learning Features
- **Real-time model improvement**: The system automatically learns from new detections
- **Adaptive training**: Model updates based on detection confidence and user feedback  
- **Self-improvement**: System gets better at person detection over time
- **Background learning**: Training happens continuously without interrupting visualization

## 📊 System Architecture & Features

### Core Components
1. **CSI Data Acquisition**
   - ESP32-CSI via UDP (recommended for beginners)
   - Nexmon firmware on Broadcom chips (advanced users)
   - Real-time CSI amplitude and phase extraction

2. **Advanced Detection Pipeline**
   - Convolutional Neural Network for person detection
   - Real-time skeleton estimation and tracking
   - Multi-person identification and re-identification (ReID)
   - Adaptive movement threshold adjustment

3. **3D Visualization System**
   - Web-based Three.js renderer with professional UI
   - Real-time 3D skeleton visualization
   - Animated CSI noise patterns on ground plane
   - Interactive camera controls and HUD overlays

4. **Machine Learning Features**
   - Continuous learning during operation
   - Automatic model improvement based on feedback
   - Self-adaptive detection thresholds
   - Person re-identification across sessions

### Real-Time Pipeline Flow
```
CSI Data Source → Signal Processing → Neural Detection → 3D Visualization
     ↓                   ↓                  ↓              ↓
ESP32/Nexmon     Amplitude/Phase      CNN Classifier   Three.js Web UI
     ↓                   ↓                  ↓              ↓  
UDP/PCap        Movement Detection   Person Tracking   Skeleton Rendering
     ↓                   ↓                  ↓              ↓
Config YAML     Adaptive Thresholding  ReID System    Activity Logging
```

## 🛠️ Installation & Setup

### Prerequisites
- Linux system (Ubuntu 18.04+ recommended)
- Python 3.8+
- WiFi adapter with monitor mode support (for Nexmon)
- ESP32 with CSI firmware (for ESP32 mode)
- CUDA-capable GPU (optional, improves training speed)

### Complete Installation
```bash
# Clone repository
git clone https://github.com/MaliosDark/wifi-3d-fusion.git
cd wifi-3d-fusion

# Install all dependencies and setup environment
bash scripts/install_all.sh

# Activate Python environment
source .venv/bin/activate

# Verify installation
python -c "import torch, numpy, yaml; print('✅ All dependencies installed')"
```

### Hardware Setup

#### Option A: ESP32-CSI Setup
1. **Flash ESP32 with CSI firmware**
   ```bash
   # Download ESP32-CSI-Tool firmware
   # Flash to ESP32 using esptool or Arduino IDE
   ```

2. **Configure ESP32**
   - Set WiFi network and password
   - Configure UDP target IP (your PC's IP)
   - Set UDP port to 5566 (or modify `configs/fusion.yaml`)

3. **Update configuration**
   ```yaml
   # configs/fusion.yaml
   source: esp32
   esp32_udp_port: 5566
   ```

#### Option B: Nexmon Setup
1. **Install Nexmon firmware**
   ```bash
   # For Raspberry Pi 4 with bcm43455c0
   git clone https://github.com/seemoo-lab/nexmon_csi.git
   cd nexmon_csi
   # Follow installation instructions for your device
   ```

2. **Enable monitor mode**
   ```bash
   sudo ip link set wlan0 down
   sudo iw dev wlan0 set type monitor
   sudo ip link set wlan0 up
   ```

3. **Update configuration**
   ```yaml
   # configs/fusion.yaml  
   source: nexmon
   nexmon_iface: wlan0
   ```

## 🎮 Running the System

### Web-Based Visualization (Recommended)
```bash
# Start the web server with real-time visualization
source .venv/bin/activate
python run_js_visualizer.py

# Optional: specify device source
python run_js_visualizer.py --source esp32
python run_js_visualizer.py --source nexmon

# Access web interface
# Open browser to: http://localhost:5000
```

### Traditional Terminal-Based
```bash
# Run with default configuration
./run_wifi3d.sh

# Run with specific source
./run_wifi3d.sh esp32
./run_wifi3d.sh nexmon

# Run with custom channel hopping (Nexmon only)
sudo IFACE=mon0 HOP_CHANNELS=1,6,11 python run_realtime_hop.py
```

### Training Mode
```bash
# Collect training data first by running the system
python run_js_visualizer.py

# Train model on collected data
python train_model.py --epochs 100 --device cuda

# Train with continuous learning enabled
python train_model.py --continuous --auto-improve

# Resume training from checkpoint
python train_model.py --resume env/weights/checkpoint_epoch_50.pth
```

## 📋 Configuration

### Main Configuration File: `configs/fusion.yaml`
```yaml
# CSI Data Source
source: esp32                    # esp32, nexmon, or dummy
esp32_udp_port: 5566            # UDP port for ESP32
nexmon_iface: wlan0             # Network interface for Nexmon

# Detection Parameters  
movement_threshold: 0.002        # Sensitivity for movement detection
debounce_seconds: 0.3           # Minimum time between detections
win_seconds: 3.0                # CSI analysis window size

# 3D Visualization
scene_bounds: [[-2,2], [-2,2], [0,3]]  # 3D scene boundaries
rf_res: 64                      # RF field resolution
alpha: 0.6                      # Visualization transparency

# Machine Learning
enable_reid: true               # Enable person re-identification
reid:
  checkpoint: env/weights/who_reid_best.pth
  seq_secs: 2.0                # Sequence length for ReID
  fps: 20.0                    # Processing framerate

# Advanced Features  
enable_pose3d: false            # 3D pose estimation (experimental)
enable_nerf2: false             # Neural RF fields (experimental)
```

## 🔧 Advanced Features

### Continuous Learning System
The system includes an advanced continuous learning pipeline that:

1. **Monitors detection confidence** in real-time
2. **Automatically collects training samples** from high-confidence detections  
3. **Updates the model** in the background without interrupting visualization
4. **Adapts detection thresholds** based on environment characteristics
5. **Improves person re-identification** over time

### Model Training Pipeline
```python
# Example: Custom training script
from train_model import WiFiTrainer, TrainingConfig

# Configure training
config = TrainingConfig(
    batch_size=64,
    learning_rate=0.001,  
    epochs=200,
    continuous_learning=True,
    auto_improvement=True
)

# Initialize trainer
trainer = WiFiTrainer('configs/fusion.yaml', args)

# Start training with continuous learning
trainer.train()
```

### Real-Time Performance Optimization
- **Multi-threaded processing**: Separate threads for data acquisition, processing, and visualization
- **Adaptive frame rates**: Automatically adjusts processing speed based on system load
- **Memory management**: Efficient CSI buffer management for long-running sessions
- **GPU acceleration**: CUDA support for neural network inference and training

## 🌐 Web Interface Features

### Professional Dashboard
- **Real-time CSI metrics**: Signal variance, amplitude, activity levels
- **Person detection status**: Count, confidence, positions
- **Skeleton visualization**: 3D animated skeletons with joint tracking
- **System performance**: FPS, memory usage, processing time
- **Activity logging**: Real-time event log with timestamps

### Interactive 3D Scene
- **Manual camera controls**: Orbit, zoom, pan with mouse
- **Ground noise visualization**: Animated circular wave patterns
- **Skeleton rendering**: Full 3D human skeletons for detected persons
- **Real-time updates**: Live data streaming at 10 FPS

### HUD Information Panels
```
╔════ 🖥️ SYSTEM METRICS ════╗
║ FPS: 60 | Frame: 1234      ║
║ Processing: 15.2ms         ║  
║ Memory: 245.7 MB           ║
╚════════════════════════════╝

╔════ 📡 CSI STATUS ════╗
║ CSI: Active            ║
║ Persons: 3 detected    ║
║ Skeletons: 3 active    ║
║ Updated: 14:23:45      ║
╚════════════════════════╝
```

## 🧪 Testing & Validation

### Test Detection System
```bash
# Test with dummy data
python test_person_visualization.py

# Test CSI processing
python test_csi_diag.py

# Test skeleton generation
python test_radar_skeleton.py
```

### Validate Model Performance
```bash
# Evaluate trained model
python tools/eval_reid.py --checkpoint env/weights/best_model.pth

# Record test sequences
python tools/record_reid_sequences.py --duration 60

# Simulate CSI data for testing
python tools/simulate_csi.py --samples 1000
```

## 📊 Data Collection & Management

### CSI Data Storage
```
env/
├── csi_logs/              # Raw CSI data files (*.pkl)
├── logs/                  # System and training logs  
├── weights/               # Trained model checkpoints
└── visualization/         # Web interface files
    ├── index.html         # Main dashboard
    ├── js/app.js         # Visualization logic
    └── css/style.css     # UI styling
```

### Training Data Organization
```
data/
├── reid/                  # Person re-identification data
│   ├── person_000/       # Individual person sequences
│   ├── person_001/
│   └── ...
├── splits/               # Training/validation splits
│   ├── train.txt
│   ├── val.txt  
│   └── gallery.txt
└── logs/                 # Training history and metrics
```

## 🚨 Troubleshooting

### Common Issues

#### 1. No CSI Data Received
```bash
# Check ESP32 connection
ping <ESP32_IP>

# Verify UDP port
netstat -ulnp | grep 5566

# Test with dummy data
python run_js_visualizer.py --source dummy
```

#### 2. Monitor Mode Issues (Nexmon)
```bash
# Reset interface
sudo ip link set wlan0 down
sudo iw dev wlan0 set type managed  
sudo ip link set wlan0 up

# Re-enable monitor mode
sudo ip link set wlan0 down
sudo iw dev wlan0 set type monitor
sudo ip link set wlan0 up
```

#### 3. Training Fails
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Reduce batch size for limited memory
python train_model.py --batch-size 16

# Use CPU training
python train_model.py --device cpu
```

#### 4. Web Interface Issues
```bash
# Check if server is running
curl http://localhost:5000/data

# Clear browser cache and reload
# Check browser console for JavaScript errors (F12)

# Restart server
pkill -f run_js_visualizer.py
python run_js_visualizer.py
```

### Debug Logging
Enable verbose logging for troubleshooting:
```bash
# Set debug mode
export WIFI3D_DEBUG=1

# Run with verbose output
python run_js_visualizer.py --verbose

# Check log files
tail -f env/logs/wifi3d_*.log
```

## 🤝 Contributing

### Development Setup
```bash
# Fork repository and clone
git clone https://github.com/your-username/wifi-3d-fusion.git
cd wifi-3d-fusion

# Create development branch
git checkout -b feature/your-feature

# Install development dependencies
pip install -e .[dev]

# Run tests
python -m pytest tests/
```

### Code Style
- Follow PEP 8 for Python code
- Use type hints where possible
- Document functions with docstrings
- Add unit tests for new features

## 📝 License & Citation

This project is licensed under the MIT License. If you use this work in research, please cite:

```bibtex
@misc{wifi3d-fusion,
  title={WiFi-3D-Fusion: Advanced CSI-Based Person Detection and Visualization},
  author={MaliosDark},
  year={2025},
  url={https://github.com/MaliosDark/wifi-3d-fusion}
}
```

<p align="center">
  <img src="docs/img/wifi-3d-fusion.png" width="950" alt="WiFi-3D-Fusion — Layered Neural Network Architecture"/>
</p>

# WiFi-3D-Fusion

<p align="center">
  <!-- Project -->
  <img src="https://img.shields.io/badge/Project-WiFi--3D--Fusion-blue?style=for-the-badge&logo=github" />
  <img src="https://img.shields.io/badge/Powered%20By-MaliosDark-black?style=for-the-badge&logo=starship&logoColor=white" />
</p>

<p align="center">
  <!-- Languages -->
  <img src="https://img.shields.io/badge/Language-Python%20%7C%20C++-3776AB?style=for-the-badge&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Framework-PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" />
  <img src="https://img.shields.io/badge/Device-ESP32-E7352C?style=for-the-badge&logo=espressif&logoColor=white" />
</p>

<p align="center">
  <!-- Tools & Libs -->
  <img src="https://img.shields.io/badge/Open3D-Viewer-0A7EEE?style=for-the-badge&logo=opengl&logoColor=white" />
  <img src="https://img.shields.io/badge/Scapy-Capture-FFD43B?style=for-the-badge&logo=wireshark&logoColor=black" />
  <img src="https://img.shields.io/badge/tcpdump-Pcap-888888?style=for-the-badge&logo=linux&logoColor=white" />
  <img src="https://img.shields.io/badge/Nexmon-CSI-8A2BE2?style=for-the-badge&logo=gnu-bash&logoColor=white" />
  <img src="https://img.shields.io/badge/OpenMMLab-mmcv%20%7C%20mmdet-FF6F00?style=for-the-badge&logo=opencv&logoColor=white" />
  <img src="https://img.shields.io/badge/NeRF%C2%B2-RF%20Fields-6A0DAD?style=for-the-badge&logo=ai&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?style=for-the-badge&logo=docker&logoColor=white" />
</p>


**Live local Wi-Fi sensing** with CSI: real-time motion detection + visualization, with optional bridges to:
- **Person-in-WiFi-3D** (multi-person **3D pose** from Wi-Fi) [CVPR 2024].  
- **NeRF²** (neural RF radiance fields).  
- **3D Wi-Fi Scanner** (RSSI volumetric mapping).

This monorepo is production-oriented: robust CSI ingestion from **local Wi-Fi** (ESP32-CSI via UDP, or **Nexmon** via `tcpdump` + `csiread`), a realtime movement detector, and a 3D viewer.

---

## Quick start

```bash
bash scripts/install_all.sh
# ESP32-CSI UDP (default port 5566):
./scripts/run_realtime.sh --source esp32

# Or Nexmon (requires monitor-mode iface and nexmon_csi firmware)
sudo ./scripts/run_realtime.sh --source nexmon
````

> GPU is only needed if you enable the 3D pose or NeRF² bridges. Realtime detector/visualizer runs on CPU.

---

## Hardware paths

### A) ESP32-CSI (recommended to start)

* Flash **ESP32-CSI-Tool** on an ESP32. Configure it to **send CSI via UDP** to your PC’s IP and port **5566** (or change `esp32_udp_port` in `configs/fusion.yaml`).
* This repo listens and parses JSON CSI payloads (`type: "CSI_DATA"` with `csi` array).

### B) Nexmon (Broadcom chips)

* Install **nexmon\_csi** on a compatible device (e.g., RPi 3B+/4 with bcm43455c0).
* Put your capture interface in monitor mode (e.g., `wlan0`).
* We tail a rolling pcap via `tcpdump` and parse CSI with **csiread** in near-real-time.

---

## What you get out-of-the-box

* **CSI capture**:

  * `src/csi_sources/esp32_udp.py` — ESP32-CSI UDP JSON receiver.
  * `src/csi_sources/nexmon_pcap.py` — Nexmon CSI via `tcpdump` + `csiread`.
* **Realtime analytics**:

  * `src/pipeline/realtime_detector.py` — movement detector using amplitude-variance over a sliding window.
* **3D visualization**:

  * `src/pipeline/realtime_viewer.py` — Open3D live point cloud (frequency × antenna × amplitude). It displays **real CSI** dynamics; no dummy data.

---

## Optional bridges (disabled by default)

### 1) Person-in-WiFi-3D (3D pose)

* Repo: `third_party/Person-in-WiFi-3D-repo`
* Enable in `configs/fusion.yaml`: `enable_pose3d: true`
* Place a compatible checkpoint at `env/weights/pwifi3d.pth`.
* Prepare test data under the repo’s expected structure (`data/wifipose/test_data/...`), then run:

  ```bash
  python -m src.bridges.pwifi3d_runner \
    third_party/Person-in-WiFi-3D-repo config/wifi/petr_wifi.py env/weights/pwifi3d.pth
  ```

  *(We shell out to OpenMMLab’s `tools/test.py` inside the repo.)*

### 2) NeRF² (RF field)

* Repo: `third_party/NeRF2`
* Enable in `configs/fusion.yaml`: `enable_nerf2: true`
* Train:

  ```bash
  python -m src.bridges.nerf2_runner
  ```

### 3) 3D Wi-Fi Scanner (RSSI volume)

* Repo: `third_party/3D_wifi_scanner`
* Use that tooling to generate volumetric RSSI datasets; you can integrate them into your own fusion pipeline if desired.

---

## Configuration

Edit `configs/fusion.yaml`:

* `source: esp32 | nexmon`
* `esp32_udp_port`, `nexmon_iface`, etc.
* Detector thresholds: `movement_threshold`, `win_seconds`, `debounce_seconds`.

---

## Docker

```bash
docker compose build
docker compose run --rm fusion
```

---

## Notes

* For **Nexmon**, you need `tcpdump` privileges. The Dockerfile includes it; on host, install it and run as root/sudo.
* For **Person-in-WiFi-3D**, follow that repo’s requirements (PyTorch, MMCV/MMDet). Our `scripts/install_all.sh` installs compatible versions.
* For **ESP32-CSI**, UDP JSON payloads compatible with common forks are supported.

---

### Usage (super short)

### Adaptive monitor-mode pipeline (recommended for RTL8812AU, Nexmon, or any monitor-mode interface)
```bash
sudo -E env PATH="$PWD/venv/bin:$PATH" IFACE=mon0 HOP_CHANNELS=1,6,11 python3 run_realtime_hop.py
```
This will launch the self-learning pipeline described above.


If you want the Docker path:

```bash
docker compose build
docker compose run --rm fusion
```
---

## 🔧 System Requirements & Dependencies

* **OS:** Ubuntu 22.04+ (tested with Kernel 6.14)
* **Python:** 3.12 (venv managed by `scripts/install_all.sh`)
* **GPU:** Optional (only for Pose3D/NeRF² bridges)
* **Packages (auto-installed):**

  * Base: `numpy`, `pyyaml`, `loguru`, `tqdm`, `open3d`, `opencv-python`, `einops`, `watchdog`, `pyzmq`, `matplotlib`, `csiread==1.4.1`
  * Optional Pose3D: `torch` + `torchvision` (cu118/cu121 or cpu), `openmim`, `mmengine`, `mmcv`, `mmdet`
* **System tools for capture (optional):** `tcpdump`, `tshark/wireshark`, `aircrack-ng`, `iw`

> The installer keeps Torch/`openmim` on **default PyPI** (no PyTorch index bleed) and pins `csiread` to a wheel compatible with Python 3.12.

---

## 🛠️ WiFi Adapter, Driver, and Monitor Mode Setup (RTL8812AU Example)

### Supported Adapters

For robust WiFi sensing, we recommend using a USB adapter based on the **Realtek RTL8812AU** chipset. This adapter supports both 2.4 GHz and 5 GHz bands, monitor mode, and packet injection. It is widely used for WiFi security research and is compatible with Linux distributions such as Ubuntu, Kali, and Parrot.

### Driver Installation (RTL8812AU)

The default kernel driver may not provide full monitor mode support. For best results, install the latest driver from the [aircrack-ng/rtl8812au](https://github.com/aircrack-ng/rtl8812au) repository:

```bash
sudo apt update
sudo apt install dkms git build-essential
git clone https://github.com/aircrack-ng/rtl8812au.git
cd rtl8812au
sudo make dkms_install
```

This will build and install the driver for your current kernel, enabling reliable monitor mode and packet capture.

### Enabling Monitor Mode

After installing the driver, connect your RTL8812AU adapter and identify its interface name (e.g., `wlx...`):

```bash
iw dev
iwconfig
```

To enable monitor mode and create a `mon0` interface:

```bash
sudo airmon-ng check kill
sudo airmon-ng start <your-interface>
# Or manually:
sudo ip link set <your-interface> down
sudo iw dev <your-interface> set type monitor
sudo ip link set <your-interface> up
```

Verify monitor mode:

```bash
iwconfig
```
You should see `Mode:Monitor` for `mon0` or your chosen interface.

### Verifying Packet Capture

To confirm that your interface is capturing WiFi packets in monitor mode:

```bash
sudo airodump-ng mon0
sudo tcpdump -i mon0
```

You should see networks and packets. If not, ensure there is active WiFi traffic in your environment.

### Additional Tools

For debugging and traffic generation, you may also want to install:

```bash
sudo apt install aircrack-ng tcpdump tshark
```

---

## 🧑‍💻 Running the Real-Time Pipeline with Monitor Mode


---

## 🧑‍💻 Running the Real-Time Adaptive Python Pipeline

Once your adapter is in monitor mode and capturing packets, run:

```bash
sudo -E env PATH="$PWD/venv/bin:$PATH" IFACE=mon0 HOP_CHANNELS=1,6,11 python3 run_realtime_hop.py
```

This will:
- Start live CSI/RSSI capture and analytics
- Train the detection model automatically
- Launch the Open3D viewer (robust, never blank)
- Adaptively scan and focus on the most active WiFi channels
- Show detections and all debug/status info in English

---

## 🧩 Architecture

<p align="center">
  <img src="docs/img/wifi3d_architecture.png" width="950" alt="WiFi-3D-Fusion — Layered Neural Network Architecture"/>
</p>

### High-level runtime

```mermaid
flowchart LR
  subgraph Capture
    A1(ESP32 UDP JSON):::node -->|csi_batch| B[esp32_udp.py]
    A2(Nexmon + tcpdump):::node -->|pcap| C[nexmon_pcap.py]
    A3(Monitor Radiotap):::node -->|RSSI stream| D[monitor_radiotap.py]
  end

  B & C & D --> E[realtime_detector.py]
  E --> F[fusion rf/rssi]
  F --> G[Open3D live viewer]

  classDef node fill:#0b7285,stroke:#083344,color:#fff;
```

### Model Training

<p align="center">
  <img src="docs/img/wifi3d_pipeline.png" width="950" alt="WiFi-3D-Fusion — End-to-End Pipeline from CSI to 3D Pose"/>
</p>


### Processing loop

```mermaid
sequenceDiagram
  participant SRC as CSI/RSSI Source
  participant DET as MovementDetector
  participant FUS as Fusion
  participant VIZ as Open3D Viewer

  loop Frames
    SRC->>DET: (ts, vector)
    DET-->>DET: sliding var / threshold
    DET->>FUS: events + buffers
    FUS-->>VIZ: point cloud + overlays
    VIZ-->>User: interactive 3D scene
  end
```

---

## 🛡️ Troubleshooting

* **Blank Open3D window**
  Ensure data is flowing:

  * ESP32: `sudo tcpdump -n -i any udp port 5566`
  * Nexmon: `sudo tcpdump -i wlan0 -s 0 -vv -c 20`
  * Monitor: `sudo tshark -I -i mon0 -a duration:5 -T fields -e radiotap.dbm_antsignal | head`
    Install GL if needed: `sudo apt-get install -y libgl1`

* **`openmim` not found / Torch index issues**
  Use the provided `install_all.sh` (Torch from PyTorch index only for Torch, `openmim` from PyPI).
  For Pose3D:
  `WITH_POSE=true TORCH_CUDA=cu121 bash scripts/install_all.sh`

* **`csiread` wheel mismatch**
  Python 3.12 → pin to `csiread==1.4.1` (already in requirements flow).

* **Monitor interface won’t capture**
  Kill network managers, recreate `mon0`, fix channel:
  `sudo airmon-ng check kill && bash scripts/setup_monitor.sh`

---

## 🔏 Legal / Research Notice

This repository is provided **By Malios Dark for research purposes only**.
You are responsible for complying with local laws and for using it **only on networks you own or have explicit permission to test**.


## 📚 Citations / Upstreams

1. [End-to-End Multi-Person 3D Pose Estimation with Wi-Fi (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/papers/Yan_Person-in-WiFi_3D_End-to-End_Multi-Person_3D_Pose_Estimation_with_Wi-Fi_CVPR_2024_paper.pdf)  
2. [GitHub - aiotgroup/Person-in-WiFi-3D-repo](https://github.com/aiotgroup/Person-in-WiFi-3D-repo)  
3. [NeRF2: Neural Radio-Frequency Radiance Fields (MobiCom 2023)](https://web.comp.polyu.edu.hk/csyanglei/data/files/nerf2-mobicom23.pdf)  
4. [GitHub - XPengZhao/NeRF2](https://github.com/XPengZhao/NeRF2)  
5. [GitHub - Neumi/3D_wifi_scanner](https://github.com/Neumi/3D_wifi_scanner)  
6. [Hackaday - Visualizing WiFi With A Converted 3D Printer](https://hackaday.com/2021/11/22/visualizing-wifi-with-a-converted-3d-printer/)  
7. [GitHub - StevenMHernandez/ESP32-CSI-Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool)  
8. [GitHub - citysu/csiread](https://github.com/citysu/csiread)  


---


