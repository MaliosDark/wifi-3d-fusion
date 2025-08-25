#!/bin/bash
# WiFi-3D-Fusion Training Launcher
# Quick training script with intelligent defaults

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                 WiFi-3D-Fusion Model Training               ║"
echo "║              Advanced CSI-Based Person Detection            ║" 
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo -e "${RED}❌ Virtual environment not found. Running setup...${NC}"
    bash scripts/install_all.sh
fi

# Activate virtual environment
echo -e "${YELLOW}🔧 Activating virtual environment...${NC}"
source .venv/bin/activate

# Check for existing data
CSI_COUNT=$(ls env/csi_logs/*.pkl 2>/dev/null | wc -l || echo "0")
echo -e "${BLUE}📊 Found $CSI_COUNT CSI data files${NC}"

if [ "$CSI_COUNT" -lt 10 ]; then
    echo -e "${YELLOW}⚠️  Warning: Limited training data ($CSI_COUNT files)${NC}"
    echo -e "${YELLOW}💡 Consider running data collection first:${NC}"
    echo -e "${YELLOW}   python run_js_visualizer.py${NC}"
    echo ""
fi

# Default parameters
DEVICE="auto"
SOURCE="auto"
EPOCHS=100
BATCH_SIZE=32
LEARNING_RATE=0.001
CONTINUOUS=false
AUTO_IMPROVE=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --source)
            SOURCE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --continuous)
            CONTINUOUS=true
            shift
            ;;
        --auto-improve)
            AUTO_IMPROVE=true
            CONTINUOUS=true
            shift
            ;;
        --quick)
            EPOCHS=50
            BATCH_SIZE=16
            echo -e "${YELLOW}🚀 Quick training mode: $EPOCHS epochs${NC}"
            shift
            ;;
        --intensive)
            EPOCHS=500
            BATCH_SIZE=64
            CONTINUOUS=true
            AUTO_IMPROVE=true
            echo -e "${YELLOW}🔥 Intensive training mode: $EPOCHS epochs with continuous learning${NC}"
            shift
            ;;
        --help)
            echo "WiFi-3D-Fusion Training Options:"
            echo ""
            echo "Basic usage:"
            echo "  ./train_wifi3d.sh                    # Default training"
            echo "  ./train_wifi3d.sh --quick            # Fast 50-epoch training"
            echo "  ./train_wifi3d.sh --intensive        # 500 epochs with auto-improvement"
            echo ""
            echo "Advanced options:"
            echo "  --device [cpu|cuda|auto]             # Training device (default: auto)"
            echo "  --source [esp32|nexmon|auto]         # CSI data source (default: auto)"
            echo "  --epochs N                           # Number of training epochs (default: 100)"
            echo "  --batch-size N                       # Training batch size (default: 32)"
            echo "  --lr FLOAT                           # Learning rate (default: 0.001)"
            echo "  --continuous                         # Enable continuous learning"
            echo "  --auto-improve                       # Enable automatic model improvement"
            echo ""
            echo "Examples:"
            echo "  ./train_wifi3d.sh --source esp32 --epochs 200 --continuous"
            echo "  ./train_wifi3d.sh --device cuda --intensive"
            echo "  ./train_wifi3d.sh --quick --source nexmon"
            exit 0
            ;;
        *)
            echo -e "${RED}❌ Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Auto-detect device if not specified
if [ "$DEVICE" = "auto" ]; then
    if python -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
        DEVICE="cuda"
        echo -e "${GREEN}🚀 CUDA detected - using GPU acceleration${NC}"
    else
        DEVICE="cpu"
        echo -e "${YELLOW}💻 Using CPU training (consider GPU for faster training)${NC}"
    fi
fi

# Auto-detect source from config if not specified
if [ "$SOURCE" = "auto" ]; then
    if [ -f "configs/fusion.yaml" ]; then
        SOURCE=$(python -c "import yaml; print(yaml.safe_load(open('configs/fusion.yaml'))['source'])" 2>/dev/null || echo "esp32")
        echo -e "${BLUE}📋 Detected source from config: $SOURCE${NC}"
    else
        SOURCE="esp32"
        echo -e "${YELLOW}📋 Using default source: $SOURCE${NC}"
    fi
fi

# Build training command
TRAIN_CMD="python train_model.py"
TRAIN_CMD="$TRAIN_CMD --config configs/fusion.yaml"
TRAIN_CMD="$TRAIN_CMD --device $DEVICE"
TRAIN_CMD="$TRAIN_CMD --source $SOURCE"
TRAIN_CMD="$TRAIN_CMD --epochs $EPOCHS"
TRAIN_CMD="$TRAIN_CMD --batch-size $BATCH_SIZE"
TRAIN_CMD="$TRAIN_CMD --lr $LEARNING_RATE"

if [ "$CONTINUOUS" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --continuous"
fi

if [ "$AUTO_IMPROVE" = true ]; then
    TRAIN_CMD="$TRAIN_CMD --auto-improve"
fi

# Display training configuration
echo -e "${BLUE}📋 Training Configuration:${NC}"
echo -e "   Device: ${GREEN}$DEVICE${NC}"
echo -e "   Source: ${GREEN}$SOURCE${NC}"
echo -e "   Epochs: ${GREEN}$EPOCHS${NC}"
echo -e "   Batch Size: ${GREEN}$BATCH_SIZE${NC}"
echo -e "   Learning Rate: ${GREEN}$LEARNING_RATE${NC}"
echo -e "   Continuous Learning: ${GREEN}$CONTINUOUS${NC}"
echo -e "   Auto Improvement: ${GREEN}$AUTO_IMPROVE${NC}"
echo ""

# Check dependencies
echo -e "${YELLOW}🔍 Checking dependencies...${NC}"
python -c "import torch, numpy, yaml, pickle" || {
    echo -e "${RED}❌ Missing dependencies. Installing...${NC}"
    pip install torch numpy PyYAML
}

# Create necessary directories
mkdir -p env/weights env/logs

# Start training
echo -e "${GREEN}🚀 Starting training...${NC}"
echo -e "${BLUE}Command: $TRAIN_CMD${NC}"
echo ""

# Run training with error handling
if eval $TRAIN_CMD; then
    echo ""
    echo -e "${GREEN}✅ Training completed successfully!${NC}"
    echo ""
    echo -e "${BLUE}📁 Results saved to:${NC}"
    echo -e "   Models: ${GREEN}env/weights/${NC}"
    echo -e "   Logs: ${GREEN}env/logs/${NC}"
    echo ""
    echo -e "${BLUE}🎯 Next steps:${NC}"
    echo -e "   1. Test your model: ${YELLOW}python run_js_visualizer.py${NC}"
    echo -e "   2. Real-time detection: ${YELLOW}./run_wifi3d.sh${NC}"
    echo -e "   3. Continue training: ${YELLOW}./train_wifi3d.sh --continuous${NC}"
    echo ""
    
    # Show model info
    BEST_MODEL="env/weights/best_model.pth"
    if [ -f "$BEST_MODEL" ]; then
        MODEL_SIZE=$(du -h "$BEST_MODEL" | cut -f1)
        echo -e "${GREEN}💾 Best model: $BEST_MODEL ($MODEL_SIZE)${NC}"
    fi
    
else
    echo ""
    echo -e "${RED}❌ Training failed!${NC}"
    echo -e "${YELLOW}💡 Troubleshooting tips:${NC}"
    echo -e "   - Check data availability: ${YELLOW}ls env/csi_logs/${NC}"
    echo -e "   - Try CPU training: ${YELLOW}./train_wifi3d.sh --device cpu${NC}"
    echo -e "   - Reduce batch size: ${YELLOW}./train_wifi3d.sh --batch-size 16${NC}"
    echo -e "   - Collect more data: ${YELLOW}python run_js_visualizer.py${NC}"
    echo ""
    exit 1
fi
