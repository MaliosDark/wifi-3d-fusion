from pathlib import Path
import yaml
from loguru import logger

CFG_PATH = Path("configs/fusion.yaml")

def load_cfg():
    with open(CFG_PATH, "r") as f:
        return yaml.safe_load(f)

def ensure_dirs():
    Path("env/logs").mkdir(parents=True, exist_ok=True)
    Path("env/weights").mkdir(parents=True, exist_ok=True)
    Path("env/tmp").mkdir(parents=True, exist_ok=True)

logger.add("env/logs/fusion_{time}.log", rotation="10 MB", retention="7 days")
