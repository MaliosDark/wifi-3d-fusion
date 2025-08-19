import subprocess
from pathlib import Path
from loguru import logger

def train(repo: str, config: str, python: str = "python", iters: int = 2000):
    repo = Path(repo)
    cfg = repo / config if not str(config).startswith(str(repo)) else Path(config)
    assert cfg.exists(), f"NeRF2 config not found: {cfg}"
    cmd = [python, str(repo / "nerf2_runner.py"), "--mode", "train", "--config", str(cfg), "--iters", str(iters)]
    logger.info(f"[nerf2] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
