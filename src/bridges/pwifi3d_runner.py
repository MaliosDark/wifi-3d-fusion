import subprocess, sys, os
from pathlib import Path
from loguru import logger

def run_batch(repo: str, config: str, weights: str, python: str = "python"):
    """
    Calls the repo's OpenMMLab test script on prepared data.
    Expectation:
      - You have created a dataset under third_party/Person-in-WiFi-3D-repo/data/wifipose/test_data/csi
      - weights is a valid checkpoint .pth compatible with the config
    """
    repo = Path(repo)
    cfg = repo / config
    wts = Path(weights)
    test_py = repo / "tools/test.py"
    assert cfg.exists(), f"Config not found: {cfg}"
    assert wts.exists(), f"Weights not found: {wts}"
    assert test_py.exists(), f"test.py not found in {repo}"
    cmd = [python, str(test_py), str(cfg), str(wts), "--eval", "mAP"]
    logger.info(f"[pwifi3d] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
