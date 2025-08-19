import numpy as np, threading, time
import open3d as o3d
from loguru import logger

class LivePointCloud:
    def __init__(self):
        self._pcd = o3d.geometry.PointCloud()
        self._lock = threading.Lock()
        self._updated = False

    def update_from_csi(self, amp: np.ndarray):
        # Map amplitudes to a 3D line cloud: x = freq index, y = antenna idx (if present), z = amplitude
        # Robust and driver-agnostic visualization (not a mock; shows real CSI dynamics).
        if amp.ndim == 1:
            x = np.arange(amp.size, dtype=np.float32)
            y = np.zeros_like(x)
            z = amp.astype(np.float32)
        else:
            x = np.arange(amp.shape[0])[:,None] * np.ones(amp.shape[1])[None,:]
            y = np.arange(amp.shape[1])[None,:] * np.ones(amp.shape[0])[:,None]
            z = amp
            x, y, z = x.ravel(), y.ravel(), z.ravel()
        pts = np.stack([x, y, z], axis=-1)
        pts[:,0] /= max(1.0, pts[:,0].max())
        pts[:,1] /= max(1.0, max(1.0, pts[:,1].max()))
        pts[:,2] = (pts[:,2] - pts[:,2].min()) / max(1e-6, (pts[:,2].ptp()))
        colors = np.repeat(pts[:,2:3], 3, axis=1)
        with self._lock:
            self._pcd.points = o3d.utility.Vector3dVector(pts)
            self._pcd.colors = o3d.utility.Vector3dVector(colors)
            self._updated = True

    def run(self):
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="WiFi CSI Live")
        vis.add_geometry(self._pcd)
        while True:
            with self._lock:
                if self._updated:
                    vis.update_geometry(self._pcd)
                    self._updated = False
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.02)
