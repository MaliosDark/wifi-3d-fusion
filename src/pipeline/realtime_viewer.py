# src/pipeline/realtime_viewer.py
import numpy as np, threading, time
import open3d as o3d
from loguru import logger

class LivePointCloud:
    def __init__(self, point_size: float = 4.0):
        self._pcd = o3d.geometry.PointCloud()
        self._lock = threading.Lock()
        self._updated = False
        self._point_size = float(point_size)

    def _ensure_hw(self, amp: np.ndarray) -> np.ndarray:
        """Acepta amp 1D o 2D. Si es 1D, lo reacomoda a grilla HxW 'cuadrada'."""
        amp = np.asarray(amp)
        if amp.ndim == 1:
            n = int(amp.size)
            H = int(np.floor(np.sqrt(n))) or 1
            W = int(np.ceil(n / H))
            pad = H * W - n
            if pad > 0:
                amp = np.pad(amp, (0, pad), mode='edge')
            amp = amp.reshape(H, W)
        elif amp.ndim != 2:
            # Colapsa dimensiones extra si viniera (T,D) o similar accidentalmente
            flat = amp.reshape(-1)
            return self._ensure_hw(flat)
        return amp.astype(np.float32, copy=False)

    def update_from_csi(self, amp: np.ndarray):
        """Mapea amplitudes a nube 3D: x=freq idx, y=antena/subcarrier idx, z=amplitud normalizada."""
        amp = self._ensure_hw(amp)
        H, W = amp.shape

        # Construye grilla
        x = (np.arange(H, dtype=np.float32)[:, None] * np.ones(W, dtype=np.float32)[None, :]).ravel()
        y = (np.arange(W, dtype=np.float32)[None, :] * np.ones(H, dtype=np.float32)[:, None]).ravel()
        z = amp.ravel()

        if x.size == 0:
            return  # no hay puntos

        # Normalizaciones robustas (usa np.ptp para NumPy >=2.0)
        pts = np.stack([x, y, z], axis=-1)
        # normaliza x,y a [-1,1]
        xmax = float(pts[:, 0].max()) if pts.shape[0] else 1.0
        ymax = float(pts[:, 1].max()) if pts.shape[0] else 1.0
        if xmax <= 0: xmax = 1.0
        if ymax <= 0: ymax = 1.0
        pts[:, 0] = (pts[:, 0] / xmax) * 2.0 - 1.0
        pts[:, 1] = (pts[:, 1] / ymax) * 2.0 - 1.0

        rng = float(np.ptp(pts[:, 2])) if pts.shape[0] else 0.0
        den = 1e-6 if rng == 0.0 else rng
        zmin = float(np.min(pts[:, 2])) if pts.shape[0] else 0.0
        pts[:, 2] = (pts[:, 2] - zmin) / den  # 0..1

        # Color verde neón uniforme
        colors = np.tile(np.array([0.0, 1.0, 0.3], dtype=np.float32), (pts.shape[0], 1))

        with self._lock:
            self._pcd.points = o3d.utility.Vector3dVector(pts)
            self._pcd.colors = o3d.utility.Vector3dVector(colors)
            self._updated = True

    def run(self):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="WiFi CSI — LivePointCloud", width=1280, height=800, visible=True)
        vis.add_geometry(self._pcd)

        opt = vis.get_render_option()
        opt.background_color = np.array([0, 0, 0], dtype=np.float32)
        opt.point_size = self._point_size
        opt.show_coordinate_frame = False

        # Controles simples
        def plus_cb(v):
            ro = v.get_render_option(); ro.point_size = min(20.0, ro.point_size + 1.0); return False
        def minus_cb(v):
            ro = v.get_render_option(); ro.point_size = max(1.0,  ro.point_size - 1.0); return False
        def reset_cb(v):
            v.reset_view_point(True); return False

        vis.register_key_callback(ord('+'), plus_cb)
        vis.register_key_callback(ord('='), plus_cb)
        vis.register_key_callback(ord('-'), minus_cb)
        vis.register_key_callback(ord('R'), reset_cb)

        ctr = vis.get_view_control()
        # vista isométrica suave
        iso = [-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)]
        ctr.set_front(iso); ctr.set_up([0, 1, 0]); ctr.set_lookat([0, 0, 0]); ctr.set_zoom(0.8)

        while True:
            upd = False
            with self._lock:
                if self._updated:
                    upd = True
                    self._updated = False
            if upd:
                vis.update_geometry(self._pcd)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.02)
