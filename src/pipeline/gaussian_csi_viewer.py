#!/usr/bin/env python3
"""
Realtime Gaussian Splatting Viewer for WiFi CSI + ReID + Skeletons
------------------------------------------------------------------
- Fast CSI renderer (single RGBA actor on a cached grid) in 2.5D.
- Per-person Gaussian “body” splats + COCO-ish skeleton overlay.
- ReID (enroll + infer) via your src/pipeline/realtime_identifier.py
- Sources: esp32 | nexmon (monitor/radiotap) | dummy
- Controls:
    WASD  : strafe/forward
    Q/E   : up/down
    Z/X   : zoom in/out
    C     : reset 2.5D camera
    F     : FAST mode (toggle AA/EDL off for max FPS)
- Notes:
    * Skeletons are animated mock walking cycles (no pose model).
      They are visually anchored “agents” to demonstrate people rendering.
    * ReID uses your CSI buffer and provides an ID + score to label agents.
"""

import os
import time
import numpy as np
import pyvista as pv

# ---------- Appearance / Tuning ----------
BG_COLOR = "black"

# CSI splats
CSI_NEON = np.array([0.10, 1.00, 0.40], dtype=np.float32)  # neon green
CSI_POINT_SIZE = 9
CSI_ALPHA_BASE = 0.18
CSI_ALPHA_GAIN = 0.72
CSI_Z_SCALE = 0.30
CSI_SMOOTH = 0.20     # lower -> more reactive
CSI_PLANE_Z = 0.04

# Person “body” splats
BODY_POINTS_PER_JOINT = 500
BODY_SIGMA = 0.045
BODY_POINT_SIZE = 12
BODY_ALPHA_BASE = 0.22
BODY_ALPHA_GAIN = 0.65

# Skeleton overlay
BONE_COLOR = (0.92, 0.92, 0.92)
BONE_WIDTH = 4

# Person color palette
PERSON_COLORS = [
    (0.10, 1.00, 0.40),  # neon green
    (0.20, 0.60, 1.00),  # cyan/blue
    (1.00, 0.30, 0.50),  # pink
    (1.00, 0.80, 0.20),  # amber
    (0.80, 0.80, 1.00),  # lilac
    (0.40, 1.00, 0.90),  # mint
]

# COCO-ish bones (robust: drops any edge if index >= K)
DEFAULT_BONES = [
    (5,6), (5,7), (7,9), (6,8), (8,10),
    (11,12), (11,13), (13,15), (12,14), (14,16),
    (5,11), (6,12)
]

# ---------- Helpers ----------
def _color_for_pid(pid: int) -> np.ndarray:
    c = PERSON_COLORS[pid % len(PERSON_COLORS)]
    return np.array(c, dtype=np.float32)

def _gaussian_blob(center: np.ndarray, n: int, sigma: float) -> np.ndarray:
    return (center.reshape(1,3) +
            np.random.normal(0.0, sigma, size=(n,3)).astype(np.float32))

def _iso_2p5d(plotter: pv.Plotter, zoom=1.15):
    plotter.camera_position = 'iso'
    plotter.camera.zoom(zoom)

def _lissajous_root(t, ax=0.9, ay=0.9, az=0.18, f=1.0, phase=0.0):
    x = ax * np.sin(f*0.6*t + 0.5 + phase)
    y = ay * np.sin(f*0.4*t + phase*0.7)
    z = 0.05 + az * 0.5 * (np.sin(0.7 * t + phase) * 0.5 + 0.5)
    return np.array([x,y,z], dtype=np.float32)

def _synth_skeleton(base, t, amp=0.12, K=17):
    """Simple walking cycle around base; indices compatible with DEFAULT_BONES."""
    sk = np.tile(base.reshape(1,3),(K,1)).astype(np.float32)
    sk[5] += [ -0.10,  0.05, 0.35 ];  sk[6] += [ 0.10,  0.05, 0.35 ]
    sk[7]  = sk[5] + [ -0.10,  0.00, -0.10 - amp*np.sin(2.0*t) ]
    sk[8]  = sk[6] + [  0.10,  0.00, -0.10 + amp*np.sin(2.0*t) ]
    sk[9]  = sk[7] + [ -0.06, -0.02, -0.10 ]
    sk[10] = sk[8] + [  0.06, -0.02, -0.10 ]
    sk[11] += [ -0.08, -0.02, 0.15 ]; sk[12] += [ 0.08, -0.02, 0.15 ]
    sk[13] = sk[11] + [ -0.02, -0.02, -0.15 - amp*np.sin(2.2*t) ]
    sk[14] = sk[12] + [  0.02, -0.02, -0.15 + amp*np.sin(2.2*t) ]
    sk[15] = sk[13] + [  0.00,  0.00, -0.12 ]
    sk[16] = sk[14] + [  0.00,  0.00, -0.12 ]
    return sk

# ---------- High-performance CSI + People Viewer ----------
class GaussianRealtimeView:
    """
    One plotter, two subsystems:
      - CSI plane splats (single actor with RGBA; points cached; per-frame RGBA+Z update)
      - People: each pid -> body actor (RGBA point cloud) + skeleton line actor + label
    """
    def __init__(self, window_size=(1400, 900)):
            print("[DEBUG] Inicializando ventana PyVista...")
            try:
                self.p = pv.Plotter(window_size=window_size)
                self.p.set_background(BG_COLOR)
                print("[DEBUG] Ventana PyVista creada correctamente.")
            except Exception as e:
                print(f"[CRITICAL] Error al crear ventana PyVista: {e}")
                raise

            # Quality defaults (toggle with 'F')
            self._fast = False
            try:
                self.p.enable_anti_aliasing('ssaa')
            except Exception as e:
                print(f"[WARNING] No se pudo activar anti-aliasing: {e}")
            try:
                self.p.enable_eye_dome_lighting()
            except Exception as e:
                print(f"[WARNING] No se pudo activar eye dome lighting: {e}")

            # CSI cache/actors
            self._H = self._W = None
            self._xx = self._yy = None
            self._csi_actor = None
            self._csi_ema = None

            # Per-person state
            self._people = {}  # pid -> {'actor':..., 'bones':..., 'label':...}

            self._init_floor()
            self._bind_keys()
            _iso_2p5d(self.p, zoom=1.15)
            print("[DEBUG] Mostrando ventana interactiva...")
            try:
                self.p.show(auto_close=False, interactive_update=True)
                print("[DEBUG] Ventana interactiva mostrada correctamente.")
            except Exception as e:
                print(f"[CRITICAL] Error al mostrar ventana interactiva: {e}")
                raise

    # ---------- Scene primitives ----------
    def _init_floor(self):
        plane = pv.Plane(center=(0,0,0), direction=(0,0,1),
                         i_size=4.0, j_size=4.0, i_resolution=40, j_resolution=40)
        self.p.add_mesh(plane, color=(0.08,0.08,0.08), style='wireframe', opacity=0.35)
        for x in (-2.0, -1.0, 0.0, 1.0, 2.0):
            self.p.add_mesh(pv.Line((x,-2,0), (x,2,0)), color=(0.12,0.12,0.12), line_width=1)
        for y in (-2.0, -1.0, 0.0, 1.0, 2.0):
            self.p.add_mesh(pv.Line((-2,y,0), (2,y,0)), color=(0.12,0.12,0.12), line_width=1)

    def _bind_keys(self):
        STEP = 0.18
        ZSTEP = 1.08
        def move(dx=0, dy=0, dz=0):
            cam = self.p.camera
            pos = np.array(cam.position); fp = np.array(cam.focal_point); up = np.array(cam.up)
            f = fp - pos; f /= (np.linalg.norm(f)+1e-9)
            r = np.cross(f, up); r /= (np.linalg.norm(r)+1e-9)
            u = np.cross(r, f); u /= (np.linalg.norm(u)+1e-9)
            d = r*dx + f*dy + u*dz
            cam.position = tuple(pos + d); cam.focal_point = tuple(fp + d)
        def zoom(f): self.p.camera.zoom(f)
        def fast():
            self._fast = not self._fast
            if self._fast:
                try: self.p.disable_eye_dome_lighting()
                except Exception: pass
                try: self.p.disable_anti_aliasing()
                except Exception: pass
            else:
                try: self.p.enable_eye_dome_lighting()
                except Exception: pass
                try: self.p.enable_anti_aliasing('ssaa')
                except Exception: pass

        self.p.add_key_event("w", lambda: move(dy= STEP))
        self.p.add_key_event("s", lambda: move(dy=-STEP))
        self.p.add_key_event("a", lambda: move(dx=-STEP))
        self.p.add_key_event("d", lambda: move(dx= STEP))
        self.p.add_key_event("q", lambda: move(dz= STEP))
        self.p.add_key_event("e", lambda: move(dz=-STEP))
        self.p.add_key_event("z", lambda: zoom( ZSTEP))
        self.p.add_key_event("x", lambda: zoom(1.0/ZSTEP))
        self.p.add_key_event("c", lambda: _iso_2p5d(self.p))
        self.p.add_key_event("f", fast)

    # ---------- CSI pipeline (fast RGBA actor) ----------
    @staticmethod
    def _ensure_hw(amp: np.ndarray) -> np.ndarray:
        a = np.asarray(amp)
        if a.ndim == 1:
            n = int(a.size)
            H = int(np.floor(np.sqrt(n))) or 1
            W = int(np.ceil(n / H))
            pad = H*W - n
            if pad > 0:
                a = np.pad(a, (0, pad), mode='edge')
            a = a.reshape(H, W)
        elif a.ndim != 2:
            a = a.reshape(-1); return GaussianRealtimeView._ensure_hw(a)
        return a.astype(np.float32, copy=False)

    def _prepare_csi_grid(self, H, W):
        self._H, self._W = H, W
        yy, xx = np.meshgrid(np.linspace(0, W-1, W, dtype=np.float32),
                             np.linspace(0, H-1, H, dtype=np.float32))
        self._xx = (xx / max(1.0, H-1)) * 2.0 - 1.0
        self._yy = (yy / max(1.0, W-1)) * 2.0 - 1.0
        z = np.full((H*W,), CSI_PLANE_Z, dtype=np.float32)
        pts = np.stack([self._xx.ravel(), self._yy.ravel(), z], axis=1)

        poly = pv.PolyData(pts)
        # RGBA initial buffer
        rgba = np.zeros((H*W, 4), dtype=np.uint8)
        base_rgb = (CSI_NEON * 0.5).clip(0,1)
        rgba[:,0:3] = (base_rgb * 255).astype(np.uint8)
        rgba[:,3] = 60
        poly['RGBA'] = rgba

        self._csi_actor = self.p.add_mesh(
            poly, scalars='RGBA', rgba=True,
            render_points_as_spheres=True,
            point_size=CSI_POINT_SIZE,
            show_scalar_bar=False
        )

    def update_csi_amp(self, amp: np.ndarray):
        A = self._ensure_hw(amp)
        H, W = A.shape

        if (self._csi_actor is None) or (self._H != H) or (self._W != W):
            self._csi_ema = None
            self._prepare_csi_grid(H, W)

        # EMA smoothing
        if self._csi_ema is None: self._csi_ema = A.copy()
        self._csi_ema = CSI_SMOOTH*self._csi_ema + (1.0-CSI_SMOOTH)*A
        A = self._csi_ema

        # Normalize
        amin = float(np.min(A)); rng = float(np.ptp(A)) if A.size else 0.0
        den = rng if rng > 1e-9 else 1.0
        A01 = (A - amin) / den

        # Update Z
        z = CSI_PLANE_Z + (A01 * CSI_Z_SCALE)
        pts = np.stack([self._xx.ravel(), self._yy.ravel(), z.ravel()], axis=1)
        poly: pv.PolyData = self._csi_actor.mapper.dataset  # type: ignore
        poly.points = pts

        # Update RGBA per point
        w = A01.ravel()
        rgb = (CSI_NEON.reshape(1,3) * (0.35 + 0.65*w[:,None])).clip(0,1)
        alpha = (np.clip(CSI_ALPHA_BASE + CSI_ALPHA_GAIN*w, 0.05, 0.98) * 255).astype(np.uint8)
        rgba = np.empty((H*W, 4), dtype=np.uint8)
        rgba[:,0:3] = (rgb * 255).astype(np.uint8)
        rgba[:,3] = alpha
        poly['RGBA'] = rgba  # in-place update

    # ---------- People (body RGBA cloud + skeleton + label) ----------
    def _drop_pid(self, pid: int):
        entry = self._people.pop(pid, None)
        if not entry: return
        for k in ('actor','bones','label'):
            a = entry.get(k)
            if a is not None:
                try: self.p.remove_actor(a)
                except Exception: pass

    def update_person(self, pid: int, joints3d: np.ndarray, score: float, text: str = ""):
        """
        joints3d: (K,3) float32
        score: 0..1 -> opacity/brightness
        text: label near head
        """
        K = joints3d.shape[0]
        col = _color_for_pid(pid)
        alpha = float(np.clip(BODY_ALPHA_BASE + BODY_ALPHA_GAIN*score, 0.05, 0.98))

        # Body splats (one RGBA actor for all joints)
        clouds = []; weights = []
        for j in range(K):
            cloud = _gaussian_blob(joints3d[j], BODY_POINTS_PER_JOINT, BODY_SIGMA)
            r = np.linalg.norm(cloud - joints3d[j], axis=1)
            w = np.exp(-(r*r) / (2.0*(BODY_SIGMA**2)))
            clouds.append(cloud); weights.append(w)
        cloud = np.vstack(clouds)
        w_all = np.hstack(weights)

        rgb = (col.reshape(1,3) * (0.35 + 0.65*w_all.reshape(-1,1))).clip(0,1)
        a_arr = np.clip(alpha * (0.35 + 0.65*w_all), 0.05, 0.99)
        rgba = np.zeros((cloud.shape[0],4), dtype=np.uint8)
        rgba[:,0:3] = (rgb * 255).astype(np.uint8)
        rgba[:,3] = (a_arr * 255).astype(np.uint8)

        # Build/Update actors
        entry = self._people.get(pid, {})
        # Body actor (RGBA)
        if 'actor' not in entry or entry['actor'] is None:
            poly = pv.PolyData(cloud); poly['RGBA'] = rgba
            actor = self.p.add_mesh(poly, scalars='RGBA', rgba=True,
                                    render_points_as_spheres=True,
                                    point_size=BODY_POINT_SIZE,
                                    show_scalar_bar=False)
            entry['actor'] = actor
        else:
            poly: pv.PolyData = entry['actor'].mapper.dataset  # type: ignore
            poly.points = cloud
            poly['RGBA'] = rgba

        # Skeleton lines
        lines = [(a,b) for (a,b) in DEFAULT_BONES if a < K and b < K]
        if lines:
            L = np.array(lines, dtype=np.int32)
            skel = pv.PolyData(joints3d.copy())
            skel.lines = np.hstack([np.full((L.shape[0],1),2,np.int32), L]).ravel()
            if 'bones' not in entry or entry['bones'] is None:
                entry['bones'] = self.p.add_mesh(skel, color=BONE_COLOR, line_width=BONE_WIDTH)
            else:
                entry['bones'].mapper.dataset = skel  # cheap update

        # Label near head (use joint 5 or 6 if available, else centroid)
        head = joints3d[5] if K>5 else joints3d.mean(axis=0)
        label_text = text if text else f"ID {pid} ({score:.2f})"
        if 'label' not in entry or entry['label'] is None:
            entry['label'] = self.p.add_point_labels(
                [head.tolist()],
                [label_text],
                point_size=0, font_size=14,
                text_color=(1,1,1), shape_color=(0.0,0.0,0.0),
                fill_shape=True, show_points=False
            )
        else:
            # Recreate label geometry to move text reliably
            self.p.remove_actor(entry['label'])
            entry['label'] = self.p.add_point_labels(
                [head.tolist()],
                [label_text],
                point_size=0, font_size=14,
                text_color=(1,1,1), shape_color=(0.0,0.0,0.0),
                fill_shape=True, show_points=False
            )

        self._people[pid] = entry

    def render_once(self):
        self.p.render()

    def tick(self, amp=None, persons=None):
        """
        Legacy wrapper to support scripts that expect .tick()
        - amp: np.ndarray amplitude vector (optional)
        - persons: list of (pid, joints, score, label) (optional)
        """
        if amp is not None:
            self.update_csi_amp(amp)
        if persons is not None:
            for (pid, joints, score, label) in persons:
                self.update_person(pid, joints, score, text=label)
        self.render_once()


# ---------- ReID wrapper (uses your existing module) ----------
class ReIDBridge:
    """
    Wraps src/pipeline/realtime_identifier.RealtimeIdentifier
    - Auto-enroll if REID_ENROLL_PID is set
    - Push complex CSI (amp+phase) per frame
    """
    def __init__(self, feat_dim_hint: int | None, cfg):
        self.ident = None
        self.enrolled = False
        self.cfg = cfg
        self.enroll_pid_env = os.environ.get('REID_ENROLL_PID')
        self._feat_dim_hint = feat_dim_hint

    def _ensure(self, amp_like: np.ndarray):
        if self.ident is not None: return
        try:
            from src.pipeline.realtime_identifier import RealtimeIdentifier
        except Exception as e:
            print(f"[ReID] disabled (import error): {e}")
            return
        ckpt = self.cfg.get('reid', {}).get('checkpoint', 'env/weights/who_reid_best.pth')
        seq_secs = float(self.cfg.get('reid', {}).get('seq_secs', 2.0))
        fps      = float(self.cfg.get('reid', {}).get('fps', 20.0))
        d = (amp_like.size * 2) if self._feat_dim_hint is None else (self._feat_dim_hint * 2)
        try:
            self.ident = RealtimeIdentifier(
                feat_dim=d,
                ckpt_path=ckpt,
                seq_secs=seq_secs,
                fps=fps
            )
            print(f"[ReID] initialized feat_dim={d}  ckpt={ckpt}")
        except Exception as e:
            print(f"[ReID] disabled (init error): {e}")
            self.ident = None

    @staticmethod
    def _as_complex(x: np.ndarray) -> np.ndarray:
        return x if np.iscomplexobj(x) else x.astype(np.float32).astype(np.complex64)

    def push_and_infer(self, ts: float, csi_vec: np.ndarray):
        self._ensure(np.asarray(csi_vec))
        if self.ident is None: return None
        self.ident.push(ts, self._as_complex(csi_vec))

        if (not self.enrolled) and (self.enroll_pid_env is not None):
            ok = self.ident.enroll(int(self.enroll_pid_env), shots=8)
            if ok:
                print(f"[ReID] Enrolled pid={self.enroll_pid_env}")
                self.enrolled = True

        return self.ident.infer()

# ---------- Sources ----------
def make_source(cfg):
    src_name = os.environ.get('SOURCE', cfg.get('source','esp32'))
    if src_name == 'esp32':
        from src.csi_sources.esp32_udp import ESP32UDPCSISource as Src
        port = int(cfg.get('esp32_udp_port', 5566))
        mtu  = int(cfg.get('esp32_mtu', 2000))
        return Src(port=port, mtu=mtu)
    elif src_name == 'nexmon':
        from src.csi_sources.monitor_radiotap import MonitorRadiotapSource as Src
        iface = os.environ.get('IFACE') or cfg.get('nexmon_iface','wlan0')
        return Src(iface=iface)
    else:
        # dummy sim
        from src.csi_sources.dummy_sim import DummySimSource as Src
        D   = int(os.environ.get("D", "128"))
        FPS = float(os.environ.get("FPS", "20"))
        PERS= int(os.environ.get("PERSONS", "3"))
        return Src(D=D, fps=FPS, persons=PERS)

# ---------- Config ----------
def load_cfg():
    # Light, avoids any heavy external deps:
    # If you already have src/common/config, we’ll try that first
    try:
        from src.common.config import load_cfg as _load
        return _load()
    except Exception:
        # minimal defaults
        return {
            'source': 'esp32',
            'esp32_udp_port': 5566,
            'esp32_mtu': 2000,
            'nexmon_iface': 'wlan0',
            'win_seconds': 2.0,
            'movement_threshold': 0.02,
            'debounce_seconds': 0.8,
            'enable_reid': True,
            'reid': {
                'checkpoint': 'env/weights/who_reid_best.pth',
                'seq_secs': 2.0,
                'fps': 20.0
            }
        }

# ---------- Main Loop ----------
def main():
    cfg = load_cfg()
    view = GaussianRealtimeView()
    src = make_source(cfg)
    reid = ReIDBridge(feat_dim_hint=None, cfg=cfg)

    # Animation state for mock skeletons (visual only)
    # Each recognized pid will get a persistent phase offset
    pid_phase = {}

    # Non-blocking UI render loop
    view.p.add_text("WiFi Gaussian Realtime — CSI + ReID + Skeletons",
                    font_size=11, color='white')

    t0 = time.time()
    last_reid = 0.0
    REID_PERIOD = 0.10  # seconds between inference attempts

    # Simple FPS print
    last_print = time.time()
    frames = 0

    for ts, vec in src.frames():
        # CSI amplitude and render (fast)
        amp = np.abs(vec).astype(np.float32) if np.iscomplexobj(vec) else np.asarray(vec, dtype=np.float32)
        view.update_csi_amp(amp)

        # ReID inference throttled (to keep UI smooth)
        now = time.time()
        out = None
        if (now - last_reid) >= REID_PERIOD:
            out = reid.push_and_infer(ts, vec)
            last_reid = now

        # Visualize up to N “people” (agents). We animate skeletons; label with ReID if available.
        # If no ReID yet, show a couple of agents for demo.
        persons = int(os.environ.get("PERSONS", "2"))
        if out:
            pid = int(out['pid'])
            score = float(out['score'])
            persons = max(persons, pid+1)  # ensure room for that pid
            pid_phase.setdefault(pid, (0.7*pid) % (2*np.pi))
            base = _lissajous_root(now + pid_phase[pid], f=1.0+0.05*pid, phase=pid_phase[pid])
            skel = _synth_skeleton(base, now + pid_phase[pid], amp=0.12, K=17)
            view.update_person(pid, skel, score=score, text=f"ID {pid} · {score:.2f}")
        else:
            # keep scene lively even before enrolling/infer
            for pid in range(persons):
                pid_phase.setdefault(pid, (0.7*pid) % (2*np.pi))
                base = _lissajous_root(now + pid_phase[pid], f=1.0+0.05*pid, phase=pid_phase[pid])
                skel = _synth_skeleton(base, now + pid_phase[pid], amp=0.12, K=17)
                view.update_person(pid, skel, score=0.6, text=f"ID {pid} · --")

        view.render_once()

        # crude FPS monitor
        frames += 1
        if (now - last_print) >= 1.0:
            print(f"[fps] {frames/(now-last_print):.1f}")
            frames = 0
            last_print = now

if __name__ == "__main__":
    main()
GaussianCSIViewer = GaussianRealtimeView