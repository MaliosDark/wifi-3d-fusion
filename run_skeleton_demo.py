#!/usr/bin/env python3
# 3D Gaussian Splatting + Skeletons + CSI Wavefield (PyVista/VTK)
# - Neon isometric 2.5D camera (con teclas: WASD/QE para mover, ZX zoom, C para 2.5D)
# - Grid de piso
# - Gaussian blobs por joint (personas + esqueletos)
# - Simulación de ondas CSI como splats gaussianos sobre el plano (multi-AP)

import os, time
import numpy as np
import pyvista as pv

# ---------- Config ----------
PERSON_COLORS = [
    (0.10, 1.00, 0.40),  # neon green
    (0.20, 0.60, 1.00),  # cyan/blue
    (1.00, 0.30, 0.50),  # pink
    (1.00, 0.80, 0.20),  # amber
]
BG_COLOR = "black"
BONE_COLOR = (0.9, 0.9, 0.9)

# Splatting (personas)
POINTS_PER_JOINT = 600
SPLAT_SIGMA = 0.045
SPHERE_POINT_SIZE = 14
BONE_WIDTH = 4
ALPHA_BASE = 0.22
ALPHA_GAIN = 0.65

# CSI wavefield
CSI_PLANE_Z   = 0.10     # altura del plano de ondas
CSI_RING_PTS  = 2000     # puntos por AP por frame
CSI_SIGMA     = 0.03     # grosor del anillo (gauss)
CSI_SPEED     = 0.6      # velocidad (unid/seg)
CSI_WAVELEN   = 0.60     # “longitud de onda” (espaciado entre anillos)
CSI_DECAY     = 0.35     # decaimiento de amplitud con radio
CSI_ALPHA     = 0.35     # opacidad base de las ondas
CSI_COLOR     = (0.15, 0.9, 0.9)  # cian verdoso

DEFAULT_BONES = [
    (5,6), (5,7), (7,9), (6,8), (8,10),
    (11,12), (11,13), (13,15), (12,14), (14,16),
    (5,11), (6,12)
]

# ---------- Utils ----------
def gaussian_blob(center, n=POINTS_PER_JOINT, sigma=SPLAT_SIGMA):
    return center + np.random.normal(0.0, sigma, size=(n,3)).astype(np.float32)

def color_for_pid(pid: int):
    c = PERSON_COLORS[pid % len(PERSON_COLORS)]
    return np.array(c, dtype=np.float32)

def iso_2p5d(plotter, zoom=1.15):
    plotter.camera_position = 'iso'
    plotter.camera.zoom(zoom)

# ---------- Viewer ----------
class GaussianSkeletonViewer:
    def __init__(self, background=BG_COLOR):
        self.plotter = pv.Plotter(window_size=(1400, 900))
        self.plotter.set_background(background)
        self.plotter.enable_anti_aliasing('ssaa')
        self.plotter.enable_eye_dome_lighting()

        self.person_actors = {}  # pid -> dict(alpha, rgb, bones)
        self.csi_actor_alpha = None
        self.csi_actor_rgb   = None

        self._init_floor()
        self._bind_keys()
        iso_2p5d(self.plotter, zoom=1.15)

    # --- floor / grid ---
    def _init_floor(self):
        plane = pv.Plane(center=(0,0,0), direction=(0,0,1),
                         i_size=4.0, j_size=4.0, i_resolution=40, j_resolution=40)
        self.plotter.add_mesh(plane, color=(0.08,0.08,0.08),
                              style='wireframe', opacity=0.35)
        # líneas de ejes suaves
        for x in (-2.0, -1.0, 0.0, 1.0, 2.0):
            line = pv.Line((x,-2,0), (x,2,0))
            self.plotter.add_mesh(line, color=(0.12,0.12,0.12), line_width=1)
        for y in (-2.0, -1.0, 0.0, 1.0, 2.0):
            line = pv.Line((-2,y,0), (2,y,0))
            self.plotter.add_mesh(line, color=(0.12,0.12,0.12), line_width=1)

    # --- camera controls ---
    def _bind_keys(self):
        STEP = 0.18
        ZSTEP = 1.08

        def move(dx=0, dy=0, dz=0):
            cam = self.plotter.camera
            pos = np.array(cam.position)
            fp  = np.array(cam.focal_point)
            up  = np.array(cam.up)
            # build right/forward vectors from camera
            forward = (fp - pos); forward = forward / (np.linalg.norm(forward)+1e-9)
            right = np.cross(forward, up); right = right / (np.linalg.norm(right)+1e-9)
            upn = np.cross(right, forward); upn = upn / (np.linalg.norm(upn)+1e-9)
            delta = right*dx + forward*dy + upn*dz
            cam.position = tuple(pos + delta)
            cam.focal_point = tuple(fp + delta)

        def zoom(factor):
            self.plotter.camera.zoom(factor)

        # WASD: strafe/forward, QE: subir/bajar, ZX: zoom
        self.plotter.add_key_event("w", lambda: move(dy= STEP))
        self.plotter.add_key_event("s", lambda: move(dy=-STEP))
        self.plotter.add_key_event("a", lambda: move(dx=-STEP))
        self.plotter.add_key_event("d", lambda: move(dx= STEP))
        self.plotter.add_key_event("q", lambda: move(dz= STEP))
        self.plotter.add_key_event("e", lambda: move(dz=-STEP))
        self.plotter.add_key_event("z", lambda: zoom( ZSTEP))
        self.plotter.add_key_event("x", lambda: zoom(1.0/ZSTEP))
        self.plotter.add_key_event("c", lambda: iso_2p5d(self.plotter, zoom=1.15))

    # --- people (gaussian splats + skeleton) ---
    def _remove_pid(self, pid):
        if pid in self.person_actors:
            for a in self.person_actors[pid].values():
                if a is not None:
                    try: self.plotter.remove_actor(a)
                    except Exception: pass
            self.person_actors.pop(pid, None)

    def update_person(self, pid: int, joints3d: np.ndarray, score: float = 1.0, bones=DEFAULT_BONES):
        self._remove_pid(pid)
        if joints3d is None or len(joints3d) == 0:
            return

        K = joints3d.shape[0]
        cpid = color_for_pid(pid)
        alpha = float(np.clip(ALPHA_BASE + ALPHA_GAIN * float(score), 0.02, 0.98))

        clouds, weights = [], []
        for j in range(K):
            cloud = gaussian_blob(joints3d[j])
            r = np.linalg.norm(cloud - joints3d[j], axis=1)
            w = np.exp(-(r*r) / (2.0*(SPLAT_SIGMA**2)))
            clouds.append(cloud); weights.append(w)
        cloud = np.vstack(clouds)
        w_all = np.hstack(weights)

        rgb = (cpid.reshape(1,3) * (0.35 + 0.65*w_all.reshape(-1,1))).astype(np.float32)
        alpha_arr = np.clip(alpha * (0.35 + 0.65*w_all), 0.02, 0.99).astype(np.float32)

        # Actor A: alpha por punto
        pcloud_a = pv.PolyData(cloud)
        alpha_actor = self.plotter.add_mesh(
            pcloud_a,
            color=tuple((cpid*0.5).tolist()),
            opacity=alpha_arr,  # array (N,)
            render_points_as_spheres=True,
            point_size=SPHERE_POINT_SIZE,
            show_scalar_bar=False
        )
        # Actor B: color por punto
        pcloud_b = pv.PolyData(cloud)
        pcloud_b['RGB'] = rgb
        color_actor = self.plotter.add_mesh(
            pcloud_b,
            scalars='RGB', rgb=True,
            opacity=0.35,
            render_points_as_spheres=True,
            point_size=SPHERE_POINT_SIZE-2,
            show_scalar_bar=False
        )

        # skeleton
        lines = [(a,b) for (a,b) in bones if a < K and b < K]
        if lines:
            L = np.array(lines, dtype=np.int32)
            skel = pv.PolyData(joints3d)
            skel.lines = np.hstack([np.full((L.shape[0],1),2,np.int32), L]).ravel()
            bones_actor = self.plotter.add_mesh(skel, color=BONE_COLOR, line_width=BONE_WIDTH)
        else:
            bones_actor = None

        self.person_actors[pid] = dict(alpha=alpha_actor, rgb=color_actor, bones=bones_actor)

    # --- CSI wavefield (gaussian splats en anillos expansivos por AP) ---
    def update_csi_wavefield(self, ap_positions, t: float):
        # Construir anillos en el plano z=CSI_PLANE_Z
        all_pts = []
        all_w   = []
        all_rgb = []
        for ap in ap_positions:
            ap = np.asarray(ap, dtype=np.float32)
            # radio de frente de onda principal + sub-armónicos (multiples anillos)
            bases = [CSI_SPEED * t + k*CSI_WAVELEN for k in range(3)]
            for R in bases:
                if R <= 0.02:  # ignora radios muy chicos
                    continue
                theta = np.random.uniform(0, 2*np.pi, size=(CSI_RING_PTS,))
                # base del anillo
                x = ap[0] + R * np.cos(theta)
                y = ap[1] + R * np.sin(theta)
                z = np.full_like(x, CSI_PLANE_Z)
                ring = np.stack([x,y,z], axis=1)
                # grosor (gauss alrededor del anillo)
                ring += np.random.normal(0.0, CSI_SIGMA, size=ring.shape).astype(np.float32)
                # peso radial (más fuerte en el frente), + decaimiento 1/(1+aR)
                w = np.exp(-((np.linalg.norm(ring[:,:2]-ap[:2], axis=1) - R)**2)/(2*(CSI_SIGMA**2)))
                w *= 1.0 / (1.0 + CSI_DECAY*R)
                all_pts.append(ring)
                all_w.append(w)
                # color por AP (ligeras variaciones)
                base_c = np.array(CSI_COLOR, np.float32)
                jitter = 0.08*np.random.randn(3).astype(np.float32)
                all_rgb.append(np.clip(base_c + jitter, 0.0, 1.0)[None,:].repeat(len(w), axis=0))

        if not all_pts:
            return

        pts = np.vstack(all_pts)
        w   = np.clip(np.hstack(all_w), 0.0, 1.0)
        rgb = np.vstack(all_rgb)

        # alpha por punto según energía
        alpha_arr = np.clip(CSI_ALPHA * (0.25 + 0.75*w), 0.02, 0.95).astype(np.float32)

        # --- RE-CREAR actores por frame para soportar opacidad por punto ---
        # (PyVista no permite setear prop.opacity con arrays; add_mesh sí acepta opacity=array.)
        if self.csi_actor_alpha is not None:
            try: self.plotter.remove_actor(self.csi_actor_alpha)
            except Exception: pass
            self.csi_actor_alpha = None
        if self.csi_actor_rgb is not None:
            try: self.plotter.remove_actor(self.csi_actor_rgb)
            except Exception: pass
            self.csi_actor_rgb = None

        # Actor A (alpha por punto)
        cloud_a = pv.PolyData(pts)
        self.csi_actor_alpha = self.plotter.add_mesh(
            cloud_a,
            color=tuple((np.array(CSI_COLOR)*0.45).tolist()),
            opacity=alpha_arr,
            render_points_as_spheres=True,
            point_size=11,
            show_scalar_bar=False
        )

        # Actor B (rgb tenue)
        cloud_b = pv.PolyData(pts)
        cloud_b['RGB'] = rgb
        self.csi_actor_rgb = self.plotter.add_mesh(
            cloud_b,
            scalars='RGB', rgb=True,
            opacity=0.28,
            render_points_as_spheres=True,
            point_size=10,
            show_scalar_bar=False
        )

    # --- render tick ---
    def show_blocking(self):
        self.plotter.show(interactive=True)

    def update_once(self):
        self.plotter.render()


# ---------- Demo animator ----------
def lissajous_root(t, ax=0.9, ay=0.9, az=0.18, f=1.0, phase=0.0):
    x = ax * np.sin(f*0.6*t + 0.5 + phase)
    y = ay * np.sin(f*0.4*t + phase*0.7)
    z = 0.05 + az * 0.5 * (np.sin(0.7 * t + phase) * 0.5 + 0.5)
    return np.array([x,y,z], dtype=np.float32)

def synth_skeleton(base, t, amp=0.12, K=17):
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

def run_demo():
    persons = int(os.environ.get("PERSONS", "3"))
    viewer = GaussianSkeletonViewer()
    viewer.plotter.show(auto_close=False, interactive_update=True)

    # APs (simulados) — coloca donde quieras
    APs = [(-1.1, -1.1, CSI_PLANE_Z), (1.2, -1.0, CSI_PLANE_Z), (0.0, 1.2, CSI_PLANE_Z)]

    t0 = time.time()
    while True:
        t = time.time() - t0

        # Personas animadas
        for pid in range(persons):
            base = lissajous_root(t + pid*0.6, f=1.0+0.1*pid, phase=pid*0.7)
            skel = synth_skeleton(base, t + pid*0.6, amp=0.12, K=17)
            conf = 0.55 + 0.45*np.abs(np.sin(t*1.1 + pid))
            viewer.update_person(pid, skel, score=float(conf))

        # Ondas CSI (splatting en anillos expansivos)
        viewer.update_csi_wavefield(APs, t)

        viewer.update_once()
        time.sleep(0.02)

if __name__ == "__main__":
    run_demo()
