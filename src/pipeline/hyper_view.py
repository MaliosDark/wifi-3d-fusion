# src/pipeline/hyper_view.py
import time, threading
import numpy as np
import open3d as o3d

def _cube_wire(center, size, color=(0.0, 0.6, 0.2)):
    cx, cy, cz = center; s = size/2.0
    pts = np.array([
        [cx-s, cy-s, cz-s],[cx+s, cy-s, cz-s],[cx+s, cy+s, cz-s],[cx-s, cy+s, cz-s],
        [cx-s, cy-s, cz+s],[cx+s, cy-s, cz+s],[cx+s, cy+s, cz+s],[cx-s, cy+s, cz+s],
    ], dtype=np.float32)
    lines = np.array([[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]], dtype=np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines  = o3d.utility.Vector2iVector(lines)
    col = np.tile(np.array(color, np.float32),(lines.shape[0],1))
    ls.colors = o3d.utility.Vector3dVector(col)
    return ls

def _lines_from_points(points, color=(0.0,1.0,0.4)):
    if len(points) < 2:
        ls = o3d.geometry.LineSet()
        P = np.asarray(points, np.float32) if len(points) else np.zeros((1,3),np.float32)
        ls.points = o3d.utility.Vector3dVector(P)
        ls.lines  = o3d.utility.Vector2iVector(np.zeros((0,2), np.int32))
        ls.colors = o3d.utility.Vector3dVector(np.zeros((0,3), np.float32))
        return ls
    P = np.asarray(points, np.float32)
    L = np.stack([np.arange(len(P)-1), np.arange(1,len(P))], axis=1).astype(np.int32)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(P)
    ls.lines  = o3d.utility.Vector2iVector(L)
    ls.colors = o3d.utility.Vector3dVector(np.tile(np.asarray(color,np.float32),(L.shape[0],1)))
    return ls

# Default COCO-17-ish skeleton edges (robust if fewer joints: edges with idx>=K drop automatically)
DEFAULT_BONES = [
    (5,6), (5,7), (7,9), (6,8), (8,10),    # shoulders->elbows->wrists
    (11,12), (11,13), (13,15), (12,14), (14,16),  # hips->knees->ankles
    (5,11), (6,12)  # torso links
]

class HyperHackerView:
    """
    Black bg, neon CSI cloud (isometric), AP anchors/objects, animated persons, trails,
    and **3D skeletons** per person.
    Keys: R reset, +/− point size.
    """
    def __init__(self, point_size=6.0, neon=(0.0,1.0,0.3), bones=DEFAULT_BONES):
        # CSI cloud
        self._pcd = o3d.geometry.PointCloud()
        self._pcd.points = o3d.utility.Vector3dVector(np.array([[0,0,0]], dtype=np.float32))
        self._pcd.colors = o3d.utility.Vector3dVector(np.array([neon], dtype=np.float32))
        self._neon = np.asarray(neon, dtype=np.float32)
        self._pt_size = float(point_size)

        # Objects (wire)
        self._objects = [
            _cube_wire(center=(-0.8,-0.2,0.05), size=0.25),
            _cube_wire(center=( 0.7, 0.0,0.10), size=0.35),
            _cube_wire(center=( 0.0, 0.7,0.08), size=0.20),
        ]

        # Multi-AP anchors (mock)
        self._ap_pos = np.array([[-1.0,-1.0,0.2],[1.05,-1.05,0.2],[0.0,1.1,0.2]], dtype=np.float32)
        self._ap_spheres = [o3d.geometry.TriangleMesh.create_sphere(radius=0.04, resolution=8) for _ in range(len(self._ap_pos))]
        for i,m in enumerate(self._ap_spheres):
            m.compute_vertex_normals(); m.paint_uniform_color([0.0,0.6,1.0]); m.translate(self._ap_pos[i].tolist(), relative=False)

        # People spheres + trails (keep, can be used as head marker)
        self._people = {}      # pid -> {'mesh':Mesh, 'trail':LineSet, 'trail_pts':list[np(3)], 'color':(3,), 'last':ts}
        self._trail_len = 60

        # Skeletons per PID
        self._bones = bones
        self._skel = {}        # pid -> {'joints':TriangleMesh, 'bones':LineSet, 'last':ts, 'color':(3,)}

        # threading flags
        self._lock = threading.Lock()
        self._upd_pcd = False
        self._upd_people = False
        self._upd_skel = False
        self._need_fit = True

    # ---------- CSI cloud ----------
    def update_from_csi(self, amp: np.ndarray):
        amp = np.asarray(amp)
        if amp.ndim == 1:
            n = amp.size; H = int(np.floor(np.sqrt(n))) or 1; W = int(np.ceil(n/H))
            pad = H*W - n
            if pad>0: amp = np.pad(amp, (0,pad), mode='edge')
            amp = amp.reshape(H, W)
        H, W = amp.shape
        x = (np.arange(H, dtype=np.float32)[:,None] * np.ones(W, dtype=np.float32)[None,:]).ravel()
        y = (np.arange(W, dtype=np.float32)[None,:] * np.ones(H, dtype=np.float32)[:,None]).ravel()
        z = amp.astype(np.float32).ravel()
        if x.size==0: return
        pts = np.stack([x,y,z], axis=-1)
        pts[:,0] = (pts[:,0]/max(1.0,float(np.max(pts[:,0]))))*2-1
        pts[:,1] = (pts[:,1]/max(1.0,float(np.max(pts[:,1]))))*2-1
        rng = float(np.ptp(pts[:,2])); den = 1e-6 if rng==0 else rng
        pts[:,2] = (pts[:,2]-float(np.min(pts[:,2])))/den
        cols = np.tile(self._neon.reshape(1,3),(pts.shape[0],1))
        with self._lock:
            self._pcd.points = o3d.utility.Vector3dVector(pts)
            self._pcd.colors = o3d.utility.Vector3dVector(cols)
            self._upd_pcd = True

    # ---------- People spheres/trails (optional eye candy) ----------
    def set_person(self, pid:int, score:float, pos_xyz:np.ndarray):
        score = float(np.clip(score,0.0,1.0))
        pos = np.asarray(pos_xyz, np.float32)
        with self._lock:
            entry = self._people.get(pid)
            if entry is None:
                mesh = o3d.geometry.TriangleMesh.create_sphere(radius=0.08, resolution=16)
                mesh.compute_vertex_normals()
                color = (0.2 + 0.8*score)*self._neon
                mesh.paint_uniform_color(color.tolist())
                mesh.translate(pos.tolist(), relative=False)
                trail_pts = [pos.copy()]
                trail = _lines_from_points(trail_pts, color=(0.0,1.0,0.4))
                self._people[pid] = {'mesh':mesh,'trail':trail,'trail_pts':trail_pts,'color':color,'last':time.time()}
                self._upd_people = True; self._need_fit = True
            else:
                base = o3d.geometry.TriangleMesh.create_sphere(radius=0.06+0.10*score, resolution=16)
                base.compute_vertex_normals()
                entry['mesh'].vertices = base.vertices; entry['mesh'].triangles = base.triangles
                entry['mesh'].compute_vertex_normals()
                entry['color'] = (0.2 + 0.8*score)*self._neon
                entry['mesh'].paint_uniform_color(entry['color'].tolist())
                entry['mesh'].translate((pos - np.asarray(entry['mesh'].get_center())).tolist(), relative=True)
                entry['trail_pts'].append(pos.copy())
                if len(entry['trail_pts']) > self._trail_len: entry['trail_pts'] = entry['trail_pts'][-self._trail_len:]
                newtrail = _lines_from_points(entry['trail_pts'], color=(0.0,1.0,0.4))
                entry['trail'].points = newtrail.points; entry['trail'].lines = newtrail.lines; entry['trail'].colors = newtrail.colors
                entry['last'] = time.time()
                self._upd_people = True

    # ---------- Skeletons (NEW) ----------
    def set_skeleton(self, pid:int, joints_xyz:np.ndarray, score:float|None=None):
        """
        joints_xyz: (K,3) float32 in viewer coordinates (roughly [-1,1] XY, 0..1 Z).
        score: optional confidence (0..1) → tints color.
        """
        J = np.asarray(joints_xyz, np.float32)
        color = (0.3 + 0.7*(0.0 if score is None else float(np.clip(score,0.0,1.0)))) * self._neon
        with self._lock:
            ent = self._skel.get(pid)
            if ent is None:
                # joints as small spheres: one merged mesh by sampling points (cheap) or just use a small point cloud
                # Here we use small spheres merged (simple & readable).
                joint_meshes = []
                for j in J:
                    s = o3d.geometry.TriangleMesh.create_sphere(radius=0.035, resolution=10)
                    s.compute_vertex_normals(); s.paint_uniform_color(color.tolist()); s.translate(j.tolist(), relative=False)
                    joint_meshes.append(s)
                # bones
                valid_bones = [(a,b) for (a,b) in self._bones if a < len(J) and b < len(J)]
                if len(J)==0: P = np.zeros((1,3),np.float32)
                else: P = J
                L = np.array(valid_bones, dtype=np.int32) if valid_bones else np.zeros((0,2),np.int32)
                bone_ls = o3d.geometry.LineSet()
                bone_ls.points = o3d.utility.Vector3dVector(P)
                bone_ls.lines  = o3d.utility.Vector2iVector(L)
                bone_ls.colors = o3d.utility.Vector3dVector(np.tile(color.reshape(1,3),(L.shape[0],1)))

                self._skel[pid] = {'joints':joint_meshes, 'bones':bone_ls, 'last':time.time(), 'color':color}
                self._upd_skel = True; self._need_fit = True
            else:
                # update joint spheres and bone lines
                jm = ent['joints']
                # add/remove spheres if K changed
                if len(jm) < len(J):
                    for _ in range(len(J)-len(jm)):
                        s = o3d.geometry.TriangleMesh.create_sphere(radius=0.035, resolution=10)
                        s.compute_vertex_normals()
                        jm.append(s)
                elif len(jm) > len(J):
                    del jm[len(J):]
                for k,j in enumerate(J):
                    base = o3d.geometry.TriangleMesh.create_sphere(radius=0.035, resolution=10)
                    base.compute_vertex_normals()
                    jm[k].vertices = base.vertices; jm[k].triangles = base.triangles
                    jm[k].compute_vertex_normals()
                    jm[k].paint_uniform_color(color.tolist())
                    jm[k].translate((j - np.asarray(jm[k].get_center())).tolist(), relative=True)

                valid_bones = [(a,b) for (a,b) in self._bones if a < len(J) and b < len(J)]
                L = np.array(valid_bones, dtype=np.int32) if valid_bones else np.zeros((0,2),np.int32)
                ent['bones'].points = o3d.utility.Vector3dVector(J)
                ent['bones'].lines  = o3d.utility.Vector2iVector(L)
                ent['bones'].colors = o3d.utility.Vector3dVector(np.tile(color.reshape(1,3),(L.shape[0],1)))

                ent['last'] = time.time(); ent['color'] = color
                self._upd_skel = True

    def fade_people(self, ttl=2.5):
        now = time.time()
        with self._lock:
            # spheres/trails
            kill_p = [pid for pid,e in self._people.items() if (now - e['last']) > ttl]
            for pid in kill_p: self._people.pop(pid, None)
            # skeletons
            kill_s = [pid for pid,e in self._skel.items() if (now - e['last']) > ttl]
            for pid in kill_s: self._skel.pop(pid, None)
            if kill_p: self._upd_people = True
            if kill_s: self._upd_skel = True

    # ---------- Loop ----------
    def run(self, window_name="WiFi CSI — Hyper Hacker View (Skeletons)"):
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name=window_name, width=1280, height=800, visible=True)
        vis.add_geometry(self._pcd)
        for m in self._ap_spheres: vis.add_geometry(m)
        for obj in self._objects:   vis.add_geometry(obj)

        opt = vis.get_render_option()
        opt.background_color = np.array([0,0,0], dtype=np.float32)
        opt.point_size = self._pt_size
        opt.show_coordinate_frame = False

        def reset_cb(v): v.reset_view_point(True); return False
        def plus_cb(v):  o=v.get_render_option(); o.point_size=min(22.0,o.point_size+1.0); return False
        def minus_cb(v): o=v.get_render_option(); o.point_size=max(1.0,o.point_size-1.0); return False
        vis.register_key_callback(ord('R'), reset_cb)
        vis.register_key_callback(ord('+'), plus_cb)
        vis.register_key_callback(ord('='), plus_cb)
        vis.register_key_callback(ord('-'), minus_cb)

        ctr = vis.get_view_control()
        iso = [-1/np.sqrt(3), -1/np.sqrt(3), -1/np.sqrt(3)]
        ctr.set_front(iso); ctr.set_up([0,1,0]); ctr.set_lookat([0,0,0]); ctr.set_zoom(0.85)

        added = set()
        while True:
            with self._lock:
                if self._upd_pcd:
                    vis.update_geometry(self._pcd); self._upd_pcd = False
                if self._need_fit:
                    bbox = self._pcd.get_axis_aligned_bounding_box()
                    ctr.set_lookat(bbox.get_center().tolist()); ctr.set_front(iso); ctr.set_up([0,1,0]); ctr.set_zoom(0.8)
                    self._need_fit = False

                # attach/update persons
                for pid, e in self._people.items():
                    if ('mesh',pid) not in added: vis.add_geometry(e['mesh']); added.add(('mesh',pid))
                    if ('trail',pid) not in added: vis.add_geometry(e['trail']); added.add(('trail',pid))
                    vis.update_geometry(e['mesh']); vis.update_geometry(e['trail'])

                # attach/update skeletons
                for pid, e in self._skel.items():
                    # joint spheres
                    for j_idx, jm in enumerate(e['joints']):
                        key = ('skj', pid, j_idx)
                        if key not in added:
                            vis.add_geometry(jm); added.add(key)
                        vis.update_geometry(jm)
                    # bone lines
                    key_lines = ('skl', pid)
                    if key_lines not in added:
                        vis.add_geometry(e['bones']); added.add(key_lines)
                    vis.update_geometry(e['bones'])

            vis.poll_events(); vis.update_renderer()
            time.sleep(0.02)
