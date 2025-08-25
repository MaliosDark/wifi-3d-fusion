import pyvista as pv
import numpy as np

class Gaussian3DViewer:
    def __init__(self):
        self.plotter = pv.Plotter()
        self.actors = {}
        self.gaussian_size = 0.05

    def _gaussian_blob(self, center, n_points=300):
        pts = center + self.gaussian_size * np.random.randn(n_points, 3)
        return pts

    def update_person(self, pid, keypoints3d, edges=None):
        # remove if already exists
        if pid in self.actors:
            for act in self.actors[pid]:
                self.plotter.remove_actor(act)

        actors = []
        # --- Gaussian blobs ---
        all_points = []
        for k in keypoints3d:
            all_points.append(self._gaussian_blob(k))
        all_points = np.vstack(all_points)

        cloud_actor = self.plotter.add_mesh(
            pv.PolyData(all_points),
            render_points_as_spheres=True,
            point_size=12,
            color="cyan",
            opacity=0.4
        )
        actors.append(cloud_actor)

        # --- Skeleton lines ---
        if edges is None:
            # COCO skeleton simple mock (pares de joints)
            edges = [
                (0,1),(1,2),(2,3),(3,4),
                (1,5),(5,6),(6,7),
                (1,8),(8,9),(9,10),
                (8,12),(12,13),(13,14)
            ]

        for i,j in edges:
            if i < len(keypoints3d) and j < len(keypoints3d):
                line = pv.Line(keypoints3d[i], keypoints3d[j])
                line_actor = self.plotter.add_mesh(line, color="white", line_width=3)
                actors.append(line_actor)

        self.actors[pid] = actors

    def show(self):
        self.plotter.show(interactive=True)

# --- DEMO ---
viewer = Gaussian3DViewer()
for pid in range(3):
    fake_keypoints = np.random.rand(17,3)
    viewer.update_person(pid, fake_keypoints)
viewer.show()
