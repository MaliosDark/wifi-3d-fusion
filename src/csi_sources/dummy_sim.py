# src/csi_sources/dummy_sim.py
import time, numpy as np

class DummySimSource:
    """
    Yields synthetic complex CSI vectors at ~fps. Matches (ts, vec) API.
    """
    def __init__(self, D=128, fps=20.0, persons=3):
        self.D = int(D); self.dt = 1.0/float(fps)
        rng = np.random.default_rng
        self.bias = [(rng().uniform(-0.5,0.5), rng().uniform(-0.5,0.5)) for _ in range(persons)]
        self.persons = persons
        self.t = 0
        self.cur = 0
        self.switch_every = int(3*fps)  # cambia de persona cada ~3s

    def frames(self):
        while True:
            if self.t % self.switch_every == 0:
                self.cur = (self.cur + 1) % self.persons
            fb, pb = self.bias[self.cur]
            x = np.linspace(0,1,self.D, dtype=np.float32)
            # amplitud y fase “raw”
            amp = (1.0 + 0.2*fb) * (1.0 + 0.03*np.sin(2*np.pi*(self.t/100.0)*(1.0+0.2*fb)))
            amp = amp + 0.05*np.sin(2*np.pi*x*(1.0+0.1*fb))
            amp = amp + np.random.normal(0, 0.01, size=self.D)
            phase = 2*np.pi*(pb*x) + 0.3*np.sin(2*np.pi*(self.t/80.0)*(1.5+pb)) + np.random.normal(0,0.02,size=self.D)
            vec = (amp*np.cos(phase)) + 1j*(amp*np.sin(phase))
            ts = time.time()
            yield ts, vec.astype(np.complex64)
            self.t += 1
            time.sleep(self.dt)
