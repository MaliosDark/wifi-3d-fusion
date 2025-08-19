# -*- coding: utf-8 -*-
"""code for ray marching and signal rendering
"""
import torch
import numpy as np
import torch.nn.functional as F
import scipy.constants as sc
from einops import rearrange, repeat


class Renderer():

    def __init__(self, networks_fn, **kwargs) -> None:
        """
        Parameters
        -----------------
        near : float. The near bound of the rays
        far : float. The far bound of the rays
        n_samples: int. num of samples per ray
        """

        ## Rendering parameters
        self.network_fn = networks_fn
        self.n_samples = kwargs['n_samples']
        self.near = kwargs['near']
        self.far = kwargs['far']


    def sample_points(self, rays_o, rays_d):
        """sample points along rays

        Parameters
        ----------
        rays_o : tensor. [n_rays, 3]. The origin of rays
        rays_d : tensor. [n_rays, 3]. The direction of rays

        Returns
        -------
        pts : tensor. [n_rays, n_samples, 3]. The sampled points along rays
        t_vals : tensor. [n_rays, n_samples]. The distance from origin to each sampled point
        """
        shape = list(rays_o.shape)
        shape[-1] = 1
        near, far = torch.full(shape, self.near), torch.full(shape, self.far)
        t_vals = torch.linspace(0., 1., steps=self.n_samples) * (far - near) + near  # scale t with near and far
        t_vals = t_vals.to(rays_o.device)
        pts = rays_o[...,None,:] + rays_d[...,None,:] * t_vals[...,:,None] # p = o + td, [n_rays, n_samples, 3]

        return pts, t_vals




class Renderer_spectrum(Renderer):
    """Renderer for spectrum (integral from single direction)
    """

    def __init__(self, networks_fn, **kwargs) -> None:
        """
        Parameters
        -----------------
        near : float. The near bound of the rays
        far : float. The far bound of the rays
        n_samples: int. num of samples per ray
        """
        super().__init__(networks_fn, **kwargs)



    def render_ss(self, tx, rays_o, rays_d):
        """render the signal strength of each ray

        Parameters
        ----------
        tx: tensor. [batchsize, 3]. The position of the transmitter
        rays_o : tensor. [batchsize, 3]. The origin of rays
        rays_d : tensor. [batchsize, 3]. The direction of rays
        """

        # sample points along rays
        pts, t_vals = self.sample_points(rays_o, rays_d)

        # Expand views and tx to match the shape of pts
        view = rays_d[:, None].expand(pts.shape)
        tx = tx[:, None].expand(pts.shape)

        # Run network and compute outputs
        raw = self.network_fn(pts, view, tx)    # [batchsize, n_samples, 4]
        receive_ss = self.raw2outputs(raw, t_vals, rays_d)  # [batchsize]

        return receive_ss



    def raw2outputs(self, raw, r_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values. (core part)

        Parameters
        ----------
        raw : [batchsize, n_samples, 4]. Prediction from model.
        r_vals : [batchsize, n_samples]. Integration distance.
        rays_d : [batchsize, 3]. Direction of each ray

        Return:
        ----------
        receive_signal : [batchsize]. abs(singal of each ray)
        """

        raw2alpha = lambda raw, dists: 1.-torch.exp(-raw*dists)
        # raw2phase = lambda raw, dists: torch.exp(1j*raw*dists)
        raw2phase = lambda raw, dists: raw*dists

        dists = r_vals[...,1:] - r_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).cuda().expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

        att_a, att_p, s_a, s_p = raw[...,0], raw[...,1], raw[...,2], raw[...,3]    # [N_rays, N_samples]
        att_p, s_p = torch.sigmoid(att_p)*np.pi*2, torch.sigmoid(s_p)*np.pi*2
        att_a, s_a = abs(F.leaky_relu(att_a)), abs(F.leaky_relu(s_a))
        # att_a, s_a = torch.sigmoid(att_a), torch.sigmoid(s_a)

        alpha = raw2alpha(att_a, dists)  # [N_rays, N_samples]
        phase = raw2phase(att_p, dists)

        att_i = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        path = torch.cat([r_vals[...,1:], torch.Tensor([1e10]).cuda().expand(r_vals[...,:1].shape)], -1)
        path_loss = 0.025 / path
        # phase_i = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), phase], -1), -1)[:, :-1]
        phase_i = torch.cumsum(torch.cat([torch.ones((alpha.shape[0], 1)).cuda(), phase], -1), -1)[:, :-1]
        phase_i = torch.exp(1j*phase_i)    # [N_rays, N_samples]


        receive_signal = torch.sum(s_a*torch.exp(1j*s_p)*att_i*phase_i*path_loss, -1)  # [N_rays]
        receive_signal = abs(receive_signal)

        return receive_signal




class Renderer_RSSI(Renderer):
    """Renderer for RSSI (integral from all directions)
    """

    def __init__(self, networks_fn, **kwargs) -> None:
        """
        Parameters
        -----------------
        near : float. The near bound of the rays
        far : float. The far bound of the rays
        n_samples: int. num of samples per ray
        """
        super().__init__(networks_fn, **kwargs)


    def render_rssi(self, tx, rays_o, rays_d):
        """render the RSSI for each gateway. To avoid OOM, we split the rays into chunks

        Parameters
        ----------
        tx: tensor. [batchsize, 3]. The position of the transmitter
        rays_o : tensor. [batchsize, 3]. The origin of rays
        rays_d : tensor. [batchsize, 9x36x3]. The direction of rays
        """

        batchsize, _ = tx.shape
        rays_d = torch.reshape(rays_d, (batchsize, -1, 3))    # [batchsize, 9x36, 3]
        chunks = 36
        chunks_num = 36 // chunks
        rays_o_chunk = rays_o.expand(chunks, -1, -1).permute(1,0,2) #[bs, cks, 3]
        tags_chunk = tx.expand(chunks, -1, -1).permute(1,0,2)        #[bs, cks, 3]
        recv_signal = torch.zeros(batchsize).cuda()
        for i in range(chunks_num):
            rays_d_chunk = rays_d[:,i*chunks:(i+1)*chunks, :]  # [bs, cks, 3]
            pts, t_vals = self.sample_points(rays_o_chunk, rays_d_chunk) # [bs, cks, pts, 3]
            views_chunk = rays_d_chunk[..., None, :].expand(pts.shape)  # [bs, cks, pts, 3]
            tx_chunk = tags_chunk[..., None, :].expand(pts.shape)  # [bs, cks, pts, 3]

            # Run network and compute outputs
            raw = self.network_fn(pts, views_chunk, tx_chunk)    # [batchsize, chunks, n_samples, 4]
            recv_signal_chunks = self.raw2outputs_signal(raw, t_vals, rays_d_chunk)  # [bs]
            recv_signal += recv_signal_chunks

        return recv_signal    # [batchsize,]


    def raw2outputs_signal(self, raw, r_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values.

        Parameters
        ----------
        raw : [batchsize, chunks,n_samples,  4]. Prediction from model.
        r_vals : [batchsize, chunks, n_samples]. Integration distance.
        rays_d : [batchsize,chunks, 3]. Direction of each ray

        Return:
        ----------
        receive_signal : [batchsize]. abs(singal of each ray)
        """
        wavelength = sc.c / 2.4e9
        # raw2alpha = lambda raw, dists: 1.-torch.exp(-raw*dists)
        # raw2phase = lambda raw, dists: torch.exp(1j*raw*dists)
        raw2phase = lambda raw, dists: raw + 2*np.pi*dists/wavelength
        raw2amp = lambda raw, dists: -raw*dists

        dists = r_vals[...,1:] - r_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).cuda().expand(dists[...,:1].shape)], -1)  # [batchsize, chunks, n_samples]
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)  # [batchsize,chunks, n_samples, 3].

        att_a, att_p, s_a, s_p = raw[...,0], raw[...,1], raw[...,2], raw[...,3]    # [batchsize,chunks, N_samples]
        att_p, s_p = torch.sigmoid(att_p)*np.pi*2-np.pi, torch.sigmoid(s_p)*np.pi*2-np.pi
        att_a, s_a = abs(F.leaky_relu(att_a)), abs(F.leaky_relu(s_a))

        amp = raw2amp(att_a, dists)  # [batchsize,chunks, N_samples]
        phase = raw2phase(att_p, dists)

        # att_i = torch.cumprod(torch.cat([torch.ones((al_shape[:-1], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        # phase_i = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), phase], -1), -1)[:, :-1]
        amp_i = torch.exp(torch.cumsum(amp, -1))            # [batchsize,chunks, N_samples]
        phase_i = torch.exp(1j*torch.cumsum(phase, -1))                # [batchsize,chunks, N_samples]

        recv_signal = torch.sum(s_a*torch.exp(1j*s_p)*amp_i*phase_i, -1)  # integral along line [batchsize,chunks]
        recv_signal = torch.sum(recv_signal, -1)   # integral along direction [batchsize,]

        return abs(recv_signal)




class Renderer_CSI(Renderer):
    """Renderer for CSI (integral from all directions)
    """

    def __init__(self, networks_fn, **kwargs) -> None:
        """
        Parameters
        -----------------
        near : float. The near bound of the rays
        far : float. The far bound of the rays
        n_samples: int. num of samples per ray
        """
        super().__init__(networks_fn, **kwargs)



    def render_csi(self, uplink, rays_o, rays_d):
        """render the RSSI for each gateway.

        Parameters
        ----------
        uplink: tensor. [batchsize, 52]. The uplink CSI (26 real + 26 imag)
        rays_o : tensor. [batchsize, 3]. The origin of rays
        rays_d : tensor. [batchsize, 9x36x3]. The direction of rays
        """

        rays_d = rearrange(rays_d, 'b (v d) -> b v d', d=3)    # [bs, 9x36, 3]
        batchsize, viewsize, _ = rays_d.shape
        rays_o = repeat(rays_o, 'b d -> b v d', v=viewsize) #[bs, 9x36, 3]
        uplink = repeat(uplink, 'b d -> b v d', v=viewsize)        #[bs, 9x36, 52]

        pts, t_vals = self.sample_points(rays_o, rays_d) # [bs, 9x36, pts, 3]
        views = repeat(rays_d, 'b v d -> b v p d', p=self.n_samples)  # [bs, 9x36, pts, 3]
        uplink = repeat(uplink, 'b v d -> b v p d', p=self.n_samples)  # [bs, cks, pts, 3]

        # Run network and compute outputs
        raw = self.network_fn(pts, views, uplink)    # [batchsize, 9x36, pts, 4]
        recv_signal = self.raw2outputs_signal(raw, t_vals, rays_d)  # [bs, 26]

        return recv_signal


    def raw2outputs_signal(self, raw, r_vals, rays_d):
        """Transforms model's predictions to semantically meaningful values.

        Parameters
        ----------
        raw : [batchsize, chunks,n_samples,  4]. Prediction from model.
        r_vals : [batchsize, chunks, n_samples]. Integration distance.
        rays_d : [batchsize,chunks, 3]. Direction of each ray

        Return:
        ----------
        receive_signal : [batchsize, 26]. OFDM singal of each ray
        """
        wavelength = sc.c / 2.4e9
        # raw2alpha = lambda raw, dists: 1.-torch.exp(-raw*dists)
        # raw2phase = lambda raw, dists: torch.exp(1j*raw*dists)
        raw2phase = lambda raw, dists: raw + 2*np.pi*dists/wavelength
        raw2amp = lambda raw, dists: -raw*dists

        dists = r_vals[...,1:] - r_vals[...,:-1]
        dists = torch.cat([dists, torch.Tensor([1e10]).cuda().expand(dists[...,:1].shape)], -1)  # [batchsize, chunks, n_samples]
        dists = dists * torch.norm(rays_d[...,None,:], dim=-1)  # [batchsize,chunks, n_samples, 3].

        att_a, att_p, s_a, s_p = raw[...,:26], raw[...,26:52], raw[...,52:78], raw[...,78:104]    # [batchsize,chunks, N_samples]
        att_p, s_p = torch.sigmoid(att_p)*np.pi*2-np.pi, torch.sigmoid(s_p)*np.pi*2-np.pi
        att_a, s_a = abs(F.leaky_relu(att_a)), abs(F.leaky_relu(s_a))

        dists = dists.unsqueeze(-1)

        amp = raw2amp(att_a, dists)  # [batchsize,chunks, N_samples, 26]
        phase = raw2phase(att_p, dists)

        # att_i = torch.cumprod(torch.cat([torch.ones((al_shape[:-1], 1)), 1.-alpha + 1e-10], -1), -1)[:, :-1]
        # phase_i = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)), phase], -1), -1)[:, :-1]
        amp_i = torch.exp(torch.cumsum(amp, -2))            # [batchsize,chunks, N_samples, 26]
        phase_i = torch.exp(1j*torch.cumsum(phase, -2))                # [batchsize,chunks, N_samples 26]

        recv_signal = torch.sum(s_a*torch.exp(1j*s_p)*amp_i*phase_i, -2)  # integral along line [batchsize,chunks,26]
        recv_signal = torch.sum(recv_signal, 1)   # integral along direction [batchsize, 26]

        return recv_signal




renderer_dict = {"spectrum": Renderer_spectrum, "rssi": Renderer_RSSI, "csi": Renderer_CSI}