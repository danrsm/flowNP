''' Implementation of FlowNP model (FNP).'''


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
from attrdictionary import AttrDict

from models.modules import build_mlp
from flow_matching.solver import ODESolver
def comp_posenc(dim_posenc, pos):

    shp = pos.shape
    omega = torch.arange(dim_posenc//2, dtype=torch.float).to(pos.device)
    omega = torch.pi * 2**(omega-2)
    out = pos[:, :, :, None] * omega[None, None, None, :]   
    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)
    emb = torch.concatenate([emb_sin, emb_cos], axis=-1)  # (M, D)
    return emb.reshape(shp[0], shp[1], shp[-1]*dim_posenc)

class FNP(nn.Module):
    def __init__(
        self,
        dim_x,
        dim_y,
        dim_posenc,
        d_model,
        emb_depth,
        dim_feedforward,
        nhead,
        dropout,
        num_layers,
        timesteps=100,
    ):
        super(FNP, self).__init__()      
        
        self.timesteps = timesteps
        self.fake_output_scale = 0.001  # not really used, only for backward compatiability 
        self.dim_posenc = dim_posenc

        self.predictor = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, dim_y)
        )

        self.embedder = build_mlp((dim_x+1)*dim_posenc + dim_y, d_model, d_model, emb_depth)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        def encode(xc, yc, xt, yt):
            x_y_ctx = torch.cat((xc, yc), dim=-1)
            x_y_tar = torch.cat((xt, yt), dim=-1)
            inp = torch.cat((x_y_ctx, x_y_tar), dim=1)
        
            num_ctx, num_tar = xc.shape[1], xt.shape[1]
            num_all = num_ctx + num_tar
            device = xc.device
            mask = torch.zeros(num_all, num_all, device=device)
        
            embeddings = self.embedder(inp)
            encoded = self.encoder(embeddings, mask=mask)[:, -num_tar:]
            out = self.predictor(encoded)
            return out
        self.encode = encode

        class Predict_Velocity(nn.Module):
            def forward(self, x: torch.Tensor, t: torch.Tensor, batch=None):
                yt = x.reshape(batch.xt.shape[0], batch.xt.shape[1], -1)
                if t.dim() == 0:
                    t = t.repeat((yt.shape[0], yt.shape[1], 1)).to(x.device)
                xc = torch.cat((batch.xc, torch.ones(
                    list(batch.xc.shape[:-1])+[1]).to(x.device)), dim=-1)
                xt = torch.cat((batch.xt, t), dim=-1)
                pred = encode(xc=comp_posenc(dim_posenc, xc),
                              xt=comp_posenc(dim_posenc, xt),
                              yc=batch.yc, yt=yt)
                p = pred.reshape(x.shape)
                return p
            
        self.predict_velocity = Predict_Velocity()
        self.solver = ODESolver(velocity_model=self.predict_velocity)

 
    def forward(self, batch, num_samples=None, reduce_ll=True):
        outs = AttrDict()
        if self.training:
            y0 = torch.randn_like(batch.yt).to(batch.yt.device)
            t = torch.rand(size=(y0.shape[0], y0.shape[1], 1)).to(y0.device)
            yt = t*batch.yt + (1-t)*y0
            
            pred = self.predict_velocity(yt, t, batch)
            outs.loss = nn.MSELoss()(pred, batch.yt-y0)
        else:
            y1 = batch.yt
            gaussian_log_density = MultivariateNormal(
                torch.zeros(y1.shape[2], device=y1.device), 
                torch.eye(y1.shape[2], device=y1.device)).log_prob
            log_p_y1 = self.solver.compute_likelihood(
                x_1=y1.reshape((-1, y1.shape[-1])), method='midpoint', step_size=1/self.timesteps, 
                exact_divergence=False, log_p0=gaussian_log_density, batch=batch)[1].reshape(
                    (y1.shape[0], y1.shape[1]))
            
            if reduce_ll:
                outs.tar_ll = log_p_y1.mean()
            else:
                outs.tar_ll = log_p_y1
        
        return outs

    def predict(self, xc, yc, xt, num_samples=50):
        
        batch_size = xc.shape[0]
        num_target = xt.shape[1] 
        
        xc = xc.repeat((num_samples, 1, 1))
        yc = yc.repeat((num_samples, 1, 1))
        xt = torch.cat((xc, xt.repeat((num_samples, 1, 1))), dim=1)
        yt = torch.randn((num_samples*batch_size, xt.shape[1], yc.shape[2])).to(xt.device)
        xct = torch.cat((xc, torch.ones((xc.shape[0], xc.shape[1], 1)).to(xc.device)), dim=-1)
        T = self.timesteps
        for t in range(T):
            tt = torch.tensor(t/T).repeat((yt.shape[0], yt.shape[1], 1)).to(yt.device)
            xtt = torch.cat((xt, tt), dim=-1)
            pred = self.encode(xc=comp_posenc(self.dim_posenc, xct),
                        xt=comp_posenc(self.dim_posenc, xtt),
                        yc=yc, yt=yt)
            alpha = 1+t/T*(1-t/T)
            sigma = 0.2*(t/T*(1-t/T))**0.5
            yt += (alpha*pred + sigma*torch.randn_like(yt).to(yt.device))/T
        
        return Normal(yt[:, xc.shape[1]:].reshape(num_samples, batch_size, num_target, yc.shape[2]), self.fake_output_scale)
