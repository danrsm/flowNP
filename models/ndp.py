''' Implementation of Neural Diffusion Process using same architecture as TNP and FlowNP'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, MultivariateNormal
from attrdictionary import AttrDict

from models.modules import build_mlp
from flow_matching.solver import ODESolver
from flow_matching.path.scheduler import CondOTScheduler, LinearVPScheduler
from flow_matching.path import AffineProbPath


def comp_posenc(dim_posenc, pos):

    shp = pos.shape
    omega = torch.arange(dim_posenc//2, dtype=torch.float).to(pos.device)
    omega = torch.pi * 2**(omega-2)
    out = pos[:, :, :, None] * omega[None, None, None, :]
    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)
    emb = torch.concatenate([emb_sin, emb_cos], axis=-1)  # (M, D)
    return emb.reshape(shp[0], shp[1], shp[-1]*dim_posenc)

def compute_ddpm_x0_prediction_weights(t_current_1_indexed, betas_schedule, alphas_schedule, alphas_cumprod_schedule):
  # compute the weights for the DDPM sampling step

    t_idx_0_based = t_current_1_indexed - 1
    beta_t = betas_schedule[t_idx_0_based]
    alpha_t = alphas_schedule[t_idx_0_based]
    alphabar_t = alphas_cumprod_schedule[t_idx_0_based]

    if t_current_1_indexed == 1:
        alphabar_t_minus_1 = 1.0 
    else:
        alphabar_t_minus_1 = alphas_cumprod_schedule[t_idx_0_based - 1]
    one_minus_alphabar_t = 1.0 - alphabar_t
    if t_current_1_indexed == 1:
        weight_predicted_x0 = 1.0
        weight_current_xt = 0.0
        tilde_beta_t = beta_t * (1.0 - alphabar_t_minus_1) / (1.0 - alphabar_t + 1e-9)
    else:
        weight_predicted_x0 = (np.sqrt(alphabar_t_minus_1) * beta_t) / one_minus_alphabar_t
        weight_current_xt = (np.sqrt(alpha_t) * (1.0 - alphabar_t_minus_1)) / one_minus_alphabar_t
        tilde_beta_t = ((1.0 - alphabar_t_minus_1) / one_minus_alphabar_t) * beta_t
    weight_noise_term_std = np.sqrt(tilde_beta_t)
    
    return {
        'x0': weight_predicted_x0,
        'xt': weight_current_xt,
        'noise': weight_noise_term_std,
        'tilde_beta_t': tilde_beta_t
    }



class NDP(nn.Module):
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
        gscale=900.0, #900 for GP 10000 for EMNIST
    ):
        super(NDP, self).__init__()      
        
        self.timesteps = timesteps
        self.gscale = gscale
        self.fake_output_scale = 0.001
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
        path = AffineProbPath(scheduler=LinearVPScheduler())
        
        class Predict(nn.Module):
            def forward(self, x: torch.Tensor, t: torch.Tensor, extra=None):
                if t.dim() == 0:
                    t = t.repeat((extra.xt.shape[0], 1, 1)).to(x.device)
                
                yt = x.reshape(extra.xt.shape[0], extra.xt.shape[1], -1)
                tt = t.repeat((1, yt.shape[1], 1)).to(x.device)
                xt = torch.cat((extra.xt, tt), dim=-1)
                pred = encode(xc=comp_posenc(dim_posenc, xt[:, :0]),
                              xt=comp_posenc(dim_posenc, xt),
                              yc=yt[:, :0], yt=yt)
                
                return pred.reshape(x.shape)
        
        predict_step = Predict()
        
        class VelocityModel(nn.Module):
            def forward(self, x: torch.Tensor, t: torch.Tensor, **extras) -> torch.Tensor:
                x_1_prediction = predict_step(x, t, **extras)
                return path.target_to_velocity(x_1=x_1_prediction, x_t=x,
                                                t=torch.min(t, torch.tensor([0.995]).to(t.device)))

        self.predict_velocity = VelocityModel()
        self.path = path
        self.predict_step = predict_step
        self.solver = ODESolver(velocity_model=self.predict_velocity)
 

    def forward(self, batch, num_samples=None, reduce_ll=True):
        
        
        outs = AttrDict()
        extra_all = AttrDict()
        extra_all.xt = batch.x
        extra_all.yt = batch.y
        extra_all.xc = batch.xc[:,:0]
        extra_all.yc = batch.yc[:,:0]
            
        if self.training:
            y0 = torch.randn_like(extra_all.yt).to(extra_all.yt.device)
            t = torch.rand(size=(y0.shape[0],)).to(y0.device)
        
            # sample probability path
            path_sample = self.path.sample(t=t, x_0=y0, x_1=extra_all.yt)
            pred = self.predict_step(path_sample.x_t, path_sample.t[:, None, None], extra=extra_all)
            
            
            # denoising loss
            outs.loss = torch.pow(pred - path_sample.x_1, 2).mean()
        else:
            # evaluate logl by p(target|context) = p(target, context) / p(context)
            extra_context = AttrDict()
            extra_context.xt = batch.xc
            extra_context.yt = batch.yc
            extra_context.xc = batch.xc[:,:0]
            extra_context.yc = batch.yc[:,:0]
            
            y1_all = extra_all.yt
            gaussian_log_density_all = MultivariateNormal(
                torch.zeros(y1_all.shape[2], device=y1_all.device), 
                torch.eye(y1_all.shape[2], device=y1_all.device)).log_prob
            log_p_y1_all = self.solver.compute_likelihood(
                x_1=y1_all.reshape((-1, y1_all.shape[-1])), method='midpoint', step_size=1/self.timesteps, 
                exact_divergence=False, log_p0=gaussian_log_density_all, extra=extra_all)[1].reshape(
                    (y1_all.shape[0], y1_all.shape[1]))
            y1_context = extra_context.yt
            gaussian_log_density_context = MultivariateNormal(
                torch.zeros(y1_context.shape[2], device=y1_context.device), 
                torch.eye(y1_context.shape[2], device=y1_context.device)).log_prob
            log_p_y1_context = self.solver.compute_likelihood(
                x_1=y1_context.reshape((-1, y1_context.shape[-1])), method='midpoint', step_size=1/self.timesteps, 
                exact_divergence=False, log_p0=gaussian_log_density_context, extra=extra_context)[1].reshape(
                    (y1_context.shape[0], y1_context.shape[1]))

            if reduce_ll:
                outs.all_ll = log_p_y1_all.mean()
                outs.ctx_ll = log_p_y1_context.mean()
                outs.tar_ll = (log_p_y1_all.mean()*log_p_y1_all.shape[1] - log_p_y1_context.mean()*log_p_y1_context.shape[1])/ \
                                (log_p_y1_all.shape[1] - log_p_y1_context.shape[1])
            else:
                outs.all_ll = log_p_y1_all
                outs.ctx_ll = log_p_y1_context
                outs.tar_ll = log_p_y1_all - log_p_y1_context
        
        return outs

    def predict(self, xc, yc, xt, num_samples=50):

        # generate samples guided by context points

        batch_size = xc.shape[0]
        num_target = xt.shape[1] 
        xc = xc.repeat((num_samples, 1, 1))
        yc = yc.repeat((num_samples, 1, 1))
        xt = torch.cat((xc, xt.repeat((num_samples, 1, 1))), dim=1)
        T = self.timesteps
        betas = np.arange(T)/T
        alphas = 1. - betas
        alphas_cp = np.cumprod(alphas)
        yt = torch.randn((num_samples*batch_size, xt.shape[1], yc.shape[2])).to(xt.device)
        for t in range(T):
            tt = torch.tensor(t/T).repeat((yt.shape[0], yt.shape[1], 1)).to(yt.device)
            xtt = torch.cat((xt, tt), dim=-1)
            yt2 = yt.detach().requires_grad_(True)
            with torch.enable_grad():
                y1_hat = self.encode(xc=comp_posenc(self.dim_posenc, xtt[:, :0]),
                            xt=comp_posenc(self.dim_posenc, xtt),
                            yc=yt2[:, :0], yt=yt2)
            
                loss = torch.pow(y1_hat[:, :xc.shape[1]] - yc, 2).mean()
                grad = torch.autograd.grad(loss, yt2)[0]
                    
            w = compute_ddpm_x0_prediction_weights(T-t, betas, alphas, alphas_cp)
            yt = w['xt']*yt + w['x0']*y1_hat + w['noise']*torch.randn((num_samples*batch_size, xt.shape[1], yc.shape[2])).to(xt.device)
            yt -= self.gscale*grad  
            #yt = torch.max(torch.Tensor([-1.]).to(yt.device), torch.min(torch.Tensor([1.]).to(yt.device), yt))

        return Normal(yt[:, xc.shape[1]:].reshape(num_samples, batch_size, num_target, yc.shape[2]), self.fake_output_scale)
        
