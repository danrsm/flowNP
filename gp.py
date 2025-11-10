import os
import os.path as osp
from sys import argv
import argparse
import yaml
import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from attrdictionary import AttrDict
from tqdm import tqdm
import time

from data.gp import *
from utils.misc import load_module
from utils.paths import results_path, evalsets_path
from utils.log import get_logger, RunningAverage

def eval_arg(arg):
    # convert a string argument to int, float, bool, or str
    if arg.lower() in ['true', 'false']:
        return arg.lower() == 'true'
    try:
        return int(arg)
    except ValueError:
        pass
    try:
        return float(arg)
    except ValueError:
        pass
    return arg

def main():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--mode', default='train', choices=['train', 'eval', 'eval_all_metrics', 'plot'])
    parser.add_argument('--expname', type=str, default='default')
    parser.add_argument('--expid', type=str, default="0")
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu). If None, will use cuda if available.')

    # Data
    parser.add_argument('--max_num_points', type=int, default=50)
    parser.add_argument('--kernel', type=str, default='rbf')
    parser.add_argument('--obs_noise', type=float, default='0.')
        
    # Model
    parser.add_argument('--model', type=str, default="tnpd")
    parser.add_argument('--modelconfig', type=str, default=None, help='Specifies the model configuration file')
    parser.add_argument('--model_args', type=str, default=None, help='Additional model arguments as a comma separated list of key=value pairs, e.g. key1=val1,key2=val2')
    
    # Train
    parser.add_argument('--train_seed', type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=16)
    parser.add_argument('--train_num_samples', type=int, default=4)
    parser.add_argument('--train_num_bs', type=int, default=10)
    parser.add_argument('--train_ctx10', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=10000)
    parser.add_argument('--save_freq', type=int, default=1000)

    # Eval
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--eval_num_batches', type=int, default=100)
    parser.add_argument('--eval_batch_size', type=int, default=128)
    parser.add_argument('--eval_num_samples', type=int, default=50)
    parser.add_argument('--eval_ctx10', type=bool, default=False)
    parser.add_argument('--eval_logfile', type=str, default=None)
    parser.add_argument('--timesteps', type=int, default=0)
    parser.add_argument('--ckptfile', type=str, default=None)

    # Plot
    parser.add_argument('--plot_seed', type=int, default=0)
    parser.add_argument('--plot_batch_size', type=int, default=16)
    parser.add_argument('--plot_num_samples', type=int, default=30)
    parser.add_argument('--plot_num_ctx', type=int, default=30)
    parser.add_argument('--plot_num_tar', type=int, default=10)
    parser.add_argument('--start_time', type=str, default=None)

    # OOD settings
    parser.add_argument('--eval_kernel', type=str, default=None) # defaults to train kernel
    parser.add_argument('--eval_obs_noise', type=float, default=None) # defaults to train obs_noise
    parser.add_argument('--eval_filename', type=str, default=None)
    
    args = parser.parse_args()

    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")

    if args.eval_kernel is None:
        args.eval_kernel = args.kernel
    if args.eval_obs_noise is None:
        args.eval_obs_noise = args.obs_noise
    
    if args.expid is not None:
        args.root = osp.join(results_path, 'gp', args.expname, args.kernel, args.model, args.expid)
    else:
        args.root = osp.join(results_path, 'gp', args.expname, args.kernel, args.model)

    model_cls = getattr(load_module(f'models/{args.model}.py'), args.model.upper())
    modelconfig = f'configs/gp/{args.model}.yaml' if args.modelconfig is None else args.modelconfig
    with open(modelconfig, 'r') as f:
        args.model_config = yaml.safe_load(f)
    
    if args.model == 'gtgp':
        args.model_config = {'kernel': args.kernel}
    
    # if "model_args is given in command line, update model_config with all the given args (given as a list of key value pairs)"
    if args.model_args is not None:
        model_args = args.model_args.split(',')
        for arg in model_args:
            key, value = arg.split('=')
            args.model_config[key] = eval_arg(value) # use eval to convert string to appropriate type
        
    model = model_cls(**args.model_config)
    model.to(args.device)

    if args.mode == 'train':
        train(args, model)
    elif args.mode == 'eval':
        eval(args, model)
    elif args.mode == 'plot':
        plot(args, model)

def train(args, model):
    if osp.exists(args.root + '/ckpt.tar'):
        if args.resume is None:
            raise FileExistsError(args.root)
    else:
        os.makedirs(args.root, exist_ok=True)

    with open(osp.join(args.root, f'args_{time.strftime("%Y%m%d-%H%M")}.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

    path, filename = get_eval_path(args)
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)

    torch.manual_seed(args.train_seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.train_seed)

    if args.kernel == 'rbf':
        kernel = RBFKernel()
    elif args.kernel == 'fixedrbf':
        kernel = FixedRBFKernel()
    elif args.kernel == 'matern':
        kernel = Matern52Kernel()
    elif args.kernel == 'fixedmatern':
        kernel = FixedMatern52Kernel()
    elif args.kernel == 'periodic':
        kernel = PeriodicKernel()
    else:
        raise ValueError(f'Invalid kernel {args.kernel}')

    sampler = GPSampler(kernel, ctx10=args.train_ctx10)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.num_steps)

    if args.resume:
        ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'), weights_only=False)
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_step = ckpt.step
    else:
        logfilename = os.path.join(args.root,
                f'train_{time.strftime("%Y%m%d-%H%M")}.log')
        start_step = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not args.resume:
        logger.info(f"Experiment: {args.expname}-{args.model}-{args.expid}")
        cmdline = ' '.join(argv)
        logger.info(f"Command line: {cmdline}")
        logger.info(f'Total number of parameters: {sum(p.numel() for p in model.parameters())}\n')

    for step in range(start_step, args.num_steps+1):
        model.train()
        optimizer.zero_grad()
        batch = sampler.sample(
            batch_size=args.train_batch_size,
            max_num_points=args.max_num_points,
            device=args.device)
        if args.obs_noise > 0.:
                batch['y'] += torch.randn_like(batch['y'])*args.obs_noise
                batch.yc = batch.y[:,:batch.xc.shape[-2]]
                batch.yt = batch.y[:,batch.xc.shape[-2]:]
        

        outs = model(batch, num_samples=args.train_num_samples)
        
        outs.loss.backward()
        optimizer.step()
        scheduler.step()

        for key, val in outs.items():
            ravg.update(key, val)

        if step % args.print_freq == 0:
            line = f'{args.expname}:{args.model}:{args.expid} step {step} '
            line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
            line += f"[train_loss] "
            line += ravg.info()
            logger.info(line)

            if step % args.eval_freq == 0:
                line = eval(args, model, step)
                logger.info(line + '\n')

            ravg.reset()

        if step % args.save_freq == 0 or step == args.num_steps:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.step = step + 1
            torch.save(ckpt, os.path.join(args.root, 'ckpt.tar'))

    args.mode = 'eval'
    eval(args, model, step)

def get_eval_path(args):
    path = osp.join(evalsets_path, 'gp')
    if args.eval_filename is not None:
        filename = args.eval_filename
    else:
        filename = f'{args.eval_kernel}-seed{args.eval_seed}' 
        if args.eval_ctx10:
            filename += f'-ctx10'
    filename += '.tar'
    return path, filename

def gen_evalset(args):
    if args.eval_kernel == 'rbf':
        kernel = RBFKernel()
    elif args.kernel == 'fixedrbf':
        kernel = FixedRBFKernel()
    elif args.eval_kernel == 'matern':
        kernel = Matern52Kernel()
    elif args.eval_kernel == 'fixedmatern':
        kernel = FixedMatern52Kernel()
    elif args.eval_kernel == 'periodic':
        kernel = PeriodicKernel()
    elif args.eval_kernel == 'gpdf':
        kernel = 'gpdf'
    elif args.eval_kernel == 'triangle':
        kernel = 'triangle'
    elif args.eval_kernel == 'step':
        kernel = 'step'
    else:
        raise ValueError(f'Invalid kernel {args.eval_kernel}')
    print(f"Generating Evaluation Sets with {args.eval_kernel} kernel")

    sampler = GPSampler(kernel, ctx10=args.eval_ctx10, seed=args.eval_seed)
    batches = []
    for i in tqdm(range(args.eval_num_batches), ascii=True):
        batches.append(sampler.sample(
            batch_size=args.eval_batch_size,
            max_num_points=args.max_num_points,
            device=args.device))

    torch.manual_seed(time.time())
    if args.device == 'cuda':
        torch.cuda.manual_seed(time.time())

    path, filename = get_eval_path(args)
    if not osp.isdir(path):
        os.makedirs(path)
    torch.save(batches, osp.join(path, filename))

def eval(args, model, step=None):
    # eval a trained model on log-likelihood
    if args.mode == 'eval':
        if args.model != 'gtgp':
            if args.ckptfile:
                ckpt = torch.load(args.ckptfile, map_location=args.device, weights_only=False)
            else:    
                ckpt = torch.load(os.path.join(args.root, 'ckpt.tar'), map_location=args.device, weights_only=False)
            model.load_state_dict(ckpt.model)
            if args.timesteps>0:
                model.timesteps = args.timesteps
        if args.eval_logfile is None:
            eval_logfile = f'eval_{args.eval_kernel}'
            eval_logfile += '.log'
        else:
            eval_logfile = args.eval_logfile
        os.makedirs(args.root, exist_ok=True)
        filename = os.path.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    
        
    else:
        logger = None


    path, filename = get_eval_path(args)
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)
    eval_batches = torch.load(osp.join(path, filename), weights_only=False)
    
    line = f'Evaluating on {path} {filename} with obs_noise {args.eval_obs_noise}'+'\n'

    if args.mode == "eval":
        torch.manual_seed(args.eval_seed)
        if args.device == 'cuda':
            torch.cuda.manual_seed(args.eval_seed)

    ravg = RunningAverage()
    model.eval()
    with torch.no_grad():
        for batch in tqdm(eval_batches, ascii=True):
            for key, val in batch.items():
                batch[key] = val.to(args.device)
            if args.eval_obs_noise > 0.:
                batch['y'] += torch.randn_like(batch['y'])*args.eval_obs_noise
                batch.yc = batch.y[:,:batch.xc.shape[-2]]
                batch.yt = batch.y[:,batch.xc.shape[-2]:]
        
            outs = model(batch, args.eval_num_samples)
            
            outs.n_ctx = batch.xc.shape[-2]
            outs.n_tar = batch.xt.shape[-2]

            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    if args.device == 'cuda':
        torch.cuda.manual_seed(time.time())

    line += f'{args.expname}:{args.model}:{args.expid} {args.eval_kernel} '
    line += ravg.info()

    if logger is not None:
        logger.info(line)

    plot(args, model, eval_batches, step if step is not None else '')

    return line




#####################################

def plot(args, model, batch=None, suffix=''):
    # could be broken if called outside of eval

    if batch is None:
        
        path, filename = get_eval_path(args)
        if not osp.isfile(osp.join(path, filename)):
            print('generating evaluation sets...')
            gen_evalset(args)
        batch = torch.load(osp.join(path, filename), weights_only=False)
    
        if args.eval_seed is not None:
            torch.manual_seed(args.eval_seed)
            if args.device == 'cuda':
                torch.cuda.manual_seed(args.eval_seed)

    if args.mode == 'plot':
        model.eval()
        os.makedirs(args.root, exist_ok=True)
        
    xp = torch.linspace(-2, 2, 200).to(args.device)
    with torch.no_grad():
        if type(batch) is list:
            mu = torch.zeros(args.plot_num_samples, args.plot_batch_size, xp.shape[0], 1)
            sigma = torch.zeros_like(mu)
            for b in range(args.plot_batch_size):
                bb = b % len(batch)
                bi = b // len(batch)
                py = model.predict(batch[bb].xc[bi:bi+1], batch[bb].yc[bi:bi+1],
                        xp[None,:,None], num_samples=args.plot_num_samples)
                mu[:, b:b+1], sigma[:, b:b+1] = py.mean, py.scale
            mu, sigma = mu.squeeze(0), sigma.squeeze(0)
        else:
            py = model.predict(batch.xc, batch.yc,
                    xp[None,:,None].repeat(args.plot_batch_size, 1, 1),
                    num_samples=args.plot_num_samples)
            mu, sigma = py.mean.squeeze(0), py.scale.squeeze(0)
                     
    def tnp(x):
            return x.squeeze().cpu().data.numpy()

    def batch_item(item, index, ii):
        if type(batch) is list:
            return batch[index][item][ii]
        else:
            return batch[item][index]
    
    if args.plot_batch_size > 1:
        nrows = max(args.plot_batch_size//4, 1)
        ncols = min(4, args.plot_batch_size)
        fig, axes = plt.subplots(nrows, ncols,
                figsize=(5*ncols, 5*nrows))
        axes = axes.flatten()
    else:
        fig = plt.figure(figsize=(5, 5))
        axes = [plt.gca()]

    # multi sample
    if mu.dim() == 4:
        for i, ax in enumerate(axes):
            ib = i % len(batch)
            ii = i // len(batch)
            for s in range(mu.shape[0]):
                ax.plot(tnp(xp), tnp(mu[s][i]), color='steelblue',
                        alpha=max(0.5/args.plot_num_samples, 0.3))
                ax.fill_between(tnp(xp), tnp(mu[s][i])-tnp(sigma[s][i]),
                        tnp(mu[s][i])+tnp(sigma[s][i]),
                        color='skyblue',
                        alpha=max(0.2/args.plot_num_samples, 0.02),
                        linewidth=0.0)
            ax.scatter(tnp(batch_item('xc', ib, ii)), tnp(batch_item('yc', ib, ii)),
                    color='k', label='context', zorder=mu.shape[0]+1)
    else:
        for i, ax in enumerate(axes):
            ib = i % len(batch)
            ii = i // len(batch)
            ax.plot(tnp(xp), tnp(mu[i]), color='steelblue', alpha=0.5)
            ax.fill_between(tnp(xp), tnp(mu[i]-sigma[i]), tnp(mu[i]+sigma[i]),
                    color='skyblue', alpha=0.2, linewidth=0.0)
            ax.scatter(tnp(batch_item('xc', ib, ii)), tnp(batch_item('yc', ib, ii)),
                    color='k', label='context')
            ax.scatter(tnp(batch_item('xt', ib, ii)), tnp(batch_item('yt', ib, ii)),
                    color='orchid', label='target')
            ax.legend()

    plt.tight_layout()
    plt.savefig(osp.join(args.root, f'plot_{args.eval_seed}_{suffix}.png'))
    plt.show()
    if args.mode != 'plot':
        plt.close()



if __name__ == '__main__':
    main()
