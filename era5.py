
from email import parser
import torch.multiprocessing as mp

# Set the sharing strategy to 'file_system'
mp.set_sharing_strategy('file_system')

import os, sys
import os.path as osp
import argparse
import yaml
import random
import torch
import torch.nn as nn
import math
import time
import matplotlib.pyplot as plt
from attrdictionary import AttrDict
from tqdm import tqdm
from copy import deepcopy
import numpy as np

from utils.misc import load_module, logmeanexp
from utils.paths import results_path, evalsets_path
from utils.log import get_logger, RunningAverage
from data.era5 import ERA5Dataset

import utils.helper as era5_utils

import data.era5 as dataset
from utils.helper import plot_inference_2d  # Import the plot function
from data.era5 import era5_to_task, task_to_era5
from utils.helper import plot_inference_2d,plot_temperature


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval', 'plot'], default='train',
                        help='Specifies the mode in which the script should run')
    parser.add_argument('--expid', type=str, default='default', help='Identifier for the experiment')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='Flag to resume training from the last checkpoint')
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu). If None, will use cuda if available.')
    parser.add_argument('--model', type=str, default='cnp', help='Specifies the model to use')
    parser.add_argument('--modelconfig', type=str, default=None, help='Specifies the model configuration file')
    parser.add_argument('--place', type=str,  default="US", help='Place of evaluation data')
    
    # training parameters
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for the optimizer')
    parser.add_argument('--num_epochs', type=int, default=200, help='Number of epochs for training')
    parser.add_argument('--eval_freq', type=int, default=20, help='Frequency of evaluation during training (in epochs)')
    parser.add_argument('--save_freq', type=int, default=50, help='Frequency of saving checkpoints (in epochs)')
    parser.add_argument('--ckptfile', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=100, required=False, help='Batch size.')
    parser.add_argument('--num_samples', type=int, default=4, help='Number of samples in training')  # for NP ANP BNP BANP models
    parser.add_argument('--seed', type=int, required=False, help='Seed for randomness.')
    parser.add_argument('--max_ctx_points', type=int, default=None, help='Maximum number of context points')
    parser.add_argument('--obs_noise', type=float, default='0.')
    
    # eval parameters
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_logfile', type=str, default=None)
    parser.add_argument('--eval_seed', type=int, default=42, help='Seed for evaluation')
    parser.add_argument('--eval_num_samples', type=int, default=50, help='Number of samples for evaluation') # for NP ANP BNP BANP models
    parser.add_argument('--eval_obs_noise', type=float, default=None) # defaults to obs_noise

    # plot parameters
    parser.add_argument('--plot_batch_size', type=int, default=4, help='Batch size for plotting')
    parser.add_argument('--plot_random_samples', action='store_true', default=False, help='Flag to indicate random samples for plotting')
    parser.add_argument('--plot_num_samples', type=int, default=5, help='Number of samples for plotting')
        
    args = parser.parse_args()

    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")

    if args.eval_obs_noise is None:
        args.eval_obs_noise = args.obs_noise
    
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if args.device == 'cuda':
            torch.cuda.manual_seed(args.seed)

    model_cls = getattr(load_module(f'models/{args.model}.py'), args.model.upper())
    modelconfig = args.modelconfig if args.modelconfig is not None else f'configs/era5/{args.model}.yaml'
    with open(modelconfig, 'r') as f:
        config = yaml.safe_load(f)
    model = model_cls(**config).to(args.device)

    args.root = osp.join(results_path, 'era5', args.model, args.expid)

    print(f"Number of parameters: {era5_utils.count_parameters(model, print_table=False)}")

    if args.mode == 'train':
        train(args, model)
    elif args.mode == 'eval':
        eval(args, model)
    elif args.mode == 'plot':
        plot(args, model)


def train(args, model):
    if not osp.isdir(args.root):
        os.makedirs(args.root)

    with open(osp.join(args.root, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

    min_ctx_points = 2
    args.max_ctx_points = (41 * 41) - 1 if args.max_ctx_points is None else args.max_ctx_points

    train_ds = ERA5Dataset('train', min_ctx_points, args.max_ctx_points, place='US', normalize=True, circular=True)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    print("len(train_load)", len(train_loader))

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.num_epochs)

    if args.resume:
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar', weights_only=False))
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        logfilename = ckpt['logfilename']
        start_epoch = ckpt['epoch']
    else:
        logfilename = osp.join(args.root, 'train_{}.log'.format(time.strftime('%Y%m%d-%H%M')))
        start_epoch = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    for epoch in range(start_epoch, args.num_epochs + 1):
        model.train()
        running_loss = 0.0

        model.train()
        for b in tqdm(train_loader):


            batch = era5_to_task(b, device=device)
            if args.obs_noise > 0.:
                batch['y'] += torch.randn_like(batch['y'])*args.obs_noise
                batch.yc = batch.y[:,:batch.xc.shape[-2]]
                batch.yt = batch.y[:,batch.xc.shape[-2]:]
        
            optimizer.zero_grad()
            outs = model(batch, args.num_samples)
            
            loss = outs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            for key, val in outs.items():
                ravg.update(key, val)

        line = f'{args.model}:{args.expid} epoch {epoch}/{args.num_epochs} '
        line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
        line += ravg.info()
        logger.info(line)


        if epoch % args.eval_freq == 0 or epoch == args.num_epochs:
            logger.info(f"{eval(args, model,epoch)} \n")

        ravg.reset()

        if epoch % args.save_freq == 0 or epoch == args.num_epochs:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.epoch = epoch + 1
            torch.save(ckpt, osp.join(args.root, 'ckpt.tar'))
            logger.info(f"Checkpoint saved at epoch {epoch}")


def gen_evalset(args):
    # Set random seed for reproducibility
    torch.manual_seed(args.eval_seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.eval_seed)

    # Load the dataset
    eval_ds = ERA5Dataset(mode='valid')
    eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=args.eval_batch_size, shuffle=False, num_workers=4)

    # Process the dataset into batches
    batches = []
    i = 0
    for b in tqdm(eval_loader, ascii=True):
        batches.append( era5_to_task(b))
        i = i + 1
        if i == 10:
            break

    torch.manual_seed(time.time())
    if args.device == 'cuda':
        torch.cuda.manual_seed(time.time())

    path = osp.join(evalsets_path, 'era5')
    if not osp.isdir(path):
        os.makedirs(path)

    filename = f'ERA5_{args.place}.tar'
    torch.save(batches, osp.join(path, filename))


def eval(args, model, epoch=None):
    if args.mode == 'eval':
        ckptfile = osp.join(args.root, 'ckpt.tar') if args.ckptfile is None else args.ckptfile
        print(f'loading checkpoint from {ckptfile}')
        ckpt = torch.load(ckptfile, weights_only=False)
        model.load_state_dict(ckpt['model'])
        if args.eval_logfile is None:
            eval_logfile = f'eval_{"ERA5"}-{args.place}'
            eval_logfile += '.log'
        else:
            eval_logfile = args.eval_logfile
        filename = osp.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None


    path = osp.join(evalsets_path, 'era5')
    filename = f'{"ERA5"}_{args.place}'
    filename += '.tar'
    if not osp.isfile(osp.join(path, filename)):
        print('generating evaluation sets...')
        gen_evalset(args)

    eval_batches = torch.load(osp.join(path, filename), weights_only=False)

    torch.manual_seed(args.eval_seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.eval_seed)

    ravg = RunningAverage()
    model.eval()


    with torch.no_grad():
        for batch in tqdm(eval_batches, ascii=True):
            print('stats:', batch.x.mean(axis=[1, 0]), '+-', batch.x.std(axis=[1, 0]), batch.y.mean(axis=[1, 0]), '+-', batch.y.std(axis=[1, 0]))
            for key, val in batch.items():
                batch[key] = val.to(args.device)

            if args.eval_obs_noise > 0.:
                batch['y'] += torch.randn_like(batch['y'])*args.eval_obs_noise
                batch.yc = batch.y[:,:batch.xc.shape[-2]]
                batch.yt = batch.y[:,batch.xc.shape[-2]:]
        
            outs = model(batch, args.eval_num_samples)
            
            for key, val in outs.items():
                ravg.update(key, val)


    torch.manual_seed(time.time())
    if args.device == 'cuda':
        torch.cuda.manual_seed(time.time())

    line = f'{args.model}:{args.expid} {"ERA5"}_{args.place} '

    line += ravg.info()

    if logger is not None:
        logger.info(line)

    return line


def plot(args, model, batch=None, suffix=''):
    torch.multiprocessing.set_sharing_strategy('file_system')
    eval_ds = None
    if batch is None:
        if args.eval_seed is not None:
            torch.manual_seed(args.eval_seed)
            if args.device == 'cuda':
                torch.cuda.manual_seed(args.eval_seed)

        args.max_ctx_points = (41 * 41) - 1 if args.max_ctx_points is None else args.max_ctx_points

        eval_ds = ERA5Dataset('valid', 2, args.max_ctx_points, place='US', normalize=True, circular=True)
        eval_loader = torch.utils.data.DataLoader(eval_ds, batch_size=args.plot_batch_size, shuffle=False, num_workers=4)

        if args.plot_random_samples:
            batches = list(eval_loader)
            random_index = random.randint(0, len(batches) - 1)
            batch = batches[random_index]
        else:
            batch = next(iter(eval_loader))

        batch = era5_to_task(batch, max_ctx_points=args.max_ctx_points, device=args.device)



    if args.mode == 'plot':
        print("Loading model from checkpoint")
        suffix = "ckpt"
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'), weights_only=False)
        model.load_state_dict(ckpt.model)
        model.eval()

    print("model has been loaded")

    results = []
    with torch.no_grad():
        if isinstance(batch, list):
            for b in range(args.plot_batch_size):
                if args.plot_random_samples:
                    bb = random.randint(0, len(batch) - 1)
                    bi = random.randint(0, len(batch[bb]['xc']) - 1)
                else:
                    bb = b % len(batch)
                    bi = b // len(batch)

                py = model.predict(batch[bb]['xc'][bi:bi+1], batch[bb]['yc'][bi:bi+1], batch[bb]['xt'][bi:bi+1], num_samples=args.plot_num_samples)
                yt = py.mean
                context, concat = task_to_era5(batch[bb])
                results.append((context, concat))
        else:
            py = model.predict(batch.xc, batch.yc, batch.xt, num_samples=args.plot_num_samples)
            Means, Covs = py.mean, py.variance
            mean_0 = Means.mean(dim=0)
            plot_inference_2d(batch.xc[0].cpu(), batch.yc[0, :, [2, 3]].cpu(),
                              X_Target=batch.xt[0].cpu(), Y_Target=batch.yt[0, :, [2, 3]].cpu(),
                              Predict=mean_0[0,:,[2,3]].detach().cpu(), Cov_Mat=Covs[0][0,:,[2,3]].detach().cpu(),
                              title="", size_scale=2, ellip_scale=0.8, quiver_scale=15, plot_points=False,out_path=args.root)
            """
            # Translate the data back to original scale
            """
            batch.x,batch.y = eval_ds.translater.translate_to_original_scale(batch.x.cpu(), batch.y.cpu())
            batch.xc,batch.yc = eval_ds.translater.translate_to_original_scale(batch.xc.cpu(), batch.yc.cpu())
            batch.xt,batch.yt = eval_ds.translater.translate_to_original_scale(batch.xt.cpu(), batch.yt.cpu())
            _, mean_0 = eval_ds.translater.translate_to_original_scale(batch.xt.cpu(), Means[0].cpu())

            plot_temperature(batch.xc[0].cpu(), batch.yc[0].cpu(),batch.xt[0].cpu(), batch.yt[0].cpu(),mean_0[0,:,:].detach().cpu(),save_path=args.root+"/temperature_plot.png")

if __name__ == '__main__':
    main()

