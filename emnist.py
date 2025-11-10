import os
import os.path as osp
import argparse
import yaml
import torch
import numpy as np
import time
from attrdictionary import AttrDict
from tqdm import tqdm
from copy import deepcopy
from PIL import Image

from data.image import img_to_task, task_to_img
from data.emnist import EMNIST
from utils.misc import load_module, eval_arg
from utils.paths import results_path, evalsets_path
from utils.log import get_logger, RunningAverage

def main():
    parser = argparse.ArgumentParser()

    # Experiment
    parser.add_argument('--mode', choices=['train', 'eval', 'plot', 'plot_samples'], default='train')
    parser.add_argument('--expid', type=str, default='default')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--device', type=str, default=None, help='Device to use (cuda/cpu). If None, will use cuda if available.')

    # Data
    parser.add_argument('--max_num_points', type=int, default=200)
    parser.add_argument('--class_range', type=int, nargs='*', default=[0,10])#[10,47])

    # Model
    parser.add_argument('--model', type=str, default="tnpd")
    parser.add_argument('--model_args', type=str, default=None, help='Additional model arguments as a comma separated list of key=value pairs, e.g. key1=val1,key2=val2')
    
    # Train
    parser.add_argument('--pretrain', action='store_true', default=False)
    parser.add_argument('--train_seed', type=int, default=0)
    parser.add_argument('--train_batch_size', type=int, default=100)
    parser.add_argument('--train_num_samples', type=int, default=4)
    parser.add_argument('--train_num_bs', type=int, default=10)
    parser.add_argument('--obs_noise', type=float, default=0.0)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--save_freq', type=int, default=10)

    # Eval
    parser.add_argument('--eval_seed', type=int, default=0)
    parser.add_argument('--eval_num_bs', type=int, default=50)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--eval_num_samples', type=int, default=50)
    parser.add_argument('--eval_logfile', type=str, default=None)
    parser.add_argument('--ckptfile', type=str, default=None)

    # Plot
    parser.add_argument('--plot_seed', type=int, default=1)
    parser.add_argument('--plot_num_imgs', type=int, default=16)
    parser.add_argument('--plot_num_samples', type=int, default=5)
    parser.add_argument('--plot_num_samples_show', type=int, default=5)
    parser.add_argument('--plot_num_bs', type=int, default=50)
    parser.add_argument('--plot_num_ctx', type=int, default=100)
    parser.add_argument('--start_time', type=str, default=None)

    # OOD settings
    parser.add_argument('--t_noise', type=float, default=None)
    parser.add_argument('--eval_obs_noise', type=float, default=None) # default is same as obs_noise

    args = parser.parse_args()

    # Set device
    if args.device is None:
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {args.device}")

    
    if args.eval_obs_noise is None:
        args.eval_obs_noise = args.obs_noise

    if args.expid is not None:
        args.root = osp.join(results_path, 'emnist', args.model, args.expid)
    else:
        args.root = osp.join(results_path, 'emnist', args.model)

    model_cls = getattr(load_module(f'models/{args.model}.py'), args.model.upper())
    with open(f'configs/emnist/{args.model}.yaml', 'r') as f:
        args.model_config = yaml.safe_load(f)
    if args.pretrain:
        assert args.model == 'tnpa'
        args.model_config['pretrain'] = args.pretrain

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
    elif args.mode == 'plot_samples':
        plot_samples(args, model)

def train(args, model):
    if osp.exists(args.root + '/ckpt.tar'):
        if args.resume is None:
            raise FileExistsError(args.root)
    else:
        os.makedirs(args.root, exist_ok=True)

    with open(osp.join(args.root, 'args.yaml'), 'w') as f:
        yaml.dump(args.__dict__, f)

    train_ds = EMNIST(train=True, class_range=args.class_range)
    train_loader = torch.utils.data.DataLoader(train_ds,
        batch_size=args.train_batch_size,
        shuffle=True, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=len(train_loader)*args.num_epochs)

    if args.resume:
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'), weights_only=False)
        model.load_state_dict(ckpt.model)
        optimizer.load_state_dict(ckpt.optimizer)
        scheduler.load_state_dict(ckpt.scheduler)
        logfilename = ckpt.logfilename
        start_epoch = ckpt.epoch
    else:
        logfilename = osp.join(args.root, 'train_{}.log'.format(
            time.strftime('%Y%m%d-%H%M')))
        start_epoch = 1

    logger = get_logger(logfilename)
    ravg = RunningAverage()

    if not args.resume:
        logger.info('Total number of parameters: {}\n'.format(
            sum(p.numel() for p in model.parameters())))

    for epoch in range(start_epoch, args.num_epochs+1):
        model.train()
        for (x, _) in tqdm(train_loader, ascii=True):
            x = x.to(args.device)
            batch = img_to_task(x,
                max_num_points=args.max_num_points)
            optimizer.zero_grad()

            if args.obs_noise > 0.:
                batch.y += torch.randn_like(batch.y)*args.obs_noise
                batch.yc = batch.y[:, :batch.xc.shape[-2]]
                batch.yt = batch.y[:, batch.xc.shape[-2]:]
        
            outs = model(batch, num_samples=args.train_num_samples)
            
            outs.loss.backward()
            optimizer.step()
            scheduler.step()

            for key, val in outs.items():
                ravg.update(key, val)

        line = f'{args.model}:{args.expid} epoch {epoch} '
        line += f'lr {optimizer.param_groups[0]["lr"]:.3e} '
        line += ravg.info()
        logger.info(line)


        if epoch % args.save_freq == 0 or epoch == args.num_epochs:
            ckpt = AttrDict()
            ckpt.model = model.state_dict()
            ckpt.optimizer = optimizer.state_dict()
            ckpt.scheduler = scheduler.state_dict()
            ckpt.logfilename = logfilename
            ckpt.epoch = epoch + 1
            torch.save(ckpt, osp.join(args.root, 'ckpt.tar'))

        if epoch % args.eval_freq == 0 or epoch == args.num_epochs:
            logger.info(eval(args, model) + '\n')

        ravg.reset()

def gen_evalset(args):

    torch.manual_seed(args.eval_seed)
    if args.device == 'cuda':
        torch.cuda.manual_seed(args.eval_seed)

    eval_ds = EMNIST(train=False, class_range=args.class_range)
    eval_loader = torch.utils.data.DataLoader(eval_ds,
            batch_size=args.eval_batch_size,
            shuffle=False, num_workers=0)

    batches = []
    for x, _ in tqdm(eval_loader, ascii=True):
        batches.append(img_to_task(
            x, max_num_points=args.max_num_points,
            t_noise=args.t_noise)
        )

    torch.manual_seed(time.time())
    if args.device == 'cuda':
        torch.cuda.manual_seed(time.time())

    path = osp.join(evalsets_path, 'emnist')
    if not osp.isdir(path):
        os.makedirs(path)

    c1, c2 = args.class_range
    filename = f'{c1}-{c2}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += f'_seed{args.eval_seed}'
    filename += '.tar'

    torch.save(batches, osp.join(path, filename))

def eval(args, model):
    if args.mode == 'eval':
        if args.ckptfile:
            ckpt = torch.load(args.ckptfile, map_location='cuda', weights_only=False)
        else:    
            ckpt = torch.load(osp.join(args.root, 'ckpt.tar'), weights_only=False)
        model.load_state_dict(ckpt.model)
        if args.eval_logfile is None:
            c1, c2 = args.class_range
            eval_logfile = f'eval_{c1}-{c2}'
            if args.t_noise is not None:
                eval_logfile += f'_{args.t_noise}'
            eval_logfile += '.log'
        else:
            eval_logfile = args.eval_logfile
        filename = osp.join(args.root, eval_logfile)
        logger = get_logger(filename, mode='w')
    else:
        logger = None

    path = osp.join(evalsets_path, 'emnist')
    c1, c2 = args.class_range
    filename = f'{c1}-{c2}'
    if args.t_noise is not None:
        filename += f'_{args.t_noise}'
    filename += f'_seed{args.eval_seed}'
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
            for key, val in batch.items():
                batch[key] = val.to(args.device)
        
            if args.eval_obs_noise > 0.:
                batch['y'] += torch.randn_like(batch['y'])*args.eval_obs_noise
                batch.yc = batch.y[:, :batch.xc.shape[-2]]
                batch.yt = batch.y[:, batch.xc.shape[-2]:]

            if args.model in ["np", "anp"]:
                outs = model(batch, args.eval_num_samples)
            else:
                outs = model(batch)

            for key, val in outs.items():
                ravg.update(key, val)

    torch.manual_seed(time.time())
    if args.device == 'cuda':
        torch.cuda.manual_seed(time.time())

    c1, c2 = args.class_range
    line = f'{args.model}:{args.expid} {c1}-{c2} '
    if args.t_noise is not None:
        line += f'tn {args.t_noise} '
    line += ravg.info()

    if logger is not None:
        logger.info(line)

    return line


def plot(args, model):
    if args.mode == 'plot':
        if args.ckptfile:
            ckpt = torch.load(args.ckptfile, map_location=args.device, weights_only=False)
        else:    
            ckpt = torch.load(osp.join(args.root, 'ckpt.tar'), weights_only=False)
        model.load_state_dict(ckpt.model)

    eval_ds = EMNIST(train=False, class_range=args.class_range)
    torch.manual_seed(args.plot_seed)
    rand_ids = torch.randperm(len(eval_ds))[:args.plot_num_imgs]
    test_data = [eval_ds[i][0] for i in rand_ids]
    test_data = torch.stack(test_data, dim=0).to(args.device)
    batch = img_to_task(test_data, max_num_points=None, num_ctx=args.plot_num_ctx, target_all=True, ctx_mode=0)
    
    model.eval()
    with torch.no_grad():
        outs = model.predict(batch.xc, batch.yc, batch.xt, num_samples=args.eval_num_samples)
        
    mean = outs.mean
    # shape: (num_samples, 1, num_points, 1)
    if mean.dim() == 4:
        mean = mean.mean(dim=0)

    task_img, completed_img = task_to_img(batch.xc, batch.yc, batch.xt, mean, shape=(1,28,28))
    _, orig_img = task_to_img(batch.xc, batch.yc, batch.xt, batch.yt, shape=(1,28,28))

    task_img = (torch.min(torch.Tensor([1.]), torch.max(torch.Tensor([0.]), task_img)) * 255).int().cpu().numpy().transpose(0,2,3,1)
    completed_img = (completed_img * 255).int().cpu().numpy().transpose(0,2,3,1)
    orig_img = (torch.min(torch.Tensor([1.]), torch.max(torch.Tensor([0.]), orig_img)) * 255).int().cpu().numpy().transpose(0,2,3,1)

    c1, c2 = args.class_range
    save_dir = osp.join(args.root, f'plots_{c1}-{c2}')
    os.makedirs(save_dir, exist_ok=True)

    samples = []
    for s in range(args.plot_num_samples_show):
        s = outs.mean[s]
        _, s_img = task_to_img(batch.xc, batch.yc, batch.xt, s, shape=(1,28,28))    
        samples.append((s_img * 255).int().cpu().numpy().transpose(0,2,3,1))

    all_imgs = np.concatenate([orig_img, task_img] + samples + [completed_img], axis=2)
        
    for i in range(args.plot_num_imgs):
        Image.fromarray(orig_img[i].astype(np.uint8)).resize((128,128),Image.BILINEAR).save(save_dir + '/%d_orig.jpg' % (i+1))
        Image.fromarray(task_img[i].astype(np.uint8)).resize((128,128),Image.BILINEAR).save(save_dir + '/%d_task.jpg' % (i+1))
        Image.fromarray(completed_img[i].astype(np.uint8)).resize((128,128),Image.BILINEAR).save(save_dir + '/%d_completed.jpg' % (i+1))
        for s in range(args.plot_num_samples_show):
            Image.fromarray(samples[s][i].astype(np.uint8)).resize((128,128),Image.BILINEAR).save(save_dir + '/%d_sample_%d.jpg' % (i+1, s+1))

        Image.fromarray(all_imgs[i].astype(np.uint8)).resize((128*(3+args.plot_num_samples_show),128),Image.BILINEAR).save(save_dir + '/%d_combined.jpg' % (i+1))
    Image.fromarray(all_imgs.reshape([-1, all_imgs.shape[2], all_imgs.shape[3]]).astype(np.uint8)).resize(
        (128*(3+args.plot_num_samples_show), 128*args.plot_num_imgs),Image.BILINEAR).save(save_dir + '/all_combined.jpg')

def plot_samples(args, model):
    if args.mode == 'plot_samples':
        ckpt = torch.load(osp.join(args.root, 'ckpt.tar'), weights_only=False)
        model.load_state_dict(ckpt.model)

    eval_ds = EMNIST(train=False, class_range=args.class_range)
    torch.manual_seed(args.plot_seed)
    rand_ids = torch.randperm(len(eval_ds))[:args.plot_num_imgs]
    test_data = [eval_ds[i][0] for i in rand_ids]
    test_data = torch.stack(test_data, dim=0).to(args.device)
    
    list_num_ctx = [10, 20, 50, 100, 150]
    batches = [img_to_task(test_data, max_num_points=None, num_ctx=i, target_all=True) for i in list_num_ctx]
    all_samples = []
    
    model.eval()
    with torch.no_grad():
        for batch in batches:
            samples = model.sample(batch.xc, batch.yc, batch.xt, num_samples=args.eval_num_samples)
            all_samples.append(samples)

    c1, c2 = args.class_range
    save_dir = osp.join(args.root, f'sample_plots_{c1}-{c2}')
    os.makedirs(save_dir, exist_ok=True)

    # save original images
    _, orig_img = task_to_img(batches[-1].xc, batches[-1].yc, batches[-1].xt, batches[-1].yt, shape=(1,28,28)) # (num_imgs, 32, 32, 3)
    orig_img = (orig_img * 255).int().cpu().numpy().transpose(0,2,3,1)
    for i in range(args.plot_num_imgs):
        Image.fromarray(orig_img[i].astype(np.uint8)).resize((128,128),Image.BILINEAR).save(save_dir + '/%d_orig.jpg' % (i+1))
    
    for i in range(len(list_num_ctx)):
        num_ctx = list_num_ctx[i]
        batch = batches[i]
        samples = all_samples[i]

        for j in range(args.eval_num_samples):
            task_img, completed_img = task_to_img(batch.xc, batch.yc, batch.xt, samples[j], shape=(1,28,28)) # (num_imgs, 32, 32, 3)

            task_img = (task_img * 255).int().cpu().numpy().transpose(0,2,3,1)
            completed_img = (completed_img * 255).int().cpu().numpy().transpose(0,2,3,1)

            for k in range(args.plot_num_imgs):
                Image.fromarray(task_img[k].astype(np.uint8)).resize((128,128),Image.BILINEAR).save(save_dir + '/%d_task_%d_ctx.jpg' % (k+1, num_ctx))
                Image.fromarray(completed_img[k].astype(np.uint8)).resize((128,128),Image.BILINEAR).save(save_dir + '/%d_completed_%d_ctx_%d_samples.jpg' % (k+1, num_ctx, j+1))

if __name__ == '__main__':
    main()
