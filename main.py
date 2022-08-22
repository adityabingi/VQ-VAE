import os
import time
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from modules import VQVAE
from dataset import CIFAR10

class VQVAETrainer:

    def __init__(self, args, device):

        self.device = device
        self.model = VQVAE(
                     in_dim = 3,
                     hdim = args.n_hiddens,
                     res_hdim = args.n_residual_hiddens,
                     n_res_blocks = args.n_residual_layers,
                     n_embed = args.n_embeddings,
                     embed_dim = args.embedding_dim,
                     beta = args.beta
                     )
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr =args.learning_rate)
        self.step = 0

    def train_step(self, x, data_var):

        x = x.to(self.device)

        self.optimizer.zero_grad()
        x_hat, embedding_loss = self.model(x)

        recon_loss = torch.mean((x_hat - x)**2) / data_var
        total_loss = recon_loss + embedding_loss

        total_loss.backward()
        self.optimizer.step()

        return total_loss, recon_loss, embedding_loss

    def test_step(self, val_data):

        with torch.no_grad():
            recon_loss, embed_loss= 0., 0.
            for x, _ in val_data:
                x = x.to(self.device)
                x_hat, embedding_loss = self.model(x)
                recon_loss += torch.mean((x_hat - x)**2)
                embed_loss += embedding_loss

            recon_loss /= len(val_data)
            embed_loss /= len(val_data)

        return recon_loss, embed_loss

    def reconstruct_imgs(self, imgs):

        with torch.no_grad():
            imgs = imgs.to(self.device)
            generated, _ = self.model(imgs)

        return generated.cpu()

    def save(self, dir_path):

        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step
            }, os.path.join(dir_path, f'ckpt_{self.step}.pt'))

    def restore(self, ckpt_path):
        with open(ckpt_path, 'rb') as f:
            state_dict = torch.load(f)
            self.model.load_state_dict(state_dict['model'])
            self.optimizer.load_state_dict(state_dict['optimizer'])
            self.step = state_dict['step']


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--generate", action='store_true')
    parser.add_argument("--dataset",  type=str, default='CIFAR10')
    parser.add_argument("--exp_name", type=str, default='vq_vae_cifar', help='name of experiment')
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_updates", type=int, default=25000)
    parser.add_argument("--n_hiddens", type=int, default=128)
    parser.add_argument("--n_residual_hiddens", type=int, default=32)
    parser.add_argument("--n_residual_layers", type=int, default=2)
    parser.add_argument("--embedding_dim", type=int, default=64)
    parser.add_argument("--n_embeddings", type=int, default=512)
    parser.add_argument("--beta", type=float, default=.25)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--ckpt_interval", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--restore", action='store_true')
    parser.add_argument("--restore_path", type=str, help='checkpoint path to restore')
  
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'results/')

    if not (os.path.exists(results_dir)):
        os.makedirs(results_dir)

    logdir = args.exp_name + '_' + time.strftime("%d-%m-%Y-%H-%M-%S")
    logdir = os.path.join(results_dir, logdir)

    if not (os.path.exists(logdir)):
        os.makedirs(logdir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU")
        torch.cuda.manual_seed(args.seed)
    else:
        device = torch.device('cpu')

    if args.dataset == 'CIFAR10':
        dataset = CIFAR10(args.batch_size)
    else:
        raise NotImplementedError

    train_data, train_var = dataset.train_dataloader()
    val_data = dataset.val_dataloader()

    trainer = VQVAETrainer(args, device)

    writer = SummaryWriter(logdir)
    sample_imgs, _ = next(iter(val_data))
    grid = make_grid(sample_imgs, nrow=8, range=(-1, 1), normalize=True)
    save_image(grid, os.path.join(logdir, 'original.jpg'))
    writer.add_image('original', grid, 0)

    if args.train:

        if args.restore:
            trainer.restore(args.restore_path)
            if trainer.step !=0:
                trainer.step+=1
        start_time = time.time()

        for i in range(trainer.step, args.num_updates):
            
            x, _ = next(iter(train_data))
            total_loss, recon_loss, embedding_loss = trainer.train_step(x, train_var)

            if i%args.log_interval == 0:

                print(f"At iteration {i}")
                print("################################")
                print(f"Recon Loss at {i} is {recon_loss.item():.6f}")
                print(f"Embedding Loss at {i} is {embedding_loss.item():.6f}")
                print(f"Total Loss at {i} is {total_loss.item():.6f}")

                time_since_start = time.time() - start_time
                print(f"Time Since Start {time_since_start:.6f}")

                writer.add_scalar('recon_loss', scalar_value=recon_loss.item(), global_step=i)
                writer.add_scalar('embedding_loss', scalar_value=embedding_loss.item(), global_step=i)
                writer.add_scalar('total_loss', scalar_value=total_loss.item(), global_step=i)

            if i% args.ckpt_interval==0:

                test_recon_loss, test_embed_loss= trainer.test_step(val_data)

                print("Testing")
                print("------------------------------")
                print(f"Recon Loss at {i} is {test_recon_loss.item():.6f}")
                print(f"Embedding Loss at {i} is {test_embed_loss.item():.6f}")

                writer.add_scalar('test_recon_loss', scalar_value=test_recon_loss.item(), global_step=i)
                writer.add_scalar('test_embed_loss', scalar_value=test_embed_loss.item(), global_step=i)

                reconstructed = trainer.reconstruct_imgs(sample_imgs)
                grid = make_grid(reconstructed.cpu(), nrow=8, range=(-1, 1), normalize=True)
                writer.add_image(f'reconstruction at epoch {i}', grid, i)
                save_image(grid, os.path.join(logdir, f'recon_at_{i}.jpg'))
                trainer.save(logdir)

            trainer.step += 1

    if args.test:
        test_recon_loss, test_embed_loss = trainer.test_step(val_data)
        print("Testing")
        print("------------------------------")
        print(f"Recon Loss at {i} is {test_recon_loss.item():.6f}")
        print(f"Embedding Loss at {i} is {test_embed_loss.item():.6f}")

    if args.generate:
        sample_imgs, _ = next(iter(val_data))
        reconstructed = trainer.reconstruct_imgs(sample_imgs)
        save_image(make_grid(reconstructed.cpu(), nrow=8, range=(-1, 1), normalize=True), \
                                                os.path.join(logdir, f'recon_infer.jpg'))

if __name__ == '__main__':
    main()
