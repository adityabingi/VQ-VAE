
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class VQVAE(nn.Module):
    """ Vector Quantized Variational Auto Encoder -- module"""
    def __init__(self, in_dim, hdim, res_hdim, n_res_blocks, n_embed, embed_dim, beta):

        super().__init__()

        self.encoder = Encoder(in_dim, hdim, res_hdim, n_res_blocks)
        self.pre_vq_conv = nn.Conv2d(hdim, embed_dim, kernel_size=1, stride=1)
        self.quantizer = Quantizer(n_embed, embed_dim, beta)
        self.decoder = Decoder(embed_dim, hdim, res_hdim, n_res_blocks)

    def forward(self, x):

        z = self.encoder(x)
        z_e = self.pre_vq_conv(z)
        z_q, embedding_loss = self.quantizer(z_e)
        x_hat = self.decoder(z_q)

        return x_hat, embedding_loss


class Encoder(nn.Module):

    """ Convolutional encoder for vq-vae"""
    def __init__(self, in_dim, hdim, res_hdim, n_res_blocks,):

        super().__init__()

        self.conv = nn.Sequential(
                      nn.Conv2d(in_dim, hdim//2, kernel_size = 4, stride=2, padding =1),
                      nn.ReLU(),
                      nn.Conv2d(hdim//2, hdim, kernel_size = 4, stride=2, padding =1),
                      nn.ReLU(),
                      nn.Conv2d(hdim, hdim, kernel_size=3, stride=1, padding=1),
                      ResStack(hdim, hdim, n_res_blocks, res_hdim))
        
    def forward(self, x):

        x = self.conv(x)
        return x

class Decoder(nn.Module):

    """ ConvTranspose Decoder for vq-vae"""
    def __init__(self, in_dim, hdim, res_hdim, n_res_blocks):

        super().__init__()

        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(in_dim, hdim, kernel_size =3, stride=1, padding =1),
            ResStack(hdim, hdim, n_res_blocks, res_hdim),
            nn.ConvTranspose2d(hdim, hdim//2, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hdim//2, 3, kernel_size=4, stride=2, padding=1),
            )

    def forward(self, x):

        x = self.conv_transpose(x)
        return x

class Quantizer(nn.Module):

    """ Latent codebook quantization """
    def __init__(self, n_embed, embed_dim, beta):

        super().__init__()

        self.n_embed = n_embed
        self.embed_dim = embed_dim
        self.beta = beta

        self.embed = nn.Embedding(self.n_embed, self.embed_dim)
        self.embed.weight.data.uniform_(-1.0/self.n_embed, 1.0/self.n_embed)


    def forward(self, x):

        z = x.permute(0, 2, 3,1).contiguous()
        z_flatten = z.view(-1, self.embed_dim)

        dist = torch.sum(z_flatten **2, dim=1, keepdim=True) + \
               torch.sum(self.embed.weight ** 2, dim=1) - \
               2 * torch.matmul(z_flatten, self.embed.weight.t())

        min_enc_indxs = torch.argmin(dist, dim=1).unsqueeze(1)

        min_enc = torch.zeros(min_enc_indxs.shape[0], self.n_embed).to(device= z.device)
        min_enc.scatter_(1, min_enc_indxs, 1)

        z_q = torch.matmul(min_enc, self.embed.weight).view(z.shape)

        loss = torch.mean((z_q.detach() - z) ** 2 ) + self.beta * torch.mean((z_q -z.detach())**2)
        
        z_q = z + (z_q -z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q, loss

class ResBlock(nn.Module):
    """ Single residual layer"""
    def __init__(self, in_dim, hdim, res_hdim):

        super().__init__()

        self.res_block = nn.Sequential(
                    nn.ReLU(),
                    nn.Conv2d(in_dim, res_hdim, kernel_size = 3, stride=1, padding=1, bias=False),
                    nn.ReLU(),
                    nn.Conv2d(res_hdim, hdim, kernel_size = 1, stride=1, bias=False),
                    )

    def forward(self, x):

        x = x + self.res_block(x)
        return x

class ResStack(nn.Module):
    """ Residual Layers Stack"""

    def __init__(self, in_dim, hdim, res_hdim, n_blocks):

        super().__init__()

        self.n_blocks = n_blocks
        self.res_stack = nn.ModuleList([ResBlock(in_dim, hdim, res_hdim)]*self.n_blocks)

    def forward(self, x):
        for layer in self.res_stack:
            x = layer(x)
        x = F.relu(x)
        return x
