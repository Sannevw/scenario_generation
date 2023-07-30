import numpy as np
import torch
import torch.nn as nn
import time
from collections import defaultdict
import os
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import statistics
from scipy.stats import norm
from gigalib.datasets import SpaTiaLPoseDataset
from gigalib.utils import *
from torch.autograd import Variable

print('Torch uses {} threads'.format(torch.get_num_threads()))


def save(model, save_path=None, dataset=None):
    try:
        print("Saved model successfully!")
        model.y_definition = [dataset.y_offset_task, dataset.y_offset_safety]
        torch.save(model, save_path)
    except NameError:
        print("Please specify a valid save path to save the model to")


def load(save_path=None):
    try:
        print("Loaded model successfully!")
        return torch.load(save_path)
    except NameError:
        print("Please specify a valid save path to load the model from")


def set_config(model_params, train_params):
    torch.manual_seed(model_params['SEED'])
    #     if torch.cuda.is_available():
    #         torch.cuda.manual_seed(seed)       
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # N_e = [dataset.shape[1], 256, 256, 256, 128, 128, 64, 64] # size of hidden layer
    # N_z = 20 # dimension of latent space 
    # N_d = [64,64,128,128,256,256,256,dataset.shape[1]] 

    if model_params['name'] == 'ae':
        print("Creating: ", model_params["name"])
        model = AE(
            encoder_layer_sizes=model_params['N_e'],
            latent_size=model_params['N_z'],
            decoder_layer_sizes=model_params['N_d'],
            adverserial_layer_sizes=model_params['N_a'],
            conditional=model_params['conditional'],
            num_labels=model_params['N_c']).to(device)

        # get pytorch dataset loader
        E_t = torch.optim.Adam(model.E.parameters(), lr=train_params['lr'])
        E_g = torch.optim.Adam(model.E.parameters(), lr=train_params['lr'])
        D_t = torch.optim.Adam(model.D.parameters(), lr=train_params['lr'])
        A_t = torch.optim.Adam(model.A.parameters(), lr=train_params['lr'])

        # optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])
    model_config = {
        #         'data_loader': data_loader,
        'E_t': E_t,
        'E_g': E_g,
        'D_t': D_t,
        'A_t': A_t,
        'device': device,
    }
    return model, model_config


class AE(nn.Module):

    ## code inspired from https://blog.paperspace.com/adversarial-autoencoders-with-pytorch/

    def __init__(self, encoder_layer_sizes, latent_size, decoder_layer_sizes, adverserial_layer_sizes,
                 conditional=False, num_labels=0):

        super().__init__()

        if conditional:
            assert num_labels > 0

        assert type(encoder_layer_sizes) == list
        assert type(latent_size) == int
        assert type(decoder_layer_sizes) == list

        self.latent_size = latent_size

        self.E = Encoder(
            encoder_layer_sizes, latent_size, False, num_labels)
        self.D = Decoder(
            decoder_layer_sizes, latent_size, conditional, num_labels)
        self.A = Discriminator(
            adverserial_layer_sizes, latent_size, False, 0)

    # encode and decode the input
    def forward(self, x, c=None):
        #         print("X: ", x.shape)
        #         print("C: ", c.shape)
        means, log_var = self.encoder(x, c)
        z = self.reparameterize(means, log_var)
        recon_x = self.decoder(z, c)

        return recon_x, means, log_var, z

    def reparameterize(self, mu, log_var):

        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)

        return mu + eps * std

        # Sample from the latent space

    def inference(self, z, c=None):

        recon_x = self.D(z, c)

        return recon_x

    def loss_fn(self, recon_x, x, mean, log_var, beta):
        # MSE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')
        MSE = torch.nn.functional.mse_loss(recon_x, x, reduction='sum')

        #     print(log_var.shape)
        # KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=-1).sum()
        # KLD = torch.nn.functional.kl_div(recon_x, x, reduction='sum')
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        KLD = torch.sum(torch.abs(mean + eps * std))
        return (MSE) + (beta * KLD)

    def train(self, data_loader, logs=defaultdict(lambda: defaultdict(list)), sample_id=0):
        epochs = self.train_params['epochs']
        N_batch = self.train_params['N_batch']
        learning_rate = self.train_params['lr']
        #         data_loader = self.model_config['data_loader']
        #         optimizer = self.model_config['optimizer']
        device = self.model_config['device']
        conditional = self.model_params['conditional']
        print_every = self.train_params['print_every']

        E_t = self.model_config['E_t']
        E_g = self.model_config['E_g']
        D_t = self.model_config['D_t']
        A_t = self.model_config['A_t']

        for epoch in range(epochs):

            tracker_epoch = defaultdict(lambda: defaultdict(dict))

            for iteration, (x, y) in enumerate(data_loader):

                x, y = x.to(device), y.to(device)
                # optimizer.zero_grad()
                E_t.zero_grad()
                D_t.zero_grad()

                valid = Variable(torch.FloatTensor(x.shape[0], 1).fill_(1.0))  # , requires_grad=False)
                fake = Variable(torch.FloatTensor(x.shape[0], 1).fill_(0.0))  # , requires_grad=False)

                # ENCODE
                z = self.E(x, y)
                x_rec = self.D(z, y)
                recon_loss = torch.nn.functional.mse_loss(x_rec, x, reduction='sum')
                recon_loss.backward()
                E_t.step()
                D_t.step()

                # Adversarial
                A_t.zero_grad()
                E_g.zero_grad()
                self.E.eval()

                z_real_gauss = Variable(torch.randn(x.shape[0], self.model_params['N_z']) * 1)  # Sample from N(0,5)

                z_fake_gauss = self.E(x, y)
                # Compute discriminator outputs and loss
                D_real_gauss, D_fake_gauss = self.A(z_real_gauss), self.A(z_fake_gauss)
                # torch.nn.functional.binary_cross_entropy
                D_loss_real = torch.nn.functional.binary_cross_entropy(D_real_gauss, valid, reduction='sum')
                D_loss_fake = torch.nn.functional.binary_cross_entropy(D_fake_gauss, fake, reduction='sum')
                # D_loss_gauss = -torch.mean(torch.log(D_real_gauss + 0.0000001) + torch.log(1 - D_fake_gauss + 0.0000001))
                D_loss_gauss = (D_loss_real + D_loss_fake) / 2.
                D_loss_gauss.backward()  # Backpropagate loss
                A_t.step()  # Apply optimization step

                self.E.train()
                z_fake_gauss = self.E(x, y)
                D_fake_gauss = self.A(z_fake_gauss)

                # G_loss = -torch.mean(torch.log(D_fake_gauss + 0.0000001))
                G_loss = torch.nn.functional.binary_cross_entropy(D_fake_gauss, valid, reduction='sum')
                G_loss.backward()
                E_g.step()
                logs[1]['loss'].append(D_loss_gauss.item())
                logs[2]['loss'].append(G_loss.item())

                logs[0]['loss'].append(recon_loss.item())

                if iteration % print_every == 0:
                    print(
                        "Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Recon Loss {:9.4f}, Generator Loss {:9.4f}, Disc Loss {:9.4f}".format(
                            epoch, epochs, iteration, len(data_loader) - 1, recon_loss.item() / N_batch,
                                                      G_loss.item() / N_batch, D_loss_gauss.item() / N_batch), end='\r')

        # plot loss over time
        plt.clf()
        plt.plot(np.array(logs[0]['loss']) / N_batch, '-b')
        plt.plot(np.array(logs[1]['loss']), '-r')
        plt.plot(np.array(logs[2]['loss']), '-g')
        plt.show()

        print("loss in the end is: ", logs[sample_id]['loss'][-1] / N_batch)

        return logs


class Encoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.conditional = conditional

        self.MLP = nn.Sequential()

        for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
            if i == 0 and self.conditional:
                in_size += num_labels
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            if i < len(layer_sizes) - 2:
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

    def forward(self, x, c=None):
        if self.conditional:
            x = torch.cat((x, c), dim=-1)

        x = self.MLP(x)

        return x


class Discriminator(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels, sigm=True):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            #             self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i),
                                    module=nn.LeakyReLU(0.2, inplace=True))  # (in_size, out_size))
            else:
                self.MLP.add_module(name="A{:d}".format(i), module=nn.Sigmoid() if sigm else nn.Tanh())

    def forward(self, z, c=None):

        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x


class Generator(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels, sigm=True):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            #             self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())  # (in_size, out_size))
            else:
                self.MLP.add_module(name="A{:d}".format(i), module=nn.Sigmoid() if sigm else nn.Tanh())

    def forward(self, z, c=None):

        if self.conditional:
            z = torch.cat((z, c), dim=-1)

        x = self.MLP(z)

        return x


class GAN(nn.Module):
    # code inspired from
    # https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/cgan/cgan.py
    # https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

    def __init__(self, layer_g, layer_d, N_z, N_x, N_c):

        super().__init__()

        self.G = Generator(layer_g, N_z, True, N_c, True)
        self.D = Discriminator(layer_d, N_x, True, N_c, True)

    # Sample from the latent space
    def inference(self, z, c=None):

        recon_x = self.G(z, c)

        return recon_x

    # train
    def train(self, data_loader, dataset, logs=defaultdict(lambda: defaultdict(list)), sample_id=0):
        epochs = self.train_params['epochs']
        N_batch = self.train_params['N_batch']
        print_every = self.train_params['print_every']
        logs = defaultdict(lambda: defaultdict(list))
        sample_id = 0
        device = self.model_config['device']

        G_op = self.model_config['G_op']
        D_op = self.model_config['D_op']

        for epoch in range(epochs):

            tracker_epoch = defaultdict(lambda: defaultdict(dict))

            for iteration, (x, y) in enumerate(data_loader):

                x, y = x.to(device), y.to(device)

                # Adversarial ground truths
                valid = Variable(torch.FloatTensor(x.shape[0], 1).fill_(1.0))  # , requires_grad=False)
                fake = Variable(torch.FloatTensor(x.shape[0], 1).fill_(0.0))  # , requires_grad=False)

                # -----------------
                #  Train Generator
                # -----------------
                G_op.zero_grad()

                # Sample noise and labels as generator input
                z = Variable(torch.randn(x.shape[0], self.model_params['N_z']) * 1)
                if self.model_params['N_c'] == 1:
                    gen_labels = Variable(torch.rand(x.shape[0], 1) * dataset.numpy[1].max())
                if self.model_params['N_c'] == 2:
                    gen_labels_task = Variable(torch.rand(x.shape[0], 1) * dataset.numpy[1][:, 0].max())
                    gen_labels_safety = Variable(torch.rand(x.shape[0], 1) * dataset.numpy[1][:, 1].max())
                    gen_labels = torch.cat((gen_labels_task, gen_labels_safety), dim=1)

                # Generate a batch of images
                gen_poses = self.G(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                validity = self.D(gen_poses, gen_labels)
                g_loss = torch.nn.functional.binary_cross_entropy(validity, valid, reduction='sum')

                g_loss.backward()  # retain_graph=True)
                G_op.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                D_op.zero_grad()

                # Loss for real poses
                validity_real = self.D(x, y)  # binary_cross_entropy
                d_real_loss = torch.nn.functional.binary_cross_entropy(validity_real, valid, reduction='sum')
                # print(d_real_loss)

                # Loss for fake poses
                validity_fake = self.D(gen_poses.detach(), gen_labels)
                d_fake_loss = torch.nn.functional.binary_cross_entropy(validity_fake, fake, reduction='sum')
                # print(d_fake_loss)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                # d_loss = adv_loss(validity_real, valid)#(adv_loss(validity_real, valid) + adv_loss(validity_fake, fake))# / 2

                d_loss.backward()  # retain_graph=True)
                D_op.step()

                logs[0]['loss'].append(g_loss.item())
                logs[1]['loss'].append(d_loss.item())

                if iteration % print_every == 0:
                    print(
                        "Epoch {:02d}/{:02d} Batch {:04d}/{:d}, Generator Loss {:9.4f}, Discriminator Loss {:9.4f}".format(
                            epoch, epochs, iteration, len(data_loader) - 1, g_loss.item() / N_batch,
                                                      d_loss.item() / N_batch), end='\r')

        # plot loss over time
        plt.clf()
        plt.plot(np.array(logs[0]['loss']) / N_batch, '-b')
        plt.plot(np.array(logs[1]['loss']) / N_batch, '-r')
        plt.show()

        print("loss in the end is: ", logs[sample_id]['loss'][-1] / N_batch)

        return logs

    @classmethod
    def set_config(cls, model_params, train_params):

        torch.manual_seed(model_params['SEED'])
        #     if torch.cuda.is_available():
        #         torch.cuda.manual_seed(seed)       
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # N_e = [dataset.shape[1], 256, 256, 256, 128, 128, 64, 64] # size of hidden layer
        # N_z = 20 # dimension of latent space 
        # N_d = [64,64,128,128,256,256,256,dataset.shape[1]] 

        if model_params['name'] == 'gan':
            print("Creating: ", model_params["name"])
            model = GAN(
                layer_g=model_params['N_g'],
                layer_d=model_params['N_d'],
                N_z=model_params['N_z'],
                N_x=model_params['N_x'],
                N_c=model_params['N_c']).to(device)

        D_op = torch.optim.Adam(model.D.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))
        G_op = torch.optim.Adam(model.G.parameters(), lr=train_params['lr'], betas=(0.5, 0.999))

        # optimizer = torch.optim.Adam(model.parameters(), lr=train_params['lr'])
        model_config = {
            #         'data_loader': data_loader,
            'D_op': D_op,
            'G_op': G_op,
            'device': device,
        }
        return model, model_config


class Decoder(nn.Module):

    def __init__(self, layer_sizes, latent_size, conditional, num_labels):

        super().__init__()

        self.MLP = nn.Sequential()

        self.conditional = conditional
        if self.conditional:
            input_size = latent_size + num_labels
        else:
            input_size = latent_size

        for i, (in_size, out_size) in enumerate(zip([input_size] + layer_sizes[:-1], layer_sizes)):
            self.MLP.add_module(
                name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
            #             self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())
            if i + 1 < len(layer_sizes):
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())  # (in_size, out_size))
            else:
                self.MLP.add_module(name="A{:d}".format(i), module=nn.Sigmoid())

    def forward(self, z, c=None):

        if self.conditional:
            z = torch.cat((z, c), dim=-1)
        x = self.MLP(z)

        return x
