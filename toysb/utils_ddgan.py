import torch as th
from torch.optim import AdamW, lr_scheduler
import torch.nn.functional as F
from torch_ema import ExponentialMovingAverage
import os
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import torchvision
from utils import compute_gaussian_product_coef, unsqueeze_xdim

from tqdm.auto import tqdm
from IPython.display import clear_output

from toySB.denoising_diffusion_gan.EMA import EMA

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def make_beta_schedule_ddgan(n_timestep=4, linear_start=1e-4, linear_end=2e-4):
    betas = (
        th.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=th.float64) ** 2
    )
    return betas.numpy()

def create_symmetric_beta_schedule_ddgan(n_timestep, *args, **kwargs):
    import numpy as np
    betas = make_beta_schedule_ddgan(n_timestep=n_timestep, *args, **kwargs)
    betas = np.concatenate([betas[:(n_timestep + 1)//2], np.flip(betas[:n_timestep//2])])
    return betas

def load_from_ckpt(net, opt, logger):
    checkpoint = th.load(opt.ckpt_path, map_location="cpu")
    net.load_state_dict(checkpoint['net'])
    logger.info(f"[Net] Loaded network ckpt: {opt.ckpt_path}!")
    ema = ExponentialMovingAverage(net.parameters(), decay=opt.ema)
    ema.load_state_dict(checkpoint["ema"])
    logger.info(f"[Ema] Loaded ema ckpt: {opt.ckpt_path}!")
    
    net.to(opt.device)
    ema.to(opt.device)
    return net, ema

def save_gif(path_to_imgs, path_to_save, range_list):
    import imageio

    path_to_imgs = Path(path_to_imgs)
    images = [imageio.imread(str(path_to_imgs / f"{i}.png")) for i in range_list]
    imageio.mimsave(path_to_save, images)

def build_range_list(path):
    return sorted([int(p.name.split(".")[0]) for p in path.glob("*")], reverse = True)

def expand_to_planes(input, shape):
    return input[..., None, None].repeat([1, 1, shape[2], shape[3]])

def frames_to_video(resulting_video_path, frames_path, fps = 5):
    import cv2
    frames_path = Path(frames_path)
    img_shape = cv2.imread(str(frames_path / '0.png')).shape
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    print("Original fourcc: ", fourcc)
    video = cv2.VideoWriter(resulting_video_path, fourcc, fps, (img_shape[1], img_shape[0]))
    
    for img_name in build_range_list(frames_path):
        img = cv2.imread(str(frames_path / f"{img_name}.png"))
        video.write(img)

    video.release()

def train_ddgan(
        netG, netD, optimizerG, optimizerD, schedulerG, schedulerD, diff_scheduler, train_dataloader,
        n_epoch=1, device=torch.device('cuda'), resume_checkpoint_path=None, lazy_reg=1, r1_gamma=1e-2,
        visualize_every=1000, print_every=100, save_content_every=None, exp_path = None, save_ckpt_every=None,
        use_ema=True, ema_decay=0.999, **kwargs,):
    if use_ema:
        optimizerG = EMA(optimizerG, ema_decay=ema_decay)


    if resume_checkpoint_path is not None:
        checkpoint_file = resume_checkpoint_path
        checkpoint = torch.load(checkpoint_file, map_location=device)

        # load G
        netG.load_state_dict(checkpoint['netG_dict'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        schedulerG.load_state_dict(checkpoint['schedulerG'])

        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])
        schedulerD.load_state_dict(checkpoint['schedulerD'])

    def gan_pred_x0_fn(x_tp1, t):
        latent_z = torch.randn((x_tp1.shape[0], netG.z_dim), device=device)
        t = t * torch.ones((x_tp1.shape[0],), dtype=torch.int64, device=device)
        return netG(x_tp1.detach(), t, latent_z)

    history = {
        'D_loss': [],
        'G_loss': [],
    }
    history = dotdict(history)

    pbar = tqdm(total=n_epoch * len(train_dataloader))
    iteration = 0

    for epoch in range(n_epoch):
        for x0_real, x1_real in train_dataloader():
            #########################
            # Discriminator training
            #########################
            for p in netD.parameters():
                p.requires_grad = True

            netD.zero_grad()

            ###################################
            # Sample real data
            x0_real, x1_real = sampler.sample()
            x0_real = x0_real.to(device)
            x1_real = x1_real.to(device)

            ###################################
            # Sample timesteps
            t = torch.randint(0, diff_scheduler.num_timesteps - 1, (x0_real.size(0),), device=device)
            tp1 = t + 1

            ###################################
            # Sample pairs
            x_t, x_tp1 = diff_scheduler.q_sample_pair(t, tp1, x0_real, x1_real, ot_ode=False)
            x_t.requires_grad = True

            ###################################
            # Optimizing loss on real data
            D_real = netD(x_t, t, x_tp1.detach()).view(-1)

            errD_real = F.softplus(-D_real)
            errD_real = errD_real.mean()

            errD_real.backward(retain_graph=True)

            ###################################
            # R_1(\phi) regularization
            if lazy_reg is not None and iteration % lazy_reg == 0:
                grad_real = torch.autograd.grad(
                    outputs=D_real.sum(), inputs=x_t, create_graph=True,
                )[0]
                grad_penalty = (
                    grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                ).mean()

                grad_penalty = r1_gamma / 2 * grad_penalty
                grad_penalty.backward()

            ###################################
            # Sample vector from latent space
            # for generation
            latent_z = torch.randn((x_tp1.shape[0], netG.z_dim), device=device)

            ###################################
            # Sample fake output
            # (x_tp1 -> x_0 -> x_t)
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_t_fake = diff_scheduler.p_posterior(t, tp1, x_tp1, x_0_predict, ot_ode=False)

            ###################################
            # Optimize loss on fake data
            output = netD(x_t_fake, t, x_tp1.detach()).view(-1)

            errD_fake = F.softplus(output)
            errD_fake = errD_fake.mean()
            errD_fake.backward()

            errD = errD_real + errD_fake

            history.D_loss.append(errD.item())

            ###################################
            # Update weights of netD
            optimizerD.step()

            #############################################################

            #########################
            # Generator training
            #########################
            for p in netD.parameters():
                p.requires_grad = False
            netG.zero_grad()

            ###################################
            # Sample timesteps
            t = torch.randint(0, diff_scheduler.num_timesteps - 1, (x0_real.size(0),), device=device)
            tp1 = t + 1

            ###################################
            # Sample pairs
            x_t, x_tp1 = diff_scheduler.q_sample_pair(t, tp1, x0_real, x1_real, ot_ode=False)

            ###################################
            # Sample vector from latent space
            # for generation
            latent_z = torch.randn((x_tp1.shape[0], netG.z_dim), device=device)

            ###################################
            # Sample fake output
            # (x_tp1 -> x_0 -> x_t)
            x_0_predict = netG(x_tp1.detach(), t, latent_z)
            x_t_fake = diff_scheduler.p_posterior(t, tp1, x_tp1, x_0_predict, ot_ode=False)

            ###################################
            # Optimize loss on fake data
            output = netD(x_t_fake, t, x_tp1.detach()).view(-1)

            ###################################
            # Update weights of netG
            errG = F.softplus(-output)
            errG = errG.mean()

            errG.backward()
            optimizerG.step()
            history.G_loss.append(errG.item())

            # LR-Scheduling step
            schedulerG.step()
            schedulerD.step()

            if visualize_every is not None and (iteration + 1) % visualize_every == 0:
                x0_fake = diff_scheduler.ddgan_sampling(torch.arange(0, 5), gan_pred_x0_fn, x1_real, ot_ode=False)[1][:, -1].detach().cpu().numpy().squeeze()
                x0_real = x0_real.detach().cpu().numpy().squeeze()

                clear_output(wait=True)
                plt.figure(figsize=(10, 8))
                plt.subplot(2, 2, 1)
                plt.plot(history.G_loss)
                plt.title('Generator loss')

                plt.subplot(2, 2, 3)
                plt.plot(history.D_loss)
                plt.title('Discriminator loss')

                plt.subplot(2, 2, 4)
                plt.scatter(x0_real[:, 0], x0_real[:, 1], s=1, label='real')
                plt.legend()
                plt.gca().set_aspect('equal', adjustable='box')
                xlim = plt.gca().get_xlim()
                ylim = plt.gca().get_ylim()

                plt.subplot(2, 2, 2)
                plt.scatter(x0_fake[:, 0], x0_fake[:, 1], s=1, label='fake')
                plt.title('Visualization')
                plt.gca().set_aspect('equal', adjustable='box')
                plt.gca().set_xlim(xlim)
                plt.gca().set_ylim(ylim)
                plt.legend()
                plt.tight_layout()

                plt.show()

            if print_every is not None and (iteration + 1) % print_every == 0:
                print('iteration: {} | G Loss: {} | D Loss: {}'.format(iteration, errG.item(), errD.item()))

            if exp_path is not None and save_content_every is not None and (iteration + 1) % save_content_every == 0:
                print('Saving content.')
                content = {
                    'netG_dict': netG.state_dict(),
                    'optimizerG': optimizerG.state_dict(),
                    'schedulerG': schedulerG.state_dict(),
                    'netD_dict': netD.state_dict(),
                    'optimizerD': optimizerD.state_dict(),
                    'schedulerD': schedulerD.state_dict(),
                }

                torch.save(content, os.path.join(exp_path, 'content.pth'))

            if exp_path is not None and save_ckpt_every is not None and (iteration + 1) % save_ckpt_every == 0:
                if use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)

                torch.save(netG.state_dict(), os.path.join(exp_path, 'netG_{}.pth'.format(iteration)))
                torch.save(netD.state_dict(), os.path.join(exp_path, 'netD_{}.pth'.format(iteration)))
                if use_ema:
                    optimizerG.swap_parameters_with_ema(store_params_in_ema=True)
            
            pbar.update(1)
            iteration += 1

    pbar.close()

    if exp_path is not None:
        # Save model in the end
        if use_ema:
            optimizerG.swap_parameters_with_ema(store_params_in_ema=True)

        torch.save(netG.state_dict(), os.path.join(exp_path, 'netG_final.pth'))
        torch.save(netD.state_dict(), os.path.join(exp_path, 'netD_final.pth'))

        if use_ema:
            optimizerG.swap_parameters_with_ema(store_params_in_ema=True)