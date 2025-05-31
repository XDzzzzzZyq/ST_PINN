import gc
import io
import os
import glob
import time

from model.mutils import get_model
from model.ema import ExponentialMovingAverage
from simulate.simulate import Simulator
import losses
import datasets

import torch
import torchvision.transforms.functional as F
from utils import save_checkpoint, load_checkpoint, restore_checkpoint


def poisson_resample(model, simulator, genes, in_tissue, s, l, n_sample=64):
    genes = genes.repeat(n_sample, 1, 1, 1)
    in_tissue = in_tissue.repeat(n_sample, 1, 1, 1)
    sample = simulator.simulate_end(genes, in_tissue)

    r = 99
    noise = torch.randn_like(in_tissue)
    noise = F.gaussian_blur(noise, (r, r), (s, s)) * l
    sample.f = sample.f + noise * sample.f
    
    info = (in_tissue, ) #if config.model.conditional else None
    sol, ts = simulator.reverse(model, sample, info, rtol=1e-4, atol=1e-5, num_sample=1)

    return noise, sol, sample

def deblur(config, workdir, tardir):
    config.param.Re_min = config.param.Re_max = 1004.1
    config.data.poisson_ratio_min = config.data.poisson_ratio_max = 0.3
    config.param.t0 = 0.0

    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    simulator = Simulator(config)
    model = get_model(config).to(config.device)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)

    # TODO: tardir, now just for debug purpose
    B = config.training.batch_size = 1
    train_ds, eval_ds = datasets.get_dataset(config)
    batch, target = next(iter(eval_ds))

    n_sample = 64
    batch = batch.to(config.device).float()
    in_tissue, density, total, genes = batch[:, 0:1], batch[:, 1:2], batch[:, 2:3], batch[:, 3:4]

    mode = 1
    if mode == 0:
        s = 4.0  # TODO: Infered Sigma
        l = 10.0

        noise, sol, sample = poisson_resample(model, simulator, genes, in_tissue, s, l, n_sample)
        pred = sol[1].mean(dim=0).unsqueeze(0)
        print(sol.shape)

        import matplotlib.pyplot as plt
        fig, axe = plt.subplots(nrows=2, ncols=5, figsize=(40, 20))
        for i in range(5):
            axe[0,i].imshow(sol[1, i, 0].cpu())
        axe[1, 0].imshow(total[0, 0].cpu())
        axe[1, 1].imshow(noise[0, 0].cpu())
        axe[1, 2].imshow(sample.f[0, 0].cpu())
        axe[1, 3].imshow(pred[0, 0].cpu())
        axe[1, 4].imshow(in_tissue[0, 0].cpu())

        os.makedirs(f"{workdir}/plots/resample", exist_ok=True)
        plt.savefig(f"{workdir}/plots/resample/resample | s : {s} | f : {l}.png")
    elif mode == 1:
        s_list = [7.5, 10.0, 15, 20.0]
        l_list = [2.5, 5.0, 7.5, 10.0]
        mse_loss = torch.nn.MSELoss()

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(s_list), len(l_list), figsize=(16, 16))  # 4x4 grid

        vmin, vmax = density.min().item(), density.max().item()
        loss_table = torch.ones(len(s_list), len(l_list))

        for i, s in enumerate(s_list):
            for j, l in enumerate(l_list):
                ax = axes[i, j]  # access subplot at row i, column j

                # Run the function with current parameters
                noise, sol, sample = poisson_resample(model, simulator, genes, in_tissue, s, l)
                pred = sol[1].mean(dim=0).unsqueeze(0)

                # Customize this part depending on what you want to visualize
                # For example, show the predicted sample (assuming 1D or 2D)
                ax.imshow(pred[0, 0].cpu(), vmin=vmin, vmax=vmax)
                loss = mse_loss(pred, density).item()

                ax.set_title(f's={s}, l={l}, loss={loss / 1e-3:.3f}e-3', fontsize=10)
                loss_table[i, j] = loss
                
                print(f"{s} : {l} : loss={loss}")
                ax.axis('off')  # or customize axes if needed

        noise, sol, sample = poisson_resample(model, simulator, genes, in_tissue, 1.0, 0.0)
        pred = sol[1].mean(dim=0).unsqueeze(0)
        loss = mse_loss(pred, density).item()
        print(f"no preturb : loss={loss / 1e-3:.3f}e-3")
        ori_loss = mse_loss(genes, density).item()
        print(f"Original : loss={ori_loss / 1e-3:.3f}e-3")
        nd_loss = mse_loss(sample.f, density).item()
        print(f"No reverse : loss={nd_loss / 1e-3:.3f}e-3")

        plt.tight_layout()
        os.makedirs(f"{workdir}/plots/resample", exist_ok=True)
        plt.savefig(f"{workdir}/plots/resample/grid | Re : {config.param.Re_min} | Poi : {config.data.poisson_ratio_max}.png")

        fig, axe = plt.subplots(nrows=1, ncols=5, figsize=(40, 10))
        axe[0].imshow(density[0, 0].cpu())
        axe[1].imshow(genes[0, 0].cpu())
        axe[2].imshow(sample.f[0, 0].cpu())
        axe[3].imshow(sol[1, 0, 0].cpu())
        axe[4].imshow(loss_table)
        fig.text(0.5, 0.02, f"No preturb est : loss = {loss / 1e-3:.3f}e-3 | Original : loss = {ori_loss / 1e-3:.3f}e-3 | No reverse : loss = {nd_loss / 1e-3:.3f}e-3 | t= = {config.param.t0}",
         ha='center', va='center', fontsize=12)
        plt.savefig(f"{workdir}/plots/resample/grid | Re : {config.param.Re_min} | Poi : {config.data.poisson_ratio_max} compare.png")

        



if __name__ == '__main__':

    import datasets
    from config.combine_configs import get_config
    config = get_config()
    config.training.batch_size = 1
    config.training.sample_per_sol = 32
    config.param.t0 = -1.0

    workdir = 'workdir/simu2'
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint-pretrain.pth")
    simulator = Simulator(config)
    model = get_model(config).to(config.device)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)

    train_ds, eval_ds = datasets.get_dataset(config)
    batch, target = next(iter(eval_ds))

    batch = batch.to(config.device).float()
    in_tissue, density, total, genes = batch[:, 0:1], batch[:, 1:2], batch[:, 2:3], batch[:, 3:4]
    info = (in_tissue, ) #if config.model.conditional else None
    B, N, W, H = genes.shape
    samples = simulator.simulate(genes.reshape(B*N, 1, W, H), in_tissue.repeat(N,1,1,1), shuffle=False)

    mode = 2
    if mode == 0:
        import matplotlib.pyplot as plt
        
        state = samples[int((len(samples)-1) * 0.8)]
        print(state.t.item())

        def draw(stride, exp):
            sol, pred = simulator.reverse_euler(model, state, info, stride=stride, exp=exp)
            vmin, vmax = sol[0][0, 0].min().item(), sol[0][0, 0].max().item()
            fig, axe = plt.subplots(nrows=2, ncols=len(sol)+1, figsize=((len(sol)+1)*10, 20))
            for i in range(len(sol)):
                axe[0,i].imshow(sol[i][0, 0].cpu(), vmin=vmin, vmax=vmax)
                axe[1,i].imshow(pred[i][0, 0].cpu())
            axe[0,-1].imshow(total[0, 0].cpu(), vmin=vmin, vmax=vmax)
            axe[1,-1].imshow(torch.zeros_like(pred[i][0, 0]).cpu(), vmin=vmin, vmax=vmax)

            print(f"loss of s={stride} | exp={exp} : {((pred[-1] - total) ** 2).mean()}")

            plt.savefig(f"{workdir}/plots/reverse/reverse_re | t={state.t.item():.2f} | s={stride} | exp={exp}.png")
        
        draw(100, 0.1)
        draw(100, 0.3)
        draw(100, 0.5)
        draw(10, 0.1)
        draw(10, 0.3)
        draw(10, 0.5)
        draw(1, 1.0)
        print(f"loss of nothing : {((state.f - total) ** 2).mean()}")
    elif mode == 1:
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
        
        state = samples[int((len(samples)-1) * 0.5)]
        state = samples[-1]
        print(state.t.item())

        def draw():
            sol, ts = simulator.reverse(model, state, info, rtol=1e-4, atol=1e-5)
            vmin, vmax = total[0, 0].min().item(), total[0, 0].max().item()
            
            n = len(sol) + 1  # number of images (sol + total)
            fig = plt.figure(figsize=(n * 10 + 1.5, 10))  # add space for colorbar
            spec = gridspec.GridSpec(nrows=1, ncols=n+1, width_ratios=[1]*n + [0.05], wspace=0.05)

            for i in range(len(sol)):
                ax = fig.add_subplot(spec[0, i])
                im = ax.imshow(sol[i][0, 0].cpu(), vmin=vmin, vmax=vmax)
                ax.set_title(f"t = {ts[i].item():.2f}", fontsize=14)
                ax.axis("off")

            ax = fig.add_subplot(spec[0, len(sol)])
            ax.set_title("Total", fontsize=14)
            ax.axis("off")
            im = ax.imshow(genes[0, 0].cpu(), vmin=vmin, vmax=vmax)

            # Add colorbar in the final column
            cbar_ax = fig.add_subplot(spec[0, -1])
            fig.colorbar(im, cax=cbar_ax)

            os.makedirs(f"{workdir}/plots/reverse", exist_ok=True)
            plt.savefig(f"{workdir}/plots/reverse/reverse_blk | t={state.t.item():.2f}d.png")
            print(genes[0, 0].sum(), sol[-1][0, 0].sum())
        
        draw()
    elif mode == 2:
        tl = []
        r = []
        os.makedirs(f"{workdir}/plots/pred", exist_ok=True)
        for idx, sample in enumerate(samples):
            t, f, v, p, df_dt = sample.get()
            with torch.no_grad():
                if model.conditional:
                    pred = model(f, t.to(f.device), info)
                else:
                    pred = model(torch.cat([*info, f], dim=1), t.to(f.device))

            import matplotlib.pyplot as plt
            fig, axe = plt.subplots(nrows=2, ncols=4, figsize=(30, 10))
            axe[0][0].imshow(f[0, 0].cpu())
            axe[0][1].imshow(v[0, 0].cpu())
            axe[0][2].imshow(v[0, 1].cpu())
            axe[0][3].imshow(p[0, 0].cpu())
            vmin, vmax = df_dt[0, 0].min().item(), df_dt[0, 0].max().item()
            axe[1][0].imshow(df_dt[0, 0].cpu(), vmin=vmin, vmax=vmax)
            axe[1][1].imshow(pred[0, 0].cpu())
            axe[1][2].imshow(in_tissue[0, 0].cpu())
            axe[1][3].imshow(genes[0, 0].cpu())

            p = pred[0, 0].abs()
            g = df_dt[0, 0].abs()
            print(p.mean(), g.mean(), p.mean()/g.mean())
            tl.append(sample.t.item())
            r.append((p.mean()/g.mean()).item())
            fig.text(0.5, 0.02, f"t={t.item():.2f}",
                ha='center', va='center', fontsize=12)

            plt.savefig(f"{workdir}/plots/pred/simulate i={idx+1}.png")
            plt.close()

        d = {"t":tl, "r":r}
        torch.save(d, 'analysis.pth')