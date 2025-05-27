import gc
import io
import os
import glob
import time

import logging
from model import unet, unet_lite
from model.ema import ExponentialMovingAverage
from simulate.simulate import Simulator
import losses
import datasets

import torch
from torch.utils import tensorboard
from torchvision.utils import make_grid, save_image
from utils import save_checkpoint, load_checkpoint, restore_checkpoint

def train(config, workdir):
    """Runs the training pipeline.

    Args:
        config: Configuration to use.
        workdir: Working directory for checkpoints and TF summaries. If this
        contains checkpoint training will be resumed from the latest checkpoint.
    """
    torch.manual_seed(config.seed)

    # Create directories for experimental logs
    tb_dir = os.path.join(workdir, "tensorboard")
    os.makedirs(tb_dir, exist_ok=True)
    writer = tensorboard.SummaryWriter(tb_dir)

    # Initialize model.
    model = unet_lite.Unet(config).to(config.device)
    simulator = Simulator(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)

    # Create checkpoints directory
    checkpoint_dir = os.path.join(workdir, "checkpoints")
    # Intermediate checkpoints to resume training after pre-emption in cloud environments
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(os.path.dirname(checkpoint_meta_dir), exist_ok=True)
    # Resume training when intermediate checkpoints are detected
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)
    initial_step = int(state['step'])

    # Build data iterators
    train_ds, eval_ds = datasets.get_dataset(config)
    train_iter = iter(train_ds)     # pytype: disable=wrong-arg-types
    eval_iter = iter(eval_ds)       # pytype: disable=wrong-arg-types

    # Build one-step training and evaluation functions
    optimize_fn = losses.get_optimize_fn(config)
    train_step_fn = losses.get_shooting_step_fn(simulator, train=True, optimize_fn=optimize_fn)
    eval_step_fn = losses.get_shooting_step_fn(simulator, train=False)

    num_train_steps = config.training.n_iters
    print("num_train_steps", num_train_steps)

    # In case there are multiple hosts (e.g., TPU pods), only log to host 0
    logging.info("Starting training loop at step %d." % (initial_step,))

    for step in range(initial_step, num_train_steps+1):
        try:
            batch, target = next(train_iter)
        except StopIteration:
            train_iter = iter(train_ds)

        batch = batch.to(config.device).float()
        in_tissue, density, total, genes = batch[:, 0:1], batch[:, 1:2], batch[:, 2:3], batch[:, 3:]
        B, N, W, H = genes.shape
        info = (in_tissue.repeat(N,1,1,1), ) #if config.model.conditional else None
        # Execute one training step
        samples = simulator.simulate(genes.reshape(B*N, 1, W, H), in_tissue.repeat(N,1,1,1), shuffle=False)
        loss = train_step_fn(state, samples, info)

        if step % config.training.log_freq == 0:
            logging.info("step: %d, training_loss: %.5e" % (step, loss.item()))
            writer.add_scalar("training_loss", loss, step)

        # Save a temporary checkpoint to resume training after pre-emption periodically
        if step != 0 and step % config.training.snapshot_freq_for_preemption == 0:
            save_checkpoint(checkpoint_meta_dir, state)

        # Report the loss on an evaluation dataset periodically
        if step % config.training.eval_freq == 0:
            try:
                batch, _ = next(eval_iter)
            except StopIteration:
                eval_iter = iter(eval_ds)
            batch = batch.to(config.device).float()
            in_tissue, density, total, genes = batch[:, 0:1], batch[:, 1:2], batch[:, 2:3], batch[:, 3:]
            B, N, W, H = genes.shape
            info = (in_tissue.repeat(N,1,1,1), ) #if config.model.conditional else None
            samples = simulator.simulate(genes.reshape(B*N, 1, W, H), in_tissue.repeat(N,1,1,1), shuffle=False)
            eval_loss = eval_step_fn(state, samples, info)
            logging.info("step: %d, eval_loss: %.5e" % (step, eval_loss.item()))
            writer.add_scalar("eval_loss", eval_loss.item(), step)

        # Save a checkpoint periodically and generate samples if needed
        if step != 0 and step % config.training.snapshot_freq == 0 or step == num_train_steps:
            # Save the checkpoint.
            save_step = step // config.training.snapshot_freq
            save_checkpoint(os.path.join(checkpoint_dir, f'checkpoint_{save_step}.pth'), state)
            print(f">>> checkpoint_{save_step}.pth saved")


if __name__ == '__main__':

    import datasets
    from config.default_configs import get_config
    config = get_config()
    config.training.batch_size = 1
    config.data.poisson_ratio_max = 0.1
    config.param.t0 = 0.0

    workdir = 'workdir/adjoint'
    checkpoint_meta_dir = os.path.join(workdir, "checkpoints-meta", "checkpoint.pth")
    simulator = Simulator(config)
    model = unet_lite.Unet(config).to(config.device)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)

    train_ds, eval_ds = datasets.get_dataset(config)
    batch, target = next(iter(eval_ds))

    batch = batch.to(config.device).float()
    in_tissue, density, total, genes = batch[:, 0:1], batch[:, 1:2], batch[:, 2:3], batch[:, 3:]
    B, N, W, H = genes.shape
    samples = simulator.simulate(genes.reshape(B*N, 1, W, H), in_tissue.repeat(N,1,1,1), shuffle=False)
    info = (in_tissue.repeat(N,1,1,1), ) #if config.model.conditional else None

    mode = 1
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
            im = ax.imshow(total[0, 0].cpu(), vmin=vmin, vmax=vmax)

            # Add colorbar in the final column
            cbar_ax = fig.add_subplot(spec[0, -1])
            fig.colorbar(im, cax=cbar_ax)

            os.makedirs(f"{workdir}/plots/reverse", exist_ok=True)
            plt.savefig(f"{workdir}/plots/reverse/reverse_blk | t={state.t.item():.2f}d.png")
            print(total[0, 0].sum(), sol[-1][0, 0].sum())
        
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
            axe[1][3].imshow(total[0, 0].cpu())

            p = pred[0, 0].abs()
            g = df_dt[0, 0].abs()
            print(p.mean(), g.mean(), p.mean()/g.mean())
            tl.append(sample.t.item())
            r.append((p.mean()/g.mean()).item())

            plt.savefig(f"{workdir}/plots/pred/simulate i={idx+1} | t={t.item():.2f}.png")
            plt.close()

        d = {"t":tl, "r":r}
        torch.save(d, 'analysis.pth')
