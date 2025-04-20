import gc
import io
import os
import glob
import time

import logging
from model import unet, unet_lite
from model.ema import ExponentialMovingAverage
from simulate import Simulator
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
    train_step_fn = losses.get_step_fn(simulator, train=True, optimize_fn=optimize_fn)
    eval_step_fn = losses.get_step_fn(simulator, train=False)

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
        in_tissue, total, genes = batch[:, 0:1], batch[:, 1:2], batch[:, 2:]
        N = genes.shape[1]                      # TODO: flatten all multi-genes
        info = (in_tissue, total)
        # Execute one training step
        samples = simulator.simulate(genes, in_tissue)
        loss = 0
        for sample in samples:
            loss += train_step_fn(state, sample, info)
        loss /= config.training.sample_per_sol

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
            in_tissue, total, genes = batch[:, 0:1], batch[:, 1:2], batch[:, 2:]
            N = genes.shape[1]                      # TODO: flatten all multi-genes
            info = (in_tissue, total)
            samples = simulator.simulate(genes, in_tissue)
            eval_loss = 0
            for sample in samples:
                eval_loss += eval_step_fn(state, sample, info)
            eval_loss /= config.training.sample_per_sol
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
    from config.default_configs import get_default_configs
    config = get_default_configs()

    checkpoint_meta_dir = os.path.join("workdir/test", "checkpoints-meta", "checkpoint.pth")
    simulator = Simulator(config)
    model = unet_lite.Unet(config).to(config.device)
    simulator = Simulator(config)
    ema = ExponentialMovingAverage(model.parameters(), decay=config.model.ema_rate)
    optimizer = losses.get_optimizer(config, model.parameters())
    state = dict(optimizer=optimizer, model=model, ema=ema, step=0)
    state = restore_checkpoint(checkpoint_meta_dir, state, config.device)

    train_ds, eval_ds = datasets.get_dataset(config)
    batch, target = next(iter(eval_ds))

    batch = batch.to(config.device).float()
    in_tissue, total, genes = batch[:, 0:1], batch[:, 1:2], batch[:, 2:]
    samples = simulator.simulate(genes, in_tissue)

    if True:
        state = samples[0]
        print(state.t.item())
        sol = simulator.reverse(model, state.f, state.t)

        import matplotlib.pyplot as plt
        fig, axe = plt.subplots(nrows=1, ncols=len(sol), figsize=(30, 10))
        for i, ax in enumerate(axe):
            ax.imshow(sol[i][0, 0].cpu())

        plt.savefig(f"plots/reverse/reverse | t={state.t.item():.2f}.png")
    else:
        for idx, sample in enumerate(samples):
            t, f, v, p, df_dt = sample.get()
            with torch.no_grad():
                pred = model(f, t)

            import matplotlib.pyplot as plt
            fig, axe = plt.subplots(nrows=2, ncols=4, figsize=(30, 10))
            axe[0][0].imshow(f[0, 0].cpu())
            axe[0][1].imshow(v[0, 0].cpu())
            axe[0][2].imshow(v[0, 1].cpu())
            axe[0][3].imshow(p[0, 0].cpu())

            axe[1][0].imshow(df_dt[0, 0].cpu())
            axe[1][1].imshow(pred[0, 0].cpu())
            axe[1][2].imshow((df_dt[0, 0]-pred[0, 0]).cpu())
            axe[1][3].imshow(total[0, 0].cpu())

            plt.savefig(f"plots/simulate/simulate i={idx+1} | t={t.item():.2f}.png")