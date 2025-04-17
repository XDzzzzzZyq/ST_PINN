from op import ns_step
import torch
import random

class Simulator:
    class State:
        def __init__(self,
            t: float,
            f: torch.Tensor,
            v: torch.Tensor,
            p: torch.Tensor,
            df_dt: torch.Tensor
        ):
            self.t = t
            self.f = f
            self.v = v
            self.p = p
            self.df_dt = df_dt
        
        def get(self):
            return self.t, self.f, self.v, self.p, self.df_dt


    def __init__(self, config):
        self.param = config.param
        self.sample_per_sol = config.training.sample_per_sol

    def simulate(self, genes, in_tissue, shuffle=True):
        num_steps = int((self.param.t2 - self.param.t1) / self.param.dt)
        assert num_steps >= self.sample_per_sol

        sample_at = torch.randperm(num_steps)[:self.sample_per_sol]

        f = genes                                   # TODO: flatten all multi-genes
        v = (1-in_tissue).repeat(1,2,1,1) * 0.001   # TODO: Random sampling
        p = in_tissue * 0.001

        dt, dx, Re = self.param.dt, self.param.dx, self.param.Re_min
        df_dx, df_dy = ns_step.diff(f, dx)
        dv_dx, dv_dy = ns_step.diff(v, dx)
        
        result = []

        for idx, t in enumerate(torch.linspace(self.param.t1, self.param.t2, num_steps)):
            v, dv_dx, dv_dy = ns_step.update_velocity(v, dv_dx, dv_dy, p, dt, dx, Re)
            # v = ns_step.vorticity_confinement(v, 1.0, dt, dx) # TODO: stability
            p = ns_step.update_pressure(p, v, dt, dx)
            f, df_dx, df_dy, df_dt = ns_step.update_density(f, df_dx, df_dy, v, dt, dx, Re)

            if torch.isnan(df_dt).any():
                print(f"nan detacted at step {idx} : {t}, total sample: {len(result)}")
                return result

            if idx in sample_at:
                result.append(self.State(t[None].to(f.device), f, v, p, df_dt))
        
        if shuffle:
            random.shuffle(result)
        return result

if __name__ == '__main__':

    import datasets
    from config.default_configs import get_default_configs
    config = get_default_configs()

    simulator = Simulator(config)

    train_ds, eval_ds = datasets.get_dataset(config)
    train_iter = iter(train_ds)
    batch, target = next(train_iter)

    batch = batch.to(config.device).float()
    in_tissue, total, genes = batch[:, 0:1], batch[:, 1:2], batch[:, 2:]
    samples = simulator.simulate(genes, in_tissue)

    for idx, sample in enumerate(samples):
        t, f, v, p, df_dt = sample.get()
        import matplotlib.pyplot as plt
        fig, axe = plt.subplots(nrows=2, ncols=4, figsize=(10, 30))
        axe[0][0].imshow(f[0, 0].cpu())
        axe[0][1].imshow(v[0, 0].cpu())
        axe[0][2].imshow(v[0, 1].cpu())
        axe[0][3].imshow(p[0, 0].cpu())

        axe[1][0].imshow(df_dt[0, 0].cpu())
        axe[1][1].imshow(in_tissue[0, 0].cpu())
        axe[1][2].imshow(total[0, 0].cpu())
        axe[1][3].imshow(total[0, 0].cpu())

        plt.savefig(f"plots/simulate/simulate i={idx+1} | t={t.item():.2f}.png")