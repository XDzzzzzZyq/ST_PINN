from op import ns_step
import torch
import random, math

class NODE(torch.nn.Module):
    def __init__(self, model, info=None):
        super(NODE, self).__init__()
        self.model = model
        self.info = info
    def forward(self, t, f):
        if t.ndim == 0:
            t = t.unsqueeze(0)
        if self.info is not None:
            f = torch.cat([*self.info, f], dim=1)
        df_dt = self.model(f, t.to(f.device))
        return df_dt

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
        v = (1-in_tissue).repeat(1,2,1,1) * random.uniform(self.param.v_min, self.param.v_max)
        p = in_tissue * random.uniform(self.param.p_min, self.param.p_max)

        # TODO: Spatially variate Re 
        Re = math.exp(random.uniform(math.log(self.param.Re_min), math.log(self.param.Re_max)))
        dt, dx = self.param.dt, self.param.dx
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
    
    def reverse(self, model, state, info, rtol=1e-7, atol=1e-9):
        from torchdiffeq import odeint

        t = torch.linspace(state.t.item(), self.param.t0, 6)
        node = NODE(model, info)
        noise = torch.randn_like(state.f) * 0.01 * info[0]
        with torch.no_grad():
            sol = odeint(node, state.f + noise, t, rtol=rtol, atol=atol)
            return sol

    def reverse_euler(self, model, state, info):
        num_steps = int((state.t.item() - self.param.t0) / self.param.dt)
        f = state.f
        node = NODE(model, info)
        result = [state.f]
        with torch.no_grad():
            for idx, t in enumerate(torch.linspace(state.t.item(), self.param.t0, num_steps)):
                f = f - node(t, f) * self.param.dt
                if idx % (num_steps//5) == 0:
                    result.append(f)
        result.append(f)
        return result

    def reverse_shooting(self, model, state1, state2, info, min_step=1, max_step=6):
        num_steps = int((state1.t.item() - state2.t.item()) / self.param.dt)
        num_steps = min(max(num_steps, min_step), max_step)
        f = state1.f
        node = NODE(model, info)
        ts = torch.linspace(state1.t.item(), state2.t.item(), num_steps+1)
        dt = abs(ts[1] - ts[0])
        for t in ts[:-1]:
            df_dt = node(t, f)
            f = f - df_dt * dt
        return f


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