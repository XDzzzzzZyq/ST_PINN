from op import ns_step
import torch

class Simulator:
    class State:
        def __init__(self,
            t: float,
            f: torch.Tensor,
            v: torch.Tensor,
            p: torch.Tensor,
            dfdt: torch.Tensor
        ):
            self.t = t
            self.f = f
            self.v = v
            self.p = p
            self.dfdt = dfdt


    def __init__(self, config):
        self.param = config.param
        self.sample_per_sol = config.training.sample_per_sol

    def simulate(self, genes, in_tissue):
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

        return result
