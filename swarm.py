import math
from abc import ABC
from typing import Tuple
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Swarm(ABC):
    def __init__(
            self, num_agents: int, dim: int, *,
            init_velocity_scale: float=0.01,
            init_position_scale: float=1,
        ):
        self.num_agents = num_agents
        self.dim = dim
        self.positions = (torch.rand((self.num_agents, self.dim),
                                     dtype=torch.double) * (init_position_scale))
        self.velocities = init_velocity_scale * \
            (torch.rand((self.num_agents, self.dim), dtype=torch.double) - 0.5)

    def step(self):
        pass


    def simulate(self, n_steps):
        assert self.dim == 2
        fig, ax  = plt.subplots()
        ln = ax.scatter(self.positions[:, 0], self.positions[:, 1], c=torch.arange(0, self.num_agents))

        def init():
            return ln,

        def update(_):
            self.step()
            ln.set_offsets(self.positions)
            # ax.relim()
            x, y = self.positions[:, 0], self.positions[:, 1]
            ax.set_xlim(x.min() - 0.25, x.max() + 0.25)
            ax.set_ylim(y.min() - 0.25, y.max() + 0.25)
            return ln,

        anim = FuncAnimation(fig, update, frames=torch.arange(0, n_steps), init_func=init, blit=False)
        plt.show()


class BirdFlock(Swarm):
    def __init__(
        self, *args,
        p: float = 2.,
        **kwargs,
    ):
        super(BirdFlock, self).__init__(*args, **kwargs)
        assert p >= 0
        self.p = p

    def peturb(self):
        return 0.

    def step(self):
        self.positions += self.velocities
        dists = torch.cdist(self.positions, self.positions, self.p)
        dists[torch.arange(self.num_agents), torch.arange(self.num_agents)] = math.inf
        nearest_neighbors = dists.min(dim=1).indices
        self.velocities = self.velocities[nearest_neighbors] + self.peturb()

class CrazyBirdFlock(BirdFlock):
    def __init__(
        self, *args,
        craziness: float = 0.001,
        **kwargs,
    ):
        super(CrazyBirdFlock, self).__init__(*args, **kwargs)
        self.craziness = craziness

    def peturb(self):
        return (torch.rand(self.velocities.shape) - 0.5) * self.craziness

class RoostingBirdFlock(BirdFlock):
    def __init__(
        self, *args,
        roost: Tuple[float, float] = (0.5, 0.5),
        roost_strength: float = 0.1,
        **kwargs,
    ):
        super(RoostingBirdFlock, self).__init__(*args, **kwargs)
        self.roost_strength = roost_strength
        self.roost = torch.tensor(roost, dtype = torch.double)

    def peturb(self):
        roost_dist = self.roost - self.positions
        return (roost_dist / roost_dist.norm(dim=1).reshape(-1, 1)) * self.roost_strength


class CornfieldSwarm(Swarm):
    def __init__(
        self, *args,
        cornfield: Tuple[float, ...] = (1, 1),
        p_increment: float = 0.001,
        g_increment: float = 0.001,
        **kwargs,
    ):
        super(CornfieldSwarm, self).__init__(*args, **kwargs)
        assert len(cornfield) == self.dim
        self.cornfield = torch.tensor(cornfield, dtype=torch.double)
        self.p_increment = p_increment 
        self.g_increment = g_increment 
        self.p_best_loss = torch.full((self.num_agents,), math.inf, dtype=torch.double)
        self.p_best_pos = self.positions.clone()

    def loss(self, pos: torch.Tensor) -> torch.Tensor:
        return (pos - self.cornfield).abs().sum(dim=1)

    def step(self):
        self.positions += self.velocities
        loss = self.loss(self.positions)
        p_improvement = loss < self.p_best_loss
        self.p_best_loss = torch.where(p_improvement, loss, self.p_best_loss)
        self.p_best_pos = torch.where(p_improvement.reshape(-1, 1), self.positions, self.p_best_pos)
        g_best_loss, g_best_idx = self.p_best_loss.min(dim=0)
        g_best_pos = self.p_best_pos[g_best_idx]
        local_adjustment = (self.positions - self.p_best_pos).sign() * -1 * torch.rand(self.positions.shape, dtype=torch.double) * self.p_increment
        global_adjustment = (self.positions - g_best_pos).sign() * -1 * torch.rand(self.positions.shape, dtype=torch.double) * self.g_increment
        self.velocities += local_adjustment + global_adjustment



if __name__ == '__main__':
    flock = CornfieldSwarm(50, 2, g_increment=0.01, p_increment = 0.005, init_velocity_scale=0.)
    flock.simulate(100)
