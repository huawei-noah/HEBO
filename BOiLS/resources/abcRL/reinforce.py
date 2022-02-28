##
# @file reinforce.py
# @author Keren Zhu
# @date 10/30/2019
# @brief The REINFORCE algorithm
#

import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
import bisect
import random
from dgl.nn.pytorch import GraphConv
import dgl

from resources.abcRL.env import EnvGraph


class GCN(torch.nn.Module):
    def __init__(self, in_feats, hidden_size, out_len, allow_zero_in_degree: bool = True):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size, allow_zero_in_degree=allow_zero_in_degree)
        self.conv2 = GraphConv(hidden_size, hidden_size, allow_zero_in_degree=allow_zero_in_degree)
        self.conv3 = GraphConv(hidden_size, hidden_size, allow_zero_in_degree=allow_zero_in_degree)
        self.conv4 = GraphConv(hidden_size, out_len, allow_zero_in_degree=allow_zero_in_degree)

    def forward(self, g):
        h = self.conv1(g, g.ndata['feat'])
        h = torch.relu(h)
        h = self.conv2(g, h)
        h = torch.relu(h)
        h = self.conv3(g, h)
        h = torch.relu(h)
        h = self.conv4(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return torch.squeeze(hg)


class FcModel(nn.Module):
    def __init__(self, numFeats, outChs):
        super(FcModel, self).__init__()
        self._numFeats = numFeats
        self._outChs = outChs
        self.fc1 = nn.Linear(numFeats, 32)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(32, outChs)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x


class FcModelGraph(nn.Module):
    def __init__(self, numFeats, outChs):
        super(FcModelGraph, self).__init__()
        self._numFeats = numFeats
        self._outChs = outChs
        self.fc1 = nn.Linear(numFeats, 32 - 4)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(32, 32)
        self.act2 = nn.ReLU()
        self.fc3 = nn.Linear(32, outChs)
        self.gcn = GCN(6, 12, 4)

    def forward(self, x, graph):
        graph_state = self.gcn(graph)
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(torch.cat((x, graph_state), 0))
        x = self.act2(x)
        x = self.fc3(x)
        return x


class PiApprox(object):
    """
    n dimensional continous states
    m discret actions
    """

    def __init__(self, dimStates, numActs, alpha, network):
        """
        @brief approximate policy pi(. | st)
        @param dimStates: Number of dimensions of state space
        @param numActs: Number of the discrete actions
        @param alpha: learning rate
        @param network: a pytorch model
        """
        self._dimStates = dimStates
        self._numActs = numActs
        self._alpha = alpha
        self._network = network(dimStates, numActs)
        # self._network.cuda()
        self._optimizer = torch.optim.Adam(self._network.parameters(), alpha, [0.9, 0.999])
        # self.tau = 0.5  # temperature for gumbel_softmax

    def __call__(self, s, graph, phaseTrain=True):
        self._network.eval()
        # s = torch.from_numpy(s).float() #.cuda()
        out = self._network(s, graph)
        # interval = (out.max() - out.min()).data
        # out = (out - out.min().data) / interval
        # normal = self.normalizeLogits(out)
        # probs = F.gumbel_softmax(out, dim=-1, tau = self.tau, hard=True)
        probs = F.softmax(out, dim=-1)
        if phaseTrain:
            m = Categorical(probs)
            action = m.sample()
        else:
            action = torch.argmax(out)

        return action.data.item()

    def update(self, s, graph, a, gammaT, delta):
        self._network.train()
        prob = self._network(s, graph)  # .cuda())
        # logProb = -F.gumbel_softmax(prob, dim=-1, tau = self.tau, hard=True)
        logProb = torch.log_softmax(prob, dim=-1)
        loss = -gammaT * delta * logProb
        self._optimizer.zero_grad()
        loss[a].backward()
        self._optimizer.step()

    def episode(self):
        # self._tau = self._tau * 0.98
        pass

    def save(self, path: str) -> None:
        torch.save(self._network.state_dict(), path)


class Baseline(object):
    """
    The dumbest baseline: a constant for every state
    """

    def __init__(self, b):
        self.b = b

    def __call__(self, s):
        return self.b

    def update(self, s, G):
        pass


class BaselineVApprox(object):
    """
    The baseline with approximation of state value V
    """

    def __init__(self, dimStates, alpha, network):
        """
        @brief approximate policy pi(. | st)
        @param dimStates: Number of dimensions of state space
        @param numActs: Number of the discret actions
        @param alpha: learning rate
        @param network: a pytorch model
        """
        self._dimStates = dimStates
        self._alpha = alpha
        self._network = network(dimStates, 1)
        # self._network.cuda()
        self._optimizer = torch.optim.Adam(self._network.parameters(), alpha, [0.9, 0.999])

    def __call__(self, state):
        self._network.eval()
        return self.value(state).data

    def value(self, state):
        # state = torch.from_numpy(state).float()
        out = self._network(state)
        return out

    def update(self, state, G):
        self._network.train()
        vApprox = self.value(state)
        loss = (torch.tensor([G]) - vApprox[-1]) ** 2 / 2
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def save(self, path: str) -> None:
        torch.save(self._network.state_dict(), path)


class Trajectory(object):
    """
    @brief The experience of a trajectory
    """

    def __init__(self, states, rewards, actions, value):
        self.states = states
        self.rewards = rewards
        self.actions = actions
        self.value = value

    def __lt__(self, other):
        return self.value < other.value


class Reinforce(object):
    def __init__(self, env: EnvGraph, gamma: float, pi, baseline, mem_length: int = 4):
        self._env = env
        self._gamma = gamma
        self._pi = pi
        self._baseline = baseline
        self.mem_trajectory = []  # the memorized trajectories. sorted by value
        self.mem_length = mem_length
        self.sum_rewards = []

    def genTrajectory(self, phaseTrain: bool = True):
        self._env.reset()
        state = self._env.state()
        term = False
        states, rewards, actions = [], [0], []
        while not term:
            action = self._pi(state[0], state[1], phaseTrain)
            term = self._env.takeAction(action)
            nextState = self._env.state()
            nextReward = self._env.reward()
            states.append(state)
            rewards.append(nextReward)
            actions.append(action)
            state = nextState
            if len(states) > 20:
                term = True
        return Trajectory(states, rewards, actions, self._env.curStatsValue())

    def episode(self, phaseTrain=True):
        trajectory = self.genTrajectory(
            phaseTrain=phaseTrain)  # Generate a trajectory of episode of states, actions, rewards
        self.updateTrajectory(trajectory, phaseTrain)
        self._pi.episode()
        return self._env.returns()

    def updateTrajectory(self, trajectory, phaseTrain=True):
        states = trajectory.states
        rewards = trajectory.rewards
        actions = trajectory.actions
        bisect.insort(self.mem_trajectory, trajectory)  # memorize this trajectory
        self.lenSeq = len(states)  # Length of the episode
        for tIdx in range(self.lenSeq):
            G = sum(self._gamma ** (k - tIdx - 1) * rewards[k] for k in range(tIdx + 1, self.lenSeq + 1))
            state = states[tIdx]
            action = actions[tIdx]
            baseline = self._baseline(state[0])
            delta = G - baseline
            self._baseline.update(state[0], G)
            self._pi.update(state[0], state[1], action, self._gamma ** tIdx, delta)
        self.sum_rewards.append(sum(rewards))
        print(sum(rewards))

    def replay(self):
        for idx in range(min(self.mem_length, int(len(self.mem_trajectory) / 10))):
            if len(self.mem_trajectory) / 10 < 1:
                return
            upper = min(len(self.mem_trajectory) / 10, 30)
            r1 = random.randint(0, upper)
            self.updateTrajectory(self.mem_trajectory[idx])
