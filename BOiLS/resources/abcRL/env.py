##
# @file env.py
# @author Keren Zhu
# @date 10/25/2019
# @brief The environment classes
#

# 2021.11.10-Remove one comment
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

import abc_py as abcPy
import numpy as np
from resources import abcRL as GE
import torch


class EnvNaive2(object):
    """
    @brief the overall concept of environment, the different. use the compress2rs as target
    """

    def __init__(self, aigfile):
        self._abc = abcPy.AbcInterface()
        self._aigfile = aigfile
        self._abc.start()
        self.lenSeq = 0
        self._abc.read(self._aigfile)
        initStats = self._abc.aigStats()  # The initial AIG statistics
        self.initNumAnd = float(initStats.numAnd)
        self.initLev = float(initStats.lev)
        self.resyn2()  # run a compress2rs as target
        self.resyn2()
        resyn2Stats = self._abc.aigStats()
        totalReward = self.statValue(initStats) - self.statValue(resyn2Stats)
        self._rewardBaseline = totalReward / 20.0  # 18 is the length of compress2rs sequence
        print("baseline num AND ", resyn2Stats.numAnd, " total reward ", totalReward)

    def resyn2(self):
        self._abc.balance(l=False)
        self._abc.rewrite(l=False)
        self._abc.refactor(l=False)
        self._abc.balance(l=False)
        self._abc.rewrite(l=False)
        self._abc.rewrite(l=False, z=True)
        self._abc.balance(l=False)
        self._abc.refactor(l=False, z=True)
        self._abc.rewrite(l=False, z=True)
        self._abc.balance(l=False)

    def reset(self):
        self.lenSeq = 0
        self._abc.end()
        self._abc.start()
        self._abc.read(self._aigfile)
        self._lastStats = self._abc.aigStats()  # The initial AIG statistics
        self._curStats = self._lastStats  # the current AIG statistics
        self.lastAct = self.numActions() - 1
        self.lastAct2 = self.numActions() - 1
        self.lastAct3 = self.numActions() - 1
        self.lastAct4 = self.numActions() - 1
        self.actsTaken = np.zeros(self.numActions())
        return self.state()

    def close(self):
        self.reset()

    def step(self, actionIdx):
        self.takeAction(actionIdx)
        nextState = self.state()
        reward = self.reward()
        done = False
        if (self.lenSeq >= 20):
            done = True
        return nextState, reward, done, 0

    def takeAction(self, actionIdx):
        """
        @return true: episode is end
        """
        self.lastAct4 = self.lastAct3
        self.lastAct3 = self.lastAct2
        self.lastAct2 = self.lastAct
        self.lastAct = actionIdx
        # self.actsTaken[actionIdx] += 1
        self.lenSeq += 1
        if actionIdx == 0:
            self._abc.balance(l=False)  # b
        elif actionIdx == 1:
            self._abc.rewrite(l=False)  # rw
        elif actionIdx == 2:
            self._abc.refactor(l=False)  # rf
        elif actionIdx == 3:
            self._abc.rewrite(l=False, z=True)  # rw -z
        elif actionIdx == 4:
            self._abc.refactor(l=False, z=True)  # rs
        elif actionIdx == 5:
            self._abc.end()
            return True
        else:
            raise ValueError(actionIdx)

        # update the statitics
        self._lastStats = self._curStats
        self._curStats = self._abc.aigStats()
        return False

    def state(self):
        """
        @brief current state
        """
        oneHotAct = np.zeros(self.numActions())
        np.put(oneHotAct, self.lastAct, 1)
        lastOneHotActs = np.zeros(self.numActions())
        lastOneHotActs[self.lastAct2] += 1 / 3
        lastOneHotActs[self.lastAct3] += 1 / 3
        lastOneHotActs[self.lastAct] += 1 / 3
        stateArray = np.array([self._curStats.numAnd / self.initNumAnd, self._curStats.lev / self.initLev,
                               self._lastStats.numAnd / self.initNumAnd, self._lastStats.lev / self.initLev])
        stepArray = np.array([float(self.lenSeq) / 20.0])
        combined = np.concatenate((stateArray, lastOneHotActs, stepArray), axis=-1)
        # combined = np.expand_dims(combined, axis=0)
        # return stateArray.astype(np.float32)
        return torch.from_numpy(combined.astype(np.float32)).float()

    def reward(self):
        if self.lastAct == 5:  # term
            return 0
        return self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline
        # return -self._lastStats.numAnd + self._curStats.numAnd - 1
        if (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.lev < self._lastStats.lev):
            return 2
        elif (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.lev == self._lastStats.lev):
            return 0
        elif (self._curStats.numAnd == self._lastStats.numAnd and self._curStats.lev < self._lastStats.lev):
            return 1
        else:
            return -2

    def numActions(self):
        return 5

    def dimState(self):
        return 4 + self.numActions() * 1 + 1

    def returns(self):
        return [self._curStats.numAnd, self._curStats.lev]

    def statValue(self, stat):
        return float(stat.numAnd) / float(self.initNumAnd)  # + float(stat.lev)  / float(self.initLev)
        # return stat.numAnd + stat.lev * 10

    def curStatsValue(self):
        return self.statValue(self._curStats)

    def seed(self, sd):
        pass

    def compress2rs(self):
        self._abc.compress2rs()


class EnvGraph:
    """
    @brief the overall concept of environment, the different. use the compress2rs as target
    """

    def __init__(self, aigfile):
        self._abc = abcPy.AbcInterface()
        self._aigfile = aigfile
        self._abc.start()
        self._abc.read(self._aigfile)
        initStats = self._abc.aigStats()  # The initial AIG statistics
        self.current_len_seq = 0
        self.initNumAnd = float(initStats.numAnd)
        self.initLev = float(initStats.lev)
        self.resyn2()  # run a compress2rs as target
        self.resyn2()
        resyn2Stats = self._abc.aigStats()
        # TODO: take into account num_nodes and depth
        totalReward = self.statValue(initStats) - self.statValue(resyn2Stats)
        self._rewardBaseline = totalReward / 20.0  # 20 is the length of compress2rs sequence
        print("baseline num AND ", resyn2Stats.numAnd, " total reward ", totalReward)

    def resyn2(self):
        self._abc.balance(l=False)
        self._abc.rewrite(l=False)
        self._abc.refactor(l=False)
        self._abc.balance(l=False)
        self._abc.rewrite(l=False)
        self._abc.rewrite(l=False, z=True)
        self._abc.balance(l=False)
        self._abc.refactor(l=False, z=True)
        self._abc.rewrite(l=False, z=True)
        self._abc.balance(l=False)

    def reset(self):
        self.current_len_seq = 0
        self._abc.end()
        self._abc.start()
        self._abc.read(self._aigfile)
        self._lastStats = self._abc.aigStats()  # The initial AIG statistics
        self._curStats = self._lastStats  # the current AIG statistics
        self.lastAct = self.numActions() - 1
        self.lastAct2 = self.numActions() - 1
        self.lastAct3 = self.numActions() - 1
        self.lastAct4 = self.numActions() - 1
        self.actsTaken = np.zeros(self.numActions())
        return self.state()

    def close(self):
        self.reset()

    def step(self, actionIdx):
        self.takeAction(actionIdx)
        nextState = self.state()
        reward = self.reward()
        done = False
        if (self.current_len_seq >= 20):
            done = True
        return nextState, reward, done, 0

    def takeAction(self, actionIdx):
        """
        @return true: episode is end
        """
        self.lastAct4 = self.lastAct3
        self.lastAct3 = self.lastAct2
        self.lastAct2 = self.lastAct
        self.lastAct = actionIdx
        # self.actsTaken[actionIdx] += 1
        self.current_len_seq += 1

        if actionIdx == 0:
            self._abc.balance(l=False)  # b
        elif actionIdx == 1:
            self._abc.rewrite(l=False)  # rw
        elif actionIdx == 2:
            self._abc.refactor(l=False)  # rf
        elif actionIdx == 3:
            self._abc.rewrite(l=False, z=True)  # rw -z
        elif actionIdx == 4:
            self._abc.refactor(l=False, z=True)  # rs
        elif actionIdx == 5:
            self._abc.end()
            return True
        else:
            raise ValueError(actionIdx)

        # update the statitics
        self._lastStats = self._curStats
        self._curStats = self._abc.aigStats()
        return False

    def state(self):
        """
        @brief current state
        """
        oneHotAct = np.zeros(self.numActions())
        np.put(oneHotAct, self.lastAct, 1)
        lastOneHotActs = np.zeros(self.numActions())
        lastOneHotActs[self.lastAct2] += 1 / 3
        lastOneHotActs[self.lastAct3] += 1 / 3
        lastOneHotActs[self.lastAct] += 1 / 3
        stateArray = np.array([self._curStats.numAnd / self.initNumAnd, self._curStats.lev / self.initLev,
                               self._lastStats.numAnd / self.initNumAnd, self._lastStats.lev / self.initLev])
        stepArray = np.array([float(self.current_len_seq) / 20.0])
        combined = np.concatenate((stateArray, lastOneHotActs, stepArray), axis=-1)
        combined_torch = torch.from_numpy(combined.astype(np.float32)).float()
        graph = GE.extract_dgl_graph(self._abc)
        return combined_torch, graph

    def reward(self):
        # TODO: check reward
        if self.lastAct == 5:  # term
            return 0
        return self.statValue(self._lastStats) - self.statValue(self._curStats) - self._rewardBaseline
        # return -self._lastStats.numAnd + self._curStats.numAnd - 1
        # if (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.lev < self._lastStats.lev):
        #     return 2
        # elif (self._curStats.numAnd < self._lastStats.numAnd and self._curStats.lev == self._lastStats.lev):
        #     return 0
        # elif (self._curStats.numAnd == self._lastStats.numAnd and self._curStats.lev < self._lastStats.lev):
        #     return 1
        # else:
        #     return -2

    def numActions(self):
        # TODO: add actions
        return 5

    def dimState(self):
        return 4 + self.numActions() * 1 + 1

    def returns(self):
        return [self._curStats.numAnd, self._curStats.lev]

    def statValue(self, stat):
        # TODO: handle this
        return float(stat.lev) / float(self.initLev)
        return float(stat.numAnd) / float(self.initNumAnd)  # + float(stat.lev)  / float(self.initLev)
        # return stat.numAnd + stat.lev * 10

    def curStatsValue(self):
        return self.statValue(self._curStats)

    def seed(self, sd):
        pass

    def compress2rs(self):
        self._abc.compress2rs()
