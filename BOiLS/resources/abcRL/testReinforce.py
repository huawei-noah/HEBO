##
# @file testReinforce.py
# @author Keren Zhu
# @date 10/31/2019
# @brief The main for test REINFORCE
#

import os
import sys
from datetime import datetime
from pathlib import Path

ROOT_PROJECT = str(Path(os.path.realpath(__file__)).parent.parent)
sys.path[0] = ROOT_PROJECT

from utils.utils_save import get_storage_root, save_w_pickle
import resources.abcRL.reinforce as RF
from resources.abcRL.env import EnvGraph

import time


class AbcReturn:
    def __init__(self, returns):
        self.numNodes = float(returns[0])
        self.level = float(returns[1])

    def __lt__(self, other):
        if (int(self.level) == int(other.level)):
            return self.numNodes < other.numNodes
        else:
            return self.level < other.level

    def __eq__(self, other):
        return int(self.level) == int(other.level) and int(self.numNodes) == int(self.numNodes)


def testReinforce(filename, ben):
    now = datetime.now()
    ref_t = time.time()
    times = []
    dateTime = now.strftime("%m/%d/%Y, %H:%M:%S") + "\n"
    print("Time ", dateTime)
    env = EnvGraph(filename)
    # vApprox = Linear(env.dimState(), env.numActions())
    vApprox = RF.PiApprox(env.dimState(), env.numActions(), 8e-4, RF.FcModelGraph)
    # baseline = RF.Baseline(0)
    vbaseline = RF.BaselineVApprox(env.dimState(), 3e-3, RF.FcModel)
    reinforce = RF.Reinforce(env, 0.9, vApprox, vbaseline)

    lastfive = []
    lines = []
    resultName = os.path.join(get_storage_root(), 'abcRL', ben)
    os.makedirs(resultName, exist_ok=True)
    for idx in range(200):
        returns = reinforce.episode(phaseTrain=True)
        seqLen = reinforce.lenSeq
        line = "iter " + str(idx) + " returns " + str(returns) + " seq Length " + str(seqLen)
        lines.append(line)
        if idx >= 195:
            lastfive.append(AbcReturn(returns))
        print(line)
        times.append(time.time() - ref_t)
        ref_t = time.time()
        # reinforce.replay()
        if idx % 10 == 9:
            print(f'save times and lines in {resultName}')
            save_w_pickle(times, resultName, 'exec_times.pkl')
            save_w_pickle(lines, resultName, 'lines.pkl')

    # lastfive.sort(key=lambda x : x.level)
    lastfive = sorted(lastfive)
    with open(os.path.join(resultName, 'best.txt'), 'a') as andLog:
        line = ""
        line += str(lastfive[0].numNodes)
        line += " "
        line += str(lastfive[0].level)
        line += "\n"
        andLog.write(line)
    rewards = reinforce.sum_rewards


if __name__ == "__main__":
    """
    env = Env("./data/i10.aig")
    vbaseline = RF.BaselineVApprox(4, 3e-3, RF.FcModel)
    for i in range(10000000):
        with open('log', 'a', 0) as outLog:
            line = "iter  "+ str(i) + "\n"
            outLog.write(line)
        vbaseline.update(np.array([2675.0 / 2675, 50.0 / 50, 2675. / 2675, 50.0 / 50]), 422.5518 / 2675)
        vbaseline.update(np.array([2282. / 2675,   47. / 50, 2675. / 2675,   47. / 50]), 29.8503 / 2675)
        vbaseline.update(np.array([2264. / 2675,   45. / 50, 2282. / 2675,   45. / 50]), 11.97 / 2675)
        vbaseline.update(np.array([2255. / 2675,   44. / 50, 2264. / 2675,   44. / 50]), 3 / 2675)
    """
    # testReinforce("./data/MCNC/Combinational/blif/dalu.blif", "dalu")
    # testReinforce("./data/MCNC/Combinational/blif/prom1.blif", "prom1")
    # testReinforce("./data/MCNC/Combinational/blif/mainpla.blif", "mainpla")
    # testReinforce("./data/ISCAS/blif/c5315.blif", "c5315")
    # testReinforce("./data/ISCAS/blif/c6288.blif", "c6288")
    # testReinforce("./data/MCNC/Combinational/blif/apex1.blif", "apex1")
    # testReinforce("./data/MCNC/Combinational/blif/bc0.blif", "bc0")
    # testReinforce("./data/i10.aig", "i10")
    # testReinforce("./data/ISCAS/blif/c1355.blif", "c1355")
    # testReinforce("./data/ISCAS/blif/c7552.blif", "c7552")

    # aig_path = "./data/MCNC/Combinational/blif/k2.blif"
    for designs in ['adder', 'square', 'sin']:
        aig_path = f"./data/epfl-benchmark/arithmetic/{designs}.blif"
        aig_path = os.path.join(ROOT_PROJECT, aig_path)
        testReinforce(aig_path, designs)
