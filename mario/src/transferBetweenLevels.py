#!/usr/bin/python

# Mario Environment Loader

import random
import numpy as np
import rlglue.RLGlue as RLGlue
import matplotlib.pyplot as plt
from consoleTrainerHelper import *


def trainAgent():
    whichTrainingMDP = 0
    episodesToRun = 1
    totalSteps = 0
    exp = 1.0
    raw_results = []
    print RLGlue.RL_agent_message("freeze_learning")
    for i in xrange(episodesToRun):
        if (i % 100 == 0):
            loadMario(True, False, i/100, 0, 1, whichTrainingMDP)
        RLGlue.RL_agent_message("set_exploring " + str(exp))
        RLGlue.RL_episode(2000)
        thisSteps = RLGlue.RL_num_steps()
        # print "Total steps in episode %d is %d" % (i, thisSteps)
        thisReturn = RLGlue.RL_return()
        if (thisReturn > 50.0):
            thisReturn = 10.0
        # print "Total return in episode %d is %f" % (i, thisReturn)
        raw_results.append(thisReturn)
        totalSteps += thisSteps
    print RLGlue.RL_agent_message("unfreeze_learning")
    print "Total steps : %d\n" % (totalSteps)
    results1 = []
    for i in xrange(100, episodesToRun):
        if (i % 100 == 0):
            results1.append(sum(raw_results[i-100:i])/100.0)

    raw_results = []
    for i in xrange(episodesToRun):
        if (i % 100 == 0):
            loadMario(True, False, i/100, 0, 1, whichTrainingMDP)
            if (exp > 0.1):
                exp -= 0.05
            RLGlue.RL_agent_message("set_exploring " + str(exp))
        RLGlue.RL_episode(2000)
        thisSteps = RLGlue.RL_num_steps()
        # print "Total steps in episode %d is %d" % (i, thisSteps)
        thisReturn = RLGlue.RL_return()
        if (thisReturn > 50.0):
            thisReturn = 10.0
        # print "Total return in episode %d is %f" % (i, thisReturn)
        raw_results.append(thisReturn)
        totalSteps += thisSteps
    print "Total steps : %d\n" % (totalSteps)
    results2 = []
    for i in xrange(100, episodesToRun):
        if (i % 100 == 0):
            results2.append(sum(raw_results[i-100:i])/100.0)

    plt.plot(results1, color='red', label='Random')
    plt.plot(results2, color='blue', label='Neural Q-Network')
    plt.plot(np.array(results2) - np.array(results1), color='green', label='Difference')
    plt.xlabel('Level Seed')
    plt.ylabel('Mean Total Reward over 100 Episodes')
    plt.legend()
    plt.show()

    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax1.plot(results1, color='red', label='Random')
    # ax1.plot(results2, color='blue', label='Neural Q-Network')
    # ax2.plot(np.array(results2) - np.array(results1), color='green', label='Difference')

    # ax1.set_xlabel('Level Seed')
    # ax1.set_ylabel('Mean Total Reward over 100 Episodes')
    # ax2.set_ylabel('Mean Q-NN Improvement over 100 Episodes')
    # plt.legend()
    # plt.show()


def testAgent():
    episodesToRun = 50
    totalSteps = 0
    RLGlue.RL_agent_message("freeze learning")
    for i in xrange(episodesToRun):
        RLGlue.RL_episode(2000)
        thisSteps = RLGlue.RL_num_steps()
        # thisReturn = RLGlue.RL_return()
        # if (i % 100 == 0):
        #     print "Total steps in episode %d is %fd" % (i, thisSteps)
        #     print "Total return in episode %d is %f" % (i, thisReturn)
        totalSteps += thisSteps
    print "Total steps : %d\n" % (totalSteps)
    RLGlue.RL_agent_message("unfreeze learning")


def main():
    whichTrainingMDP = 0
    '''
    Parameter definition:
    fast - determines if Mario runs very fast or at playable-speed.
    Set it to true to train your agent,
    false if you want to actually see what is going on.
    dark - make Mario visible or not. Set it to true to make it invisible when training.
    levelSeed - determines Marios behavior
    levelType - 0..2: outdoors/subterranean/other
    levelDifficulty - 0..10, how hard it is.
    instance - 0..9, determines which Mario you run.
    '''
    loadMario(True, False, 3, 0, 1, whichTrainingMDP)

    RLGlue.RL_init()

        #RLGlue.RL_agent_message("load_policy agents/exampleAgent.dat")

    trainAgent()

    #RLGlue.RL_agent_message("save_policy agents/exampleAgent.dat")

    #testAgent()

    RLGlue.RL_cleanup()

if __name__ == "__main__":
    main()
