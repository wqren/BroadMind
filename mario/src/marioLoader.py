#!/usr/bin/python

# Mario Environment Loader

import random
import rlglue.RLGlue as RLGlue
from consoleTrainerHelper import *

def trainAgent():
        episodesToRun = 50
        totalSteps = 0
        for i in range(episodesToRun):
                RLGlue.RL_episode(2000)
		thisSteps = RLGlue.RL_num_steps()
                print "Total steps in episode %d is %d" %(i, thisSteps)
                thisReturn = RLGlue.RL_return()
                print "Total return in episode %d is %d" %(i, thisReturn)
                totalSteps += thisSteps
        print "Total steps : %d\n" % (totalSteps)

def testAgent():
        episodesToRun = 50
	totalSteps = 0
        RLGlue.RL_agent_message("freeze learning");
        for i in range(episodesToRun):
                RLGlue.RL_episode(2000)
                thisSteps = RLGlue.RL_num_steps()
                print "Total steps in episode %d is %d" %(i, thisSteps)
                thisReturn = RLGlue.RL_return()
                print "Total return in episode %d is %d" %(i, thisReturn)
                totalSteps += thisSteps
        print "Total steps : %d\n" % (totalSteps)
	RLGlue.RL_agent_message("unfreeze learning");

def main():
	whichTrainingMDP = 1
        '''
	Parameter definition:
	fast - determines if Mario runs very fast or at playable-speed. Set it to true to train your agent, false if you want to actually see what is going on.
	dark - make Mario visible or not. Set it to true to make it invisible when training.
	levelSeed - determines Marios behavior 
	levelType - 0..2: outdoors/subterranean/other
	levelDifficulty - 0..10, how hard it is. 
	instance - 0..9, determines which Mario you run.	
	'''
	loadMario(False, False, random.randint(0,1000), 0, 10, whichTrainingMDP);

	RLGlue.RL_init()

	trainAgent()

	#testAgent()

	RLGlue.RL_cleanup()

if __name__ == "__main__":
	main()
