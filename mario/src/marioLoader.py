#!/usr/bin/python

# Mario Environment Loader

import rlglue.RLGlue as RLGlue
from consoleTrainerHelper import *

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
	loadMario(False, False, 121, 0, 10, whichTrainingMDP);

	RLGlue.RL_init()
	episodesToRun = 1000
	totalSteps = 0
	for i in range(episodesToRun):
		RLGlue.RL_episode(20000)
		thisSteps = RLGlue.RL_num_steps()
		print "Total steps in episode %d is %d" %(i, thisSteps)
		totalSteps += thisSteps
	print "Total steps : %d\n" % (totalSteps)
	RLGlue.RL_cleanup()

if __name__ == "__main__":
	main()
