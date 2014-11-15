# Mario Environment Loader

import rlglue.RLGlue as RLGlue
from consoleTrainerHelper import *

def main():
	whichTrainingMDP = 1
	loadMario(True, True, 121, 0, 10, whichTrainingMDP);

	RLGlue.RL_init()
	episodesToRun = 10
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
