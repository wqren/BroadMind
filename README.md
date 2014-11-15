BroadMind
=========

CIS 519 term project

##Setting up the environment

1.) Build RLGlue Core
```bash
$ cd external/rlglue-3.04
$ ./configure
$ make
$ sudo make install
```

2.) Build RLGlue C Codec
```bash
$ cd external/c-codec-2.0
$ ./configure
$ make
$ sudo make install
```

3.) Build RLGlue Python Codec
```bash
$ cd external/python-codec/src
$ python setup.py install
```
 Add <path_to_BroadMind>/external/python-codec to PYTHONPATH environment variable


4.) Build ALE (with RLGlue Agent)
Install external libsdl dependency for visualization support.
https://www.libsdl.org
If you choose not to do this step, you must disable USE_SDL in the makefile that you use later. You also will have to skip the "-display_screen true" command line option when executing ALE.
If you run into trouble with this (I did on Mac), try installing using brew:
```bash
$ brew install sdl2
$ brew install sdl_gfx
$ brew install sdl_image
```
```bash
$ cd external/ale0.4_-2.4/ale_0_4
```
 IF OSX -> 
```bash
$ cp makefile.mac makefile
```
 IF UNIX -> 
```bash 
$ cp makefile.unix makefile
```
```bash
$ make
$ cd doc/examples
$ make rlglueAgent
```

5.) (Optional) Build RL 2009 Competition
```bash
$ cd external/15-rl-competition-2009
$ bash install.bash
```

Documentation for using external systems (RLGlue, ALE, 2009 Comp) are in the docs directory. 

##Running Off-the-Shelf Experiments
This project uses the RL-Glue framework as its cornerstone. RL-Glue involves 4 components to run:
- rl_glue (Creates a core server for communication)
- An Agent (e.g. A random agent, an AI for a particular game, etc.)
- An Environment (e.g. ALE with a ROM, Infinite Mario, Tetris)
- An Experiment (Instructions for how to run the process.)

An RL-Glue process does not start until the master hears from the other 3 components.

###2009 AI Competition (Mario)

The 2009 AI Competition has a few out-of-the-box experiments that can be run. In 2 terminals:
```bash
$ cd external/15-rl-competition-2009/trainers/guiTrainerJava
$ bash run.bash
```
```bash
$ cd external/15-rl-competition-2009/agents/marioAgentJava
$ bash run.bash
```

The first executes the main rl_glue core, as well as an environment and experiment (through a GUI loader). The second loads a Mario AI. The GUI does not load until both are connected. Once the GUI is open, load the GeneralizedMario Environment, configure parameters, then select Load Experiment.

Other demo agents are available to run for the other environments. There are additional trainers that can be used without the GUI.

###Arcade Learning Environment

We also have an ALE demonstration through RL-Glue running our Pitfall platformer. In 4 terminals:
```bash
$ rl_glue
```
```bash
$ cd external/ale_0.4-2.4/ale_0_4/doc/examples
$ ./RLGlueAgent
```
```bash
$ cd external/ale_0.4-2.4/ale_0_4/doc/examples
$ ./RLGlueExperiment
```
```bash
$ cd external/ale_0.4-2.4/ale_0_4
$ ./ale -display_screen true -game_controller rlglue roms/pitfall.bin
```

Additional Atari ROMS can be added to the roms directory, though ALE is looking for them to be named a particular way. See http://yavar.naddaf.name/ale/list_of_current_games.html for a list of supported ROMS, and what the expected names are.

##Running Our Experiments

These are instructions for running the agents and experiments for our project.

###Mario
First you must build the Mario experiment.
```bash
cd mario
make
```

Then you can run a python executable that boots up multiple processes for RL-Glue, the mario environment, a mario experiment, and our mario agent.
```bash
python run_mario_exp.py
```

Right now, our Mario agent can parse the observations from the RLGlue interface from the Mario environment, and it selects actions randomly. All of the python files that need to be modified are within the src directory. The mario agent itself can be run with the 2009 Competition GUI trainer by running:
```bash
./mario_agent
```

##Currently Supported Atari 2600 Platformers
- kung_fu_master
- frostbite
- kangaroo
- pitfall
- pitfall2
- hero
- montezuma_revenge

##Relevant Links

[Arcade Learning Environment Paper](http://arxiv.org/pdf/1207.4708v2.pdf)

[Arcade Learning Environment Website](http://www.arcadelearningenvironment.org)

[RL-Glue Paper](http://www.jmlr.org/papers/volume10/tanner09a/tanner09a.pdf)

[RL-Glue Wiki](http://glue.rl-community.org/wiki/Main_Page)

[DeepMind Deep Q-Networks](http://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

[An Open Source Deep Q-Learning Implemtation](https://github.com/spragunr/deep_q_rl)

[Mario AI Competition 2009](http://julian.togelius.com/Togelius2010The.pdf)

[A-Star Mario AI Project](https://github.com/jumoel/mario-astar-robinbaumgarten)

[Gaussian Process for RL in Trajectory Tracking Controllers](http://mlg.eng.cam.ac.uk/pub/pdf/HalRasMac11.pdf)

[Nonlinear Systems with Gaussian Processes](http://mlg.eng.cam.ac.uk/pub/pdf/HalRasMac12.pdf)