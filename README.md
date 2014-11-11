BroadMind
=========

CIS 519 term project

##Setting up the environment

1.) Build RLGlue Core
 -   cd external/rlglue-3.04
 -   ./configure
 -   make
 -   sudo make install

2.) Build RLGlue C Codec
 -  cd external/c-codec-2.0
 -  ./configure
 -  make
 -  sudo make install

3.) Build RLGlue Python Codec
 -  cd external/python-codec/src
 -  python setup.py install
 -  Add <path_to_BroadMind>/external/python-codec to PYTHONPATH environment variable

4.) Build ALE (with RLGlue Agent)
 -  cd external/ale0.4_-2.4/ale_0_4
 -  IF OSX -> cp makefile.mac makefile
 -  IF UNIX -> cp makefile.unix makefile
 -  make
 -  cd doc/examples
 -  make rlglueAgent

5.) (Optional) Build RL 2009 Competition
 -  cd external/15-rl-competition-2009
 -  bash install.bash

##Possible Atari 2600 Platformers (in order of difficulty)
 1.) Kung Fu Master
 2.) Frostbite
 3.) Kangaroo
 4.) Pitfall
 5.) H.E.R.O.
 6.) Montezuma Revenge

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