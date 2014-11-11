##################################
Setting up the environment
##################################

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
