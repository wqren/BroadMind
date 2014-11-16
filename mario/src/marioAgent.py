import random
import time
import string
from rlglue.agent.Agent import Agent
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.agent import AgentLoader as AgentLoader
import numpy as np
from qnn import QNN

class Monster:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.sx = 0
        self.sy = 0
        self.type = -1
        self.typeName = ""
        self.winged = False

monsterNames = ["Mario", "Red Koopa", "Green Koopa", "Goomba", "Spikey", "Piranha Plant", "Mushroom", "Fire Flower", "Fireball", "Shell", "Big Mario", "Fiery Mario"]

'''
* Valid tiles:
* M - the tile mario is currently on. there is no tile for a monster.
* $ - a coin
* b - a smashable brick
* ? - a question block
* | - a pipe. gets its own tile because often there are pirahna plants
*     in them
* ! - the finish line
* And an integer in [1,7] is a 3 bit binary flag
*  first bit is "cannot go through this tile from above"
*  second bit is "cannot go through this tile from below"
*  third bit is "cannot go through this tile from either side"
'''
tileEncoder = {'\0': -1, '1': -1, '2': -1, '3': -1, '4': -1, '5': -1, '6': -1, '7': -1, 'M': 0, '$': 2, 'b': 2, '?': 2, '|': 2, '!': 2, ' ': 1, '\n': 1}
#TODO: Maybe have a set of these encodings that we can switch between?

class MarioAgent(Agent):
	
    def agent_init(self,taskSpecString):
        self.learn_mode = 1 #O = Random, 1 = Neural Q-Function
        self.policy_frozen = False
        self.total_steps = 0
        self.trial_start = 0.0
        self.total_steps = 0
        self.step_number = 0
        self.gamma = 0.6 #Discount rate for future rewards
        self.exp = 0.95 #Exploration factor
        self.state_dim_x = 20
        self.state_dim_y = 12
        self.last_state = None
        self.last_action = None
        random.seed(0)
        if (self.learn_mode == 1):
            self.Q = QNN(nactions=12, input_size=(self.state_dim_x*self.state_dim_y), alpha=0.2)
	
    def agent_start(self,observation):
        self.step_number = 0
        self.trial_start = time.clock()
        return self.getAction(observation)
	
    def agent_step(self,reward, observation):
        #self.printFullState(observation)
        #self.printMarioState(observation)
        self.step_number += 1
        self.total_steps += 1
        act = self.getAction(observation)
        if (!self.policy_frozen):
            self.update(observation, act, reward)
        self.last_state = observation
        self.last_action = act
        return act
	
    def agent_end(self,reward):
        time_passed = time.clock() - self.trial_start
        print "ended after " + str(self.total_steps) + " total steps"
        print "average " + str(self.step_number/time_passed) + " steps per second"
	
    def agent_cleanup(self):
        pass
	
    def agent_freeze(self):
        pass
	
    def agent_message(self,inMessage):
        if inMessage.startswith("freeze learning"):
            self.policy_frozen=True
            return "message understood, policy frozen"
        if inMessage.startswith("unfreeze learning"):
            self.policy_frozen=False
            return "message understood, policy unfrozen"
        if inMessage.startswith("freeze exploring"):
            self.exp = 0.0
            return "message understood, exploring frozen"
        if inMessage.startswith("unfreeze exploring"):
            self.exp = 0.95
            return "message understood, exploring unfrozen"
        if inMessage.startswith("save policy"):
            splitString=inMessage.split(" ");
            self.saveQFun(splitString[1]);
            print "Saved.";
            return "message understood, saving policy"
        if inMessage.startswith("load policy"):
            splitString=inMessage.split(" ")
            self.loadQFun(splitString[1])
            print "Loaded."
            return "message understood, loading policy"
        return None

    def getMonsters(self, observation):
        monsters = []
        i = 0
        while (1+2*i < len(observation.intArray)):
            m = Monster()
            m.type = observation.intArray[1+2*i]
            m.winged = (observation.intArray[2+2*i]) != 0
            m.x = observation.doubleArray[4*i];
            m.y = observation.doubleArray[4*i+1];
            m.sx = observation.doubleArray[4*i+2];
            m.sy = observation.doubleArray[4*i+3];
            m.typeName = monsterNames[m.type]
            monsters.append(m)
            i += 1
        return monsters

    def getMario(self, observation):
        monsters = self.getMonsters(observation)
        for i in xrange(len(monsters)):
            if (monsters[i].type == 0 or monsters[i].type == 10 or monsters[i].type == 11):
                return monsters[i]
        return None

    def getTileAt(self, x, y, observation):
        if (x < 0):
            return '7'
        y = 16 - y
        x -= observation.intArray[0]
        if (x<0 or x>21 or y<0 or y>15):
            return '\0';
        index = y*22+x;
        return observation.charArray[index];        

    #Handy little function for debugging by printing out the char array
    def printFullState(self, observation):
        print "-----------------"
        for yi in xrange(16):
            out_string = ""
            for xi in xrange(22):
                out_string += observation.charArray[yi*22+xi]
            print out_string

    #Handy little functio for debugging by printing out Mario's (x,y) position
    def printMarioState(self, observation):
        mar = self.getMario(observation)
        print "Mario X: " + str(mar.x) + " Mario Y: " + str(mar.y)

    #Handy little function for debugging by printing out the encoded state that is being send to the NN
    def printEncodedState(self, s):
        print "-----------------"
        for yi in xrange(self.state_dim_y):
            out_string = ""
            for xi in xrange(self.state_dim_x):
                out_string += (str(s[yi*self.state_dim_x+xi]) + " ")
            print out_string

    def getAction(self, observation):
        monsters = self.getMonsters(observation)
        mario = self.getMario(observation)

        if (self.learn_mode == 0):
            act = self.randomAction(observation)
        elif (self.learn_mode == 1):
            act = self.qnnAction(observation)

	return act

    def update(self, observation, action, reward):
        if (self.learn_mode == 1):
            self.qnnUpdate(observation, action, reward)

    def stateEncoder(self, observation):
        s = []
        #Determine Mario's current position. Everything is relative to Mario
        mar = self.getMario(observation)
        mx = int(mar.x)
        my = 15 - int(mar.y)
        #Update based on the environment
        for yi in xrange(self.state_dim_y):
            for xi in xrange(self.state_dim_x):
                x = mx + xi - int(self.state_dim_x/2.0)
                y = my + yi - int(self.state_dim_y/2.0)
                if (x < 0 or x > 21 or y < 0 or y > 15):
                    s.append(-1)
                    continue
                s.append(tileEncoder[observation.charArray[y*22+x]])
        #Add monsters
        monsters = self.getMonsters(observation)
        for mi in xrange(len(monsters)):
            if (monsters[mi].type == 0 or monsters[mi].type == 10 or monsters[mi].type == 11): #skip mario
                continue
            monx = int(monsters[mi].x)
            mony = 15 - int(monsters[mi].y)
            x = monx - mx + int(self.state_dim_x/2.0)
            y = mony - my + int(self.state_dim_y/2.0)
            if (x < 0 or x >= self.state_dim_x or y < 0 or y >= self.state_dim_y): #skip monsters farther away
                continue
            s[y*self.state_dim_x + x] = -2
        return s

    def actionEncoder(self, act):
        a = 1*act.intArray[2] + 2*act.intArray[1] + 4*(act.intArray[0]+1)
        return a

    def actionDecoder(self, a):
        act = Action()
        act.intArray.append(int(a/4.0) - 1)
        act.intArray.append(int(a/2.0) % 2)
        act.intArray.append(a % 2)
        return act

    def randomAction(self, observation):
        act = Action()
        #The first control input is -1 for left, 0 for nothing, 1 for right
        act.intArray.append(random.randint(-1,1))
        #The second control input is 0 for nothing, 1 for jump
        act.intArray.append(random.randint(0,1))
        #The third control input is 0 for nothing, 1 for speed increase
        act.intArray.append(random.randint(0,1))        
        return act

    def qnnAction(self, observation):
        s = self.stateEncoder(observation)
        #self.printEncodedState(s)
        if (random.random()>self.exp):
            a = np.argmax(self.Q(s))
        else:
            #TODO: Bias the exploration towards moving forward?
            a  = random.randint(0,12)
        act = self.actionDecoder(a)
        return act

    def qnnUpdate(self, observation, action, reward):
        if (self.last_state == None or self.last_action == None):
            return
        s = self.stateEncoder(observation)
        ls = self.stateEncoder(self.last_state)
        a = self.actionEncoder(action)
        la = self.actionEncoder(self.last_action)
        #TODO: Use a flag between using a SARSA update or the regular Q update
        #TODO: Do we need some experience replay function like DeepMind?
        target_value = reward + self.gamma*max(self.Q(s)) #Q-Learning
        self.Q.Update(ls, la, target_value)
        #target_value = reward + self.gamma*self.Q(s, a) #SARSA 
        #self.Q.Update(ls, la, target_value)

    def saveQFun(self, fileName):
        '''
        TODO: Implement saving off NN with pickle
        '''
        pass

    def loadQFun(self, fileName):
        '''
        TODO: Implement loading NN with pickle
        '''
        pass

if __name__=="__main__":        
    AgentLoader.loadAgent(MarioAgent())
