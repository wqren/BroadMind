import random
import time
from rlglue.agent.Agent import Agent
from rlglue.types import Action
from rlglue.types import Observation
from rlglue.agent import AgentLoader as AgentLoader

import string

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

class MarioAgent(Agent):
	
    def agent_init(self,taskSpecString):
        self.total_steps = 0
        self.trial_start = 0.0
        self.total_steps = 0
        self.step_number = 0
        self.last_actions = []
        self.this_actions = []
        random.seed(0)
	
    def agent_start(self,observation):
        self.step_number = 0
        self.trial_start = time.clock()
        return self.getAction(observation)
	
    def agent_step(self,reward, observation):
        self.step_number += 1
        self.total_steps += 1
        return self.getAction(observation)
	
    def agent_end(self,reward):
        time_passed = time.clock() - self.trial_start
        if (len(self.this_actions) > 7):
            self.last_actions = self.this_actions[0:len(self.this_actions)-7]
        else:
            self.last_actions = []
        self.this_actions = [];
        print "ended after " + str(self.total_steps) + " total steps"
        print "average " + str(self.step_number/time_passed) + " steps per second"
	
    def agent_cleanup(self):
        pass
	
    def agent_freeze(self):
        pass
	
    def agent_message(self,inMessage):
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

    '''
    * Returns the char representing the tile at the given location.
    * If unknown, returns '\0'.
    *
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
    def getTileAt(self, x, y, observation):
        if (x < 0):
            return '7'
        y = 16 - y
        x -= observation.intArray[0]
        if (x<0 or x>21 or y<0 or y>15):
            return '\0';
        index = y*22+x;
        return observation.charArray[index];        

    def getAction(self,observation):
        monsters = self.getMonsters(observation)
        mario = self.getMario(observation)

        #TODO: Add a learning component here using the input observations to determine
        #the action in a non-random way

        act = Action()
        #The first control input is -1 for left, 0 for nothing, 1 for right
        act.intArray.append(random.randint(-1,1))
        #The second control input is 0 for nothing, 1 for jump
        act.intArray.append(random.randint(0,1))
        #The third control input is 0 for nothing, 1 for speed increase
        act.intArray.append(random.randint(0,1))
        self.this_actions.append(act)
	return act

if __name__=="__main__":        
    AgentLoader.loadAgent(MarioAgent())
