from nn import NeuralNet

class QNN():
    """
    Neural Network implementation of Q-Function
    nactions: the number of actions
    input_size: the number of inputs
    nelemns: the number of integers that can be input on each element of the input state
    alpha: learning rate
    """
    
    def __call__(self,s,a=None):
        """ implement here the returned Qvalue of state (s) and action(a)
        e.g. Q.GetValue(s,a) is equivalent to Q(s,a)
        """
        if a==None:
            return self.GetValue(s)
        return self.GetValue(s,a)

    def __init__(self, nactions, input_size, alpha=0.1):
        lay = [input_size, int((nactions+input_size)/2.0), nactions]
        self.Q = NeuralNet(layers=lay, epsilon=0.04, learningRate=alpha)
        
    def GetValue(self, s, a=None):
        """ Return the Q(s,a) value of state (s) for action (a)
        or al values for Q(s)
        """
        out = self.Q.propagate(s)
        if (a==None):
            return out
        return out[a]


    def Update(self, s, a, v):
        """ update action value for action(a)
        """
        self.Q.propagateAndUpdate(s, a*v)
