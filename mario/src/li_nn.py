import numpy as np

class NeuralNet:

    def __init__(self, layers, epsilon=0.12, learningRate = 0.1):
        '''
        Constructor
        Arguments:
        	layers - a numpy array of L integers (L is # layers in the network)
        	epsilon - one half the interval around zero for setting the initial weights
        	learningRate - the learning rate for backpropagation
        '''
        self.layers = layers
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.numEpochs = 1
        self.nclasses = None
        self.activation = None
        self.regLambda = 0.1
        self.converge = 0.01
        
        # random.rand gives random number between 0,1, need between -eps, +eps
        length = len(layers)
        tht = np.empty((length-1), dtype = object)
        for i in range(0,length-1):
            tht[i]=np.random.rand(layers[i+1],layers[i]+1)            
        tht = tht * 2. * epsilon - epsilon
        self.theta = np.array([])
        for i in range(0,length-1):
            self.theta = np.append(self.theta, tht[i].flatten())
        

    def update(self, X, Y, a):
        '''
        Used to backpropagate the and correct based on an input x, prediction a, and truth Y
        X is a d-dimentional vector, 
        '''
        layers = self.layers
        epsilon = self.epsilon
        length = len(layers)
        
        # decode vector theta
        theta = np.empty((length-1),dtype=object)
        theta_vec = self.theta
        
        flag = 0        
        for i in range(0,length-1):
            size = layers[i+1] * (layers[i]+1)
            theta[i]=theta_vec[flag:(flag+size)].reshape(layers[i+1],-1)
            flag = flag+size
        
        
        # start loop with numEpoch = 10
        for iteration in range(0,self.numEpochs):
            # record vector for converge condition
            old_vec = np.copy(theta_vec)
            grad = np.empty((length-1), dtype=object)
            derivative = np.empty((length-1),dtype=object)
            cost_p1 = 0
            n = 1
            
            # perform forward propogation
            error = np.empty((length),dtype=object)
            out_fprop = a
            
            # increment cost func
            for l in range(0,len(Y)):
                if (out_fprop[l][0] == 1):
                    out_fprop[l] = out_fprop[l] - 0.0001
                suml = Y[l] * np.log(out_fprop[l][0]) + (1. -Y[l]) * np.log(1. - out_fprop[l][0])
                cost_p1 += suml

            # activation 0~ l-1, without bias node
            activation = self.activation
            error[length-1] = activation[length-1] - Y.reshape(-1,1)
            # calculate error for each layer
            for j in range(length-2,0,-1):
                p1 = np.dot(theta[j].T,error[j+1])
                p2 = np.multiply(activation[j], (1-activation[j]))
                error[j]= np.multiply(p1,p2)
            
            error[Y==0.0] == 0.0 #IMPORTANT: This step makes us only update the action that we observed   
            
            # calculate gradient
            if grad[length-2] == None:
                grad[length-2] = np.dot(error[length-1],activation[length-2].T)
            else:
                grad[length-2] = grad[length-2] + np.dot(error[length-1],activation[length-2].T)
        
            for k in range(length-3,-1,-1):
                if grad[k] == None:
                    grad[k] = np.dot(error[k+1][1:],activation[k].T)

                else:
                    grad[k] = grad[k] + np.dot(error[k+1][1:],activation[k].T)

            # compute partial derivative
            for i in range(length-2,-1,-1):
                row, col = theta[i].shape
                # print "theta[%d]" %i, theta[i].shape
                # print grad[i].shape
                temp = np.c_[np.zeros((row,1)),self.regLambda * theta[i][:,1:]]
                # temp = np.vstack((np.zeros((1,col)),self.regLambda * theta[i][1:,:]))
                # derivative[i] =1./n * (grad[i] + temp)
                derivative[i] = 1./ n * (grad[i] + temp)

            # update weights
            for i in range(0,length-1):
                theta[i] = theta[i] - self.learningRate * derivative[i]
        
            # print "theta_update: \n", theta
        
            # unroll matrices into a single vector
            theta_vec = np.array([])
            for i in range(0,length-1):
                theta_vec = np.append(theta_vec, theta[i].flatten())
            
            # record new vector
            new_vec = np.copy(theta_vec)
            # calculate cost func
            para1 = -1. / n
            cost_p1 = cost_p1 * para1
            para2 = self.regLambda / (2. * n)
            cost_reg = para2 * (np.linalg.norm(theta_vec) ** 2)

            cost_func = cost_p1 + cost_reg
            print "%d cost_func: %.4f" % (iteration,cost_func)
            print np.linalg.norm(new_vec - old_vec)

            if (np.linalg.norm(new_vec - old_vec) < self.converge) or (iteration == self.numEpochs-1):
                self.theta = np.copy(theta_vec)
                # print "################", self.theta
                break
              

    def propagate(self, X):
        '''
        Used the model to predict weighted output values for instance x
        Arguments:
            x is a d-dimenstional numpy array
        Returns:
            a c-dimensional numpy array of the strength of each output
        '''
        d = len(X)
        
        # vector theta
        theta_fit = self.theta
        a = self.forwardprop(X,theta_fit)
        
        return a
        

    def forwardprop(self, X, theta):
        '''
        take in parameters vector theta | theta_1 ~ theta_(L-1)
        take in instance(s) X
        used by predict() and backprop()
        return output(s) regarding input instance(s)
        '''
        layers = self.layers
        length = len(layers)
        
        # decode vector theta
        theta_mat = np.empty((length-1),dtype=object)

        flag = 0        
        for i in range(0,length-1):
            size = layers[i+1] * (layers[i]+1)
            theta_mat[i]=theta[flag:(flag+size)].reshape(layers[i+1],-1)
            flag = flag+size
                
        layerX = np.empty((length),dtype=object)
        
        # add bias node to layer0, now 1* d+1
        # need to use sigmoid func
        layerX[0] = X
        layerX[0] = np.append(1,layerX[0]).reshape(-1,1)
            
        for j in range(1,length):
            layerX[j] = self.sigmoid(np.dot(theta_mat[j-1],layerX[j-1]))
            # add bias node
            if j< (length -1) :
                layerX[j] = np.append(1,layerX[j]).reshape(-1,1)
        
        # store each layer
        self.activation = layerX
        output = layerX[length-1]
        # print layerX
        
        return output
                
        
    
    def sigmoid(self, z):
        # return value of sigmoid function
        # z: n * 1 vector
        M = 1. + np.exp(-1. * z)
        result = np.divide(1.,M, dtype=float)
        
        for i in range(0,len(result)):
            if result[i] == 0:
                result[i] += 0.0001
        
        return result

    def propagateAndUpdate(self, x, y):
        '''
        Used to forward propagate a prediction based on input x, and update against truth y
        '''
        a = self.propagate(x)
        self.update(x, y, a)
        