'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton
    Modified Zhi Li
'''

import numpy as np
import Image
from sklearn.preprocessing import label_binarize

class NeuralNet:

    def __init__(self, layers, learningRate, epsilon=0.12, numEpochs=100):
        '''
        Constructor
        Arguments:
        	layers - a numpy array of L-2 integers (L is # layers in the network)
        	epsilon - one half the interval around zero for setting the initial weights
        	learningRate - the learning rate for backpropagation
        	numEpochs - the number of epochs to run during training
        '''
        self.layers = layers
        self.epsilon = epsilon
        self.learningRate = learningRate
        self.numEpochs = numEpochs
        self.nclasses = None
        self.theta = None
        self.theta_m = None
        self.activation = None
        self.regLambda = 0.1
        self.converge = 0.01
      

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy array
            y is an n-dimensional numpy array
        '''
        layers = self.layers
        epsilon = self.epsilon
        
        length = len(layers) + 2
        
        n,d = X.shape
        # get classes y
        temp = y.astype(int)
        countY = np.bincount(temp)
        iY = np.nonzero(countY)[0]
        dictY = np.vstack(zip(iY,countY[iY]))
        n_classes, _ = dictY.shape
        self.nclasses = n_classes
        
        # random.rand gives random number between 0,1, need between -eps, +eps
        theta = np.empty((length-1), dtype = object)
        theta[0]=np.random.rand(layers[0],d+1)
        theta[length-2]= np.random.rand(n_classes,layers[length-3]+1)
        for i in range(1,length-2):
            theta[i]=np.random.rand(layers[i],layers[i-1]+1)            
        # * 2 * eps - eps
        theta = theta * 2. * epsilon - epsilon
        # print "theta_first:\n",theta        
   
        # unroll matrices into a single vector
        theta_vec = np.array([])
        for i in range(0,length-1):
            theta_vec = np.append(theta_vec, theta[i].flatten())
        
        # start loop with numEpoch = 100
        for iteration in range(0,self.numEpochs):
            # record vector for converge condition
            old_vec = np.copy(theta_vec)
            
            grad = np.empty((length-1), dtype=object)
            derivative = np.empty((length-1),dtype=object)
            Y = label_binarize(y,classes=np.arange(n_classes))
            cost_p1 = 0

            # loop through every instance in X   
            for i in range(0,n):
                # debug printout
                # print "%d Ytrain: %d " %(i,y[i])
                # print "   Binarilize:", Y[i]
                
                error = np.empty((length),dtype=object)
                # perform forward propogation
                out_fprop = self.forwardprop(X[i,:], theta_vec)
                # increment cost func
                for l in range(0,len(Y[i])):
                    if (out_fprop[l][0] == 1):
                        out_fprop[l] = out_fprop[l] - 0.0001
                    suml = Y[i][l] * np.log(out_fprop[l][0]) + (1. -Y[i][l]) * np.log(1. - out_fprop[l][0])
                    cost_p1 += suml

                # print out_fprop
                # activation 0~ l-1, without bias node
                activation = self.activation
                error[length-1] = activation[length-1] - Y[i].reshape(-1,1)
                # calculate error for each layer
                for j in range(length-2,0,-1):
                    p1 = np.dot(theta[j].T,error[j+1])
                    p2 = np.multiply(activation[j], (1-activation[j]))
                    error[j]= np.multiply(p1,p2)
                    
                # calculate gradient
                if grad[length-2] == None:
                    grad[length-2] = np.dot(error[length-1],activation[length-2].T)
                else:
                    grad[length-2] = grad[length-2] + np.dot(error[length-1],activation[length-2].T)

                # debug printout gradient 
                # print "%d gradient change: " %i, np.dot(error[length-1],activation[length-2].T)
            
                for k in range(length-3,-1,-1):
                    if grad[k] == None:
                        # #debug
                        # print "##", activation[k].shape
                        # print "@@", error[k+1].shape
                        grad[k] = np.dot(error[k+1][1:],activation[k].T)
                        # print "%d gradient change: " %i, np.dot(error[k+1][1:],activation[k].T)

                    else:
                        grad[k] = grad[k] + np.dot(error[k+1][1:],activation[k].T)
                        # print "%d gradient change: " %i, np.dot(error[k+1][1:],activation[k].T)
                    

                
                # ##################
                # # gradient checking
                # c = 1e-4
                # check_ar = np.zeros((len(theta_vec)))
                #
                # for icheck in range(0,len(theta_vec)):
                #     cost_left = 0.
                #     cost_right = 0.
                #     check_vec_left = np.copy(theta_vec)
                #     check_vec_right = np.copy(theta_vec)
                #     check_vec_left[icheck] = check_vec_left[icheck] - c
                #     check_vec_right[icheck] = check_vec_right[icheck] + c
                #
                #     # if icheck <3:
                #     #     print check_vec_right
                #     #     print check_vec_left
                #
                #     left = self.forwardprop(X[i,:], check_vec_left)
                #     right = self.forwardprop(X[i,:], check_vec_right)
                #
                #     for l1 in range(0,len(Y[i])):
                #         if (left[l1][0] == 1):
                #             left[l1] = left[l1] - 0.0001
                #         if (right[l1][0] == 1):
                #             right[l1] = right[l1] - 0.0001
                #         suml1 = Y[i][l1] * np.log(left[l1][0]) + (1. -Y[i][l1]) * np.log(1. - left[l1][0])
                #         suml2 = Y[i][l1] * np.log(right[l1][0]) + (1. -Y[i][l1]) * np.log(1. - right[l1][0])
                #         cost_left += suml1 * -1.
                #         cost_right += suml2 * -1.
                #
                #     check_ar[icheck] = (cost_right - cost_left)/ (2. * c)
                #
                # print "check: ", check_ar[10025:].reshape(-1,26)
                # print "check: ", check_ar[:10025].reshape(25,-1)
                # ##################
                
                
            # compute partial derivative
            for i in range(length-2,-1,-1):
                row, col = theta[i].shape
                # print "theta[%d]" %i, theta[i].shape
                # print grad[i].shape
                temp = np.c_[np.zeros((row,1)),self.regLambda * theta[i][:,1:]]
                # temp = np.vstack((np.zeros((1,col)),self.regLambda * theta[i][1:,:]))
                derivative[i] =1./n * (grad[i] + temp)
        
            # print "deri: ", derivative
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
            # print "##############", derivative
            para1 = -1. / n
            cost_p1 = cost_p1 * para1
            para2 = self.regLambda / (2. * n) 
            cost_reg = para2 * (np.linalg.norm(theta_vec) ** 2)
        
            cost_func = cost_p1 + cost_reg
            # print "%d cost_func: %.4f" % (iteration,cost_func)
            # print np.linalg.norm(new_vec - old_vec)

            if (np.linalg.norm(new_vec - old_vec) < self.converge) or (iteration == self.numEpochs-1):
                self.theta = np.copy(theta_vec)
                self.theta_m = np.copy(theta)
                # print "################", self.theta
                break
              

    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy array
        Returns:
            an n-dimensional numpy array of the predictions
        '''
        n,d = X.shape
        
        # vector theta
        theta_fit = self.theta
        result = np.zeros((n))
        
        for i in range(0,n):
            predict = self.forwardprop(X[i,:],theta_fit)
            # debug
            result[i]= np.argmax(predict)
            # if i<3 :
            #     print "output: ", predict
            #     print "predict: ", result[i]
        
        return result
        

    def forwardprop(self, X, theta):
        '''
        take in parameters vector theta | theta_1 ~ theta_(L-1)
        take in instance(s) X
        used by predict() and backprop()
        return output(s) regarding input instance(s)
        '''
        d = len(X)
        layers = self.layers
        length = len(layers) + 2
        n_classes = self.nclasses
        
        # decode vector theta
        theta_mat = np.empty((length-1),dtype=object)
        flag = layers[0]*(d+1)
        
        theta_mat[0] = theta[0:flag]
        theta_mat[0] = theta_mat[0].reshape(layers[0],-1)
        
        for i in range(1,length-2):
            size = layers[i] * (layers[i-1]+1)
            theta_mat[i]=theta[flag:(flag+size)].reshape(layers[i],-1)
            flag = flag+size
        theta_mat[length-2]= theta[flag:].reshape(n_classes,-1)
                
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
        # print "@@@@@@@@"
        # print z.shape
        
        for i in range(0,len(result)):
            if result[i] == 0:
                result[i] += 0.0001
        
        return result
        
    
    def visualizeHiddenNodes(self, filename):
        '''
        CIS 519 ONLY - outputs a visualization of the hidden layers
        Arguments:
            filename - the filename to store the image
        '''
        # get the hidden layer, 25 * 400
        hid_layer = self.theta_m[0][:,1:]
        
        # print "shape of hidden layer:", hid_layer.shape
        # reshape to 20 * 20
        visualize = np.empty((25), dtype = object)
        # store 25 array--> 25 img 
        img_group = np.empty((25),dtype=object)
        
        for i in range(0,25):
            # remap 400 * 1
            remap = hid_layer[i].flatten()
            max_v = remap[remap.argmax()]
            min_v = remap[remap.argmin()]
            # (x - min)/(max- min)  normalize to 0~1
            remap = np.divide((remap - min_v), (max_v - min_v), dtype=float)
            visualize[i] = remap.reshape(20,20)
            
            img_group[i] = Image.fromarray(np.uint8(visualize[i] * 255))
            # img_group[i].show() #show 25 img
        
        # background
        new_im = Image.new('RGB', (112,112),"black")
        
        for i in xrange(2,100,22):
            for j in xrange(2,100,22):
                # give no. of stored img
                loc = (i-2)/22 + ((j-2)/22 * 5)
                #paste the image at location i,j:
                new_im.paste(img_group[loc], (i,j))

        # new_im.show()
        
        # save file name
        savename = filename + ".png"
        new_im.save(savename)

        