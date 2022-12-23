# -*- coding: utf-8 -*-


# nn with n input and 4 hidden neurons(1 hidden layer)
import numpy as np
def sigmoid(x):
    return 1/(1+np.e**(-x))

def derivSigmoid(x):
    return (sigmoid(x)*(1-sigmoid(x)))

def mse(yT,yP):
    return ((yT-yP)**2).mean()

class Neuron():
    def __init__(self, bias, weights):
        self.bias = bias
        self.weights = weights 
    
    def feedForward(self,x):
        return np.dot(x, self.weights)+self.bias
    
class NeuralNetwork():
    
    def __init__(self):
        weights = np.random.rand(4,6)
        bias = np.random.rand(5)
        outputWeights = np.random.rand(4)
        self.n1 = Neuron(bias[0], weights[0])
        self.n2 = Neuron(bias[1], weights[1])
        self.n3 = Neuron(bias[2], weights[2])
        self.n4 = Neuron(bias[3], weights[3])
        self.o1 = Neuron(bias[4], outputWeights)
    
    def predict(self,x):
        res = []
        for i in range(len(x)):
            h1 = sigmoid(self.n1.feedForward(x[i]))
            h2 = sigmoid(self.n2.feedForward(x[i]))
            h3 = sigmoid(self.n3.feedForward(x[i]))
            h4 = sigmoid(self.n4.feedForward(x[i]))
            res.append(sigmoid(self.o1.feedForward(np.array([h1,h2,h3,h4]))))
        return res
    
    def train(self,data, yTrues):
        
        eta = 0.1
        epochs = 1000
        
        for epoch in range(epochs):
            preds = []
            for x, y in zip(data, yTrues):
                sum_h1 = self.n1.feedForward(x)
                sum_h2 = self.n2.feedForward(x)
                sum_h3 = self.n3.feedForward(x)
                sum_h4 = self.n4.feedForward(x)
                h1 = sigmoid(sum_h1)
                h2 = sigmoid(sum_h2)
                h3 = sigmoid(sum_h3)
                h4 = sigmoid(sum_h4)
                out = self.o1.feedForward(np.array([h1,h2,h3,h4]))
                yPred = sigmoid(out)
                preds.append(yPred)
                
                d_l_d_ypred = -2*(y-yPred)
                derout = derivSigmoid(out)
                derh1 = derivSigmoid(sum_h1)
                derh2 = derivSigmoid(sum_h2)
                derh3 = derivSigmoid(sum_h3)
                derh4 = derivSigmoid(sum_h4)
                
                # all h
                
                d_h1_d_w1 = x[0]*derh1
                d_h1_d_w2 = x[1]*derh1
                d_h1_d_w3 = x[2]*derh1
                d_h1_d_w4 = x[3]*derh1
                d_h1_d_w5 = x[4]*derh1
                d_h1_d_w6 = x[5]*derh1
                d_h1_b1 = derh1
                
                d_h2_d_w7  = x[0]*derh2
                d_h2_d_w8  =  x[1]*derh2
                d_h2_d_w9  =  x[2]*derh2
                d_h2_d_w10 = x[3]*derh2
                d_h2_d_w11 = x[4]*derh2
                d_h2_d_w12 = x[5]*derh2
                d_h2_b2 = derh2
                
                d_h3_d_w13 = x[0]*derh3
                d_h3_d_w14 = x[1]*derh3
                d_h3_d_w15 = x[2]*derh3
                d_h3_d_w16 = x[3]*derh3
                d_h3_d_w17 = x[4]*derh3
                d_h3_d_w18 = x[5]*derh3
                d_h3_b3 = derh3
                
                d_h4_d_w19 = x[0]*derh4
                d_h4_d_w20 = x[1]*derh4
                d_h4_d_w21 = x[2]*derh4
                d_h4_d_w22 = x[3]*derh4
                d_h4_d_w23 = x[4]*derh4
                d_h4_d_w24 = x[5]*derh4
                d_h4_b4 = derh4
                
                # all out
                
                d_out_d_w25 = derout*h1
                d_out_d_w26 = derout*h2
                d_out_d_w27 = derout*h3
                d_out_d_w28 = derout*h4
                d_out_d_h1 = derout*self.o1.weights[0]
                d_out_d_h2 = derout*self.o1.weights[1]
                d_out_d_h3 = derout*self.o1.weights[2]
                d_out_d_h4 = derout*self.o1.weights[3]
                d_out_d_b = derout
                
                
                #update weights and biases
                
                self.n1.weights[0] -= eta*d_l_d_ypred*d_h1_d_w1*d_out_d_h1
                self.n1.weights[1] -= eta*d_l_d_ypred*d_h1_d_w2*d_out_d_h1
                self.n1.weights[2] -= eta*d_l_d_ypred*d_h1_d_w3*d_out_d_h1
                self.n1.weights[3] -= eta*d_l_d_ypred*d_h1_d_w4*d_out_d_h1
                self.n1.weights[4] -= eta*d_l_d_ypred*d_h1_d_w5*d_out_d_h1
                self.n1.weights[5] -= eta*d_l_d_ypred*d_h1_d_w6*d_out_d_h1
                self.n1.bias -= eta*d_l_d_ypred*d_h1_b1*d_out_d_h1
                
                self.n2.weights[0] -= eta*d_l_d_ypred*d_h2_d_w7*d_out_d_h2
                self.n2.weights[1] -= eta*d_l_d_ypred*d_h2_d_w8*d_out_d_h2
                self.n2.weights[2] -= eta*d_l_d_ypred*d_h2_d_w9*d_out_d_h2
                self.n2.weights[3] -= eta*d_l_d_ypred*d_h2_d_w10*d_out_d_h2
                self.n2.weights[4] -= eta*d_l_d_ypred*d_h2_d_w11*d_out_d_h2
                self.n2.weights[5] -= eta*d_l_d_ypred*d_h2_d_w12*d_out_d_h2
                self.n2.bias -= eta*d_l_d_ypred*d_h2_b2*d_out_d_h2
                
                self.n3.weights[0] -= eta*d_l_d_ypred*d_h3_d_w13*d_out_d_h3
                self.n3.weights[1] -= eta*d_l_d_ypred*d_h3_d_w14*d_out_d_h3
                self.n3.weights[2] -= eta*d_l_d_ypred*d_h3_d_w15*d_out_d_h3
                self.n3.weights[3] -= eta*d_l_d_ypred*d_h3_d_w16*d_out_d_h3
                self.n3.weights[4] -= eta*d_l_d_ypred*d_h3_d_w17*d_out_d_h3
                self.n3.weights[5] -= eta*d_l_d_ypred*d_h3_d_w18*d_out_d_h3
                self.n3.bias -= eta*d_l_d_ypred*d_h3_b3*d_out_d_h3
                
                self.n4.weights[0] -= eta*d_l_d_ypred*d_h4_d_w19*d_out_d_h4
                self.n4.weights[1] -= eta*d_l_d_ypred*d_h4_d_w20*d_out_d_h4
                self.n4.weights[2] -= eta*d_l_d_ypred*d_h4_d_w21*d_out_d_h4
                self.n4.weights[3] -= eta*d_l_d_ypred*d_h4_d_w22*d_out_d_h4
                self.n4.weights[4] -= eta*d_l_d_ypred*d_h4_d_w23*d_out_d_h4
                self.n4.weights[5] -= eta*d_l_d_ypred*d_h4_d_w24*d_out_d_h4
                self.n4.bias -= eta*d_l_d_ypred*d_h4_b4*d_out_d_h4
                
                self.o1.weights[0] -= eta*d_l_d_ypred*d_out_d_w25
                self.o1.weights[1] -= eta*d_l_d_ypred*d_out_d_w26
                self.o1.weights[2] -= eta*d_l_d_ypred*d_out_d_w27
                self.o1.weights[3] -= eta*d_l_d_ypred*d_out_d_w28
                self.o1.bias -= eta*d_l_d_ypred*d_out_d_b
            if epoch % 10 == 0:
                loss =  mse(np.array(yTrues), np.array(preds))
                print("Epoch: {} loss : {}".format(epoch, loss))
    
                
nn = NeuralNetwork()
target = np.array([1,0,1])
X = np.array([[11,12,10,7,9,6],[-5,-6,0,1,-2,1],[7,9,8,11,9,7]])
nn.train(X,target)
print(nn.predict(np.array([[11,12,9,6,8,5],[-5,-6,0,2,-4,1]])))
              
                
                
                

                
                
                
                
                
                
                
                
                
                
                
        
        
