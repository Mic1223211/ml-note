from datetime import datetime
import numpy as np
import sys

class Sigmoid:
    def forward(self,x):
        return 1.0 / (1.0 + np.exp(-x))

    def backward(self,x,top_diff):
        output = self.forward(xx)
        return (1.0 -output)*output*top_diff

class Tanh:
    def forward(self,x):
        return np.tanh(x)

    def backward(self,x,top_diff):
        output = self.forward(x)
        return (1.0 - np.square(output)) * top_diff

class MultiplyGate:
    def forward(self,W,x):
        return np.dot(W,x)

    def backward(self,W,x,dz):
        dW = np.asarray(np.dot(np.transpose(np.asmatrix(dz)),np.asmatrix(x)))
        dx = np.dot(np.transpose(W),dz)
        return dW,dx
class AddGate:
    def forward(self,x1,x2):
        return x1 +x2
    def backward(self,x1,x2,dz):
        dx1 = dz * np.ones_like(x1)
        dx2 = dz * np.ones_like(x2)
        return dx1,dx2

class Softmax:
    def predict(self,x):
        exp_scores = np.exp(x)
        return exp_scores/np.sum(exp_scores)

    def loss(self,x,y):
        probs = self.predict(x)
        return -np.log(probs[y])

    def diff(self,x,y):
        probs = self.predict(x)
        probs[y] -= 1.0
        return probs

mulGate = MultiplyGate()
addGate = AddGate()
activation = Tanh()

class RNNLayer:
    def forward(self,x,prev_s,U,W,V):
        self.mulu = mulGate.forward(U,x)
        self.mulw = mulGate.forward(W,prev_s)
        self.add = addGate.forward(self.mulw,self.mulu)
        self.s = activation.forward(self.add)
        self.mulv = mulGate.forward(V,self.s)

    def backward(self,x,prev_s,U,W,V,diff_s,dmulu):
        self.forward(x,prev_s,U,W,V)
        dV,dsv = mulGate.backward(V,self.s,dmulv)
        ds = dsv + diff_s
        dadd = activation.backward(self.add,ds)
        dmulw,dmulu = addGate.backward(self.mulw,self.mulu,dadd)
        dW,dprev_s = mulGate.backward(W,prev_s,dmulw)
        dU,dx = mulGate.backward(U,x,dmulu)
        return (dprev_s,dU,dW,dV)


class Model:
    def __init__(self,word_dim,hidden_dim=100,bptt_truncate=4):
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.U = np.random.uniform(-np.sqrt(1./word_dim),np.sqrt(1./word_dim),(hidden_dim,word_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(hidden_dim,hidden_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim),np.sqrt(1./hidden_dim),(word_dim,hidden_dim))

    '''
        forward propagation (predicting word probabilities)
        x is one single data,and a batch of data
        for example x = [0,179,341,416], then its y = [179,341,416,1]
    '''
    def forward_propagation(self,x):
        #The total number of time steps
        T = len(x)
        layers = []
        prev_s = np.zeros(self.hidden_dim)
        #For each time step..
        for t in range(T):
            layer = RNNLayer()
            input = np.zeros(self.word_dim)
            input[x[t]] = 1
            layer.forward(input,prev_s,self.U,self.W,self.V)
            prev_s = layer.s
            layers.append(layer)
        return layers

    def predict(self,x):
        output = Softmax()
        layers = self.forward_propagation(x)
        return [np.argmax(output.predict(layer.mulv)) for layer in layers]

    def calculate_loss(self,x,y):
        assert len(x) == len(y)
        output = Softmax()
        layers = self.forward_propagation(x)
        loss = 0.0
        for i,layer in enumerate(layers):
            loss += output.loss(layer.mulv,y[i])
        return loss/ float(len(y))

    def calculate_total_loss(self,X,Y):
        loss = 0.0
        for i in range(len(Y)):
            loss += self.calculate_loss(X[i],Y[i])
        return loss/float(len(Y))

    def bptt(self,x,y):
        assert len(x) == len(y)
        output = Softmax()
        layers = self.forward_propagation(x)
        dU = np.zeros(self.U.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)

        T = len(layers)
        prev_s_t = np.zeros(self.hidden_dim)
        diff_s = np.zeros(self.hidden_dim)
        for t in range(0,T):
            dmulv = output.diff(layers[t].mulv,y[t])
            input = np.zeros(self.word_dim)
            input[x[t]] = 1
            dprev_s,dU_t,dW_t,dV_t = layers[t].backward(input,prev_s_t,self.U,self.W,self.V,diff_s,dmulv)
            prev_s_t = layers[t].s
            dmulv = np.zeros(self.word_dim)
            for i in range(t-1,max(-1,t-self.bptt_truncate-1),-1):
                input = np.zeros(self.word_dim)
                input[x[i]] = 1
                prev_s_i = np.zeros(self.hidden_dim) if i == 0  else layers[i-1].s
                dprev_s,dU_i,dW_i,dV_i = layers[i].backward(input,prev_s_i,self.U,self.W,self.V,dprev_s,dmulv)
                dU_t += dU_i
                dW_t += dW_i

            dV += dV_t
            dU += dU_t
            dW += dW_t
        return (dU,dW,dV)

    def sgd_step(self,x,y,learning_rate):
        dU,dW,dV = self.bptt(x,y)
        self.U -= learning_rate*dU
        self.V -= learning_rate *dV
        self.W -= learning_rate *dW

    def train(self,X,Y,learning_rate=0.005,nepoch=100,evaluate_loss_after=5):
        num_examples_seen = 0
        losses = []
        for epoch in range(nepoch):
            if (epoch % evaluate_loss_after == 0):
                loss = self.calculate_total_loss(X,Y)
                losses.append((num_examples_seen,loss))
                time = datetime.now.strftime('%Y-%m-%d %H:%M:%S')
                print('%s: loss after num_examples_seen=%d epoch %d:%f ' % (time,num_examples_seen,epoch,loss))
                #Adjust the learning rate if loss increases
                if len(losses) > 1 and losses[-1][1] > losses[-2][1]:
                    learning_rate = learning_rate*0.5
                    print('setting learning rate to %f' % learning_rate)
                sys.stdout.flush()
            #For each training example
            for i in range(len(Y)):
                self.sgd_step(X[i],Y[i],learning_rate)
                num_examples_seen +=1
        return losses
