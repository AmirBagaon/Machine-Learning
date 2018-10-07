#204313100
import numpy as np

#Global
totalPixels = 784
possibleAnswers = 10 # 0-9 items
unzip = lambda a: zip(*a)


#Activation Functions
sigmoid = lambda x: 1 / (1 + np.exp(-x))
tanh = lambda  x: ((np.exp(x)-np.exp(-x)) / (np.exp(x)+np.exp(-x)))
RelU = lambda x: np.maximum(x, 0)

#SoftMax Func
def softMax(x):
    temp = np.exp(x - np.max(x))
    return temp / (np.sum(temp))

#Back Propogation Func
def bprop(fprop_cache):
  x, z1, h1, z2, h2 = [fprop_cache[key] for key in ('x', 'z1', 'h1', 'z2', 'h2')]
  dz2 = (h2 - y)  # dL/dz2
  dW2 = np.dot(dz2, h1.T)  # dL/dz2 * dz2/dw2
  db2 = dz2  # dL/dz2 * dz2/db2
  dz1 = np.dot(fprop_cache['W2'].T, (h2 - y)) * sigmoid(z1) * (1 - sigmoid(z1))  # dL/dz2 * dz2/dh1 * dh1/dz1 #SigmoidDiv
  dW1 = np.dot(dz1, x.reshape(784,1).T)  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/dw1
  db1 = dz1  # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
  return {'b1': db1, 'W1': dW1, 'b2': db2, 'W2': dW2}

#Forward Propogation Func
def forwardP(params, x):
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    # Add W and bias
    z1 = np.dot(W1, x).reshape(hidden_layer, 1) + b1

    # Activation func
    #h1 = tanh(z1)
    #h1 = ReLU(z1)
    h1 = sigmoid(z1)

    # Again and softmax
    z2 = np.dot(W2, h1) + b2
    h2 = softMax(z2)

    # Return all values
    ret = {'x': x, 'answer': np.argmax(h2), 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2}
    for key in params:
        ret[key] = params[key]
    return ret

#Create Test Function
def createTest():
    target = open("test.pred", 'w')
    test_x = np.loadtxt("test_x_int")
    test_x = test_x / 255

    for i in range(len(test_x)):
        xi = test_x[i]
        fprop_cache = forwardP(params, xi)
        answer = str(fprop_cache['answer'])
        target.write(answer + '\n')
    target.close()


#Load data and shuffle
train_x = np.loadtxt("train_x")
maxPixVal = 255 #To normalize
train_x = train_x / maxPixVal
train_y = np.loadtxt("train_y")
length = len(train_y)

#Create hyperparams, including epochs, learning rate and hidden layer num
lrnRate = 0.007
epochs = 30
hidden_layer = 80

#Create W1,b1,W2,b2 randomally
W1 = np.random.uniform(-0.05,0.05,[hidden_layer, totalPixels])     # First layer to hidden layer
b1 = np.random.uniform(-0.05,0.05,[hidden_layer,1]) # Bias of Vec Amuda of hidden_layer which we will add to hidden_layer
W2 = np.random.uniform(-0.05,0.05,[possibleAnswers,hidden_layer]) # Hiden+layer to final layer
b2 = np.random.uniform(-0.05,0.05,[possibleAnswers,1]) # Bias of Vec Amuda of hidden_layer which we will add to final layer

params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

#Learning part
print "Start epochs"
for i in range(epochs):
    #Attach and shuffle examples and labels
    temp = zip(train_x, train_y)
    np.random.shuffle(temp)
    train_x, train_y = unzip(temp)
    #Train
    for j in range(length):
        # Take vec that represent clothes-classes, set it as zeros, and turn the label to 1
        y = np.zeros((possibleAnswers, 1))
        index = int(train_y[j])
        y[index] = 1.0
        #Take 1 example
        x = train_x[j]
        # Do Forward Propogation
        fprop_cache = forwardP(params, x)
        # And Back Propogation
        bprop_cache = bprop(fprop_cache)

        #And Finaly, SGD
        params['b1'] = params['b1'] - np.dot(bprop_cache['b1'],lrnRate)
        params['b2'] = params['b2'] - np.dot(bprop_cache['b2'],lrnRate)
        params['W1'] = params['W1'] - np.dot(bprop_cache['W1'],lrnRate)
        params['W2'] = params['W2'] - np.dot(bprop_cache['W2'],lrnRate)
    print "Epoc ", i, "Finished"

#Test part
createTest()
