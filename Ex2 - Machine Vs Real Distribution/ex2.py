import numpy as np
import matplotlib.pyplot as plt


group = [1, 2, 3] #Our set of 'a', which will be 1, 2 or 3.


def createTraningSet(tSet):
    """
    Creates and returns the Traning set
    :param tSet: The training set
    :return: the set
    """
    numOfPoints = 100
    for a in group:
        # sample 100 of each class
        for num in np.random.normal(2 * a, 1, numOfPoints):
            tSet.append((num, a))
    return tSet


def createPlot(w, b):
    """
    Create the plot
    :param w: weights
    :param b: bias
    """
    points = np.linspace(0, 10, 100, endpoint=False)  #Make deviation of 100 points
    trueDist =[]
    trainingDist = []

    for p in points:
        trueDist.append( density(p, 2, 1) / ( density(p, 2, 1) + density(p, 4, 1) + density(p, 6, 1)) )
        trainingDist.append(softmax(w * p + b)[0])

    #Create the Plot
    plt.title('Ex2 Distributions')
    plt.xlabel("scalar")
    plt.ylabel("Probabilty")
    plt.plot(points, trueDist, color='orange', linestyle='solid', linewidth=2)
    plt.plot(points, trainingDist, color='green', linestyle='dashdot',linewidth=2)
    plt.legend(('True dist.', 'Estimated prob.'))
    plt.show()


def density(x, mean, standartDev):
    """
    The true density function
    :return: The result after calculation
    """
    temp = - (np.power(x - mean, 2) / 2 * np.power(standartDev, 2))
    result = (1 / (standartDev * np.sqrt(2 * np.pi)))* np.exp(temp)
    return result


def softmax(num):
    #Calculate and return the result of softmax of num
    e = np.exp(num)
    result = e / np.sum(e)
    return result


def calculate(tSet):
    #Calculating and training data and make the graph
    epochs = 1000
    learningRate = 0.01
    length = len(group)
    w = np.random.random((length, 1)) #weight
    b = np.random.random((length, 1)) #bias

    i = 1
    while i <= epochs:
        i = i + 1
        # Shuffle again
        np.random.shuffle(tSet)
        for x, y in tSet:
            index = y - 1
            temp = softmax(w * x + b)

            # Calculate w^t, which equeal to w^(t-1) - rate*dw
            dw = x * temp
            dw[index] = x * temp[index] - x
            w = w - learningRate * dw

            # Calculate b^t, which equeal to b^(t-1) - rate*db
            db = temp
            db[index] = temp[index] - 1
            b = b - learningRate * db

    createPlot(w, b)




tSet = []
createTraningSet(tSet)
calculate(tSet)