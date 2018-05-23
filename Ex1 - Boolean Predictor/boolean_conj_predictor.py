import sys

def fillArray(size):
    '''
    Returns array with 'size' size full with 1's.
    :param size: the size of the wanted array
    :return: array full of 1
    '''
    temp = []
    for i in range(size):
        temp.append('1')
    return temp

def Y_roof(positive, negative, instance):
    '''
    Return the state for Y_ROOF (Y_HAT)
    :param positive: The array that represents the positive vars in the hypotasis
    :param negative: The array that represents the negative vars in the hypotasis
    :param instance: the part of the Domain
    :return: 0 for false, 1 for true
    '''
    for i, x in enumerate(instance):
        if (x == '1' and positive[i] == '1') or (x == '0' and negative[i] == '1'):
            return '0'
    return '1'

#First we check that user did insert the path of the file
if len(sys.argv)!=2 :
    print "Invalid args. The command should be: 'python filename.py filepath' "
    sys.exit()

#Lists of Domain and Label
X = []
Y = []

#Get file path
file_path = sys.argv[1]

try:
    with open(file_path, "r") as dataBase:
        for line in dataBase:
            line = line.split()
            X.append(line[:-1])
            Y.append(line[-1])
    d = len(X[0])

    #Fill 2 arrays with 1. They will represent the vars
    positive = fillArray(d)
    negative = fillArray(d)

    #Start the algorithm
    countT = -1
    for instanceT in X:
        countT = countT + 1
        if Y[countT] == '1' and Y_roof(positive, negative, instanceT) == '0':
            index = -1
            for x in instanceT:
                index = index + 1
                if x == '1':
                    negative[index] = '0'
                if x == '0':
                    positive[index] ='0'

    #Now prepare the output
    #For cell with 1 in positive array, we will print the positive var
    #And for 1 in negative array, we will print not(var)
    toWrite = ""
    for x in range(d):
        if positive[x] == '1':
            toWrite += 'X' + str(x + 1) + ','
        if negative[x] == '1':
            toWrite += "not(X" + str(x + 1) + "),"
    toWrite= toWrite[:-1]
    print toWrite
    with open('output.txt', 'w') as output:
        output.write(toWrite)


except:
    print "Could not found file"
