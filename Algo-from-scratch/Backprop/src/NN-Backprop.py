#import cv2  # Uncomment if you have OpenCV and want to run the real-time demo
import numpy as np



def getLoss(Y, yhat):
    n = Y.shape[0]
    loss = ((-1 /(2* n)) * np.sum(Y * np.log(yhat)))
    return loss

def getZ(X, W, B):
    return (X.dot(W))+B
    
def getRelu(z):
    relu = np.maximum(0,z)
    return relu

def gradRelu (z):
    z[z<=0] = 0
    z[z>0] = 1
    return z

def getHiddenlayerSizes(num_layer, layer_sizes):
    if(len(layer.sizes)!= num_layer):
        raise Exception('number of sizes of layers is not equal to the number of layers')
    
    hidden_layer_sizes = {}
    
    for i in range(num_layer):
        hidden_layer_sizes[i] = layer_sizes[i]

    return hidden_layer_sizes

def softmaxCalc2 (z):
   
    softmax = (np.exp(z.T) / np.sum(np.exp(z), axis=1)).T
    return softmax

def softmaxCalc(Y):
    '''
    exp_Y = np.exp(Y)
    #print("Shape y:",Y.shape,exp_Y.shape)
    softmax_Y=exp_Y.T/np.sum(exp_Y,axis=1)
    '''
        
    x=Y
    z = x - np.max(x)
    
    numerator = np.exp(z)
    denominator = np.sum(numerator,axis=1)
    softmax = numerator.T/denominator
    return softmax.T
    
def findBestHyperParam(hiddenLayers, layer_sizes, learningRates, batch_sizes, epochs, alpha):
    #use itertools to generate combinations of the parameters
    j=0

# def init_mat(X, Y, hidden_units, hidden_layers):
#     W = []
#     B = []
#     w = np.zeros((X.shape[1], hidden_units))
#     b = np.zeros(hidden_units)
#     W.append(w)
#     B.append(b)
    
#     for i in range(hidden_layers-1):
#         w = np.zeros((hidden_units, hidden_units))
#         b = np.zeros(hidden_units)
#         W.append(w)
#         B.append(b)
        
#     w = np.zeros((hidden_units, Y.shape[1]))
#     b = np.zeros(Y.shape[0])
#     W.append(w)
#     B.append(b)
#     return W,B

def init_weights(num_inputs, num_output, num_hidden, num_units):
    """
    Returns a list of 2D arrays. Theses entries correlate to the weights for the 
    neural netowrk layer at that index. All values are initialized to zero.
    """
    weights = []
    
    weights.append(np.sqrt(1/num_units)*np.random.randn(num_inputs, num_units))
    for _ in range(1, num_hidden):
        weights.append(np.sqrt(1/num_units)*np.random.randn(num_units, num_units))
    weights.append(np.sqrt(1/num_units)*np.random.randn(num_units, num_output))
    return weights

def init_biases(num_output, num_hidden, num_units):
    """
    Returns a list of vectors representing the biases of the layer.
    """
    biases = []
#     for _ in range(num_hidden):
#         biases.append(0.01* np.random.randn(num_units))
#     biases.append(0.01 * np.random.randn(num_output))
    
    for layer in range(num_hidden):
        biases.append(np.zeros((num_units)))
    biases.append(np.zeros((num_output)))
    return biases

def forwardProp(X, W, B, hiddenLayers):
    #z = getZ(X, W, B)
    h = X
    zs = []
    hs = []
    j = 0
    for i in range(hiddenLayers):
        
        z = np.dot(h, W[i]) + B[i]
        h = getRelu(z)
        zs.append(z)
        hs.append(h)
        j=j+1
    
    z = np.dot(h, W[j]) + B[j]
    
    h = softmaxCalc(z)
    zs.append(z)
    hs.append(h)
    return zs, hs
        
        
def backProp(y, X, Z, A, W, B, hLayers, eta):
    g = A[-1]-y
    
    dw = np.dot(A[-2].T, g)
    db = np.sum(g, axis=0)
    dws = [dw]
    dbs = [db]
    
    for i in range(hLayers, 0, -1):
        g = np.dot(g, W[i].T)
        g = gradRelu(Z[i-1]) * g
        
        if(i!=1):
            dw = np.dot(A[i-2].T, g)
            db = np.sum(g, axis=0)
            dws.append(dw)
            dbs.append(db)
        else:
            dw = np.dot(X.T, g)
            db = np.sum(g, axis=0)
            dws.append(dw)
            dbs.append(db)
    dws = dws[::-1]
    dbs = dbs[::-1]
    for j in range(len(W)-1, 0, -1):
#         print("backprop i", j)
#         print("W[{}].shape = {}".format(j, W[j].shape))
#         print("dws[{}].shape = {}".format(j, dws[j].shape))
#         W[j] = W[j] - eta* (1/X.shape[0]) * dws[j]
#         B[j] = B[j] - eta * (1/X.shape[0]) * dbs[j].reshape(B[j].shape)
        W[j] = W[j] - eta * dws[j]
        B[j] = B[j] - eta * dbs[j].reshape(B[j].shape)
        
    return W,B

def SGD (X, Y, W, B, epochs, batch_size, hiddenLayers, alpha, eta, test_X, test_Y):
    
    for i in range(epochs):
        batch = _yield_minibatches_idx(batch_size,X,shuffle=True)
        #print("batch=",batch)
        for j in batch:
            x = X[j]
            y = Y[j]
            Z, A = forwardProp(x, W, B, hiddenLayers)
            W, B = backProp(y, x, Z, A, W, B, hiddenLayers, eta)
        
        # getting loss for the epoch
        z, h = forwardProp(test_X, W, B, hiddenLayers)
        loss = getLoss(test_Y, h[hiddenLayers])
        print("Epoch = {}, Loss= {}".format(i, loss))
          
    return W,B

def _yield_minibatches_idx(n_batches, X, shuffle=True):
    indices = np.arange(X.shape[0])

    n_batches = X.shape[0]/n_batches
    
    if shuffle:
        indices = np.random.permutation(indices)
        if n_batches > 1:
            remainder = X.shape[0] % n_batches

            if remainder:
                minis = np.array_split(indices[:-remainder], n_batches)
                minis[-1] = np.concatenate((minis[-1], indices[-remainder:]), axis=0)
            else:
                minis = np.array_split(indices, n_batches)

        else:
            minis = (indices,)

        for idx_batch in minis:
            yield idx_batch
    


def reportCosts (w, trainingFaces, trainingLabels, testingFaces, testingLabels, alpha = 0.):
    traningloss=J(w, trainingFaces, trainingLabels, alpha)
    validationloss=J(w, testingFaces, testingLabels, alpha)
    print("training loss",traningloss.shape)
    print("validationloss loss",validationloss.shape)
    print ("Training cost: {}".format(J(w, trainingFaces, trainingLabels, alpha)))
    print ("Testing cost:  {}".format(J(w, testingFaces, testingLabels, alpha)))

def getAccuracy(y_pred,y_actual):
    pred = np.argmax(y_pred, axis=1)

    actual = np.argmax(y_actual, axis=1)

    accuracy = np.sum(pred == actual)
    accuracy = accuracy/len(pred)

    return accuracy

def train_model(X, Y, hidden_layers, hidden_units, epochs, batch_size, eta, test_X, test_Y):
    
    W = init_weights(X.shape[1], Y.shape[1], hidden_layers, hidden_units)
    B = init_biases(Y.shape[1], hidden_layers, hidden_units)
    
    #print(W[3])
    W, B = SGD(X, Y, W, B, epochs, batch_size, hidden_layers, alpha, eta, test_X, test_Y)
    
    z, h = forwardProp(test_X, W, B, hidden_layers)
    print("Accuracy  =", getAccuracy(h[hidden_layers], test_Y))
    return W,B
    
if __name__ == "__main__":
    # Load data
    if ('train_X' not in globals()):  # In ipython, use "run -i homework2_template.py" to avoid re-loading of data
        train_X = np.load("data/mnist_train_images.npy")
        train_Y = np.load("data/mnist_train_labels.npy")
        val_X = np.load("data/mnist_validation_images.npy")
        val_Y = np.load("data/mnist_validation_labels.npy")

    alpha = 1E-5
    epochs = 150
    eta = 0.005
    batch_size = 10
    hidden_layers = 5
    hidden_units = 50
    np.random.seed(1234)
    
    W, B = train_model(train_X, train_Y, hidden_layers, hidden_units, epochs, batch_size, eta, val_X, val_Y)
  
    #print("W,B=", len(W),len(B), W[1].shape, len(B[0]))
#     w1 = method2(trainingFaces, trainingLabels, epochs, batch_size, alpha, neta)

#     for w in [ w1 ]:
#         reportCosts(w, trainingFaces, trainingLabels, testingFaces, testingLabels)
#     yhat = softmaxCalc(testingFaces, w1)
#     accuracy = accuracy_score(testingLabels.argmax(axis=1),yhat.argmax(axis=1))
#     print("accuracy=", accuracy)

     
    
    #detectSmiles(w3)  # Requires OpenCV
