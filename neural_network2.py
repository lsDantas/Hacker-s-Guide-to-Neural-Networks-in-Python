import math
import random

# Data

data = []
labels = []
data.append([1.2, 0.7])
labels.append(1)
data.append([-0.3, -0.5])
labels.append(-1)
data.append([3.0, 0.1])
labels.append(1)
data.append([-0.1, -1.0])
labels.append(-1)
data.append([-1.0, 1.1])
labels.append(-1)
data.append([2.1, -3])
labels.append(1)

# Parameters

a1 = random.random() - 0.5 # A random number between -0,5 and 0.5
b1 = random.random() - 0.5
c1 = random.random() - 0.5

a2 = random.random() - 0.5
b2 = random.random() - 0.5
c2 = random.random() - 0.5

a3 = random.random() - 0.5
b3 = random.random() - 0.5
c3 = random.random() - 0.5

a4 = random.random() - 0.5
b4 = random.random() - 0.5
c4 = random.random() - 0.5
d4 = random.random() - 0.5

num_correct = 0
for iteration in range(0,400):
    i = math.floor(random.random() * len(data))
    x = data[i][0]
    y = data[i][1]
    label = labels[i]

    # Compute Forward Pass
    n1 = max(0, a1*x + b1*y +c1) # Activation of 1st hidden neuron
    n2 = max(0, a2*x + b2*y + c2) # 2nd neuron
    n3 = max(0, a3*x, b3*y + c3) # 3rd neuron
    score = a4*n1 + b4*n2 + c4*n3 + d4 #score

    # Compute the pull on top
    pull = 0.0
    if(label == 1 and score < 1):
        pull = 1 # We want higher output! Pull up.
    if(label == -1 and score > -1):
        pull = -1 # We want lower output! Pull down.

    if(score > 0):
        predicted_label = 1
    if(score < 0):
        predicted_label = -1
    if(predicted_label == label):
        num_correct += 1

    # Now compute backward pass to all parameters of the Model

    # Backprop through the last "score" neuron
    dscore = pull
    da4 = n1 * dscore
    dn1 = a4 * dscore
    db4 = n2 * dscore
    dn2 = b4 * dscore
    dc4 = n3 * dscore
    dn3 = c4 * dscore
    dd4 = 1.0 * dscore

    # Backprop the ReLU non-linearities, in placeholder
    # i.e. just set gradients to zero if the neurons did not "fire"

    if(dn3 == 0):
        dn3 = 0
    if(dn2 == 0):
        dn2 = 0
    if(dn1 == 0):
        dn1 = 0

    # Backprop to parameters of neuron 1
    da1 = x * dn1
    db1 = y * dn1
    dc1 = 1.0 * dn1

    # Backprop to parameters of neuron 2
    da2 = x * dn2
    db2 = y * dn2
    dc2 = 1.0 * dn2

    # Backprop to parameters of neuron 3
    da3 = x * dn3
    db3 = y * dn3
    dc3 = 1.0 * dn3

    # Phew! End of backprop!
    # Note we could have also backpropped into x,y
    # but we do not need these gradients. We only use the
    # gradients on our parameters in the parameter
    # update, and we discard x and y.

    # Add the pulls from the regularization, tugging all
    # the multiplicative paramaters (i.e. not the
    # biases) downward, proportional to their value

    da1 += -a1
    da2 += -a2
    da3 += -a3

    db1 += -b1
    db2 += -b2
    db3 += -b3

    da4 += -a4
    da4 += -a4
    da4 += -a4

    # Finally, do the parameter update
    step_size = 0.01
    a1 += step_size * da1
    b1 += step_size * db1
    c1 += step_size * dc1
    a2 += step_size * da2
    b2 += step_size * db2
    c2 += step_size * dc2
    a3 += step_size * da3
    b3 += step_size * db3
    c3 += step_size * dc3
    a4 += step_size * da4
    b4 += step_size * db4
    c4 += step_size * dc4
    d4 += step_size * dd4

    # Wow this is tedious, please use for loops in prod.
    # We're done!
    if( iteration % 25 == 0):
        print("Accuracy: " + str(num_correct/(iteration+1)))
