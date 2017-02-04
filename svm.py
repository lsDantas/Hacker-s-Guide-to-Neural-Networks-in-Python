import math
import random

class Unit:
    def __init__(self, value, grad):
        # Value computed in the forward pass
        self.value = value
        # The derivative of circuit output w.r.t. this unit, computed in backward pass
        self.grad = grad
class MultiplyGate(object):
    def __init__(self):
        pass

    def forward(self, u0, u1):
        # Store pointers to input Units u0 and u1 and output unit utop
        self.u0 = u0
        self.u1 = u1
        self.utop = Unit(u0.value * u1.value, 0.0)

        return self.utop

    def backward(self):
        # Take the gradient in output unit and chain  # it with the local gradients, which we
        # derived for multiply gates before, then
        # write those gradients to those Units.
        self.u0.grad += self.u1.value * self.utop.grad
        self.u1.grad += self.u0.value * self.utop.grad

class AddGate(object):
    def __init__(self):
        pass

    def forward(self, u0, u1):
        # Store pointers to input units
        self.u0 = u0
        self.u1 = u1
        self.utop = Unit(u0.value + u1.value, 0.0)
        return self.utop

    def backward(self):
        # Add Gate. Derivative wrt both inputs is 1
        self.u0.grad += 1 * self.utop.grad
        self.u1.grad += 1 * self.utop.grad

class SigmoidGate(object):
    def __init__(self):
        pass
    # Helper Function
    def sig(self, x):
        return 1 / (1 + math.exp(-x))

    def forward(self, u0):
        self.u0 = u0
        self.utop = Unit(self.sig(self.u0.value), 0.0)
        return self.utop

    def backward(self):
        s = self.sig(self.u0.value)
        self.u0.grad += (s * (1 - s)) * self.utop.grad

class Circuit():
    def __init__(self):
        self.mulG0 = MultiplyGate()
        self.mulG1 = MultiplyGate()
        self.addG0 = AddGate()
        self.addG1 = AddGate()

    def forward(self, x,y,a,b,c):
        self.ax = self.mulG0.forward(a,x)
        self.by = self.mulG1.forward(b,y)
        self.axpby = self.addG0.forward(self.ax, self.by)
        self.axpbypc = self.addG1.forward(self.axpby, c)
        return self.axpbypc

    def backward(self, gradient_top):
        self.axpbypc.grad = gradient_top
        self.addG1.backward()
        self.addG0.backward()
        self.mulG1.backward()
        self.mulG0.backward()

# SVM class
class SVM():
    def __init__(self):
        # Random Initial Parameter Values
        self.a = Unit(1.0, 0.0)
        self.b = Unit(-2.0, 0.0)
        self.c = Unit(-1.0, 0.0)

        self.circuit = Circuit()

    def forward(self, x,y):
        # Assume x and y are Units
        self.unit_out = self.circuit.forward(x,y, self.a, self.b, self.c)
        return self.unit_out

    def backward(self, label):
        # Label is +1 or -1

        # Reset pulls on a,b,c
        self.a.grad = 0.0
        self.b.grad = 0.0
        self.c.grad = 0.0

        # Compute the pull based on what the circuit output was
        pull = 0.0
        if(label == 1 and self.unit_out.value < 1):
            # The score was too low: pull up.
            pull = 1
        if(label == -1 and self.unit_out.value > -1):
            # The score was too high for a positive example: pull down.
            pull = -1
        self.circuit.backward(pull)

        # Add regularization pull for parameters: toward zero and proportional to value
        self.a.grad += -self.a.value
        self.b.grad += -self.b.value

    def learnFrom(self, x,y,label):
        # Forward pass (set .value in all Units)
        self.forward(x,y)
        # Backward Pass (set .grad in all Units)
        self.backward(label)
        # Parameters respond to through
        self.parameterUpdate()

    def parameterUpdate(self):
        step_size = 0.01
        self.a.value += step_size * self.a.grad
        self.b.value += step_size * self.b.grad
        self.c.value += step_size * self.c.grad

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

svm = SVM()
# A function that computes the classification accuracy
def evalTrainingAccuracy():
    num_correct = 0
    for i in range(0, len(data)):
        x = Unit(data[i][0], 0.0)
        y = Unit(data[i][1], 0.0)

        true_label = labels[i]
        # See if the prediction matches the provided label
        if(svm.forward(x, y).value > 0):
            predicted_label = 1
        else:
            predicted_label = -1

        if(predicted_label == true_label):
            num_correct += 1

    return num_correct/len(data)

# The Learning Loop
for iter in range(0, 400):
    # Pick a Random Data Point
    i = math.floor(random.random() * len(data))
    x = Unit(data[i][0], 0.0)
    y = Unit(data[i][1], 0.0)
    label = labels[i]
    svm.learnFrom(x, y, label)

    if( iter % 25 == 0):
        # Every 10 interations...
        print("Training accuracy at iter " + str(iter) + ": " + str(evalTrainingAccuracy()))
