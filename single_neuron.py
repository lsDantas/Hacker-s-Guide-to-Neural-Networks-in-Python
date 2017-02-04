import math

# Every unit corresponds to a wire in the diagrams
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

# Create the Input Units
a = Unit(1.0, 0.0)
b = Unit(2.0, 0.0)
c = Unit(-3.0, 0.0)
x = Unit(-1.0, 0.0)
y = Unit(3.0, 0.0)

# Create the gates

mulG0 = MultiplyGate()
mulG1 = MultiplyGate()
addG0 = AddGate()
addG1 = AddGate()
sG0 = SigmoidGate()

# Do the Forward Pass
def forwardNeuron():
    ax = mulG0.forward(a,x) # a*x = -1
    by = mulG1.forward(b, y) # b*y = 6
    axpby = addG0.forward(ax, by) # a*x + b*y = 5
    axpbypc = addG1.forward(axpby, c) # a*x + b*y + c = 2
    s = sG0.forward(axpbypc) # sig(a*x + b*y + c) = 0.8808
    return s

s = forwardNeuron()
print("Results: %f" % s.value)

# Compute Gradient
s.grad = 1.0
sG0.backward() # Writes gradient into axpbypc
addG1.backward() # Writes gradients into axpby and c
addG0.backward() # Writes gradients into ax and by
mulG1.backward() # Writes gradients into b and y
mulG0.backward() # Writes gradients into a and x

step_size = 0.01
a.value += step_size * a.grad # a.grad is -0.105
b.value += step_size * b.grad # b.grad is 0.315
c.value += step_size * c.grad # c.grad is 0.105
x.value += step_size * x.grad # x.grad is 0.105
y.value += step_size * y.grad # y.grad is 0.210

s = forwardNeuron();
print("Circuit output after on Backprop: %f" % s.value)


# Checking the Gradient
def forwardCircuitFast(a,b,c,x,y):
    return 1/(1 + math.exp( - (a*x + b*y + c)))

a,b,c,x,y = 1,2,-3,-1,3

h = 0.0001

a_grad = (forwardCircuitFast(a+h,b,c,x,y) - forwardCircuitFast(a,b,c,x,y))/h
b_grad = (forwardCircuitFast(a,b+h,c,x,y) - forwardCircuitFast(a,b,c,x,y))/h
c_grad = (forwardCircuitFast(a,b,c+h,x,y) - forwardCircuitFast(a,b,c,x,y))/h
x_grad = (forwardCircuitFast(a,b,c,x+h,y) - forwardCircuitFast(a,b,c,x,y))/h
y_grad = (forwardCircuitFast(a,b,c,x,y+h) - forwardCircuitFast(a,b,c,x,y))/h

gradient_check = [a_grad, b_grad, c_grad, x_grad, y_grad]
print(gradient_check)
