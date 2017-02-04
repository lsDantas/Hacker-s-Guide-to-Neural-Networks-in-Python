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

class Circuit():
    def __init__(self):
        self.mulG0 = MultiplyGate()
        self.mulG1 = MultiplyGate()
        self.addG0 = AddGate()
        self.addG1 = AddGate()

    def forward(x,y,a,b,c,):
        self.ax = self.mulG0.forward(a,x)
        self.by = self.mulG1.foward(b,y)
        self.axpby = self.addG0.foward(self.ax, self.by)
        self.axpbypc = self.addG1.forward(self.axpby, c)
        return this.axpbyc

    def backward(gradient_top):
        self.axpbypc.grad = gradient_top
        self.addG1.backward()
        self.addG0.backward()
        self.mulG1.backward()
        self.mulG0.backward()
