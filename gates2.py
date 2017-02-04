#Strategy 2 - Numerical Gradient

def forwardMultiplyGate(x, y):
    return x * y

x = -2
y = 3

out = forwardMultiplyGate(x,y)
h = 0.0001

#Compute Derivative with Respect to X
xph = x + h #1.9999
out2 = forwardMultiplyGate(xph,y) #-5.9997
x_derivative = (out2 - out)/h #3.0

#Compute Derivative with Respect to Y
yph = y + h #3.0001
out3 = forwardMultiplyGate(x, yph) #-6.0002
y_derivative = (out3 - out)/h #-2.0

print("Derivative of X: %f\nDerivative of Y: %f\n" % (x_derivative, y_derivative))

#Apply Gradient
step_size = 0.01
x = x + step_size * x_derivative # X becomes -1.97
y = y + step_size * y_derivative # Y becomes 2.98
out_new = forwardMultiplyGate(x,y)

print("Results: %f" % out_new)
