#Strategy 3 - Analytic Gradient

def forwardMultiplyGate(x, y):
    return x * y

x = -2
y = 3

out = forwardMultiplyGate(x,y) # Before: -6
h = 0.0001

x_gradient = y # By our complex mathematical derivation above
y_gradient = x

step_size = 0.01
x += step_size * x_gradient # -2.03
y += step_size * y_gradient # 2.98

out_new = forwardMultiplyGate(x,y) #-5.87. Higher output! Nice.

print("Results: %f" % out_new)
