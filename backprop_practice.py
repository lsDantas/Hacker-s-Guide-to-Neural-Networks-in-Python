import math

# Multiply Gate
x = a * b
# and given gradient on x(dx), we saw that in backprop we would compute:
da = b * dx
db = a * dx

# Add Gate
x = a + b
da = 1.0 * dx
db = 1.0 * dx

# Let's compute x = a + b + c in two steps:
q = a + b # Gate 1
x = q + c # Gate 2

# Backward Pass:
dc = 1.0 * dx # Backprop Gate 2
dq = 1.0 * dx
da = 1.0 * dq # Backprop Gate 1
db = 1.0 * dq

x = a + b + c
da = 1.0 * dx
db = 1.0 * dx
dc = 1.0 * dx

# Combining gates
x = a * b + c
# Given dx, backprop in-one-sweep would be =>
da = b * dx
db = a * dx
dc = 1.0 * dx

def sig(x):
    return 1 / (1 + math.exp(-x))

# Let's do our neuron in two steps:
q = a*x + b*y + c
f = sig(q) # Sig is the sigmoid function
# and now the backward pass, we are given df, and:
df = 1
dq = (f * (1 - f)) * df
# and now we chain it to the inputs
da = x * dq
dx = a * dq
dy = b * dq
db = y * dq
dc = 1.0 * dq

x = a*a + b*b + c*c
da = 2 * a * dx
db = 2 * b * dx
dc = 2 * c * dx

x = math.pow(((a * b + c) * d), 2)
#pow(x,2) squares the input JS
