#Strategy 1 - Random Local Search

import math
import random

def forwardMultiplyGate(x, y):
    return x * y

x = -2
y = 3

# Try changin x,y randomly small amounts and keep track of what works best
tweak_amount = 0.01
best_out = -math.inf
best_x = x
best_y = y

for k in range(0,100):
    # Tweak x a bit
    x_try = x + tweak_amount * (random.random() * 2 - 1)
    # Tweak y a bit
    y_try = y + tweak_amount * (random.random() * 2 - 1)
    out = forwardMultiplyGate(x_try, y_try)
    if(out>best_out):
        #Best Improvement yet! Keep track of the x and y
        best_out = out
        best_x = x_try
        best_y = y_try

print("Best X: %f\nBest Y: %f\n" % (best_x, best_y))

results = forwardMultiplyGate(best_x, best_y)
print("Results: %f" % results)
