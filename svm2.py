import math

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

# Initial Parameters
a = 1
b = -2
c = -1
for iteration in range(0,400):
    # Pick a Random Data Point
    i = math.floor(math.random() * len(data))
    x = data[i][0]
    y = data[i][1]
    label = labels[i]

    # Compute Pull
    score = a*x + b*y + c
    pull = 0.0
    if(label == 1 and score < 1):
        pull = 1
    if(label == -1 and score > -1):
        pull = -1

    # Compute Gradient and Update Parameters
    step_size = 0.01
    a += step_size * (x * pull - a)
    b += step_size * (y * pull - b)
    c += step_size * (1 * pull)
