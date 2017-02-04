x = [ [1.2, 0.7], [-0.3, 0.5], [3,2.5]] # Array of 2-Dimensional Data
y = [1, -1, 1] # Array of labels
w = [0.1, 0.2, 0.3] # Example: random numbers
alpha = 0.1 # regularization strength

def cost(x, y, w):
    total_cost = 0.0 # L, in SVM loss function above
    n = len(x)
    for i in range(0, n):
        # Loop over all data points and compute their score
        xi = x[i]
        score = w[0] * xi[0] + w[1] * xi[1] + w[2]

        # Accumlate cost based on how compatible the score is
        # with the label
        yi = y[i]
        costi = max(0, -yi * score + 1)
        print("Example " + str(i) + ": xi = (" + str(xi) + ") and label = " + str(yi))
        print("  score computed to be " + str(("%.3f" % score)))
        print(" => cost computed to be " + str(("%.3f" % costi)))
        total_cost += costi

    reg_cost = alpha * (w[0]*w[0] + w[1]*w[1])
    print("Regularization cost current model is " + str(("%.3f" % reg_cost)))
    total_cost += reg_cost

    print("Total cost is " + str(("%.3f" % total_cost)))
    return total_cost

cost(x,y,w)
