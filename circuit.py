def forwardMultiplyGate(a,b):
    return a * b

def forwardAddGate(a,b):
    return a + b

def forwardCircuit(x,y,z):
    q = forwardAddGate(x,y)
    f = forwardMultiplyGate(q,z)
    return f

x = -2
y = 5
z = -4
q = forwardAddGate(x,y) # q is 3
f = forwardCircuit(x,y,z) #Result is -12

# Gradient of the Multiply Gate with respect to its inputs
# wrt is short for "with respect to"
derivative_f_wrt_z = q # 3
derivative_f_wrt_q = z # -4

# Derivative of the Add Gate with respect to its inputs
derivative_q_wrt_x = 1.0
derivative_q_wrt_y = 1.0

# Chain Rule
derivative_f_wrt_x = derivative_q_wrt_x * derivative_f_wrt_q # -4
derivative_f_wrt_y = derivative_q_wrt_y * derivative_f_wrt_q # -4

# Final Gradient, from above [-4, -4, 3]
gradient_f_wrt_xyz = [derivative_f_wrt_x, derivative_f_wrt_y, derivative_f_wrt_z]

# Let the inputs responde to the force/tug
step_size = 0.01
x = x + step_size * derivative_f_wrt_x # -2.04
y = y + step_size * derivative_f_wrt_y # 4.96
z = z + step_size * derivative_f_wrt_z # -3.97

# Our circuit now better give higher output:
q = forwardAddGate(x,y)
f = forwardMultiplyGate(q,z)

print("Result: %f" % f)

# Numerical Gradient Check

x = -2
y = 5
z = -4

h = 0.0001
x_derivative = (forwardCircuit(x + h, y, z) - forwardCircuit(x,y,z))/h # -4
y_derivative = (forwardCircuit(x, y + h, z) - forwardCircuit(x,y,z))/h # -4
z_derivative = (forwardCircuit(x, y, z + h) - forwardCircuit(x,y,z))/h

gradient = [x_derivative, y_derivative, z_derivative]
print("Gradient: ")
print(gradient)
