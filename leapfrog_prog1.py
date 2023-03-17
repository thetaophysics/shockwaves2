import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants and variables
e = 1.0
N = 5
m = N - 1
beta = 0.1
delX = 2.0 / m
delT = beta * (delX / e)

t_max = 0.15
n = round(t_max / delT)    # n is integer where t_max = n * delT
print(f"delT = {delT}, delX = {delX}, n = {n}")
u0 = np.zeros(N, float) # amplitude for time t
u = np.zeros(N, float) # amplitude for t+h

f = open('prog1_1.dat', 'w')

def initial_wave(inDex):
    return 3.0 * np.sin(3.14*inDex*delX)

# Boundary conditions:
# u[0], u[m] = 0, 0

f.write('j = 0\n')
for i in range (m+1):   #initialize the sinusoidal wave at t = 0 or j = 0
    u0[i] = initial_wave(i)
    f.write(str(u0[i])+'\n')

f1 = open('prog1_2.dat', 'w')
for j in range(1,n):  # Name of the first row where j=1,2,3,...n
    f1.write("j = %d    " %j)
f1.write("\n")

for j in range(1, n):    #time
    for i in range(N):     # position i = 0, 1, 2, ... (m - 2)
        if (i == 0): u[i] = 0   #BC
        elif (i == m): u[i] = 0   #BC
        else: u[i] = u0[i] - (beta / 4.0) * ((u0[i+1])**2 - (u0[i-1])**2)
        f1.write(str(u[i]) + "\n")
    u0 = u


f.close()
f1.close()



        


        

