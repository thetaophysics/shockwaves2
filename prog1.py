import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
from matplotlib.animation import FuncAnimation

def main():
    e = 1.0
    N = 101  #number of grid points for position
    beta = 0.5 #stability condition
    dx = 2.0 / (N - 1)
    dt = beta * (dx / e)
    t_max = 0.30
    N_t = round(t_max / dt)    # t_max = n * dt for # of grid points for time
    print(f'N_t = {N_t}')
    u0 = np.zeros(N, float) # amplitude for time t
    u = np.zeros((N, N_t), float) # amplitude for t+h

    ini_wave(u, u0, N, dx)
    u[0, :] = 0
    u[N-1, :] = 0

    leapfrog = np.transpose(LeapFrog(u, u0, N, N_t, dt, dx, beta))
    lax_wend = np.transpose(Lax_Wendroff(u, u0, N, N_t, dt, dx, beta))
    
    #Print to file
    np.savetxt('shock_lf.txt', leapfrog, fmt='%f')
    np.savetxt('shock_lw.txt', lax_wend, fmt='%f')
    Z1 = np.loadtxt("shock_lf.txt",float) 
    Z2 = np.loadtxt("shock_lw.txt",float) 
    part_b(Z2)
    part_c(Z1)
    part_c(Z2)
    return

def ini_wave(u, u0, N, dx):
    for i in range(N):
        u0_i = 3.0 * np.sin(np.pi*i*dx)
        u0[i] = u0_i
    u[:,0] = u0
    return u

def LeapFrog(u, u0, N, N_t, dt, dx, beta):
    for j in range(1, N_t-1):
        for i in range(1, N-1):
            u[i, j] = u[i, j-1] - (beta / 4.0) * (pow(u[i+1, j-1], 2) - pow(u[i-1,j-1],2))
            # if (i == 5):
            #     print(u[i, j])
    return u

def Lax_Wendroff(u, u0, N, N_t, dt, dx, beta):
    for j in range(1, N_t-1):
        for i in range(1, N-1):
            a = u[i, j-1]
            b = pow(u[i+1, j-1], 2) - pow(u[i-1,j-1],2)
            c = (u[i,j-1] - u[i+1,j-1]) * (pow(u[i+1,j-1],2) - pow(u[i, j-1],2)) + (u[i,j-1] - u[i-1,j-1]) * (pow(u[i-1,j-1],2) - pow(u[i,j-1],2))
            u[i,j] = a - (beta / 4.0) * b + (pow(beta,2) / 8.0) * c
            # if (i == 9):
            #     print(u[i,j])
    return u

def part_b(Z):
    # 3D plots
    xmax=2.0    # x [unitless]
    ymax=0.15   # time [unitless]

    yA = np.linspace(0,ymax,Z.shape[0]) 
    xA = np.linspace(0,xmax,Z.shape[1])
    xA, yA = np.meshgrid(xA, yA)

    fig1 = plt.figure(1)
    ax1 = fig1.add_subplot(projection='3d')
    surf1 = ax1.plot_surface(xA, yA, Z, cmap=cm.coolwarm,linewidth=0, antialiased=False)
    ax1.set_title("Shock Waves by Lax_Wendroff (t=0.15)")
    ax1.set_xlabel("position")
    ax1.set_ylabel("time")
    ax1.set_zlabel("height")
    fig1.colorbar(surf1, shrink=0.5, aspect=5)

    return plt.show()

def part_c(Z):
    x_final=2.0                         
    T_final = 0.30

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set(xlim=(0, x_final), ylim=(-20,20))
    ax.set_title("Shock Waves by Lax_Wendroff (b = 0.5)")
    ax.set_xlabel('x')
    ax.set_ylabel('u(x,t)')

    x = np.linspace(0, x_final, Z.shape[1])
    t = np.linspace(0, T_final, Z.shape[0])
    X2, T2 = np.meshgrid(x, t)

    line = ax.plot(x, Z[0, :], color='k', lw=2)[0]

    def animate(i):
        line.set_ydata(Z[i, :])
    
    nim = FuncAnimation(fig, animate, interval=50, frames=len(t)-1,repeat=False)

    plt.draw()
    return plt.show()

if __name__ == '__main__':
    main()









