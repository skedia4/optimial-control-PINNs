#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 12:11:47 2022

@author: shubham
"""
from casadi import *
import numpy as np
import random
import matplotlib.pyplot as plt

#for more in range(10):
    
T = 1# #######################################################################Time horizon
N = 20 ########################################################################### Iterations

x = MX.sym('x',3)
u= MX.sym('u',2)

xdot = vertcat(u[0]*cos(x[2]), u[0]*sin(x[2]), u[1])

L = dot(x,x) + 1*dot(u,u)


M = 4 ########################################################################## RK4 steps per interval
DT = T/N/M
f = Function('f', [x, u], [xdot, L])
X0 = MX.sym('X0', 3)
U = MX.sym('U',2)
X = X0
Q = 0
for j in range(M):
    k1, k1_q = f(X, U)
    k2, k2_q = f(X + DT/2 * k1, U)
    k3, k3_q = f(X + DT/2 * k2, U)
    k4, k4_q = f(X + DT * k3, U)
    X=X+DT/6*(k1 +2*k2 +2*k3 +k4)
    Q = Q + DT/6*(k1_q + 2*k2_q + 2*k3_q + k4_q)
F = Function('F', [X0, U], [X, Q],['x0','p'],['xf','qf'])




w=[]
w0 = []
lbw = []
ubw = []
J = 0
g=[]
lbg = []
ubg = []

int_state = [0,1,pi/2] #0,1,-pi/2,0,0###############################################################initial states

Xk = MX(int_state) 
Uk = MX.sym('U', 2*N)
for k in range(N):
   
    lbw += [-inf,-inf]
    ubw += [inf,inf]
    
    u_rand= random.uniform(-10,10)
    psi_rand=random.uniform(-3,3)
    
    #a_rand=0 ; psi_rand=0 ################################################################### zero controls or no random controls
    
    #w0 += [a_rand,psi_rand] ################################################################initial controls

    # Integrate till the end of the interval
    Fk = F(x0=Xk, p=vertcat(Uk[2*k],Uk[2*k+1]))
    Xk = Fk['xf']
    Qk = Fk['qf']

    J=J+Fk['qf']
    
    # # Add inequality constraint
    # g += [Xk[3],Xk[4]]                                     #[[Xk[3],Xk[4]]]
          
    # lbg += [-5,-pi/3]
    # ubg += [5,pi/3]

J=J+1*Fk['qf']



#x_temp=vcat(w)
#g_temp=vcat(g)
prob = {'f': J, 'x': Uk}
solver = nlpsol('solver', 'ipopt', prob)

sol = solver(x0=w0, lbx=lbw, ubx=ubw)


w_opt = sol['x']

# Plot the solution
u_opt = w_opt
x_opt = [int_state]  #0,3,-pi/2,0,0################################################################initial states
for k in range(N):
    Fk = F(x0=x_opt[-1], p=vertcat(u_opt[2*k],u_opt[2*k+1]))
    x_opt += [Fk['xf'].full()]

x1_opt = [r[0] for r in x_opt]
x2_opt = [r[1] for r in x_opt]
x3_opt = [r[2] for r in x_opt]

u_dense = u_opt.full()
u1_opt=[]
u2_opt=[]
i=0
for r in u_dense:
    if i%2==0:
        u1_opt+=[r]
    else:
        u2_opt+=[r]
    i=i+1
    

tgrid = [T/N*k for k in range(N+1)]

plt.subplot(2, 1, 1)
#plt.clf()
plt.plot(tgrid, x1_opt)
plt.plot(tgrid, x2_opt)
plt.plot(tgrid, x3_opt)
plt.grid()
plt.title('Plot of system states with time')
plt.xlabel('Time')
plt.legend(['x','y','theta'])
#plt.savefig("Canvas12.png",dpi=500)


plt.subplot(2, 1, 2)
#plt.clf()
plt.step(tgrid[:-1], u1_opt)
plt.step(tgrid[:-1], u2_opt)
plt.grid()
plt.xlabel('Time')
plt.title('Control inputs with time')
plt.legend(['Velocity','Yaw rate'])
#plt.savefig("Canvas13.png",dpi=500)

plt.tight_layout()
plt.show()

plt.plot(x1_opt,x2_opt )
plt.show()

