# -*- coding: utf-8 -*-
"""
Created on Sat Dec  3 12:54:32 2022

@author: CASTLEC6

#function for solving the two-body problem and three-body problem using rk4

"""

from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
import numpy as np
import time as time 
import numba
tic = time.perf_counter()

##############################################################################
# initial condition state vectors and masses

G = 6.67430e-20  # Universal gravitational constant km**3/(kg * s**2)

# masses and constants
m_1 = 1e26  # mass of Object 1 in kg
m_2 = 1e26  # mass of Object 2 in kg
m_3 = 1e26 # mass of Object 3 in kg

# intial position and velocities
R_1_0 = np.array((0, 25000, 0))  # position of object 1 km
R_2_0 = np.array((0, -25000, 0))  # postion of object 2 km
R_3_0 = np.array((0, 0, 0))  # position of object 3 km
dotR_1_0 = np.array((-15, 15, -5))  # velocity of object 1 km/s
dotR_2_0 = np.array((15, -15, -5))  # velocity of object 2 km/s
dotR_3_0 = np.array((0, 0, 15))  # velocity of object 3 km/s


#time conditons
t_0 = 0  # initial time seconds
t_f = 48*3600  # final time seconds
h = 0.5 #step size seconds
t_points = np.linspace(t_0, t_f, int(t_f/h +1))


y_02 = np.hstack((R_1_0, R_2_0, dotR_1_0, dotR_2_0))

##############################################################################
#Two-body problem starts 

@numba.jit
def twobody_problem(t, y):
    """Calculate the motion of a two-body system in an inertial reference frame

    The state vector y is:

    1. Position of m_1
    2. Position of m_2
    3. Velocity components of m_1
    4. Velocity components of m_2
    """
    # Get the six coordinates for m_1 and m_2 from the state vector
    R_1 = y[:3]
    R_2 = y[3:6]

    # Fill the derivative vector with zeros
    ydot = np.zeros_like(y)

    # Set the first 6 elements of the derivative the current velocity
    ydot[:6] = y[6:]

    # Calculate the accelerations 
    r = np.sqrt(np.sum(np.square(R_2 - R_1)))
    ddot = G * (R_2 - R_1) / r ** 3
    ddotR_1 = m_2 * ddot
    ddotR_2 = -m_1 * ddot

    ydot[6:9] = ddotR_1
    ydot[9:] = ddotR_2
    return ydot


#solve using RK4
def rk4sys(dydt,tspan,y0,h,*args):
    """
   RK4 method for solving a system of ODEs
    input:
        dydt = function name that evaluates the derivatives
        tspan = array of [ti,tf] where
        ti and tf are the initial and final times
        y0 = initial conditions
        h = step size
        * args = additional argument to be passed to dydt
    output:
        t = array of time points
        y = array of numerical approximatin at the time point
    """
    ti = tspan[0] ; tf = tspan[1]
    if not(tf>ti+h): return 'upper limit must be greater than lower limit'
    t = []
    t.append(ti)  # start the t array with ti
    nsteps = int((tf-ti)/h)
    for i in range(nsteps):  # add the rest of the t values
        t.append((i+1)*h)
    n = len(t)
    if t[n-1] < tf:  # check if t array is short of tf
        t.append(tf)
        n = n+1
    neq = len(y0)
    y = np.zeros((n,neq))  # set up 2-D array for dependent variables
    for j in range(neq):
        y[0,j] = y0[j]  #  set first elememts to initial conditions
    for i in range(n-1):  # 4th order RK
        hh = t[i+1] - t[i]
        k1 = dydt(t[i],y[i,:],*args)
        ymid = y[i,:] + k1*hh/2.
        k2 = dydt(t[i]+hh/2.,ymid,*args)
        ymid = y[i,:] + k2*hh/2.
        k3 = dydt(t[i]+hh/2.,ymid,*args)
        yend = y[i,:] + k3*hh
        k4 = dydt(t[i]+hh,yend,*args)
        phi = (k1 + 2.*(k2+k3) + k4)/6.
        y[i+1,:] = y[i,:] + phi*hh
    return t,y

def dydtsys(t,y):
    n = len(y)
    dy = np.zeros((n))
    dy[0] = -2.*y[0]**2 +2.*y[0] + y[1] - 1.
    dy[1] = -y[0] -3*y[1]**2 +2.*y[1] + 2.
    return dy

#testing rk45
# ti = 0. ; tf = 2.
# tspan = np.array([ti,tf])
# h = 0.01
# y0 = np.array([2.,0.])
# t,y = rk4sys(dydtsys,tspan,y0,h)
# import pylab
# pylab.plot(t,y[:,0],c='k',label='y1')
# pylab.plot(t,y[:,1],c='k',ls='--',label='y2')
# pylab.grid()
# pylab.xlabel('t')
# pylab.ylabel('y')
# pylab.legend()



#solve using built-in functions
sol = solve_ivp(twobody_problem, [t_0, t_f], y_02, method='RK23', t_eval=t_points)
y_built_in = sol.y.T

#solving using RK45
t, y = rk4sys(twobody_problem, [t_0, t_f], y_02, h)

R_1 = y[:, :3]  # km
R_2 = y[:, 3:6]  # km
V_1 = y[:, 6:9]  # km/s
V_2 = y[:, 9:]  # km/s
barycenter = (m_1 * R_1 + m_2 * R_2) / (m_1 + m_2)  # km

#graphing two body problem
#Inertial frame plot
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot3D(R_1[:, 0], R_1[:, 1], R_1[:, 2], label="m_1")
ax.plot3D(R_2[:, 0], R_2[:, 1], R_2[:, 2], label="m_2")
ax.plot3D(barycenter[:, 0], barycenter[:, 1], barycenter[:, 2], label="Barycenter")
ax.legend()
ax.plot3D(R_1[-1, 0], R_1[-1, 1], R_1[-1, 2], color='b', marker='o')
ax.plot3D(R_2[-1, 0], R_2[-1, 1], R_2[-1, 2], color='orange', marker='o')
ax.plot3D(barycenter[-1, 0], barycenter[-1, 1], barycenter[-1, 2], label="Barycenter", color='r', marker='o')


# barycenter centered frame
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
R1_rel_COG = R_1 - barycenter
R2_rel_COG = R_2 - barycenter
ax.plot(R1_rel_COG[:, 0], R1_rel_COG[:, 1], R1_rel_COG[:, 2], label="m_1")
ax.plot(R2_rel_COG[:, 0], R2_rel_COG[:, 1], R2_rel_COG[:, 2], label="m_2")
ax.plot(0, 0, 0, "ro", label="Barycenter")
ax.legend()


# m1 centered frame
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
R2_rel_R1 = R_2 - R_1
COG_rel_R1 = barycenter - R_1
ax.plot(R2_rel_R1[:, 0], R2_rel_R1[:, 1], R2_rel_R1[:, 2], label="m_2")
ax.plot(COG_rel_R1[:, 0], COG_rel_R1[:, 1], COG_rel_R1[:, 2], label="Barycenter")
ax.plot(0, 0, 0, "ro", label="m_1")
ax.legend()


##############################################################################

#Three body problem

#fun test run 
# m_1 = 1e26  # mass of Object 1 in kg
# m_2 = 1e25  # mass of Object 2 in kg
# m_3 = 1e25 # mass of Object 3 in kg
# R_1_0 = np.array((0, 0, 0))  # position of object 1 km
# R_2_0 = np.array((0, 3000, 7000))  # postion of object 2 km
# dotR_1_0 = np.array((5, 0, 0))  # velocity of object 1 km/s
# dotR_2_0 = np.array((-5, 20, 10))  # velocity of object 2 km/s
# R_3_0 = np.array((0, 7000, -3000))  # position of object 3 km
# dotR_3_0 = np.array((10, 10, -25))  # velocity of object 3 km/s

#test conditions 2
# 6 or 3 hours half a second very cool
# m_1 = 1e26  # mass of Object 1 in kg
# m_2 = 1e26  # mass of Object 2 in kg
# m_3 = 3e26 # mass of Object 3 in kg
# R_1_0 = np.array((0, 0, 0))  # position of object 1 km
# R_2_0 = np.array((0, 3000, 14000))  # postion of object 2 km
# dotR_1_0 = np.array((5, 0, 35))  # velocity of object 1 km/s
# dotR_2_0 = np.array((-5, 20, 10))  # velocity of object 2 km/s
# R_3_0 = np.array((0, 17000, -3000))  # position of object 3 km
# dotR_3_0 = np.array((10, 0, -10))  # velocity of object 3 km/s

#very cool test 3
# m_1 = 1e26  # mass of Object 1 in kg
# m_2 = 1e26  # mass of Object 2 in kg
# m_3 = 1e26 # mass of Object 3 in kg

# R_1_0 = np.array((0, 25000, 0))  # position of object 1 km
# R_2_0 = np.array((0, -25000, 0))  # postion of object 2 km
# R_3_0 = np.array((0, 0, 0))  # position of object 3 km
# dotR_1_0 = np.array((-15, 15, -5))  # velocity of object 1 km/s
# dotR_2_0 = np.array((15, -15, -5))  # velocity of object 2 km/s
# dotR_3_0 = np.array((0, 0, 15))  # velocity of object 3 km/s

#Test slow perfect balanced circle
# m_1 = 1e26  # mass of Object 1 in kg
# m_2 = 1e26  # mass of Object 2 in kg
# m_3 = 1e26 # mass of Object 3 in kg

# #two-body intial position and velocities
# R_1_0 = np.array((0, 100000, 0))  # position of object 1 km
# R_2_0 = np.array((0, 0, 100000))  # postion of object 2 km
# R_3_0 = np.array((100000, 0, 0))  # position of object 3 km
# dotR_1_0 = np.array((5, 0, -5))  # velocity of object 1 km/s
# dotR_2_0 = np.array((-5, 5, 0))  # velocity of object 2 km/s
# dotR_3_0 = np.array((0, -5, 5))  # velocity of object 3 km/s

##############################################################################
#derivative functions
@numba.jit
def Three_body(t,y3):
    """Calculate the motion of a two-body system in an inertial reference frame

    The state vector y is:

    1. Position of object 1
    2. Position of object 2
    3. Position of object 3
    4. Velocity components of object 1
    5. Velocity components of object 2
    6. Velocity components of object 3
    """
    # Get the six coordinates for m_1 and m_2 from the state vector
    R_1 = y3[:3]
    R_2 = y3[3:6]
    R_3 = y3[6:9]

    # Fill the derivative vector with zeros
    ydot3 = np.zeros_like(y3)

    # Set the first 6 elements of the derivative the current velocity
    ydot3[:9] = y3[9:]

    # Calculate the accelerations 
    r12 = np.sqrt(np.sum(np.square(R_2 - R_1)))
    r23 = np.sqrt(np.sum(np.square(R_2 - R_3)))
    r31 = np.sqrt(np.sum(np.square(R_3 - R_1)))
    
    #Two-body gravitional Acceleration 
    ddot21 = G * (R_2 - R_1) / r12 ** 3
    ddot23 = G * (R_2 - R_3) / r23 ** 3
    ddot31 = G * (R_3 - R_1) / r31 ** 3
    
    #Three-body total Acceleration calculation
    ddotR_1 = m_2 * ddot21 + m_3 * ddot31
    ddotR_2 = -m_1 * ddot21 + -m_3 * ddot23
    ddotR_3 = -m_1 * ddot31 + m_2 * ddot23
    
    ydot3[9:12] = ddotR_1
    ydot3[12:15] = ddotR_2
    ydot3[15:] = ddotR_3
    
    return ydot3


#initial condition of three body problem R1,R2,R3,V1,V2,V3
y_03 = np.hstack((R_1_0, R_2_0, R_3_0, dotR_1_0, dotR_2_0, dotR_3_0))

#solve using built-in functions
sol = solve_ivp(Three_body, [t_0, t_f], y_03, method='RK45', t_eval=t_points)
y3_built_in = sol.y.T

#solving using RK45
t, y3 = rk4sys(Three_body, [t_0, t_f], y_03, h)


R_1 = y3[:, :3]  # km
R_2 = y3[:, 3:6]  # km
R_3 = y3[:, 6:9]  # km
V_1 = y3[:, 9:12]  # km/s
V_2 = y3[:, 12:15]  # km/s
V_2 = y3[:, 15:]  # km/s
barycenter3 = (m_1 * R_1 + m_2 * R_2 + m_3 * R_3) / (m_1 + m_2 + m_3)  # km

#graphing
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot3D(R_1[:, 0], R_1[:, 1], R_1[:, 2], label="m_1")
ax.plot3D(R_2[:, 0], R_2[:, 1], R_2[:, 2], label="m_2")
ax.plot3D(R_3[:, 0], R_3[:, 1], R_3[:, 2], label="m_3")
ax.plot3D(barycenter3[:, 0], barycenter3[:, 1], barycenter3[:, 2], label="Barycenter", color='r')
ax.legend()

ax.plot3D(R_1[-1, 0], R_1[-1, 1], R_1[-1, 2], color='b', marker='o')
ax.plot3D(R_2[-1, 0], R_2[-1, 1], R_2[-1, 2], color='orange', marker='o')
ax.plot3D(R_3[-1, 0], R_3[-1, 1], R_3[-1, 2], color='g', marker='o')
ax.plot3D(barycenter3[-1, 0], barycenter3[-1, 1], barycenter3[-1, 2], label="Barycenter", color='r', marker='o')

toc = time.perf_counter()

print('Run Time: ', toc-tic)
