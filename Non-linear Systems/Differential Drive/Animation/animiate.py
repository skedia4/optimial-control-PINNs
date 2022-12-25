#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 25 22:04:06 2022

@author: shubham
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from svgpathtools import svg2paths
from svgpath2mpl import parse_path
import numpy as np
from matplotlib import animation
import numpy as np
import os

path = os.getcwd()
parent = os.path.dirname(path)
fname = parent+'/sample_state_trajectories/trajec_1.npy'


global car_marker
car_path, attributes = svg2paths('car3.svg')
car_marker = parse_path(attributes[0]['d'])
car_marker.vertices -= car_marker.vertices.mean(axis=0)


x = np.linspace(0,2*np.pi,1000)
y = np.sin(x)


with open(fname, 'rb') as f:
    states = np.load(f)
    
x = states[:,0]
y = states[:,1]
th = states[:,2]*180/3.141


print(x.min(),x.max())
print(y.min(),y.max())
#car_marker = car_marker.transformed(mpl.transforms.Affine2D().rotate_deg(-90))
#plt.plot(x,y, 'og', marker=car_marker,markersize=50)


fig = plt.figure(figsize=(10,10), dpi = 100)
ax = plt.axes(xlim=(-0.5, 1), ylim=(-0.2, 1.2))
#ax = plt.axes()
ax.axis('off')
line, = ax.plot([], [], 'go', )
line2, = ax.plot([], [], 'r--', lw=2)

start, = ax.plot([], [], 'go', )
goal, = ax.plot([], [], 'bo', )


# initialization function: plot the background of each frame
def init():
    line.set_data([], [])
    line2.set_data([], [])
    start.set_data([], [])
    goal.set_data([], [])
    
    
    global car_marker, car_marker1
    car_marker1 = car_marker.transformed(mpl.transforms.Affine2D().rotate_deg(th[0]-90))
    line.set_marker(marker=car_marker)
    line.set_markersize(100)
    
    car_marker2 = car_marker.transformed(mpl.transforms.Affine2D().rotate_deg(th[0]-90))
    start.set_marker(marker=car_marker2)
    start.set_markersize(100)
    
    
    car_marker3 = car_marker.transformed(mpl.transforms.Affine2D().rotate_deg(-90))
    goal.set_marker(marker=car_marker3 )
    goal.set_markersize(100)
    
    return line, line2, start, goal

# animation function.  This is called sequentially
def animate(i):
    global car_marker1
    line.set_data(x[i+1], y[i+1])
    car_marker1 = car_marker1.transformed(mpl.transforms.Affine2D().rotate_deg(th[i+1]-th[i]))
    line.set_marker(marker=car_marker1)
    
    line2.set_data(x[:i+1], y[:i+1])
    
    start.set_data([x[0]], [y[0]])
    
    goal.set_data([0, 0])
    

    return line, line2, start, goal


# call the animator.  blit=True means only re-draw the parts that have changed.
anim = animation.FuncAnimation(fig, animate, init_func=init,
                               frames=len(x)-1, interval=1, blit=True, repeat=False)

writervideo = animation.PillowWriter(fps= 60)
anim.save("./output/"+fname.split("/")[-1]+"v3.gif", writer=writervideo)

plt.show()
