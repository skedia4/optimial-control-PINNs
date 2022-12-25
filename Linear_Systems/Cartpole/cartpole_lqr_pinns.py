import gym
import math 
import gym_cartpole_continuous
import control
import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt

import torch
from torch import nn


def save_frames_as_gif(frames, path='./', filename='cartpole_lqr_pinns.gif'):

    # Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / 72.0,
               frames[0].shape[0] / 72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(
        plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=60)


def lqr_policy(observation):

    # cost function

    Q = np.identity(4)
    R = np.identity(1)

    # linearization
    A = np.array([[0, 1, 0, 0], [0, 0, -0.98, 0],
                 [0, 0, 0, 1], [0, 0, 21.56, 0]])

    B = np.array([[0, 1, 0, -2]]).T

    K, S, E = control.lqr(A, B, Q, R)

    action = -1*np.dot(K, observation)

    if action >= 1:
        return np.array([1])
    elif action <= -1:
        return np.array([-1])
    else:
        return action

class ffnn(nn.Module):
    """basic FF network for approximating functions"""
    def __init__(self, nn_width=10, num_hidden=3):
        super().__init__()
        
        self.layer_first = nn.Linear(5, nn_width)
        
        layers = []
        for _ in range(num_hidden):
            layers.append(nn.Linear(nn_width, nn_width))
        self.layer_hidden = nn.ModuleList(layers)
        
        self.layer_last = nn.Linear(nn_width, 1)
        
    def forward(self, x):
        activation = nn.Tanh()
        u = activation(self.layer_first(x))
        for hidden in self.layer_hidden:
            u = activation(hidden(u))
        u = self.layer_last(u)
        return u


def u(model,x):

    B=torch.tensor([[0, 1, 0, -2]], dtype=torch.float64).T
    v=model(x)
    temp = torch.ones((x.shape[0],1),requires_grad=True)
    v_x = torch.autograd.grad(v,x,grad_outputs=temp,retain_graph=True,create_graph=True)[0]
    v_x = v_x[:,1:].squeeze(0).unsqueeze(-1)
    B_numpy= B.detach().numpy()
    v_x_numpy=v_x.detach().numpy()
    u= B_numpy.T @ v_x_numpy

    return -0.5*u.squeeze(-1)

model = ffnn(64,3)
model.load_state_dict(torch.load('cartpole_mod.pth',map_location='cpu'))

env = gym.make('CartPoleContinuous-v0')

M = 1.0
m = 0.1
l = 0.5  
g = 9.8

frames = []
observation = env.reset()

obs_list=[]
policy=[]

for t in range(50):
    frames.append(env.render(mode="rgb_array"))
    x=torch.tensor(np.concatenate([np.array([t*0.02]), observation]),
    dtype=torch.float64,requires_grad=True).unsqueeze(0).float()

    #u_= lqr_policy(observation)   #uncomment to run baseline controller(lqr)
    u_=u(model,x)
    
    observation=env.step(u_)[0]

    obs_list.append(observation[2])
    policy.append(u_)

env.close()

t = np.linspace(0,50*0.02,50)
plt.plot(t,obs_list)
plt.xlabel('Time (t)')
plt.ylabel(r'Angle (in radians)')
plt.legend([r'$\theta$'])
plt.savefig('Trajectory-pinns.png', format = 'png')
plt.show()

plt.plot(t,policy,'r')
plt.xlabel('Time (t)')
plt.ylabel('Control input (u)')
plt.legend([r'$u$'])
plt.savefig('Control-input-pinns.png', format = 'png')
plt.show()

#save_frames_as_gif(frames)



