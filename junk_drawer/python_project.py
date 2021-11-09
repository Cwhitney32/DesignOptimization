import logging
import math
import random
import numpy as np
import time
import torch as t
import torch.nn as nn
from torch import optim
from torch.nn import utils
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
logger = logging.getLogger(__name__)

FRAME_TIME = 0.1  # time interval
BOOST_ACCEL_X = 0.18  # thrust constant
BOOST_ACCEL_Y = 0.18  # thrust constant
BOOST_ACCEL_mX = -0.18  # thrust constant
BOOST_ACCEL_mY = -0.18  # thrust constant
GRAVITY_ACCEL = 0.12  # gravity constant
BOOST_ACCEL_Z = 0.18  # thrust constant
drag = 0.02

class Dynamics(nn.Module):
    def __init__(self):
        super(Dynamics, self).__init__()
    @staticmethod
    def forward(state, action):
        xthrust_t=FRAME_TIME*t.tensor([0.,1.,0.,0.,0.,0.])
        xthrust_action=t.matmul(t.tensor([BOOST_ACCEL_X,0.,0.,0.,0.]), t.transpose(action,0,1))
        xthrust_t= t.transpose(xthrust_t.unsqueeze(0),0,1)
        xthrust=xthrust_t*xthrust_action
        xthrust=t.transpose(xthrust,0,1)
        xmthrust_t=FRAME_TIME*t.tensor([0.,1.,0.,0.,0.,0.])
        xmthrust_action=t.matmul(t.tensor([0.,BOOST_ACCEL_mX,0.,0.,0.]), t.transpose(action,0,1))
        xmthrust_t= t.transpose(xmthrust_t.unsqueeze(0),0,1)
        xmthrust=xmthrust_action*xmthrust_t
        xmthrust=t.transpose(xmthrust,0,1)
        ythrust_t=FRAME_TIME*t.tensor([0.,0.,0.,1.,0.,0.])
        ythrust_action=t.matmul(t.tensor([0., 0.,BOOST_ACCEL_Y,0.,0.]), t.transpose(action,0,1))
        ythrust_t= t.transpose(ythrust_t.unsqueeze(0),0,1)
        ythrust=ythrust_action*ythrust_t
        ythrust=t.transpose(ythrust,0,1)
        ymthrust_t = FRAME_TIME*t.tensor([0.,0.,0.,1.,0.,0.])
        ymthrust_action = t.matmul(t.tensor([0., 0.,0.,BOOST_ACCEL_mY,0.]), t.transpose(action,0,1))
        ymthrust_t= t.transpose(ymthrust_t.unsqueeze(0),0,1)
        ymthrust=ymthrust_action*ymthrust_t
        ymthrust=t.transpose(ymthrust,0,1)
        delta_state_gravity = t.tensor([0.,0.,0.,0.,0.,GRAVITY_ACCEL * FRAME_TIME])
        delta_state_drag = t.tensor([0.,0.,0.,0.,0., -1 * drag * FRAME_TIME])
        delta_state_t = FRAME_TIME*t.tensor([0.,0.,0.,0.,0.,-1.])
        delta_state_action =  t.matmul(t.tensor([0., 0.,0.,0.,BOOST_ACCEL_Z]), t.transpose(action,0,1))
        delta_state_t = t.transpose(delta_state_t.unsqueeze(0),0,1)
        delta_state =delta_state_action*delta_state_t
        delta_state=t.transpose(delta_state,0,1)
        state = state + xthrust + xmthrust + ythrust + ymthrust + delta_state + delta_state_gravity + delta_state_drag
        state = t.transpose(state,0,1)
        step_mat = t.tensor([[1., FRAME_TIME,0.,0., 0., 0.],
                            [0., 1.,0.,0.,0.,0.],
                             [0.,0.,1.,FRAME_TIME,0.,0.],
                             [0.,0.,0.,1.,0.,0.],
                             [0.,0.,0.,0.,1.,FRAME_TIME],
                             [0.,0.,0.,0.,0.,1.]])
        state = t.matmul(step_mat,state)
        state= t.transpose(state,0,1)
        return state

class Controller(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_output):
        """
        dim_input: # of system states
        dim_output: # of actions
        dim_hidden: up to you
        """
        super(Controller, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_output),
            # You can add more layers here
            nn.Sigmoid()
        )

    def forward(self, state):
        action = self.network(state)
        return action

class Simulation(nn.Module):
    def __init__(self, controller, dynamics, T):
        super(Simulation, self).__init__()
        self.state = self.initialize_state()
        self.controller = controller
        self.dynamics = dynamics
        self.T = T
        self.action_trajectory = []
        self.state_trajectory = []
    def forward(self, state):
        self.action_trajectory = []
        self.state_trajectory = []
        for _ in range(T):
            action = self.controller.forward(state)
            state = self.dynamics.forward(state, action)
            self.action_trajectory.append(action)
            self.state_trajectory.append(state)
        return self.error(state)
    @staticmethod
    def initialize_state():
        state = t.tensor([[4.,.2,-3.,.2,1.,0.],[4.,.2,-3.,.2,1.,0.]])
        return t.tensor(state, requires_grad=False).float()
    def error(self, state):
##Laplacian multipliers were investigated for convergence imrovement
        l1=10
        l2=1
        l3=10
        l4=1
        l5=10
        l6=1
        return t.sum(l1*state[:,0]**2 + l2*state[:,1]**2 + l3*state[:,2]**2 + l4*state[:,3]**2 + l5*state[:,4]**2 + l6*state[:,5]**2)

losslist=[]
itterlist=[]
class Optimize:
    def __init__(self, simulation):
        self.simulation = simulation
        self.parameters = simulation.controller.parameters()
        self.optimizer = optim.LBFGS(self.parameters, lr=0.01)
        #self.optimizer = optim.SGD(self.parameters, lr=0.1, momentum=0.9)
    def step(self):
        def closure():
            loss = self.simulation(self.simulation.state)
            self.optimizer.zero_grad()
            loss.sum().backward()
            return loss
        self.optimizer.step(closure)
        return closure()
    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.step()
            losslist.append(loss.detach())
            itterlist.append(epoch+1)
            #print('[%d] loss: %.3f' % (epoch + 1, loss))
            if (epoch == 0 or epoch == (epochs/2)-1 or epoch==epochs-1):
                self.visualize(epoch,loss)
    
    def visualize(self,epoch,loss):
        data = np.array([self.simulation.state_trajectory[i].detach().numpy() for i in range(self.simulation.T)])
        Position_X = data[:,0, 0]
        Position_Y = data[:,0, 2]
        Velocity_X = data[:,0, 1]
        Velocity_Y = data[:,0, 3]
        Postion_Z = data[:,0, 4]
        Velocity_Z = data[:,0, 5]
        fig = plt.figure(figsize=(12, 12))
        fig.suptitle('Vissuliztion of solution at itteration [%d] \nloss=%.3f' % (epoch + 1, loss))
        gs = GridSpec(nrows=2, ncols=2)
        ax0 = fig.add_subplot(gs[0, 0])
        ax0.plot(Position_X, Position_Y)
        ax0.set_title('Horizonatal Positioning')
        ax0.set_ylabel('Position Y')
        ax0.set_xlabel('Position X')
        ax1 = fig.add_subplot(gs[1, 0])
        ax1.plot(Postion_Z, Velocity_Z)
        ax1.set_title('Vertical Positioning')
        ax1.set_ylabel('Position Z')
        ax1.set_xlabel('Velocity Z')
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.quiver(Position_X, Position_Y,Velocity_X,Velocity_Y,units='xy' ,scale=1)
        ax2.set_title('Horizontal Velocity Vector')
        ax2.set_ylabel('loss')
        ax2.set_xlabel('itteration')
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(itterlist,losslist)
        ax3.set_title('loss vs. iteration')
        ax3.set_ylabel('loss')
        ax3.set_xlabel('itteration')

T = 100  # number of time steps
dim_input = 6 # state space dimensions
dim_hidden = 6  # latent dimensions
dim_output = 5  # action space dimensions
d = Dynamics()  # define dynamics
c = Controller(dim_input, dim_hidden, dim_output)  # define controller
s = Simulation(c, d, T)  # define simulation
o = Optimize(s)  # define optimizer
o.train(40)  # solve the optimization problem