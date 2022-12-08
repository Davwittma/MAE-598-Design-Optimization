import logging
import math as m

import numpy as np
import torch as t
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# environment parameters

FRAME_TIME = .1  # time interval
GRAVITY_ACCEL = 10  # gravity constant
BOOST_ACCEL = 150  # thrust constant
ROTATION_THRUST = 100  # Angular Thrust constant


# define system dynamics
# Notes:
# 0. You only need to modify the "forward" function
# 1. All variables in "forward" need to be PyTorch tensors.
# 2. All math operations in "forward" has to be differentiable, e.g., default PyTorch functions.
# 3. Do not use inplace operations, e.g., x += 1. Please see the following section for an example that does not work.

class Dynamics(nn.Module):

    def __init__(self):
        super(Dynamics, self).__init__()

    @staticmethod
    def forward(state, action):
        """
        action: thrust or no thrust
        action[0] use orientation to adjust x_dot
        action[1] Use vertical thruster
        action[2] Orient the rocket vertically using side thrusters

        state[0] = x            position of x
        state[1] = y            speed of x
        state[2] = x_dot        position of y
        state[3] = y_dot        speed of y
        state[4] = angle        orientation of the rocket wrt the +vertical axis
        state[5] = angle_dot    Change in angle wrt +vert axis
        """

        # Apply gravity (acting in negative direction, ie downward) Note: Here gravity is used to change velocity
        # which is the second element of the state vector Normally, we would do x[1] = x[1] + gravity * delta_time
        # but this is not allowed in PyTorch since it overwrites one variable (x[1]) that is part of the
        # computational graph to be differentiated. Therefore, I define a tensor dx = [0., gravity * delta_time],
        # and do x = x + dx. This is allowed...

        # Note: Gravity only acting on y_dot, leaving all other terms zero and will accommodate them as follows:

        # Thrust

        # Note: Same reason as above. Need a 2-by-1 tensor.
        # To directionalize the thrust, we must incorporate the components of theta into the thrust direction:

        # delta_x = BOOST_ACCEL * FRAME_TIME * t.sin(state[4]) * action[0] * t.tensor([0, 1, 0, 0, 0, 0])
        # delta_y = (BOOST_ACCEL * FRAME_TIME * t.cos(state[4]) * action[0] - delta_state_gravity) * t.tensor(
        #    [0, 0, 0, 1, 0, 0])
        # delta_w = ROTATION_ACCEL * FRAME_TIME * action[1] * (2 * action[2] - 1) * t.tensor([0, 0, 0, 0, 0, 1])
        delta_state_gravity = t.tensor([0., 0., 0., -GRAVITY_ACCEL * FRAME_TIME, 0., 0.])

        # Thrust
        # Note: Same reason as above. Need a 2-by-1 tensor.
        delta_state = ((BOOST_ACCEL * FRAME_TIME * t.tensor([0., 0., -1., 0., 0., 0.]) * action[0])
                       + (BOOST_ACCEL * FRAME_TIME * t.tensor([0., 0., 0., 1., 0., 0.]) * action[1])
                       + (ROTATION_THRUST * FRAME_TIME * t.tensor([0., 0., 0., 0., 0., 1.]) * action[2]))
        # Angle

        # Update velocity
        # print('a')

        state = state + delta_state + delta_state_gravity

        # Update state
        # Note: Same as above. Use operators on matr
        # ices/tensors as much as possible. Do not use element-wise operators as they are considered inplace.

        step_mat = t.tensor([[1., 0., FRAME_TIME, 0., 0., 0.],
                             [0., 1., 0., FRAME_TIME, 0., 0.],
                             [0., 0., 1., 0., 0., 0.],
                             [0., 0., 0., 1., 0., 0.],
                             [0., 0., 0., 0., 1., FRAME_TIME],
                             [0., 0., 0., 0., 0., 1.]])
        state = t.t(t.matmul(step_mat, state))
        # print('b')
        return state


# a deterministic controller Note: 0. You only need to change the network architecture in "__init__" 1. nn.Sigmoid
# outputs values from 0 to 1, nn.Tanh from -1 to 1 2. You have all the freedom to make the network wider (by
# increasing "dim_hidden") or deeper (by adding more lines to nn.Sequential) 3. Always start with something simple

class Controller(nn.Module):

    def __init__(self, dim_input, dim_hidden, dim_output):
        """
        dim_input: # of system states (6)
        dim_output: # of actions (3)
        dim_hidden: up to you (no idea, experimenting below:)
        """
        super(Controller, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_hidden),
            # You can add more layers here
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_output)
        )

    def forward(self, state):
        action = self.network(state)
        # print(action)
        return action


# the simulator that rolls out x(1), x(2), ..., x(T)
# Note:
# 0. Need to change "initialize_state" to optimize the controller over a distribution of initial states
# 1. self.action_trajectory and self.state_trajectory stores the action and state trajectories along time


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
        """
        recall:
        action: thrust or no thrust
        action[0] ang thrust on/off
        action[1] thrust on
        action[2] ang thrust CCW or CW

        state[0] = x            position of x
        state[1] = y            speed of x
        state[2] = x_dot        position of y
        state[3] = y_dot        speed of y
        state[4] = angle        orientation of the rocket wrt the +vertical axis
        state[5] = angle_dot    Change in angle wrt +vert axis
        """
        state = [200., 100., -10., 40., -45, 3]
        return t.tensor(state, requires_grad=False).float()

    def error(self, state):
        # error is the square error from all states. When landed, all states should be zero
        e = state[0] ** 2 + state[1] ** 2 + state[2] ** 2 + state[3] ** 2 + state[4] ** 2 + state[5] ** 2
        return e


# set up the optimizer Note: 0. LBFGS is a good choice if you don't have a large batch size (i.e., a lot of initial
# states to consider simultaneously) 1. You can also try SGD and other momentum-based methods implemented in PyTorch
# 2. You will need to customize "visualize" 3. loss.backward is where the gradient is calculated (d_loss/d_variables)
# 4. self.optimizer.step(closure) is where gradient descent is done


class Optimize:
    def __init__(self, simulation):
        self.simulation = simulation
        self.parameters = simulation.controller.parameters()
        self.optimizer = optim.LBFGS(self.parameters, lr=0.1)

    def step(self):
        def closure():
            loss = self.simulation(self.simulation.state)
            self.optimizer.zero_grad()
            loss.backward()
            return loss

        self.optimizer.step(closure)
        return closure()

    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.step()
            print('[%d] loss: %.3f' % (epoch + 1, loss))
            self.visualize()

    def visualize(self):  # grapher to visualize trajectory
        data = np.array([self.simulation.state_trajectory[i].detach().numpy() for i in range(self.simulation.T)])
        x = data[:, 0]
        y = data[:, 1]
        vx = data[:, 2]
        vy = data[:, 3]
        ang_disp = data[:, 4]
        ang_vel = data[:, 5]
        ft = range(self.simulation.T)

        fig, ax = plt.subplots(1, 3, tight_layout=1, figsize=(15, 5))

        ax[0].plot(x, y, c='C0', label="Coordinates")
        ax[0].set_xlabel("X-coordinates (m)")
        ax[0].set_ylabel("Y-coordinates (m)")
        ax[0].legend(frameon=0)
        ax[0].set(title='Displacement plot')

        ax[1].plot(ft, vx, c='b', label="Velocity in x")
        ax[1].plot(ft, vy, c='r', label="Velocity in y")
        ax[1].set_xlabel("Time interval (1 s)")
        ax[1].set_ylabel("Velocity (m/s)")
        ax[1].legend(frameon=0)
        ax[1].set(title='Velocity plot')

        ax[2].plot(ft, ang_disp, c='b', label="Angluar Displacement")
        ax[2].plot(ft, ang_vel, c='r', label="Angular Velocity")
        ax[2].set_xlabel("Time interval (1 s)")
        ax[2].set_ylabel("Angular Displacment (deg) and Velocity (deg/s)")
        ax[2].legend(frameon=0)
        ax[2].set(title='Angular Displacement and Velocity plot')

        plt.show()


T = 100  # number of time steps
dim_input = 6  # state space dimensions
dim_hidden = 15  # latent dimensions
dim_output = 3  # action space dimensions
d = Dynamics()  # define dynamics
c = Controller(dim_input, dim_hidden, dim_output)  # define controller
s = Simulation(c, d, T)  # define simulation
o = Optimize(s)  # define optimizer
o.train(60)  # solve the optimization problem
