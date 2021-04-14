# This is the Simulation of the LAL-SNN
import numpy as np
from Neuronator import builder, Weighting, NetworkStepper, Accelerator

# Simulation parameters --------------------------------------------------
# temporal resolution of simulation
dt = 0.001  # [s]
# duration of simulation
T = 2  # [s]
# steps of simulation
steps = int(T/dt)
# Setting up the neurons' parameters
[N, N_Indices] = builder(steps)
# Setting up the weights
Weights = Weighting()  # WeightMatrix
# load lookup table for input neurons
lookup = {}
lookup['Input'] = np.genfromtxt('InputTable.csv', delimiter=',')

Agent = {}

# "World" settings
Agent['Centre'] = np.zeros((steps + 1, 2))  # Centrepoint of the Agent
Phi = np.zeros((1, steps + 1))

# muscle activity
Agent['Acceleration'] = np.zeros((2, steps + 1))  # rows: neurons 1,2
Agent['AA'] = np.zeros((2, steps + 1))  # AccelerationActivation

# test
F = [24, 49, 74, 99]
Agent['F'] = np.zeros((6, steps + 1))
Agent['F'][0, 0:steps] = lookup['Input'][F[3]]
Agent['F'][1, 0:steps] = lookup['Input'][F[3]]

# Simulation
for t in range(0, steps):

    ''' positions of the different parts of the agent
    Agent['LeftEye'][t,:], Agent['RightEye'][t,:], \
    Agent['LeftMotor'][t,:], Agent['RightMotor'][t,:] \
    = Bodypositions(Agent['Centre'][t,:],Phi[t])'''

    # give the visual input to the LAL - network
    N = NetworkStepper(N, N_Indices, t, dt, Agent['F'][:, t], Weights)

    # give the right output (#6) to the left motor
    Agent['Acceleration'][0, t + 1], Agent['AA'][0, t + 1] = \
    Accelerator(N['spike'][5, t], Agent['Acceleration'][0, t], Agent['AA'][0, t])
    # give the left output (#5) to the right motor
    Agent['Acceleration'][1, t + 1], Agent['AA'][1, t + 1] = \
    Accelerator(N['spike'][4, t], Agent['Acceleration'][1, t], Agent['AA'][1, t])

    '''
    # what is the new direction with which distance
    Rotation[t], Distance[t] = \
    ReactionApproach(Agent['Acceleration'][:, t], Agent['Turning'][:, t])

    # Current angle(Phi) + NewAngle(Rotation) + Noise
    Phi[t + 1] = Phi[t] + Rotation[t]

    Agent['Centre'][t + 1,:] = CentrePosition(Agent['Centre'][t,:], Phi[t + 1], Distance[t])
    '''

# visualization
'''
% for i = 1:(N_Indices(end))
% subplot(N_Indices(end), 1, i)
% bar(N.spike(i,:))
% s(i) = sum(N.spike(i,:));
% end
% pause()
% close
%
% bar(s)
% pause()
% close
%
% plot(Agent.Centre(:, 1), Agent.Centre(:, 2))
% axis
equal
% pause()
% close
'''
# Data packing

data = N['spike']