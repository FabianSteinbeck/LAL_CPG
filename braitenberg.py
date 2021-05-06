from utils import image_split
from Neuronator import builder, Weighting, NetworkStepper, Accelerator
import numpy as np




class Breitenberg:
    def __init__(self, steps, dt=0.001, overlap=10, blind=10):
        self.overlap = overlap
        self.blind = blind
        self.dt = dt
        self.step = 0
        self.Weights = Weighting()
        N, N_Indices = builder(steps)


    def get_heading(self, img):
        # give the visual input to the LAL - network
        N = NetworkStepper(N, N_Indices, t, self.dt, Agent['F'][:, t], self.Weights)

        # give the right output (#6) to the left motor
        Agent['Acceleration'][0, t + 1], Agent['AA'][0, t + 1] = \
        ml = Accelerator(N['spike'][5, t], Agent['Acceleration'][0, t], Agent['AA'][0, t])
        # give the left output (#5) to the right motor
        Agent['Acceleration'][1, t + 1], Agent['AA'][1, t + 1] = \
        mr = Accelerator(N['spike'][4, t], Agent['Acceleration'][1, t], Agent['AA'][1, t])
        return ml, mr

