from utils import image_split, cor_coef
from Neuronator import builder, Weighting, NetworkStepper, Accelerator
import numpy as np




class Breitenberg:
    def __init__(self, steps, dt=0.001, overlap=10, blind=10):
        self.overlap = overlap
        self.blind = blind
        self.dt = dt
        self.t = 0
        self.Weights = Weighting()
        self.N, self.N_Indices = builder(steps)

    def get_heading(self, img):
        # give the visual input to the LAL - network
        l, r = image_split(img, overlap=self.overlap, blind=self.blind)
        # TODO: compare left image with all left images and right image with all right images?
        N = NetworkStepper(self.N, self.N_Indices, self.t, self.dt, Agent['F'][:, t], self.Weights)

        # give the right output (#6) to the left motor
        Agent['Acceleration'][0, t + 1], Agent['AA'][0, self.t + 1] = \
        ml = Accelerator(N['spike'][5, t], Agent['Acceleration'][0, self.t], Agent['AA'][0, self.t])
        # give the left output (#5) to the right motor
        Agent['Acceleration'][1, t + 1], Agent['AA'][1, self.t + 1] = \
        mr = Accelerator(N['spike'][4, self.t], Agent['Acceleration'][1, self.t], Agent['AA'][1, self.t])
        return ml, mr

