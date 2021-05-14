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
        self.logs = {'Acceleration': np.zeros((2, steps + 1)), 'AA': np.zeros((2, steps + 1))}

    def get_motors(self, img):
        # give the visual input to the LAL - network
        l, r = image_split(img, overlap=self.overlap, blind=self.blind)
        # TODO: compare left image with all left images and right image with all right images
        # TODO: Alternative: One Infomax model for each eye.
        # Familiriaty vector is 6 elements, first left fam, second right familiarity
        f = [l, r, 0, 0, 0, 0]
        self.N = NetworkStepper(self.N, self.N_Indices, self.t, self.dt, f, self.Weights)

        # give the right output (#6) to the left motor
        self.logs['Acceleration'][0, self.t + 1], self.logs['AA'][0, self.t + 1] = \
            Accelerator(self.N['spike'][5, self.t], self.logs['Acceleration'][0, self.t], self.logs['AA'][0, self.t])
        # give the left output (#5) to the right motor
        self.logs['Acceleration'][1, self.t + 1], self.logs['AA'][1, self.t + 1] = \
            Accelerator(self.N['spike'][4, self.t], self.logs['Acceleration'][1, self.t], self.logs['AA'][1, self.t])

        self.t += 1
        ml = self.logs['Acceleration'][0, self.t + 1]
        mr = self.logs['Acceleration'][1, self.t + 1]
        return ml, mr

    def split_route_images(self):
        pass

    def matching_function(self):
        # return max familiarity for left and right
        pass

