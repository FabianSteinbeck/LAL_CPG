from utils import image_split, cor_coef
from Neuronator import builder, Weighting, NetworkStepper, Accelerator
import numpy as np


class Breitenberg:
    def __init__(self, imgs, steps, dt=0.001, overlap=30, blind=45, no_of_eyes=2):
        self.overlap = overlap
        self.blind = blind
        self.imgs = self.split_route_images(imgs)
        self.dt = dt
        self.t = 0
        self.Weights = Weighting()
        self.N, self.N_Indices = builder(steps)
        self.logs = {'Acceleration': np.zeros((2, steps + 1)), 'AA': np.zeros((2, steps + 1))}

        self.radius = 1
        self.bearing = []

    def get_motors(self, img):
        # give the visual input to the LAL - network
        l, r = self.matching_function(img)
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

    def get_heading(self, img):
        ml, mr = self.get_motors(img)
        # average velocity and angular velocity
        va = (ml + mr) / 2
        omega = (mr + ml) / (2 * self.radius)
        bearing = np.fmod(self.bearing[-1] + self.dt * omega, (2*np.pi))
        self.bearing.append(bearing)
        # angle in degrees
        angle = bearing * (180 / np.pi)
        return angle

    def get_heading2(self, img):
        '''
        Function for converting motor input to heading using the matlab function
        '''
        ml, mr = self.get_motors(img)
        r = 0.0033; # ~radius for the rotation in Melophorus bagoti
        # rotation gain
        RG = np.pi/180; # conversion deg -->rad
        RotationPart = abs(ml - mr)
        PropulsionPart = max(ml,mr) - RotationPart
        if ml > mr:
            phi = RG*(-RotationPart/r)
        elif ml == mr:
            phi = 0
        elif ml < mr:
            phi = RG*RotationPart/r

        return phi

    def split_route_images(self, imgs):
        return [image_split(im, self.overlap, self.blind) for im in imgs]

    def matching_function(self, img):
        left_sims = []
        right_sims = []
        l, r = image_split(img, overlap=self.overlap, blind=self.blind)
        for eyes in self.imgs:
            left_sims.append(cor_coef(l, eyes[0]))
            right_sims.append(cor_coef(r, eyes[1]))
        best_left = np.max(left_sims)
        best_right = np.max(right_sims)
        return best_left, best_right
    
    def set_bearing(self, deg):
        rad = deg * (np.pi / 180)
        self.bearing.append(rad)
