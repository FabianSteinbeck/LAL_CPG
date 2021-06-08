import antworld
import cv2
import numpy as np
from utils import squash_deg, pre_process, pol2cart, write_route, check_for_dir_and_create


# Old Seville data (lower res, but loads faster)
worldpath = antworld.bob_robotics_path + "/resources/antworld/world5000_gray.bin"
# z = 0.01 # m

# New Seville data
worldpath = antworld.bob_robotics_path + "/resources/antworld/seville_vegetation_downsampled.obj"
print(antworld.bob_robotics_path)
z = 1.5 # m (for some reason the ground is at ~1.5m for this world)

# agent = antworld.Agent(720, 150)
# (xlim, ylim, zlim) = agent.load_world(worldpath)
# print(xlim, ylim, zlim)


class Agent:
    def __init__(self):
        self.agent = antworld.Agent(720, 150)
        (xlim, ylim, zlim) = self.agent.load_world(worldpath)
        print(xlim, ylim, zlim)

    def get_img(self, xy, deg):
        '''
        Render a greyscale image from the antworld given an xy position and heading
        :param xy:
        :param deg:
        :return:
        '''
        self.agent.set_position(xy[0], xy[1], z)
        self.agent.set_attitude(deg, 0, 0)
        return cv2.cvtColor(self.agent.read_frame(), cv2.COLOR_BGR2GRAY)

    def update_position(self, xy, deg, r):
        rad = deg * (np.pi / 180)
        x, y = pol2cart(r, rad)

        xx = xy[0] + x
        yy = xy[1] + y

        self.agent.set_position(xx, yy, z)
        self.agent.set_attitude(deg, 0, 0)

        img = self.agent.read_frame()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return (xx, yy), img

    def test_nav(self, route, nav, r=0.05, t=100, sigma=0.1, preproc={}):
        # random initial position and heading
        # near the first location of the route
        if sigma:
            h = np.random.randint(0, 360)
            x = route['x'][0]
            x = np.random.normal(x, sigma)
            y = route['y'][0]
            y = np.random.normal(y, sigma)
            xy = (x, y)
        else:
            xy = (route['x'][0], route['y'][0])
            h = route['yaw'][0]

        # Place agent to the initial position and render the image
        img = self.get_img(xy, h)
        img = pre_process(img, preproc)

        # initialise the log variables
        headings = []
        headings.append(h)
        traj = np.empty((2, t))
        traj[0, 0] = xy[0]
        traj[1, 0] = xy[1]
        # Navigation loop
        for i in range(1, t):
            h = nav.get_heading(img)
            h = headings[-1] + h
            h = squash_deg(h)
            headings.append(h)
            # get new position and image
            xy, img = self.update_position(xy, h, r)
            img = pre_process(img, preproc)
            traj[0, i] = xy[0]
            traj[1, i] = xy[1]

        headings = np.array(headings)
        trajectory = {'x': traj[0], 'y': traj[1], 'heading': headings}
        return trajectory, nav

    def record_route(self, route, path, route_id=1):
        check_for_dir_and_create(path)
        x = route['x']
        y = route['y']
        z = route['z']
        headings = route['yaw']
        # Fixed high for now
        # TODO: in the future it may be a good idea to adap the coed to use the elevation
        #   and the pitch, roll noise
        z = 1.5
        route['filename'] = []

        for i, (xi, yi, h1) in enumerate(zip(x, y, headings)):
            self.agent.set_position(xi, yi, z)
            self.agent.set_attitude(h1, 0, 0)
            img = self.agent.read_frame()
            filename = path + "img%i.png" % i
            cv2.imwrite(filename, img)
            route['filename'].append("img%i.png" % i)

        write_route(path, route, route_id=route_id)

"""
Testing
"""
# agent = Agent()
# TODO: Run antworld agent for t1, graba  frame and pas to Breitenberg.
# TODO: update agent possition for X no of steps before getting a new frame

# How to record a route
# route = {}
# route['x'] = np.arange(10)
# route['y'] = np.arange(10)
# route['z'] = np.full(10, 1.5)
# route['yaw'] = np.full(10, 45)
# path = 'testroute/'
# agent = Agent()
# agent.record_route(route, path)


