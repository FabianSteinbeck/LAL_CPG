# This is the Simulation of the LAL-SNN
import antworld2 as aw
from utils import load_route_naw


def main():
    t = 100  # How long to
    r = 0.05
    preproc = {'shape': (180, 50)}
    agent = aw.Agent()
    path = 'test_route/route1/'
    route = load_route_naw(path, route_id=1, imgs=True)
    nav = None  ## here we need to initilize the LAL-CPG
    traj, nav = agent.test_nav(route, nav, t=t, r=r, sigma=None, preproc=preproc)


if __name__ == "__main__":
    main()