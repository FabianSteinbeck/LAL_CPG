from utils import load_route_naw, plot_route, pre_process
import antworld2 as aw
from braitenberg import Breitenberg

route_id = 1
path = 'new-antworld/exp1/route' + str(route_id) + '/'

# path = '../test_data/route' + str(route_id) + '/'
route = load_route_naw(path, route_id=route_id, imgs=True)

plot_route(route)
route_imgs = route['imgs']
route_imgs = pre_process(route_imgs, sets={'shape': (180, 50)})
# nav = seqnav.SequentialPerfectMemory

# nav = nav(route_imgs, 'mae', deg_range=(-180, 180))
agent = aw.Agent()
nav = Breitenberg(route_imgs, 50)
traj, nav = agent.test_nav(route, nav, t=20, r=0.1, preproc={'shape': (180, 50)})


plot_route(route, traj)
