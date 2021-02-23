from utils import load_route_naw, plot_route, cor_coef, image_split, display_split

path = 'test_route/route1/'
route = load_route_naw(path, route_id=1, imgs=True)

plot_route(route)


for img in route['imgs']:
    # split the image in 2 eyes
    # The overlap at the front and the blind spot at the back
    #   are both defined in degrees
    left, right = image_split(img, overlap=0, blind=0)

    # Show the 2 images
    display_split(left, right)

    # Get the similarty for each image with itself.
    # In future test we will change this
    sim_l = cor_coef(left, left)
    sim_r = cor_coef(right, right)

    # Add the LAL_CCP code here
