import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2 as cv
from scipy.spatial.distance import correlation, cdist
import math
import os


def display_split(image_l, image_r, size=(10, 10), file_name=None):
    image_l = np.squeeze(image_l)
    image_r = np.squeeze(image_r)

    fig = plt.figure(figsize=size)
    fig.add_subplot(1, 2, 1)
    plt.imshow(image_l, cmap='gray', interpolation='bilinear')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

    fig.add_subplot(1, 2, 2)
    plt.imshow(image_r, cmap='gray', interpolation='bilinear')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    # or plt.axis('off')
    plt.tight_layout(pad=0)
    if file_name: plt.savefig(file_name + ".png")
    plt.show()


def plot_route(route, traj=None, scale=70, window=None, windex=None, save=False, size=(10, 10), path=None):
    fig, ax = plt.subplots(figsize=size)
    u, v = pol2cart_headings(90 - route['yaw'])
    ax.scatter(route['x'], route['y'])
    ax.quiver(route['x'], route['y'], u, v, scale=scale)
    if window and windex:
        start = window[0]
        end = window[1]
        ax.quiver(route['x'][start:end], route['y'][start:end], u[start:end], v[start:end], color='r', scale=scale)
        ax.scatter(route['qx'][:windex], route['qy'][:windex])
    # Plot grid test points
    if 'qx' in route and window is None:
        ax.scatter(route['qx'], route['qy'])
    # Plot the trajectory of the agent when repeating the route
    if traj:
        u, v = pol2cart_headings(90 - traj['heading'])
        ax.scatter(traj['x'], traj['y'])
        ax.quiver(traj['x'], traj['y'], u, v, scale=scale)

    if save:
        fig.savefig(path + str(windex) + '.png')
        plt.close(fig)
    if not save: plt.show()


def load_route_naw(path, route_id=1, imgs=False, query=False, max_dist=0.5):
    route_data = pd.read_csv(path + 'route' + str(route_id) + '.csv', index_col=False)
    route_data = route_data.to_dict('list')
    # convert the lists to numpy arrays
    for k in route_data:
        route_data[k] = np.array(route_data[k])
    if imgs:
        imgs = []
        for i in route_data['filename']:
            img = cv.imread(path + i, cv.IMREAD_GRAYSCALE)
            imgs.append(img)
        route_data['imgs'] = imgs

    # Sample positions and images from the grid near the route for testing
    if query:
        grid = pd.read_csv(path + 'grid70.csv')
        grid = grid.to_dict('list')
        for k in grid:
            grid[k] = np.array(grid[k])

        grid_xy = np.transpose(np.array([grid['x'], grid['y']]))
        query_indexes = np.empty(0, dtype=int)
        qx = np.empty(0)
        qy = np.empty(0)
        qimg = []
        # Fetch images from the grid that are located nearby route images.
        # for each route position
        for i, (x, y) in enumerate(zip(route_data['x'], route_data['y'])):
            # get distance between route point and all grid points
            dist = np.squeeze(cdist([(x, y)], grid_xy, 'euclidean'))
            # indexes of distances within the limit
            indexes = np.where(dist <= max_dist)[0]
            # check which indexes have not been encountered before
            mask = np.isin(indexes, query_indexes, invert=True)
            # get the un-encountered indexes
            indexes = indexes[mask]
            # save the indexes
            query_indexes = np.append(query_indexes, indexes)

            for i in indexes:
                qx = np.append(qx, grid_xy[i, 0])
                qy = np.append(qy, grid_xy[i, 1])
                imgfile = path + grid['filename'][i]
                qimg.append(cv.imread(imgfile, cv.IMREAD_GRAYSCALE))

        route_data['qx'] = qx
        route_data['qy'] = qy
        route_data['qimgs'] = qimg

    return route_data


def image_split(image, overlap=None, blind=0):
    '''
    Splits an image to 2 left and right part evenly when no overlap is provided.
    :param image: Image to split. 2 dimentional ndarray
    :param overlap: Degrees of overlap between the 2 images
    :param blind: Degrees of blind visual field at the back of the agent
    :return:
    '''
    num_of_cols = image.shape[1]
    if blind:
        num_of_cols_perdegree = int(num_of_cols / 360)
        blind_pixels = blind * num_of_cols_perdegree
        blind_pixels_per_side = int(blind_pixels/2)
        image = image[:, blind_pixels_per_side:-blind_pixels_per_side]

    num_of_cols = image.shape[1]
    split_point = int(num_of_cols / 2)
    if overlap:
        num_of_cols_perdegree = int(num_of_cols / (360-blind))
        pixel_overlap = overlap * num_of_cols_perdegree
        l_split = split_point + int(pixel_overlap/2)
        r_split = split_point - int(pixel_overlap/2)
        left = image[:, :l_split]
        right = image[:, r_split:]
    else:
        left = image[:, :split_point]
        right = image[:, split_point:]

    return left, right


def rotate(d, image):
    """
    Converts the degrees into columns and rotates the image.
    Positive degrees turn the image clockwise
    and negative degrees rotate the image counter clockwise
    :param d: number of degrees the agent will rotate its view
    :param image: An np.array that we want to shift.
    :return: Returns the rotated image.
    """
    assert abs(d) <= 360

    num_of_cols = image.shape[1]
    num_of_cols_perdegree = num_of_cols / 360
    cols_to_shift = round(d * num_of_cols_perdegree)
    return np.roll(image, -cols_to_shift, axis=1)


def pol2cart(r, theta):
    '''
    Coverts polar coordinates to cartesian coordinates
    :param r: An array or single value of radial values
    :param theta: An array or single values ot angles theta
    :return:
    '''
    x = np.multiply(r, np.cos(theta))
    y = np.multiply(r, np.sin(theta))
    return x, y


def pol2cart_headings(headings):
    """
    Convert degree headings to U,V cartesian coordinates
    :param headings: list of degrees
    :return: 2D coordinates
    """
    rads = np.radians(headings)
    U, V = pol2cart(1, rads)
    return U, V


def cov(a, b):
    """
    Calculates covariance (non sample)
    Assumes flattened arrays
    :param a:
    :param b:
    :return:
    """
    assert len(a) == len(b)

    a_mean = np.mean(a)
    b_mean = np.mean(b)

    return np.sum((a - a_mean) * (b - b_mean)) / (len(a))


def cor_coef(a, b):
    """
    Calculate correlation coefficient
    :param a: A single image or vector
    :param b: A single image or vector
    :return:
    """
    a = a.flatten()
    b = b.flatten()
    return cov(a, b) / (np.std(a) * np.std(b))


def rmse(a, b):
    """
    Image Differencing Function RMSE
    :param a: A single query image
    :param b: One or more reference images
    :return:
    """
    if isinstance(b, list):
        return [np.sqrt(np.subtract(ref_img, a).mean()) for ref_img in b]

    return np.sqrt(np.subtract(b, a).mean())


def mae(a, b):
    """
    Image Differencing Function MAE
    :param a: A single query image
    :param b: One or more reference images
    :return:
    """
    if isinstance(b, list):
        return [cv.absdiff(a, img).mean() for img in b]

    return cv.absdiff(a, b).mean()


def cor_dist(a, b):
    """
    Calculates the correlation coefficient distance
    between a (list of) vector(s) b and reference vector a
    :param a: A single query image
    :param b: One or more reference images
    :return:
    """
    a = a.flatten()
    if isinstance(b, list):
        return [correlation(a, img.flatten()) for img in b]

    return correlation(a, b.flatten())


def rmf(query_img, ref_imgs, matcher=mae, d_range=(0, 360), d_step=1):
    """
    Rotational Matching Function.
    Rotates a query image and compares it with one or more reference images
    :param query_img:
    :param ref_imgs:
    :param matcher:
    :param d_range:
    :param d_step:
    :return:
    """
    assert d_step > 0
    assert not isinstance(query_img, list)
    if not isinstance(ref_imgs, list):
        ref_imgs = [ref_imgs]

    degrees = range(*d_range, d_step)
    total_search_angle = round((d_range[1] - d_range[0]) / d_step)
    sims = np.empty((len(ref_imgs), total_search_angle), dtype=np.float)

    for i, rot in enumerate(degrees):
        # rotated query image
        rqimg = rotate(rot, query_img)
        sims[:, i] = matcher(rqimg, ref_imgs)

    return sims if sims.shape[0] > 1 else sims[0]