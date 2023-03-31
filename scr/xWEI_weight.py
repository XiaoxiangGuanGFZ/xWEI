"""
manually compute the weights for Eta curves interpolation
then call the weights (import from local files)
    for xWEI computation

"""

import numpy as np
import scipy.spatial.qhull as qhull
import pandas as pd


def interp_weights(xyz, uvw, d=2):
    tri = qhull.Delaunay(xyz)
    simplex = tri.find_simplex(uvw)
    vertices = np.take(tri.simplices, simplex, axis=0)
    temp = np.take(tri.transform, simplex, axis=0)
    delta = uvw - temp[:, d]
    bary = np.einsum('njk,nk->nj', temp[:, :d, :], delta)
    return vertices, np.hstack((bary, 1 - bary.sum(axis=1, keepdims=True)))


def interpolate(values, vtx, wts, fill_value=np.nan):
    ret = np.einsum('nj,nj->n', np.take(values, vtx), wts)
    ret[np.any(wts < 0, axis=1)] = fill_value
    return ret


def weight_generate(domain_size=1713,
                    resolution=1000,
                    durations=['24', '48', '72', '96'],
                    outpath='D:/xWEI_RGW/', outnames=['vtx', 'wts']
                    ):
    """
    on the basis of the interp_weights() functions,
    we will compute the weights for domains individually
    -parameters:
    domain_size: the number (size) of the domain of interest
    durations: the durations of interest, a vector of char
    outpath: the directory where the output files should be stored
    outnames: the file names of outputted vtx and wts
    -output:

    """
    len_x = domain_size  # e.g. if you have a 200km * 200 km window
    # the number of grids in the domain
    # durations = ['01', '02', '04', '06', '12', '24', '48', '72']
    # durations = ['24', '48', '72', '96']
    durations = [int(i) for i in durations]
    # resolution = 1000

    coords = np.ones((len_x * len(durations), 2))

    x_coords = list(np.arange(1, len_x + 1)) * len(durations)

    # from https://stackoverflow.com/questions/2449077/duplicate-each-member-in-a-list
    y_coords = [val for val in durations for _ in (range(1, len_x + 1))]

    coords[:, 0] = np.log(x_coords)
    coords[:, 1] = np.log(y_coords)

    # create the refined grid on which the values will be later interpolated
    x_range = np.linspace(0, np.log(len_x), len_x)
    y_range = np.linspace(0, np.log(durations[-1]), resolution)
    grid_x, grid_y = np.meshgrid(x_range, y_range)

    uv = np.zeros([grid_x.shape[0] * grid_x.shape[1], 2])
    uv[:, 0] = grid_x.flatten()
    uv[:, 1] = grid_y.flatten()

    # interpolate the eta values to get a surface. These weights can be reused for every 200km x 200km box
    vtx, wts = interp_weights(coords[:, :2], uv)

    vtx = pd.DataFrame(vtx)
    vtx.to_csv(outpath + outnames[0] + '.csv', index=False, header=False)

    wts = pd.DataFrame(wts)
    wts.to_csv(outpath + outnames[1] + '.csv', index=False, header=False)
    print('weights generation and store: done!')


def xWEI_weight(
        df, vtx, wts, resolution=1000,
        duration_levels=['24', '48', '72', '96']
) -> float:
    """
    calculate the xWEI value by using the pre-stored weights
    -parameters:
    df: the array of Eta values, the number of columns equals the duration numbers (be defaut 4)
    vtx, wts: output from the function weight_generate()
    -return:
    xWEI: float, one-value vector
    """
    eta = np.array(df).flatten("F")
    # interpolate values on finer grid using precalculated weights and vertices
    grid_z = interpolate(eta, vtx, wts)

    dx = np.log(len(df)) / len(df)
    dy = np.log(int(duration_levels[-1])) / resolution
    xwei = np.nansum(dx * dy * grid_z)
    return xwei
