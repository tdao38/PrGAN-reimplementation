import numpy as np

import tensorflow as tf
import matplotlib.image as mpimg
import numpy as np
import scipy.misc
import sys
import re

def grid_coord(h, w, d):
    xl = np.linspace(-1.0, 1.0, w)
    yl = np.linspace(-1.0, 1.0, h)
    zl = np.linspace(-1.0, 1.0, d)

    xs, ys, zs = np.meshgrid(xl, yl, zl, indexing='ij')
    g = np.vstack((xs.flatten(), ys.flatten(), zs.flatten()))
    return g


def transform_volume(v, t):
    height = int(v.get_shape()[0])
    width = int(v.get_shape()[1])
    depth = int(v.get_shape()[2])
    grid = grid_coord(height, width, depth)

    xs = grid[0, :]
    ys = grid[1, :]
    zs = grid[2, :]

    idxs_f = np.transpose(np.stack([xs, ys, zs]))
    idxs_f = np.matmul(idxs_f, t)

    xs_t = (idxs_f[:, 0] + 1.0) * float(width) / 2.0
    ys_t = (idxs_f[:, 1] + 1.0) * float(height) / 2.0
    zs_t = (idxs_f[:, 2] + 1.0) * float(depth) / 2.0

    return tf.reshape(resample_voxels(v, xs_t, ys_t, zs_t, method='trilinear'), v.get_shape())

def rot_matrix(s):
    i = tf.cast((s+1)*4.5, 'int32')
    vp = tf.constant([0, np.pi/4.0, np.pi/2.0, 3*np.pi/4.0, np.pi, 5*np.pi/4.0, 3*np.pi/2.0, 7*np.pi/4.0, np.pi/2.0])
    hp = tf.constant([0, 0, 0, 0, 0, 0, 0, 0, np.pi/2.0])

    theta = tf.gather(vp, i)
    phi = tf.gather(hp, i)

    #theta = (s[0] + 1.0) * np.pi
    #phi = s[1] * np.pi/2.0

    sin_theta = tf.sin(theta)
    cos_theta = tf.cos(theta)
    sin_phi = tf.sin(phi)
    cos_phi = tf.cos(phi)

    ry = [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
    rx = [[1, 0, 0], [0, cos_phi, -sin_phi], [0, sin_phi, cos_phi]]

    return tf.matmul(rx, ry)
    #return ry

def get_voxel_values(v, xs, ys, zs):
    idxs = tf.cast(tf.stack([xs, ys, zs], axis=1), 'int32')
    idxs = tf.clip_by_value(idxs, 0, v.get_shape()[0])
    idxs = tf.expand_dims(idxs, 0)
    return gather_nd(v, idxs)


def resample_voxels(v, xs, ys, zs, method="trilinear"):
    if method == "trilinear":
        floor_xs = tf.floor(tf.clip_by_value(xs, 0, 64))
        floor_ys = tf.floor(tf.clip_by_value(ys, 0, 64))
        floor_zs = tf.floor(tf.clip_by_value(zs, 0, 64))

        ceil_xs = tf.ceil(tf.clip_by_value(xs, 0, 64))
        ceil_ys = tf.ceil(tf.clip_by_value(ys, 0, 64))
        ceil_zs = tf.ceil(tf.clip_by_value(zs, 0, 64))

        final_value = (
                    tf.abs((xs - floor_xs) * (ys - floor_ys) * (zs - floor_zs)) * get_voxel_values(v, ceil_xs, ceil_ys,
                                                                                                   ceil_zs) +
                    tf.abs((xs - floor_xs) * (ys - floor_ys) * (zs - ceil_zs)) * get_voxel_values(v, ceil_xs, ceil_ys,
                                                                                                  floor_zs) +
                    tf.abs((xs - floor_xs) * (ys - ceil_ys) * (zs - floor_zs)) * get_voxel_values(v, ceil_xs, floor_ys,
                                                                                                  ceil_zs) +
                    tf.abs((xs - floor_xs) * (ys - ceil_ys) * (zs - ceil_zs)) * get_voxel_values(v, ceil_xs, floor_ys,
                                                                                                 floor_zs) +
                    tf.abs((xs - ceil_xs) * (ys - floor_ys) * (zs - floor_zs)) * get_voxel_values(v, floor_xs, ceil_ys,
                                                                                                  ceil_zs) +
                    tf.abs((xs - ceil_xs) * (ys - floor_ys) * (zs - ceil_zs)) * get_voxel_values(v, floor_xs, ceil_ys,
                                                                                                 floor_zs) +
                    tf.abs((xs - ceil_xs) * (ys - ceil_ys) * (zs - floor_zs)) * get_voxel_values(v, floor_xs, floor_ys,
                                                                                                 ceil_zs) +
                    tf.abs((xs - ceil_xs) * (ys - ceil_ys) * (zs - ceil_zs)) * get_voxel_values(v, floor_xs, floor_ys,
                                                                                                floor_zs)
                    )
        return final_value

    elif method == "nearest":
        r_xs = tf.round(xs)
        r_ys = tf.round(ys)
        r_zs = tf.round(zs)
        return get_voxel_values(v, r_xs, r_ys, r_zs)

    else:
        raise NameError(method)

def gather_nd(params, indices, name=None):
    # shape = params.get_shape().as_list()
    shape = params.shape
    rank = len(shape)
    flat_params = torch.reshape(params, [-1])
    multipliers = [reduce(lambda x, y: x*y, shape[i+1:], 1) for i in range(0, rank)]
    indices_unpacked = tf.unpack(tf.transpose(indices, [rank - 1] + range(0, rank - 1), name))
    flat_indices = sum([a*b for a,b in zip(multipliers, indices_unpacked)])
    return tf.gather(flat_params, flat_indices, name=name)