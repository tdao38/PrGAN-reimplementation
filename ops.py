import torch
import numpy as np
import math


def grid_coord(h, w, d):
    xl = np.linspace(-1.0, 1.0, w)
    yl = np.linspace(-1.0, 1.0, h)
    zl = np.linspace(-1.0, 1.0, d)

    xs, ys, zs = np.meshgrid(xl, yl, zl, indexing='ij')
    g = np.vstack((xs.flatten(), ys.flatten(), zs.flatten()))
    return g


def transform_volume(v, t):
    height = int(v.shape[0])
    width = int(v.shape[1])
    depth = int(v.shape[2])
    grid = grid_coord(height, width, depth)

    xs = grid[0, :]
    ys = grid[1, :]
    zs = grid[2, :]

    idxs_f = np.transpose(np.stack([xs, ys, zs]))
    idxs_f = np.matmul(idxs_f, t)

    xs_t = (idxs_f[:, 0] + 1.0) * float(width) / 2.0
    ys_t = (idxs_f[:, 1] + 1.0) * float(height) / 2.0
    zs_t = (idxs_f[:, 2] + 1.0) * float(depth) / 2.0

    return torch.reshape(resample_voxels(v, torch.from_numpy(xs_t), torch.from_numpy(ys_t), torch.from_numpy(zs_t), method='trilinear'), v.shape)

def rot_matrix(s):
    i = (s + 1) * 4.5
    i = i.int()
    vp = [0, np.pi / 4.0, np.pi / 2.0, 3 * np.pi / 4.0, np.pi, 5 * np.pi / 4.0, 3 * np.pi / 2.0, 7 * np.pi / 4.0,
          np.pi / 2.0]
    hp = [0, 0, 0, 0, 0, 0, 0, 0, np.pi / 2.0]

    theta = vp[i]
    phi = hp[i]

    #theta = (s[0] + 1.0) * np.pi
    #phi = s[1] * np.pi/2.0

    sin_theta = math.sin(theta)
    cos_theta = math.cos(theta)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    ry = [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
    rx = [[1, 0, 0], [0, cos_phi, -sin_phi], [0, sin_phi, cos_phi]]

    return torch.Tensor(np.matmul(rx, ry))
    #return ry

def get_voxel_values(v, xs, ys, zs):
    idxs = torch.ceil((torch.stack([xs, ys, zs], axis=1)))
    idxs = torch.clamp(idxs, 0, v.shape[0]-1)
    # idxs = tf.expand_dims(idxs, 0)
    return gather_nd(v, idxs)


def resample_voxels(v, xs, ys, zs, method="trilinear"):
    if method == "trilinear":

        floor_xs = torch.floor(torch.clamp(xs, 0, 64))
        floor_ys = torch.floor(torch.clamp(ys, 0, 64))
        floor_zs = torch.floor(torch.clamp(zs, 0, 64))

        ceil_xs = torch.ceil(torch.clamp(xs, 0, 64))
        ceil_ys = torch.ceil(torch.clamp(ys, 0, 64))
        ceil_zs = torch.ceil(torch.clamp(zs, 0, 64))

        final_value = (
                    torch.abs((xs - floor_xs) * (ys - floor_ys) * (zs - floor_zs)) * get_voxel_values(v, ceil_xs, ceil_ys,
                                                                                                   ceil_zs) +
                    torch.abs((xs - floor_xs) * (ys - floor_ys) * (zs - ceil_zs)) * get_voxel_values(v, ceil_xs, ceil_ys,
                                                                                                  floor_zs) +
                    torch.abs((xs - floor_xs) * (ys - ceil_ys) * (zs - floor_zs)) * get_voxel_values(v, ceil_xs, floor_ys,
                                                                                                  ceil_zs) +
                    torch.abs((xs - floor_xs) * (ys - ceil_ys) * (zs - ceil_zs)) * get_voxel_values(v, ceil_xs, floor_ys,
                                                                                                 floor_zs) +
                    torch.abs((xs - ceil_xs) * (ys - floor_ys) * (zs - floor_zs)) * get_voxel_values(v, floor_xs, ceil_ys,
                                                                                                  ceil_zs) +
                    torch.abs((xs - ceil_xs) * (ys - floor_ys) * (zs - ceil_zs)) * get_voxel_values(v, floor_xs, ceil_ys,
                                                                                                 floor_zs) +
                    torch.abs((xs - ceil_xs) * (ys - ceil_ys) * (zs - floor_zs)) * get_voxel_values(v, floor_xs, floor_ys,
                                                                                                 ceil_zs) +
                    torch.abs((xs - ceil_xs) * (ys - ceil_ys) * (zs - ceil_zs)) * get_voxel_values(v, floor_xs, floor_ys,
                                                                                                floor_zs)
                    )
        return final_value

    # elif method == "nearest":
    #     r_xs = tf.round(xs)
    #     r_ys = tf.round(ys)
    #     r_zs = tf.round(zs)
    #     return get_voxel_values(v, r_xs, r_ys, r_zs)
    #
    # else:
    #     raise NameError(method)

def gather_nd(v, idxs, name=None):
    # shape = params.get_shape().as_list()
    # shape = params.shape
    # rank = len(shape)
    # flat_params = torch.reshape(params, [-1])
    # multipliers = [reduce(lambda x, y: x*y, shape[i+1:], 1) for i in range(0, rank)]
    # indices_unpacked = tf.unpack(tf.transpose(indices, [rank - 1] + range(0, rank - 1), name))
    # flat_indices = sum([a*b for a,b in zip(multipliers, indices_unpacked)])
    # return tf.gather(flat_params, flat_indices, name=name)
    temp = v[idxs[:, 0].type(torch.LongTensor), idxs[:, 1].type(torch.LongTensor), idxs[:, 2].type(torch.LongTensor)]

    return temp

def project(v, tau=1):
    p = v.sum(axis=2)
    p = torch.ones(p.shape) - torch.exp(-p * tau)
    img = torch.flip(p.T, [0, 1])
    return img