import torch
import numpy as np


class ShapeGenerator3D(torch.nn.Module):
    """
    Transform the input z into voxel representation
    :return: voxel represntation of the shape
    """
    def __init__(self):
        super(ShapeGenerator3D, self).__init__()
        self.z_dim = 201
        self.vox_dim = 32
        self.z = self.get_z(self.z_dim)

        # Architecture
        self.fc = torch.nn.Linear(in_features=200,  # self.z_dim-1,
                                  out_features=64 * 128 * 4 * 4 * 4)
        self.batch_norm_1 = torch.nn.BatchNorm3d(128)
        self.batch_norm_2 = torch.nn.BatchNorm3d(64)
        self.batch_norm_3 = torch.nn.BatchNorm3d(32)
        self.batch_norm_4 = torch.nn.BatchNorm3d(1)

        self.relu = torch.nn.ReLU()

        self.conv3d_1 = torch.nn.ConvTranspose3d(in_channels=128,
                                                out_channels=64,
                                                kernel_size=4,
                                                padding=1,
                                                stride=2)

        self.conv3d_2 = torch.nn.ConvTranspose3d(in_channels=64,
                                                out_channels=32,
                                                kernel_size=4,
                                                padding=1,
                                                stride=2)

        self.conv3d_3 = torch.nn.ConvTranspose3d(in_channels=32,
                                                out_channels=1,
                                                kernel_size=4,
                                                padding=1,
                                                stride=2)

        self.conv3d_4 = torch.nn.ConvTranspose3d(in_channels=1,
                                                out_channels=1,
                                                kernel_size=4,
                                                padding=1,
                                                stride=2)
        self.sigmoid = torch.nn.Sigmoid()

    def get_z(z_dim=201):
        """
        Generate the input z for the generator, drawn from a uniform distribution U(-1,1)
        :param dim: dimension of the z vector, default is 201 as stated by the paper
        :return: vector z
        """
        z = torch.FloatTensor(z_dim).uniform_(-1, 1)

        return z

    def generator(self, z):

        z_fc = self.fc(z[:200])
        z_fc_reshape = z_fc.reshape(64,128,4,4,4)
        model = torch.nn.Sequential(self.batch_norm_1,
                                    self.relu,
                                    self.conv3d_1,
                                    self.batch_norm_2,
                                    self.relu,
                                    self.conv3d_2,
                                    self.batch_norm_3,
                                    self.relu,
                                    self.conv3d_3,
                                    self.batch_norm_4,
                                    self.relu,
                                    self.conv3d_4,
                                    self.sigmoid)

        z_final = model(z_fc_reshape)
        return z_final

## TODO: YOU CAN RUN FROM HERE
# DRAFT:
z_dim=201
fc = torch.nn.Linear(in_features=200, #self.z_dim-1,
                     out_features=64*128*4*4*4)
batch_norm_1 = torch.nn.BatchNorm3d(128)
batch_norm_2 = torch.nn.BatchNorm3d(64)
batch_norm_3 = torch.nn.BatchNorm3d(32)
batch_norm_4 = torch.nn.BatchNorm3d(1)

relu = torch.nn.ReLU()

conv3d_1 = torch.nn.ConvTranspose3d(in_channels=128,
                                    out_channels=64,
                                    kernel_size=4,
                                    padding=1,
                                    stride=2)

conv3d_2 = torch.nn.ConvTranspose3d(in_channels=64,
                                    out_channels=32,
                                    kernel_size=4,
                                    padding=1,
                                    stride=2)

conv3d_3 = torch.nn.ConvTranspose3d(in_channels=32,
                                     out_channels=1,
                                     kernel_size=4,
                                     padding=1,
                                     stride=2)

conv3d_4 = torch.nn.ConvTranspose3d(in_channels=1,
                                     out_channels=1,
                                     kernel_size=4,
                                     padding=1,
                                     stride=2)
sigmoid = torch.nn.Sigmoid()

z = torch.FloatTensor(z_dim).uniform_(-1, 1)
z_out = fc(z[:200])
z_out_reshape = z_out.reshape(64,128,4,4,4)
z_norm_1 = batch_norm_1(z_out_reshape)
z_relu_1 = relu(z_norm_1)
z_conv3d_1 = conv3d_1(z_relu_1)
z_conv3d_1.shape
z_norm_2 = batch_norm_2(z_conv3d_1)
z_relu_2 = relu(z_norm_2)
z_conv3d_2 = conv3d_2(z_relu_2)
z_conv3d_2.shape
z_norm_3 = batch_norm_3(z_conv3d_2)
z_relu_3 = relu(z_norm_3)
z_conv3d_3 = conv3d_3(z_relu_3)
z_conv3d_3.shape
z_norm_4 = batch_norm_4(z_conv3d_3)
z_relu_4 = relu(z_norm_4)
z_conv3d_4 = conv3d_4(z_relu_4)
z_conv3d_4.shape
z_final = sigmoid(z_conv3d_4)
voxels = z_final.reshape(64, 64, 64, 64)


## TODO: NEED TO FIGURE OUT HOW THIS RUNS:
# for i in xrange(self.batch_size):
#     img = ops.project(ops.transform_volume(self.voxels[i], ops.rot_matrix(v[i])),
#                       self.tau)

## The structure of the function calls:
# ops.project
# |_ ops.transform_volume
# | |_grid_coord
# | |_resample_voxels
# |   |_get_voxel_values
# |     |_ gather_nd: stuck here!!!
# |_ ops.rot_matrix


batch_size = 64

v = z[-1]

w = 64
h = 64
d = 64

## ROT MATRIX FUNCTION:
s = v
i = (s+1)*4.5
i = i.int()
vp = torch.Tensor([0, np.pi / 4.0, np.pi / 2.0, 3 * np.pi / 4.0, np.pi, 5 * np.pi / 4.0, 3 * np.pi / 2.0, 7 * np.pi / 4.0, np.pi / 2.0])
hp = torch.Tensor([0, 0, 0, 0, 0, 0, 0, 0, np.pi / 2.0])

theta = vp[i]
phi = hp[i]

sin_theta = torch.sin(theta)
cos_theta = torch.cos(theta)
sin_phi = torch.sin(phi)
cos_phi = torch.cos(phi)

ry = [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
rx = [[1, 0, 0], [0, cos_phi, -sin_phi], [0, sin_phi, cos_phi]]

## grid_cord:
xl = np.linspace(-1.0, 1.0, w)
yl = np.linspace(-1.0, 1.0, h)
zl = np.linspace(-1.0, 1.0, d)

xs, ys, zs = np.meshgrid(xl, yl, zl, indexing='ij')
g = np.vstack((xs.flatten(), ys.flatten(), zs.flatten()))


## TRANSFORM VOLUME FUNCTION:
## Loop through batch size voxels[i]
v = voxels[0]
t = np.matmul(rx, ry)
grid = g

xs = grid[0, :]
ys = grid[1, :]
zs = grid[2, :]

idxs_f = np.transpose(np.stack([xs, ys, zs]))
idxs_f = np.matmul(idxs_f, t)

xs_t = (idxs_f[:, 0] + 1.0) * float(w) / 2.0
ys_t = (idxs_f[:, 1] + 1.0) * float(h) / 2.0
zs_t = (idxs_f[:, 2] + 1.0) * float(d) / 2.0



## RESAMPLE VOXELS:
v = v
xs = torch.from_numpy(xs_t)
ys = torch.from_numpy(ys_t)
zs = torch.from_numpy(ys_t)

floor_xs = torch.floor(torch.clamp(xs, 0, 64))
floor_ys = torch.floor(torch.clamp(ys, 0, 64))
floor_zs = torch.floor(torch.clamp(zs, 0, 64))

ceil_xs = torch.ceil(torch.clamp(xs, 0, 64))
ceil_ys = torch.ceil(torch.clamp(ys, 0, 64))
ceil_zs = torch.ceil(torch.clamp(zs, 0, 64))

# GET VOXEL VALUE
idxs = torch.stack([xs, ys, zs], axis=1)
idxs = torch.clamp(idxs, 0, v.shape[0])
# idxs = tf.expand_dims(idxs, 0)

