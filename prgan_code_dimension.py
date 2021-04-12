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

# DRAFT
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

v = z[-1]

w = 64
h = 64
d = 64

def grid_coord(h, w, d):
    xl = np.linspace(-1.0, 1.0, w)
    yl = np.linspace(-1.0, 1.0, h)
    zl = np.linspace(-1.0, 1.0, d)

    xs, ys, zs = np.meshgrid(xl, yl, zl, indexing='ij')
    g = np.vstack((xs.flatten(), ys.flatten(), zs.flatten()))
    return g