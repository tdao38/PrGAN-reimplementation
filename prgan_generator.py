import torch
import numpy as np
import math

# import ops


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

    return torch.reshape(resample_voxels(v, xs_t, ys_t, zs_t, method='trilinear'), v.shape)

# identical to tensorflow
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
        # xs = torch.from_numpy(xs)
        # ys = torch.from_numpy(ys)
        # zs = torch.from_numpy(zs)

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
    else:
        r_xs = torch.round(xs)
        r_ys = torch.round(ys)
        r_zs = torch.round(zs)
        final_value = get_voxel_values(v, r_xs, r_ys, r_zs)

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

def loss_generator(fake):
    # https://stackoverflow.com/questions/65458736/pytorch-equivalent-to-tf-nn-softmax-cross-entropy-with-logits-and-tf-nn-sigmoid

    # calculate sigmoid_cross_entropy_with_logits
    # not sure if this is right
    # loss = p * -torch.log(logits) + (1 - p) * -torch.log(1 - logits)

    loss = torch.square(fake - torch.ones(fake.shape))
    # low gradient

    return loss.sum()

class ShapeGenerator3D(torch.nn.Module):
    """
    Transform the input z into voxel representation
    :return: voxel represntation of the shape
    """
    def __init__(self):
        super(ShapeGenerator3D, self).__init__()
        self.z_dim = 201
        self.vox_dim = 32
        self.batch_size = 64
        self.tau = 1
        #self.z = self.get_z()
        # Architecture
        self.fc = torch.nn.Linear(in_features=200,  # self.z_dim-1,
                                  out_features=1 * 128 * 4 * 4 * 4)
        self.batch_norm_1 = torch.nn.BatchNorm3d(128)
        self.batch_norm_2 = torch.nn.BatchNorm3d(64)
        self.batch_norm_3 = torch.nn.BatchNorm3d(32)
        self.batch_norm_4 = torch.nn.BatchNorm3d(1)

        self.relu_1 = torch.nn.ReLU()
        self.relu_2 = torch.nn.ReLU()
        self.relu_3 = torch.nn.ReLU()
        self.relu_4 = torch.nn.ReLU()

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

    def get_z(self):
        """
        Generate the input z for the generator, drawn from a uniform distribution U(-1,1)
        :param dim: dimension of the z vector, default is 201 as stated by the paper
        :return: vector z
        """
        z = torch.FloatTensor(self.batch_size, self.z_dim).uniform_(-1, 1)

        return z

    def forward(self):
        self.z = self.get_z()
        z_enc = self.z[:, :self.z_dim-1]
        z_fc = self.fc(z_enc)
        z_fc_reshape = z_fc.reshape(64, 128, 4, 4, 4)
        model = torch.nn.Sequential(self.batch_norm_1,
                                    self.relu_1,
                                    self.conv3d_1,
                                    self.batch_norm_2,
                                    self.relu_2,
                                    self.conv3d_2,
                                    self.batch_norm_3,
                                    self.relu_3,
                                    self.conv3d_3,
                                    self.batch_norm_4,
                                    self.relu_4,
                                    self.conv3d_4,
                                    self.sigmoid)
        z_final = model(z_fc_reshape) * (1.0 / self.tau)
        #z_final = model(z_fc_reshape) * (1.0 / 1)
        z_final_reshape = z_final.reshape(64, 64, 64, 64)
        self.voxels= z_final_reshape
        # if the value negative (1 - blacked binvox package )
        # visualize the grayscale of the voxels



        #self.voxels = torch.zeros(64,64,64,64)

        # for i in range(z_enc.shape[0]):
        #     z = z_enc[i]
        #     z_fc = self.fc(z)
        #     z_fc_reshape = z_fc.reshape(1,128,4,4,4)
        #     model = torch.nn.Sequential(self.batch_norm_1,
        #                                 self.relu,
        #                                 self.conv3d_1,
        #                                 self.batch_norm_2,
        #                                 self.relu,
        #                                 self.conv3d_2,
        #                                 self.batch_norm_3,
        #                                 self.relu,
        #                                 self.conv3d_3,
        #                                 self.batch_norm_4,
        #                                 self.relu,
        #                                 self.conv3d_4,
        #                                 self.sigmoid)
        #     z_final = model(z_fc_reshape) * (1.0/self.tau)
        #     z_final_reshape = z_final.reshape(1, 64, 64, 64)
        #     self.voxels[i] = z_final_reshape
        # View
        v = self.z[:, self.z_dim-1]
        rendered_imgs = torch.zeros(64, 64, 64)
        for i in range(self.batch_size):
            #print("batch ", i+1)
            img = project(transform_volume(self.voxels[i], rot_matrix(v[i])), self.tau)
            rendered_imgs[i] = img
        #rendered_imgs = (rendered_imgs > 0.5).float()


    # x^2 + y^2 + z^2 - r^2 > positive or negative  sphere function
        rendered_imgs_final = rendered_imgs.unsqueeze(1)
        return rendered_imgs_final

if __name__=='__main__':
    generator = ShapeGenerator3D()
    final_images = generator.forward()

## TODO: YOU CAN RUN FROM HERE
# DRAFT:
# z_dim=201
# fc = torch.nn.Linear(in_features=64, #self.z_dim-1,
#                      out_features=64*128*4*4*4)
# batch_norm_1 = torch.nn.BatchNorm3d(128)
# batch_norm_2 = torch.nn.BatchNorm3d(64)
# batch_norm_3 = torch.nn.BatchNorm3d(32)
# batch_norm_4 = torch.nn.BatchNorm3d(1)
#
# relu = torch.nn.ReLU()
#
# conv3d_1 = torch.nn.ConvTranspose3d(in_channels=128,
#                                     out_channels=64,
#                                     kernel_size=4,
#                                     padding=1,
#                                     stride=2)
#
# conv3d_2 = torch.nn.ConvTranspose3d(in_channels=64,
#                                     out_channels=32,
#                                     kernel_size=4,
#                                     padding=1,
#                                     stride=2)
#
# conv3d_3 = torch.nn.ConvTranspose3d(in_channels=32,
#                                      out_channels=1,
#                                      kernel_size=4,
#                                      padding=1,
#                                      stride=2)
#
# conv3d_4 = torch.nn.ConvTranspose3d(in_channels=1,
#                                      out_channels=1,
#                                      kernel_size=4,
#                                      padding=1,
#                                      stride=2)
# sigmoid = torch.nn.Sigmoid()
#
# z = torch.FloatTensor(z_dim).uniform_(-1, 1)
# z_out = fc(z[:200])
# z_out_reshape = z_out.reshape(64,128,4,4,4)
# z_norm_1 = batch_norm_1(z_out_reshape)
# z_relu_1 = relu(z_norm_1)
# z_conv3d_1 = conv3d_1(z_relu_1)
# z_conv3d_1.shape
# z_norm_2 = batch_norm_2(z_conv3d_1)
# z_relu_2 = relu(z_norm_2)
# z_conv3d_2 = conv3d_2(z_relu_2)
# z_conv3d_2.shape
# z_norm_3 = batch_norm_3(z_conv3d_2)
# z_relu_3 = relu(z_norm_3)
# z_conv3d_3 = conv3d_3(z_relu_3)
# z_conv3d_3.shape
# z_norm_4 = batch_norm_4(z_conv3d_3)
# z_relu_4 = relu(z_norm_4)
# z_conv3d_4 = conv3d_4(z_relu_4)
# z_conv3d_4.shape
# z_final = sigmoid(z_conv3d_4)
# voxels = z_final.reshape(64, 64, 64, 64)
#
#
# ## TODO: NEED TO FIGURE OUT HOW THIS RUNS:
# # for i in xrange(self.batch_size):
# #     img = ops.project(ops.transform_volume(self.voxels[i], ops.rot_matrix(v[i])),
# #                       self.tau)
#
# ## The structure of the function calls:
# # ops.project
# # |_ ops.transform_volume
# # | |_grid_coord
# # | |_resample_voxels
# # |   |_get_voxel_values
# # |     |_ gather_nd: stuck here!!!
# # |_ ops.rot_matrix
#
#
# batch_size = 64
#
# v = z[-1]
#
# w = 64
# h = 64
# d = 64
#
# ## ROT MATRIX FUNCTION:
# s = v
# i = (s+1)*4.5
# i = i.int()
# # not using torch tensor
# vp = [0, np.pi / 4.0, np.pi / 2.0, 3 * np.pi / 4.0, np.pi, 5 * np.pi / 4.0, 3 * np.pi / 2.0, 7 * np.pi / 4.0, np.pi / 2.0]
# hp = [0, 0, 0, 0, 0, 0, 0, 0, np.pi / 2.0]
# import math
#
#
# theta = vp[i]
# phi = hp[i]
#
# sin_theta = math.sin(theta)
# cos_theta = math.cos(theta)
# sin_phi = math.sin(phi)
# cos_phi = math.cos(phi)
#
# ry = [[cos_theta, 0, sin_theta], [0, 1, 0], [-sin_theta, 0, cos_theta]]
# rx = [[1, 0, 0], [0, cos_phi, -sin_phi], [0, sin_phi, cos_phi]]
# # return
# torch.Tensor(np.matmul(rx, ry))
# # 3x3
# # convert that to tensor
#
#
# ## grid_cord:
# xl = np.linspace(-1.0, 1.0, w)
# yl = np.linspace(-1.0, 1.0, h)
# zl = np.linspace(-1.0, 1.0, d)
#
# xs, ys, zs = np.meshgrid(xl, yl, zl, indexing='ij')
# g = np.vstack((xs.flatten(), ys.flatten(), zs.flatten()))
#
#
# ## TRANSFORM VOLUME FUNCTION:
# ## Loop through batch size voxels[i]
# v = voxels[0]
# t = np.matmul(rx, ry)
# grid = g
#
# xs = grid[0, :]
# ys = grid[1, :]
# zs = grid[2, :]
#
# idxs_f = np.transpose(np.stack([xs, ys, zs]))
# idxs_f = np.matmul(idxs_f, t)
#
# xs_t = (idxs_f[:, 0] + 1.0) * float(w) / 2.0
# ys_t = (idxs_f[:, 1] + 1.0) * float(h) / 2.0
# zs_t = (idxs_f[:, 2] + 1.0) * float(d) / 2.0
#
#
#
# ## RESAMPLE VOXELS:
# v = v
# xs = torch.from_numpy(xs_t)
# ys = torch.from_numpy(ys_t)
# zs = torch.from_numpy(ys_t)
# # xs = xs_t
# # ys = ys_t
# # zs = ys_t
#
#
# floor_xs = torch.floor(torch.clamp(xs, 0, 64))
# floor_ys = torch.floor(torch.clamp(ys, 0, 64))
# floor_zs = torch.floor(torch.clamp(zs, 0, 64))
#
# ceil_xs = torch.ceil(torch.clamp(xs, 0, 64))
# ceil_ys = torch.ceil(torch.clamp(ys, 0, 64))
# ceil_zs = torch.ceil(torch.clamp(zs, 0, 64))
#
# # GET VOXEL VALUE
# # convert them to integer bc they are indexs
# idxs =  torch.ceil((torch.stack([xs, ys, zs], axis=1)))
# idxs = torch.clamp(idxs, 0, v.shape[0]-1)
# # idxs_new = idxs.unsqueeze(0)
#
# # get_voxel_values
# temp = v[idxs[:,0].type(torch.LongTensor), idxs[:,1].type(torch.LongTensor), idxs[:,2].type(torch.LongTensor)]
#
# final = torch.abs((xs-floor_xs)*(ys-floor_ys)*(zs-floor_zs))*temp
#
# # transform volume
# transform = torch.reshape(final, v.shape)
#
# # project
#
# p = transform.sum(axis=2)
# tau=1
# p = torch.ones(p.shape) - torch.exp(-p*tau)
#
# ##### return
# img = torch.flip(p.T, [0, 1])
#
# batch_size=2
# rendered_imgs = torch.zeros(batch_size, 64,64)
# rendered_imgs[0] = img
# rendered_imgs[1] = img
#
# # to match with the dim for real image
# # batch_size, 1, 64, 64
# rendered_imgs.unsqueeze(1).shape
#
# from PIL import Image
# img_plot = Image.fromarray(img.detach().numpy(), 'L')
# img_plot.show()