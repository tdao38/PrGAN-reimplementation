import torch
import numpy as np


class ShapeDiscriminator3D(torch.nn.Module):

    def __init__(self):
        super(ShapeDiscriminator3D, self).__init__()
        self.d_size = 128
        self.batch_size = 64
        self.image_size = (32, 32)

        # Architecture
        self.batch_norm_1 = torch.nn.BatchNorm2d(128)
        self.batch_norm_2 = torch.nn.BatchNorm2d(128*2)
        self.batch_norm_3 = torch.nn.BatchNorm2d(128*4)
        self.batch_norm_4 = torch.nn.BatchNorm2d(128*8)


        self.conv2d_1 = torch.nn.Conv2d(1, self.d_size, 5, 2, 2)
        self.conv2d_2 = torch.nn.Conv2d(self.d_size, self.d_size*2, 5, 2, 2)
        self.conv2d_3 = torch.nn.Conv2d(self.d_size * 2, self.d_size * 4, 5, 2, 2)
        self.conv2d_4 = torch.nn.Conv2d(self.d_size * 4, self.d_size * 8, 5, 2, 2)
        self.linear = torch.nn.Linear(4096, 1)
        self.leakyrelu = torch.nn.LeakyReLU(0.2)
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self, image, train_flag, reuse = False):
        # input must be a torch.randn(64, 1, 32, 32)

        reshaped_img = torch.reshape(image, [self.batch_size, 1, self.image_size[0], self.image_size[1]])
        h0 = self.conv2d_1(reshaped_img)
        h0 = self.leakyrelu(self.batch_norm_1(h0))

        h1= self.conv2d_2(h0)
        h1 = self.leakyrelu(self.batch_norm_2(h1))

        h2 = self.conv2d_3(h1)
        h2 = self.leakyrelu(self.batch_norm_3(h2))

        h3 = self.conv2d_4(h2)
        h3 = self.leakyrelu(self.batch_norm_4(h3))

        h3 = h3.reshape([64, -1])

        h4 = self.linear(h3)

        sigmoid = self.sigmoid(h4)

        return sigmoid, h4, h0

    def loss(self, logits, p):
        #https://stackoverflow.com/questions/65458736/pytorch-equivalent-to-tf-nn-softmax-cross-entropy-with-logits-and-tf-nn-sigmoid

        # calculate sigmoid_cross_entropy_with_logits
        # not sure if this is right
        loss = p * -torch.log(torch.sigmoid(logits)) + (1 - p) * -torch.log(1 - torch.sigmoid(logits))

        return loss.mean()

    def stat(self, input):
        return input.mean(dim=2) , input.std(dim=2)


def l2(a, b):
    return (torch.pow(a - b, 2)).mean()




###############################  test script begin
model = ShapeDiscriminator3D()
input = torch.randn(64, 1, 32, 32)
sigmoid, h4, h0 = model(input, False)
p = torch.ones(sigmoid.shape)
loss = model.loss(h4, p)
mean, std = model.stat(h0)


G_loss = l2(dr_mean, dl_mean)
G_loss += l2(dr_var, dl_var)
D_loss = D_loss_real + D_loss_fake



###############################  test script end
# calculate the loss
# write a scirpt that call both function
## optimzie the code
# render the image
# 

# self.G_loss = ops.l2(dr_mean, dl_mean)
#
# self.D_optim = tf.train.AdamOptimizer(1e-4, beta1=0.5).minimize(self.D_loss, var_list=self.D_vars)
# self.G_optim = tf.train.AdamOptimizer(0.0025, beta1=0.5).minimize(self.G_loss, var_list=self.G_vars)
# self.G_optim_classic = tf.train.AdamOptimizer(0.0025, beta1=0.5).minimize(self.G_loss_classic, var_list=self.G_vars)
#
#
#
# p_tensor=  tf.convert_to_tensor(torch.empty(sigmoid.shape))
# logits_logit= tf.convert_to_tensor(torch.empty(logits .shape))
#
# logits_tensor = tf.convert_to_tensor(logits)
#
# p = p_tensor
# logit = logits_logit
#
# loss = p*-torch.log(torch.sigmoid(logits)) + (1-p)*-torch.log(1-torch.sigmoid(logits))





# #https://stackoverflow.com/questions/52975843/comparing-conv2d-with-padding-between-tensorflow-and-pytorch
#
#
#
#
#         d_size = 128
#         self.image_size = 32
#         self.image_size = 32
#
#         weight1 = torch.normal(0, 0.02, size=(128, 1,5,5))
#         weight1_bias =
#         weight2 = torch.normal(0, 0.02, size=(128*2, 128,5,5))
#         weight1_bias =
#         weight3 = torch.normal(0, 0.02, size=(128 *4, 128 * 2, 5, 5))
#         weight1_bias =
#         weight4 = torch.normal(0, 0.02, size=(128 * 8, 128 * 4, 5, 5))
#         weight1_bias =
#
#         linear_w =torch.normal(0, 0.1, size=(128 * 8, 128 * 4, 5, 5))
#
#         linear_b =
#
#
#
#
#         reshaped_img = torch.reshape(a, [self.batch_size, 1, self.image_size[0], self.image_size[1]])
#
#         # weight = torch.Size([128, 1, 5, 5])
#         weight = torch.randn(128, 1, 5, 5)
#         h0 = F.conv2d(input, weight, padding=2, stride=2)
#         # add bias term
#         #torch.Size([64, 128, 16, 16])
#
#         weight2 = torch.randn(128*2, 128, 5, 5)
#         h1 = F.conv2d(h0, weight2 , padding=2, stride=2)
#
#         weight3 = torch.randn(128*4, 128*2, 5, 5)
#         h2 = F.conv2d(h1, weight3, padding=2, stride=2)
#
#         weight4 = torch.randn(128*8, 128*4, 5, 5)
#         h3 = F.conv2d(h2, weight4, padding=2, stride=2)
#
#         h3 = h3.reshape([64,-1])
#
#         m = torch.nn.Linear(4096, 1)
#
#         h4 =m(h3)
#
#         return torch.sigmoid(h4), h4, h0
#
#         # weight for the 2d initializer=tf.truncated_normal_initializer(stddev=stddev)
#         stddev = 0.02
#
#         # weight for linear w = tf.random_normal_initializer(mean=0.0, stddev=0.1)
#
#         # b  = tf.constant_initializer()
#
#         #  biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0)
#
#
#
#         # 64x32x32x1
#
#         # w = [5,5,1,128]
#         # strides = [1,2,2,1]
#         # TensorShape([64, 16, 16, 128])
#
#         # leakyrelu layer
#         # # TensorShape([64, 16, 16, 128])
#         F.conv2d(input, weights_torch, padding=2, stride=2)
#
#
#         reshaped_img = tf.reshape(image, [self.batch_size, self.image_size[0], self.image_size[1], 1])
#         # 64 x 32 x 32 x1
#
#         weights_tf.shape
#         TensorShape([5, 5, 1, 128])
#
#         h0 = tf.nn.conv2d(x, weights_tf, strides=[1, 2, 2, 1], padding='SAME')
#         #TensorShape([64, 16, 16, 128])
#
#         h0.get_shape()[-1] #- 128
#         weight = [5,5,128,128*2]
#         weights = torch.empty(5, 5, 128, 128 * 2)
#         weights_tf = tf.convert_to_tensor(weights.numpy(), dtype=tf.float32)
#
#         h0 = ops.lrelu(self.d_bn0(h0, train))
#         h1 = ops.conv2d(h0, self.d_size*2, name='d_h1_conv')
#         h1 = ops.lrelu(self.d_bn1(h1, train))
#         #TensorShape([64, 8, 8, 256])
#
#
#         h2 = ops.conv2d(h1, self.d_size*4, name='d_h2_conv')
#         weight = [5, 5, 256, 128 * 4]
#         h2 = ops.lrelu(self.d_bn2(h2, train))
#         #TensorShape([64, 4, 4, 512])
#
#
#
#        weight = [5,5,128*4, 128*8]
#
#         h3 = ops.conv2d(h2, self.d_size*8, name='d_h3_conv')
#     # TensorShape([64, 2, 2, 1024])
#
#
#         h3_tensor = ops.lrelu(self.d_bn3(h3, train))
#
#
#
#         h3 = tf.reshape(h3_tensor, [self.batch_size, -1])
#     #TensorShape([64, 4096])
#     # h3.get_shape()[1] = 4096
#
#     # weight = [4096, 1]
#     # b = [1]
#     # matmul(h4, w) + b
#
#
#
#         h4 = ops.linear(h3, h3.get_shape()[1], 1, scope='d_h5_lin')
#     # 64 x 1
#
#     sigmoid(h4)
#
#
# def truncated_normal_(tensor, mean=0, std=0.02):
#     size = [128, 1, 5, 5]
#     tmp = tensor.new_empty(size + (4,)).normal_()
#     valid = (tmp < 2) & (tmp > -2)
#     ind = valid.max(-1, keepdim=True)[1]
#     tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
#     tensor.data.mul_(std).add_(mean)


# image = np.random.uniform(-1, 1, [64, 32,32])
# a = torch.FloatTensor(image)
#
# input = torch.randn(64, 32, 32, 1)
# tf.reshape(image, [64, 32, 32, 1])
