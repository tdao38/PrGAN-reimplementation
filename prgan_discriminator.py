import torch


class ShapeDiscriminator3D(torch.nn.Module):

    def __init__(self):
        super(ShapeDiscriminator3D, self).__init__()
        self.d_size = 128
        self.batch_size = 64
        self.image_size = (64, 64)

        # Architecture
        self.batch_norm_1 = torch.nn.BatchNorm2d(128)
        self.batch_norm_2 = torch.nn.BatchNorm2d(128 * 2)
        self.batch_norm_3 = torch.nn.BatchNorm2d(128 * 4)
        self.batch_norm_4 = torch.nn.BatchNorm2d(128 * 8)

        self.conv2d_1 = torch.nn.Conv2d(1, self.d_size, 5, 2, 2)
        self.conv2d_2 = torch.nn.Conv2d(self.d_size, self.d_size * 2, 5, 2, 2)
        self.conv2d_3 = torch.nn.Conv2d(self.d_size * 2, self.d_size * 4, 5, 2, 2)
        self.conv2d_4 = torch.nn.Conv2d(self.d_size * 4, self.d_size * 8, 5, 2, 2)
        self.linear = torch.nn.Linear(16384, 1)
        self.leakyrelu = torch.nn.LeakyReLU(0.2)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, image):
        # input must be a torch.randn(64, 1, 32, 32)

        reshaped_img = torch.reshape(image, [self.batch_size, 1, self.image_size[0], self.image_size[1]])
        h0 = self.conv2d_1(reshaped_img)
        h0 = self.leakyrelu(self.batch_norm_1(h0))

        h1 = self.conv2d_2(h0)
        h1 = self.leakyrelu(self.batch_norm_2(h1))

        h2 = self.conv2d_3(h1)
        h2 = self.leakyrelu(self.batch_norm_3(h2))

        h3 = self.conv2d_4(h2)
        h3 = self.leakyrelu(self.batch_norm_4(h3))

        h3 = h3.reshape([64, -1])

        h4 = self.linear(h3)

        sigmoid = self.sigmoid(h4)

        # 64 x 1
        return sigmoid


def l2(a, b):
    return (torch.pow(a - b, 2)).mean()


def loss_discriminator(real, fake):
    # https://stackoverflow.com/questions/65458736/pytorch-equivalent-to-tf-nn-softmax-cross-entropy-with-logits-and-tf-nn-sigmoid

    # calculate sigmoid_cross_entropy_with_logits
    # not sure if this is right
    # loss = p * -torch.log(logits) + (1 - p) * -torch.log(1 - logits)

    loss = torch.square(real - torch.ones(real.shape)) + torch.square(fake)
    # low gradient

    return loss.mean()

def stat(input):
    return input.mean(dim=2), input.std(dim=2)
