from  prgan_discriminator import ShapeDiscriminator3D, l2, loss_discriminator, stat, loss_generator
from prgan_generator import ShapeGenerator3D
import matplotlib.image as mpimg
from torchvision import transforms
import glob
from torchvision.utils import save_image
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


import os
import shutil
import argparse
import numpy as np
import random
from skimage import measure
import torch
import torch.backends.cudnn as cudnn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# function to save a checkpoint during training, including the best model so far
def save_checkpoint(state, model, checkpoint_folder='checkpoints/'):
    if model == 'generator':
        filename = 'generator_best.pth.tar'
    else:
        filename = 'discriminator_best.pth.tar'
    checkpoint_file = os.path.join(checkpoint_folder, filename)
    torch.save(state, checkpoint_file)
    # shutil.copyfile(checkpoint_file, os.path.join(checkpoint_folder, 'model_best.pth.tar'))


def train(dataset_name, discriminator, generator, optimizer_Dloss, optimizer_Gloss):
    discriminator.train()  # switch to train mode
    generator.train()
    #importing files
    dataset_files = glob.glob("data/" + dataset_name  + "/*.png")
    dataset_files = np.array(dataset_files)
    n_files = dataset_files.shape[0]

    # samepling
    batch_size=64
    z_size = 201
    sample_z = np.random.uniform(-1, 1, [batch_size , z_size])
    training_step = 0

    n_iterations = 200

    best_d_error = None
    best_g_error = None

    # d_error = 9
    #trans = transforms.Compose([transforms.Resize(32)])
    for epoch in range(n_iterations):

        # randomzied the file order
        rand_idxs = np.random.permutation(range(n_files))
        #rand_idxs = np.random.permutation(128)
        # find the number of batches
        n_batches = n_files // batch_size
        #n_batches = 128 // batch_size
        # batch size = 64
        for batch_i in range(n_batches):
            print(batch_i)
            # iterate through the files

            #idxs_i = rand_idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
            idxs_i = rand_idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
            # load image batch
            imgs_batch = load_imgbatch(dataset_files[idxs_i], color=False)
            imgs_batch = torch.tensor(np.array(imgs_batch))
            # downsample the image to 32 x 32
            tData = imgs_batch

            # fake data
            fake_image = generator.forward()
            fake_image = fake_image
            # if d_error > 0.2:
            #     print("train")
            d_error, d_pred_real, d_pred_fake = train_discriminator(optimizer_Dloss, tData, fake_image, discriminator)
            # 2. Train Generator
            # Generate fake data
            fake_data = generator.forward()
            fake_image = fake_data

            # Train G
            g_error = train_generator(optimizer_Gloss, fake_image, discriminator)
            print(g_error)
            print(d_error)

            # Save best model
            if epoch == 0 and batch_i == 0:
                best_d_error = d_error
                best_g_error = g_error

            is_best_d = d_error < best_d_error
            is_best_g = g_error < best_g_error

            if is_best_d:
                best_d_error = d_error
                save_checkpoint({"state_dict": discriminator.state_dict(), "optimizer": optimizer_Dloss.state_dict()}, model="discriminator")
                print("Saved best discriminator for epoch", epoch, 'batch ', batch_i)
            if is_best_g:
                best_g_error = g_error
                save_checkpoint({"state_dict": generator.state_dict(), "optimizer": optimizer_Gloss.state_dict()}, model="generator")
                print("Saved best generator for epoch", epoch, 'batch ', batch_i)

            # Render image
            if batch_i == n_batches-1:
                # Save 2D
                save_image(fake_image, "fake" + str(epoch) + str(batch_i) + ".png")
                save_image(tData, "real" +str(epoch)+ str(batch_i) + ".png")

                # Save 3D
                filled = generator.voxels[0] < 0.2
                colors = np.array([0.4, 0.6, 0.8, 0.1])
                edgecolors = np.array([0.0, 1.0, 0.0, 0.0])
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.voxels(filled, facecolors=colors, edgecolors=edgecolors)
                fig.savefig("3d" +str(epoch)+ ".png")

def train_discriminator(optimizer, real_data, fake_data, discriminator):
    # Reset gradients
    # optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    #error_real = loss(prediction_real, torch.ones(prediction_real.shape))
    #error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    error = loss_discriminator(prediction_real,prediction_fake)
    # Calculate error and backpropagate
    #error_fake = loss(prediction_fake, torch.zeros(prediction_fake.shape))
    #error_fake.backward()

    if error > 0.2:
        print("Train discriminator")
        optimizer.zero_grad()
        error.backward(retain_graph=True)

        # 1.3 Update weights with gradients
        optimizer.step()

    # Return error and predictions for real and fake inputs
    return error, prediction_real, prediction_fake

def train_generator(optimizer, fake_image, discriminator):
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_image)
    # Calculate error and backpropagate
    # log(1-p)
    error = loss_generator(prediction)
    error.backward()
        # Update weights with gradients
    optimizer.step()
        # Return error
    return error



#
#
# # validation function
# def val(dataset, model, optimizer, args):
#     model.eval()  # switch to test mode
#     loss_count = 0.0
#     num_batch = len(dataset)
#     loss = 0
#     for i in range(num_batch):
#         data = dataset[i]  # a dict
#         xyz_tensor = data['xyz'].to(device)
#         ground_truth = data['gt_sdf'].to(device)
#         with torch.no_grad():
#             pred_sdf_tensor = model(xyz_tensor)
#             predict = torch.clamp(pred_sdf_tensor, -args.clamping_distance, args.clamping_distance)
#             truth = torch.clamp(ground_truth, -args.clamping_distance, args.clamping_distance)
#         loss = loss + (abs(predict - truth)).sum()
#         loss_count = loss_count + 1
#
#     avg_loss = loss /loss_count
#
#
#     return avg_loss
#
#
#
# # testing function
# def test(dataset, model, args):
#     model.eval()  # switch to test mode
#     num_batch = len(dataset)
#     number_samples = dataset.number_samples
#     grid_shape = dataset.grid_shape
#     IF = np.zeros((number_samples, ))
#     start_idx = 0
#     for i in range(num_batch):
#         data = dataset[i]  # a dict
#         xyz_tensor = data['xyz'].to(device)
#         this_bs = xyz_tensor.shape[0]
#         end_idx = start_idx + this_bs
#         with torch.no_grad():
#             pred_sdf_tensor = model(xyz_tensor)
#             pred_sdf_tensor = torch.clamp(pred_sdf_tensor, -args.clamping_distance, args.clamping_distance)
#         pred_sdf = pred_sdf_tensor.cpu().squeeze().numpy()
#         IF[start_idx:end_idx] = pred_sdf
#         start_idx = end_idx
#     IF = np.reshape(IF, grid_shape)
#     verts, simplices = measure.marching_cubes_classic(IF, 0)
#     x, y, z = zip(*verts)
#     colormap = ['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,191,255)']
#     fig = ff.create_trisurf(x=x,
#                             y=y,
#                             z=z,
#                             plot_edges=False,
#                             colormap=colormap,
#                             simplices=simplices,
#                             title=args.graphname)
#     plotly.offline.plot(fig, filename=args.save_file_name)
#     return
#

def main(args):
    best_loss = 2e10
    best_epoch = -1

    discriminator = ShapeDiscriminator3D()
    generator = ShapeGenerator3D()

    optimizer_Dloss = torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=0.00001, weight_decay=0.5)
    optimizer_Gloss = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=0.0025, weight_decay=0.05)

    if args.evaluate:
        print("\nEvaluation only")
        path_to_resume_generator ='checkpoints/generator_best.pth.tar'
        print("=> Loading best generator '{}'".format(path_to_resume_generator))
        generator_checkpoint = torch.load(path_to_resume_generator)
        generator.load_state_dict(generator_checkpoint['state_dict'])
        print("evaluate generator")
        # Generate 3d shape
        # 3d
        return

    if args.retrain:
        print("\nRetrain using best models")
        path_to_resume_generator = 'checkpoints/generator_best.pth.tar'
        path_to_resume_discriminator = 'checkpoints/discriminator_best.pth.tar'
        print("=> Loading best generator '{}'".format(path_to_resume_generator))
        print("=> Loading best discriminator '{}'".format(path_to_resume_discriminator))
        generator_checkpoint = torch.load(path_to_resume_generator)
        discriminator_checkpoint = torch.load(path_to_resume_discriminator)

        generator.load_state_dict(generator_checkpoint['state_dict'])
        discriminator.load_state_dict(discriminator_checkpoint['state_dict'])

        optimizer_Gloss.load_state_dict(generator_checkpoint['optimizer'])
        optimizer_Dloss.load_state_dict(discriminator_checkpoint['optimizer'])

    discriminator.to(device)
    generator.to(device)
    print("=> Will use the (" + device.type + ") device.")

    cudnn.benchmark = True

    dataset_name = "airplane64"
    train(dataset_name , discriminator, generator, optimizer_Dloss, optimizer_Gloss)

def load_imgbatch(img_paths, color=True):
    images = []
    if color:
        for path in img_paths:
            images.append(mpimg.imread(path)[:, :, 0:3])
    else:
        for path in img_paths:
            img = mpimg.imread(path)
            img = np.reshape(img, (1, img.shape[0], img.shape[1]))
            images.append(img)
    return images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PrGAN')
    parser.add_argument("-e", "--evaluate", action="store_true", help="Activate test mode - Evaluate model on val/test set (no training)")
    parser.add_argument("-r", "--retrain", action="store_true", help="Load best models and retrain")

    main(parser.parse_args())