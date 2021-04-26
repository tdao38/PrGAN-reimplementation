from  prgan_discriminator import ShapeDiscriminator3D, l2, loss, stat
from prgan_generator import ShapeGenerator3D
import matplotlib.image as mpimg
from torchvision import transforms
import glob
from torchvision.utils import save_image
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
def save_checkpoint(state, is_best, checkpoint_folder='checkpoints/', filename='checkpoint.pth.tar'):
    checkpoint_file = os.path.join(checkpoint_folder, 'checkpoint_{}.pth.tar'.format(state['epoch']))
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, os.path.join(checkpoint_folder, 'model_best.pth.tar'))


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
    trans = transforms.Compose([transforms.Resize(32)])
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
            tData = trans(imgs_batch)

            # fake data
            fake_image = generator.forward().detach()
            fake_image = trans(fake_image)

            d_error, d_pred_real, d_pred_fake = train_discriminator(optimizer_Dloss, tData, fake_image, discriminator)
            # 2. Train Generator
            # Generate fake data
            fake_data = generator.forward()
            fake_image = trans(fake_data)

            # Train G
            g_error = train_generator(optimizer_Gloss, fake_image, discriminator)
            print(g_error)
            print(d_error)

            #
            # optimizer_Dloss.zero_grad()
            # optimizer_Gloss.zero_grad()
            # optimizer_Gloss_classic.zero_grad()
            #
            # D_real, D_real_logits, D_stats_real = discriminator(tData)
            # D_fake, D_fake_logits, D_stats_fake = discriminator(fake_image)
            #
            # D_loss_real = loss(D_real, torch.ones(D_real.shape))
            # D_loss_fake = loss(D_fake, torch.zeros(D_fake.shape))
            # G_loss_classic = loss(D_fake, torch.ones(D_fake.shape))
            #
            # # discriminator optmier will stop oprtimizing after the dacc is greater than 0.8
            #
            # # save check point
            # dr_mean, dr_var = stat(D_stats_real)
            # dl_mean, dl_var = stat(D_stats_fake)
            # margin = 0.7
            # train_discriminator = True
            # dacc_real = torch.mean(D_real)
            # dacc_fake = torch.mean(torch.ones(D_fake.shape) - D_fake)
            # dacc = (dacc_real + dacc_fake) * 0.5
            # G_loss = l2(dr_mean, dl_mean)
            # G_loss += l2(dr_var, dl_var)
            # D_loss = D_loss_real + D_loss_fake
            # if dacc > margin:
            #     train_discriminator = False
            # if train_discriminator:
            #     D_loss.backward(retain_graph=True)
            # print(G_loss)
            # print(D_loss)
            # print(G_loss_classic)
            # print(dacc)
            # G_loss.backward(retain_graph=True)
            # G_loss_classic.backward()
            # if train_discriminator:
            #     optimizer_Dloss.step()
            # optimizer_Gloss.step()
            # optimizer_Gloss_classic.step()
            save_image(fake_image, "fake" + str(batch_i) + ".png")
            save_image(tData, "real" + str(batch_i) + ".png")

        filled = generator.voxels[0]> 0.9
        N = 64
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.voxels(filled, edgecolors='k')
        fig.savefig("3d" +str(epoch)+ ".png")

def train_discriminator(optimizer, real_data, fake_data, discriminator):
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, torch.ones(prediction_real.shape))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, torch.zeros(prediction_fake.shape))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

def train_generator(optimizer, fake_data, discriminator):
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, torch.ones(prediction.shape))
    error.backward()
        # Update weights with gradients
    optimizer.step()
        # Return error
    return error




        #save_image(tData, "real" + str(epoch) + ".png")
        #save_image(fake_image, "fake" + str(epoch) + ".png")

            # every 50Th save the image
            # after 50 batch, we render/ save real image, fake image , and voxels
            # ops.save_images
            # ops.save_voxels



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

def main():
    best_loss = 2e10
    best_epoch = -1

    # create checkpoint folder
    # if not isdir(args.checkpoint_folder):
    #     print("Creating new checkpoint folder " + args.checkpoint_folder)
    #     mkdir_p(args.checkpoint_folder)

    discriminator = ShapeDiscriminator3D()
    generator = ShapeGenerator3D()

    discriminator.to(device)
    generator.to(device)
    print("=> Will use the (" + device.type + ") device.")

    # cudnn will optimize execution for our network
    cudnn.benchmark = True

    # if args.evaluate:
    #     print("\nEvaluation only")
    #     path_to_resume_file = os.path.join(args.checkpoint_folder, args.resume_file)
    #     print("=> Loading training checkpoint '{}'".format(path_to_resume_file))
    #     checkpoint = torch.load(path_to_resume_file)
    #     model.load_state_dict(checkpoint['state_dict'])
    #     test_dataset = SdfDataset(phase='test', args=args)
    #     test(test_dataset, model, args)
    #     return

    optimizer_Dloss = torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=1e-4, weight_decay=0.5)
    optimizer_Gloss = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=0.00025, weight_decay=0.5)
    #optimizer_Gloss_classic = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=0.00025, weight_decay=0.5)

    # perform training!
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_Dloss,optimizer_Gloss, optimizer_Gloss_classic, args.schedule, gamma=args.gamma)

    dataset_name = "airplane64"
    train(dataset_name , discriminator, generator, optimizer_Dloss, optimizer_Gloss)
        # val_loss = val(val_dataset, model, optimizer, args)
        # scheduler.step()
        # is_best = val_loss < best_loss
        # if is_best:
        #     best_loss = val_loss
        #     best_epoch = epoch
        # save_checkpoint({"epoch": epoch + 1, "state_dict": model.state_dict(), "best_loss": best_loss, "optimizer": optimizer.state_dict()},
        #                 is_best, checkpoint_folder=args.checkpoint_folder)
        # print(f"Epoch{epoch+1:d}. train_loss: {train_loss:.8f}. val_loss: {val_loss:.8f}. Best Epoch: {best_epoch+1:d}. Best val loss: {best_loss:.8f}.")


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

    main()