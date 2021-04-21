from  prgan_discriminator import ShapeDiscriminator3D, l2
import matplotlib.image as mpimg
from torchvision import transforms
import glob


import os
import shutil
import argparse
import numpy as np
import random
import plotly
import plotly.figure_factory as ff
from skimage import measure
import torch
import torch.backends.cudnn as cudnn
from model import Decoder
from utils import normalize_pts, normalize_normals, SdfDataset, mkdir_p, isdir


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# function to save a checkpoint during training, including the best model so far
def save_checkpoint(state, is_best, checkpoint_folder='checkpoints/', filename='checkpoint.pth.tar'):
    checkpoint_file = os.path.join(checkpoint_folder, 'checkpoint_{}.pth.tar'.format(state['epoch']))
    torch.save(state, checkpoint_file)
    if is_best:
        shutil.copyfile(checkpoint_file, os.path.join(checkpoint_folder, 'model_best.pth.tar'))


def train(dataset_name, discriminator, generator, optimizer_Dloss, optimizer_Gloss, optimizer_Gloss_classic, args):
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

    n_iterations = 50
    trans = transforms.Compose([transforms.Resize(32)])
    for epoch in range(n_iterations):

        # randomzied the file order
        rand_idxs = np.random.permutation(range(n_files))
        # find the number of batches
        n_batches = n_files // batch_size
        # batch size = 64
        for batch_i in range(n_batches):
            # iterate through the files
            idxs_i = rand_idxs[batch_i * batch_size: (batch_i + 1) * batch_size]

            # load image batch
            imgs_batch = load_imgbatch(dataset_files[idxs_i], color=False)
            imgs_batch = torch.tensor(np.array(imgs_batch))
            # downsample the image to 32 x 32
            tData = trans(imgs_batch)

            # batch_z = np.random.uniform(-1, 1, [batch_size, z_size])
            # 64 x 201
            # pass batch_z to generator
            fake_image = torch.randn(tData.shape)

            optimizer_Dloss.zero_grad()
            optimizer_Gloss.zero_grad()
            optimizer_Gloss_classic.zero_grad()

            D_real, D_real_logits, D_stats_real = discriminator(tData, False)
            D_fake, D_fake_logits, D_stats_fake = discriminator(fake_image, True)

            D_loss_real = discriminator.loss(D_real_logits, torch.ones(D_real.shape))
            D_loss_fake = discriminator.loss(D_fake_logits, torch.zeros(D_fake.shape))
            G_loss_classic = discriminator.loss(D_fake_logits, torch.ones(D_fake.shape))

            # discriminator optmier will stop oprtimizing after the dacc is greater than 0.8

# save check point
            dr_mean, dr_var = discriminator.stat(D_stats_real)
            dl_mean, dl_var = discriminator.stat(D_stats_fake)

            G_loss = l2(dr_mean, dl_mean)
            G_loss += l2(dr_var, dl_var)
            D_loss = D_loss_real + D_loss_fake
            D_loss.backward()
            G_loss.backward()
            G_loss_classic.backward()
            optimizer_Dloss.step()
            optimizer_Gloss.step()
            optimizer_Gloss_classic.step()

            # after 50 batch, we render/ save real image, fake image , and voxels
            # ops.save_images
            # ops.save_voxels





# validation function
def val(dataset, model, optimizer, args):
    model.eval()  # switch to test mode
    loss_count = 0.0
    num_batch = len(dataset)
    loss = 0
    for i in range(num_batch):
        data = dataset[i]  # a dict
        xyz_tensor = data['xyz'].to(device)
        ground_truth = data['gt_sdf'].to(device)
        with torch.no_grad():
            pred_sdf_tensor = model(xyz_tensor)
            predict = torch.clamp(pred_sdf_tensor, -args.clamping_distance, args.clamping_distance)
            truth = torch.clamp(ground_truth, -args.clamping_distance, args.clamping_distance)
        loss = loss + (abs(predict - truth)).sum()
        loss_count = loss_count + 1

    avg_loss = loss /loss_count


    return avg_loss



# testing function
def test(dataset, model, args):
    model.eval()  # switch to test mode
    num_batch = len(dataset)
    number_samples = dataset.number_samples
    grid_shape = dataset.grid_shape
    IF = np.zeros((number_samples, ))
    start_idx = 0
    for i in range(num_batch):
        data = dataset[i]  # a dict
        xyz_tensor = data['xyz'].to(device)
        this_bs = xyz_tensor.shape[0]
        end_idx = start_idx + this_bs
        with torch.no_grad():
            pred_sdf_tensor = model(xyz_tensor)
            pred_sdf_tensor = torch.clamp(pred_sdf_tensor, -args.clamping_distance, args.clamping_distance)
        pred_sdf = pred_sdf_tensor.cpu().squeeze().numpy()
        IF[start_idx:end_idx] = pred_sdf
        start_idx = end_idx
    IF = np.reshape(IF, grid_shape)
    verts, simplices = measure.marching_cubes_classic(IF, 0)
    x, y, z = zip(*verts)
    colormap = ['rgb(255,105,180)', 'rgb(255,255,51)', 'rgb(0,191,255)']
    fig = ff.create_trisurf(x=x,
                            y=y,
                            z=z,
                            plot_edges=False,
                            colormap=colormap,
                            simplices=simplices,
                            title=args.graphname)
    plotly.offline.plot(fig, filename=args.save_file_name)
    return


def main(args):
    best_loss = 2e10
    best_epoch = -1

    # create checkpoint folder
    if not isdir(args.checkpoint_folder):
        print("Creating new checkpoint folder " + args.checkpoint_folder)
        mkdir_p(args.checkpoint_folder)

    discrimnator = ShapeDiscriminator3D()
    generator = ShapeGenerator3D()

    discrimnator.to(device)
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

    optimizer_Dloss = torch.optim.Adam(filter(lambda p: p.requires_grad, discrimnator.parameters()), lr=1e-4, weight_decay=0.5)
    optimizer_Gloss = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=0.0025, weight_decay=0.5)
    optimizer_Gloss_classic = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=0.0025, weight_decay=0.5)

    # perform training!
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer_Dloss,optimizer_Gloss, optimizer_Gloss_classic, args.schedule, gamma=args.gamma)

    dataset_name = "airplane64"
    train(dataset_name , discrimnator, generator, optimizer_Dloss, optimizer_Gloss, optimizer_Gloss_classic , args)
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
    parser = argparse.ArgumentParser(description='DeepSDF')

    parser.add_argument("-e", "--evaluate", action="store_true", help="Activate test mode - Evaluate model on val/test set (no training)")

    # paths you may want to adjust, but it's better to keep the defaults
    parser.add_argument("--input_pts", default="data/sphere.pts", type=str, help="Folder to save checkpoints")
    parser.add_argument("--save_file_name", default="sphere.html", type=str, help="Folder to save testing visualization")
    parser.add_argument("--checkpoint_folder", default="checkpoints/", type=str, help="Folder to save checkpoints")
    parser.add_argument("--resume_file", default="model_best.pth.tar", type=str, help="Path to retrieve latest checkpoint file relative to checkpoint folder")

    # hyperameters of network/options for training
    parser.add_argument("--weight_decay", default=1e-2, type=float, help="Weight decay/L2 regularization on weights")
    parser.add_argument("--lr", default=1e-4, type=float, help="Initial learning rate")
    parser.add_argument("--schedule", type=int, nargs="+", default=[40, 50], help="Decrease learning rate at these milestone epochs.")
    parser.add_argument("--gamma", default=0.1, type=float, help="Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestone epochs")
    parser.add_argument("--start_epoch", default=0, type=int, help="Start from specified epoch number")
    parser.add_argument("--epochs", default=80, type=int, help="Number of epochs to train (when loading a previous model, it will train for an extra number of epochs)")
    parser.add_argument("--train_batch", default=1024, type=int, help="Batch size for training")
    parser.add_argument("--train_split_ratio", default=0.8, type=float, help="ratio of training split")
    parser.add_argument("--N_samples", default=80.0, type=float, help="for each input point, N samples are used for training or validation")
    parser.add_argument("--sample_variance", default=0.0025, type=float, help="we perturb each surface point along normal direction with mean-zero Gaussian noise with the given variance")
    parser.add_argument("--clamping_distance", default=0.1, type=float, help="clamping distance for sdf")

    # various options for testing and evaluation
    parser.add_argument("--test_batch", default=2048, type=int, help="Batch size for testing")
    parser.add_argument("--grid_N", default=128, type=int, help="construct a 3D NxNxN grid containing the point cloud")
    parser.add_argument("--max_xyz", default=1.0, type=float, help="largest xyz coordinates")
    parser.add_argument('--graphname', type=str, default="Isosurface",
                        help='graphname', required=False)

    print(parser.parse_args())
    main(parser.parse_args())