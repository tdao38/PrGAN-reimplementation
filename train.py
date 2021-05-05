from prgan_discriminator import ShapeDiscriminator3D, loss_discriminator
from prgan_generator import ShapeGenerator3D, loss_generator
from utils import mkdir_p, isdir
import matplotlib.image as mpimg
import glob
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# function to save a checkpoint during training, including the best model so far
def save_checkpoint(state, model, checkpoint_folder='checkpoints/'):
    if model == 'generator':
        filename = 'generator_best.pth.tar'
    elif model == 'discriminator':
        filename = 'discriminator_best.pth.tar'
    else:
        filename = 'generator_' + str(model) + '.pth.tar'
    checkpoint_file = os.path.join(checkpoint_folder, filename)
    torch.save(state, checkpoint_file)

def train(dataset_name, discriminator, generator, optimizer_Dloss, optimizer_Gloss, args):
    discriminator.train()  # switch to train mode
    generator.train()
    #importing files
    dataset_files = glob.glob("data/" + dataset_name  + "/*.png")
    dataset_files = np.array(dataset_files)
    n_files = dataset_files.shape[0]

    # samepling
    batch_size=64
    z_size = 201

    n_iterations = 50

    best_d_error = None
    best_g_error = None

    for epoch in range(n_iterations):

        # randomzied the file order
        rand_idxs = np.random.permutation(range(n_files))
        # find the number of batches
        n_batches = n_files // batch_size
        #n_batches = 128 // batch_size
        # batch size = 64
        for batch_i in range(n_batches):
            print(batch_i)
            # iterate through the files
            idxs_i = rand_idxs[batch_i * batch_size: (batch_i + 1) * batch_size]
            # load image batch
            imgs_batch = load_imgbatch(dataset_files[idxs_i], color=False)
            imgs_batch = torch.tensor(np.array(imgs_batch))

            tData = imgs_batch

            # fake data
            fake_image = generator.forward()
            fake_image = fake_image

            d_error, d_pred_real, d_pred_fake = train_discriminator(optimizer_Dloss, tData, fake_image, discriminator)
            # 2. Train Generator
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
                save_image(fake_image, os.path.join(args.results , "fake" + str(epoch) + str(batch_i) + ".png"))
                save_image(tData,  os.path.join(args.results , "real" +str(epoch)+ str(batch_i) + ".png"))
                # Save 3D
                # filled = generator.voxels[0] < 0.2
                # colors = np.array([0.4, 0.6, 0.8, 0.1])
                # edgecolors = np.array([0.0, 1.0, 0.0, 0.0])
                # fig = plt.figure()
                # ax = fig.gca(projection='3d')
                # ax.voxels(filled, facecolors=colors, edgecolors=edgecolors)
                # fig.savefig("3d" +str(epoch)+ ".png")

        save_checkpoint({"state_dict": generator.state_dict(), "optimizer": optimizer_Gloss.state_dict()},
                                model=epoch)

def train_discriminator(optimizer, real_data, fake_data, discriminator):

    prediction_real = discriminator(real_data)

    prediction_fake = discriminator(fake_data)
    error = loss_discriminator(prediction_real,prediction_fake)

    if error > 0.2:
        print("Train discriminator")
        optimizer.zero_grad()
        error.backward(retain_graph=True)
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

def generate_3d(generator, args):
    fake_image = generator.forward()
    save_image(fake_image, os.path.join(args.results ,"fake.png"))
    # Save 3D
    filled = generator.voxels[0] < 0.2
    colors = np.array([0.4, 0.6, 0.8, 0.1])
    edgecolors = np.array([0.0, 1.0, 0.0, 0.0])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.voxels(filled, facecolors=colors, edgecolors=edgecolors)
    fig.savefig(os.path.join(args.results ,"3d_object.png"))


def main(args):
    # create checkpoint folder
    if not isdir(args.checkpoint_folder):
        print("Creating new checkpoint folder " + args.checkpoint_folder)
        mkdir_p(args.checkpoint_folder)

    # create checkpoint folder
    if not isdir(args.results):
        print("Creating new checkpoint folder " + args.results)
        mkdir_p(args.results)

    discriminator = ShapeDiscriminator3D()
    generator = ShapeGenerator3D()

    optimizer_Dloss = torch.optim.Adam(filter(lambda p: p.requires_grad, discriminator.parameters()), lr=0.00001, weight_decay=0.5)
    optimizer_Gloss = torch.optim.Adam(filter(lambda p: p.requires_grad, generator.parameters()), lr=0.005, weight_decay=0.05)

    if args.evaluate:
        print("\nEvaluation only")
        path_to_resume_generator ='checkpoints/generator_best.pth.tar'
        print("=> Loading best generator '{}'".format(path_to_resume_generator))
        generator_checkpoint = torch.load(path_to_resume_generator)
        generator.load_state_dict(generator_checkpoint['state_dict'])
        print("evaluate generator")
        # Generate 3d shape
        generate_3d(generator, args)
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
    train(dataset_name , discriminator, generator, optimizer_Dloss, optimizer_Gloss, args)

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
    parser.add_argument("--checkpoint_folder", default="checkpoints/", type=str, help="Folder to save checkpoints")
    parser.add_argument("--results", default="results/", type=str, help="Folder to save trained images")


    main(parser.parse_args())