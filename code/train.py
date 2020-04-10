import os
import pickle
import argparse
import shutil
import json

import  numpy as np

import models
import tools
import generative_model as gm

DIRECTORY = os.path.dirname(os.path.realpath(__file__))

MODEL_TYPES = {"AE": gm.AutoEncoder,
               "GAN": gm.GenerativeAdversarialNetwork,
               "AAE": gm.AdversarialAutoEncoder,
               "EGAN": gm.EncodingGenerativeAdversarialNetwork,
               "AEGAN": gm.AutoEncodingGenerativeAdversarialNetwork,
               }

def get_samples(count, x_train, x_test, test=False):
    """Sample from the dataset."""
    if test:
        return x_test[np.random.randint(0, len(x_test), count)]
    else:
        return x_train[np.random.randint(0, len(x_train), count)]


def get_noise(count, latent_dim):
    """Sample from the noise distribution."""
    return np.random.normal(0, 1, (count, latent_dim)).astype(np.float32)


def set_up_directories(experiment_name):
    """Set up directories for the experiment and return their paths."""
    results_dir = os.path.join(DIRECTORY, "..", "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    dirs = tools.build_directories(results_dir, experiment_name)
    return dirs


def set_up_test_data(results_dir, name, latent_dim, x_test):
    """Load or create and save test data."""
    test_images_file = os.path.join(results_dir, f'{name}_test_images.pkl')
    test_noise_file = os.path.join(results_dir, f'{name}_test_noise.pkl')
    if os.path.exists(test_images_file):
        with open(test_images_file, 'rb') as f:
            test_images = pickle.load(f)
    else:
        test_images = x_test[np.random.randint(0, len(x_test), 400)]
        with open(test_images_file, 'wb') as f:
            pickle.dump(test_images, f)
    if os.path.exists(test_noise_file):
        with open(test_noise_file, 'rb') as f:
            test_noise = pickle.load(f)
    else:
        test_noise = get_noise(400, latent_dim)
        with open(test_noise_file, 'wb') as f:
            pickle.dump(test_noise, f)
    return test_images, test_noise


def plot_epoch(model, dirs, epoch, test_noise=None, test_images=None):
    if "autoencode_images" in dir(model):
        filename = os.path.join(dirs["reconstructed_random"], f"r.{epoch:05d}.png")
        images = model.data_generating_function(50)
        reconstructions = model.autoencode_images(images)
        interwoven = [y for x in zip(images, reconstructions) for y in x]
        tools.plot_images(np.array(interwoven), 10, filename)
        if test_images is not None:
            filename = os.path.join(dirs["reconstructed_fixed"], f"r.{epoch:05d}.png")
            reconstructions = model.autoencode_images(test_images)
            interwoven = [y for x in zip(test_images, reconstructions) for y in x]
            tools.plot_images(np.array(interwoven), 10, filename)
    if "decode" in dir(model):
        filename = os.path.join(dirs["generated_random"], f"r.{epoch:05d}.png")
        ims = model.generate(num=100)
        tools.plot_images(ims, 10, filename)
        if test_noise is not None:
            filename = os.path.join(dirs["generated_fixed"], f"r.{epoch:05d}.png")
            ims = model.decode(test_noise[:100])
            tools.plot_images(ims, 10, filename)

def save_models(model, dirs, epoch):
    if "encoder" in dir(model):
        model.encoder.save(os.path.join(dirs['output_models'], f'g{epoch}.encoder.h5'))
    if "generator" in dir(model):
        model.generator.save(os.path.join(dirs['output_models'], f'g{epoch}.generator.h5'))
    if "discriminator_latent" in dir(model):
        model.discriminator_latent.save(os.path.join(dirs['output_models'], f'g{epoch}.discriminator_latent.h5'))
    if "discriminator_image" in dir(model):
        model.discriminator_image.save(os.path.join(dirs['output_models'], f'g{epoch}.discriminator_image.h5'))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str, required=True)
    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('-p', '--plot_every', type=int, default=1)
    parser.add_argument('-l', '--latent_dim', type=int, default=16)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-d', '--data_dir', type=str, default=None)
    parser.add_argument('-e', '--epochs', type=int, default=1)
    parser.add_argument('-s', '--steps_per_epoch', type=int, default=None)
    parser.add_argument('-f', '--parameter_file', type=str, default=None)
    parser.add_argument('-x', '--flip', type=bool, default=False)
    return parser.parse_args()


def train_model(model_type,
                dirs,
                parameter_file,
                plot_every,
                latent_dim,
                batch_size,
                epochs,
                steps_per_epoch,
                test_images,
                test_noise,
                x_train,
                x_test,
                ):
    model = model_type(
            x_train.shape[1:],
            latent_dim,
            parameter_file,
            lambda x: get_samples(x, x_train, x_test),
            lambda x: get_noise(x, latent_dim),
            )
    if not steps_per_epoch:
        steps_per_epoch = len(x_train) // batch_size
    for epoch in range(epochs):
        model.train(batch_size=batch_size,
                    batch_num=steps_per_epoch,
                    prepend=f'Epoch {epoch+1} ')
        if plot_every and (epoch + 1) % plot_every == 0:
            plot_epoch(model, dirs, epoch, test_images=test_images, test_noise=test_noise)
        if (epoch+1) % 100 == 0:
            save_models(model, dirs, epoch+1)
        save_models(model, dirs, 0)
    save_models(model, dirs, 'current')
    if "encode" in dir(model):
        tools.plot_interpolation_grid(test_images[:9],
                    test_images[1:10],
                    os.path.join(dirs['output_interpolations'], 'real.png'),
                    encoder=model.encode,
                    decoder=model.decode,
                    steps=7,
                    interpolation='linear',
                    upscale_times=0,)
    tools.plot_interpolation_grid(model.noise_generating_function(9),
                model.noise_generating_function(9),
                os.path.join(dirs['output_interpolations'], 'fake.png'),
                decoder=model.decode,
                steps=7,
                interpolation='linear',
                upscale_times=0,)



if __name__ == '__main__':
    args = get_args()
    with open(args.parameter_file, 'r') as f:
        params = json.load(f)
    shape = (np.array(params['generator']['starting_shape'][:2])
             * np.prod(params['generator']['upsampling']))
    if args.data_dir is None:
        (x_train, __), (x_test, __) = tools.get_mnist_data()
        del __
    else:
        data = tools.load_image_files(args.data_dir, reshape=shape, flip=args.flip)
        x_train = data[:int(0.9*len(data))]
        x_test = data[int(0.9*len(data)):]
        del data
    dirs = set_up_directories(f'{args.name}_{args.type}')
    shutil.copytree(DIRECTORY, os.path.join(dirs['experiment'], 'code'))
    test_images, test_noise = set_up_test_data(dirs["results"],
            args.name,
            args.latent_dim,
            x_test)
    train_model(MODEL_TYPES[args.type.upper()],
                dirs,
                args.parameter_file,
                args.plot_every,
                args.latent_dim,
                args.batch_size,
                args.epochs,
                args.steps_per_epoch,
                test_images,
                test_noise,
                x_train,
                x_test)
