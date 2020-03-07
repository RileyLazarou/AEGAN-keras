import os
import pickle
import argparse

import  numpy as np
from tensorflow.keras.optimizers import Adam

import models
import tools
import generative_model as gm

EXPERIMENT = 'MNIST_AE'
DIRECTORY = os.path.dirname(os.path.realpath(__file__))
PARAMTERS = os.path.join(DIRECTORY, "params_mnist.json")
(x_train, y_train), (x_test, y_test) = tools.get_mnist_data()
IMAGE_DIM = (28, 28, 1)

def get_samples(count, test=False):
    """Sample from the dataset."""
    if test:
        return x_test[np.random.randint(0, len(x_test), count)]
    else:
        return x_train[np.random.randint(0, len(x_train), count)]


def get_noise(count, latent_dim):
    """Sample from the noise distribution."""
    return np.random.normal(0, 1, (count, latent_dim))


def set_up_directories(experiment_name):
    results_dir = os.path.join(DIRECTORY, "..", "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    dirs = tools.build_directories(results_dir, experiment_name)
    return dirs


def set_up_test_data(results_dir):
    test_samples_file = os.path.join(results_dir, 'test_samples.pkl')
    test_noise_file = os.path.join(results_dir, 'test_noise.pkl')
    if os.path.exists(test_samples_file):
        with open(test_samples_file, 'rb') as f:
            test_samples = pickle.load(f)
    else:
        test_samples = x_test[np.random.randint(0, len(x_test), 400)]
        with open(test_samples_file, 'wb') as f:
            pickle.dump(test_samples, f)
    if os.path.exists(test_noise_file):
        with open(test_noise_file, 'rb') as f:
            test_noise = pickle.load(f)
    else:
        test_noise = get_noise(400)
        with open(test_noise_file, 'wb') as f:
            pickle.dump(test_noise, f)
    return test_samples, test_noise


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str, required=True)
    parser.add_argument('-n', '--name', type=str, required=True)
    parser.add_argument('-p', '--plot_every', type=int, default=1)
    parser.add_argument('-l', '--latent_dim', type=int, default=16)
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=1)
    return parser.parse_args()

def train_autoencoder(plot_every, latent_dim, batch_size, epochs, test_samples):
    print("Training Autoencoder...")
    model = gm.AutoEncoder(
            (28, 28, 1),
            latent_dim,
            PARAMTERS,
            get_samples,
            lambda x: get_noise(x, latent_dim),
            )
    BATCHES_PER_EPOCH = len(x_train) // batch_size
    count = 0
    for epoch in range(epochs):
        model.train(batch_size=batch_size,
                    batch_num=BATCHES_PER_EPOCH,
                    prepend=f'GAN Epoch {epoch+1} ')
        if plot_every and (epoch + 1) % plot_every == 0:
            filename = os.path.join(dirs["reconstructed_fixed"],
                                    f"r.{count:05d}.png")
            tools.plot_reconstruction(model.autoencode_images,
                    test_samples[:100],
                    10,
                    filename,
                    inverse=True)
            count += 1
    model.generator.save(os.path.join(dirs['output_models'], 'generator.h5'))
    model.encoder.save(os.path.join(dirs['output_models'], 'encoder.h5'))


def train_gan(plot_every, latent_dim, batch_size, epochs, test_samples):
    print("Training Autoencoder...")
    model = gm.GenerativeAdversarialNetwork(
        (28, 28, 1),
        latent_dim,
        PARAMTERS,
        get_samples,
        lambda x: get_noise(x, latent_dim),
        )
    BATCHES_PER_EPOCH = len(x_train) // batch_size
    count = 0
    for epoch in range(epochs):
        model.train(batch_size=batch_size,
                    batch_num=BATCHES_PER_EPOCH,
                    prepend=f'GAN Epoch {epoch+1} ')
        if plot_every and (epoch + 1) % plot_every == 0:
            filename = os.path.join(dirs["generated_random"],
                                    f"r.{count:05d}.png")
            ims = model.generate(num=100)
            tools.plot_images(ims, 10, filename, inverse=True)
            filename = os.path.join(dirs["generated_fixed"],
                                    f"r.{count:05d}.png")
            ims = model.decode(test_noise[:100])
            tools.plot_images(ims, 10, filename, inverse=True)
            count += 1
    model.generator.save(os.path.join(dirs['output_models'], 'generator.h5'))
    model.discriminator_image.save(os.path.join(dirs['output_models'],
                                                'discriminator_image.h5'))


if __name__ == '__main__':
    args = get_args()
    dirs = set_up_directories(args.name)
    test_samples, test_noise = set_up_test_data(dirs["results"])
    if args.type.upper() == "AE":
        train_autoencoder(args.plot_every,
                          args.latent_dim,
                          args.batch_size,
                          args.epochs,
                          test_samples)
    if args.type.upper() == "GAN":
        train_gan(args.plot_every,
                          args.latent_dim,
                          args.batch_size,
                          args.epochs,
                          test_samples)


#train_autoencoder()
