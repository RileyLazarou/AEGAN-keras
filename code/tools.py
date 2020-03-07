import os

from PIL import Image
import numpy as np
from tensorflow.keras.datasets.mnist import load_data

def plot_images(ims, num, filename, inverse=False):
    """
    Given a list of (equally-shaped) images
    Save them as `filename` in a `num`-by-`num` square
    """
    if np.min(ims) < 0:
        ims = (np.array(ims) + 1) / 2
    if len(ims) < num ** 2:
        indices = np.arange(len(ims))
    else:
        indices = np.arange(num ** 2)
    image_height, image_width = ims.shape[1], ims.shape[2]

    full_im = np.zeros((num * image_height, num * image_width, 4))
    for index, ims_idx in enumerate(indices):
        column = index % num
        row = index // num
        this_image = ims[ims_idx]
        if len(this_image.shape) == 2:  # No channels greyscale
            this_image = np.reshape((*this_image.shape, 1))
        if this_image.shape[2] == 1:  # One channel greyscale
            this_image = np.tile(this_image, (1, 1, 3))
        if this_image.shape[2] == 3:
            new_image = np.ones((this_image.shape[0], this_image.shape[1], 4))
            new_image[:, :, :3] = this_image
            this_image = new_image
        full_im[
            row * image_height : (row + 1) * image_height,
            column * image_width : (column + 1) * image_width,
        ] = this_image
    if inverse:
        full_im = 1 - full_im
    full_im[:, :, 3] = 1

    im = Image.fromarray(np.array(full_im * 255, dtype=np.uint8))
    im.save(filename)


def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_train = x_train / 255
    x_train = x_train * 2 - 1
    x_test = x_test.reshape((-1, 28, 28, 1))
    x_test = x_test / 255
    x_test = x_test * 2 - 1
    return (x_train, y_train), (x_test, y_test)


def plot_reconstruction(reconstruction_function, images, num, filename, inverse=False):
    reconstructions = reconstruction_function(images)
    interwoven = [y for x in zip(images, reconstructions) for y in x]
    plot_images(np.array(interwoven),
                num,
                filename,
                inverse=inverse)


def plot_generated(generator_function, latent_vectors, num, filename, inverse=False):
    images = generator_function(latent_vectors)
    plot_images(np.array(images),
                num,
                filename,
                inverse=inverse)


def upscale(im):
    new_im = np.zeros((2 * im.shape[0], 2 * im.shape[1], *im.shape[2:]))
    new_im[::2, ::2] = im
    new_im[1::2, ::2] = im
    new_im[::2, 1::2] = im
    new_im[1::2, 1::2] = im
    return new_im


def plot_interpolation_gif(start,
            finish,
            directory,
            encoder=lambda x: x,
            decoder=lambda x: x,
            frames=120,
            interpolation='linear',
            inverse=False,
            upscale_times=2):
    if not os.path.exists(directory):
        os.makedirs(directory)
    start_encoded = encoder(np.array([start]))[0]
    finish_encoded = encoder(np.array([finish]))[0]
    if interpolation == 'linear':
        space = np.linspace(start_encoded, finish_encoded, frames)
    elif interpolation == 'sigmoid':
        proportion_start = np.linspace(np.zeros(start_encoded.shape),
                                       np.ones(start_encoded.shape) * np.pi,
                                       frames)
        proportion_start = (np.cos(proportion_start) + 1) / 2
        space = (proportion_start * start_encoded
                 + (1-proportion_start) * finish_encoded)
    interpolated = decoder(space)
    start_upscaled = start
    finish_upscaled = finish
    for i in range(upscale_times):
        interpolated = np.array([upscale(x) for x in interpolated])
        start_upscaled = upscale(start_upscaled)
        finish_upscaled = upscale(finish_upscaled)
    plot_images([start_upscaled], 1, os.path.join(directory, f"start.png"), inverse=inverse)
    plot_images([finish_upscaled], 1, os.path.join(directory, f"finish.png"), inverse=inverse)
    for index, im in enumerate(interpolated):
        this_filename = os.path.join(directory, f"interp.{index:05d}.png")
        plot_images([im], 1, this_filename, inverse=inverse)


    os.system(
        f"ffmpeg -y -r 20 -i {os.path.join(directory, 'interp.%05d.png')}"
        f" -crf 15 {os.path.join(directory, 'interp.mp4')}"
    )

def plot_interpolation_grid(starts,
            finishes,
            filename,
            encoder=lambda x: x,
            decoder=lambda x: x,
            steps=7,
            interpolation='linear',
            inverse=False,
            upscale_times=2):
    starts_encoded = encoder(starts)
    finishes_encoded = encoder(finishes)
    ims = []
    for i in range(len(starts)):
        start = starts[i]
        finish = finishes[i]
        start_encoded = starts_encoded[i]
        finish_encoded = finishes_encoded[i]
        space = np.linspace(start_encoded, finish_encoded, steps)
        images = decoder(space)
        for __ in range(upscale_times):
            start = upscale(start)
            finish = upscale(finish)
        ims.append(start)
        for image in images:
            for __ in range(upscale_times):
                image = upscale(image)
            ims.append(image)
        ims.append(finish)
    plot_images(ims, steps+2, filename, inverse=inverse)


def plot_sampling(noise_function,
            generator,
            filename,
            num=10,
            upscale_times=2,
            inverse=False):
    images = generator(noise_function(num*num))
    for i in range(upscale_times):
        images = [upscale(x) for x in images]
    images = np.array(images)
    plot_images(images, num, filename, inverse=inverse)


def build_directories(base_dir, experiment):
    dirs = {}
    dirs["results"] = base_dir
    dirs["experiment"] = os.path.join(base_dir, experiment)
    dirs["training"] = os.path.join(dirs["experiment"], 'training')
    dirs["generated"] = os.path.join(dirs["training"], 'generated')
    dirs["generated_random"] = os.path.join(dirs["generated"], 'random')
    dirs["generated_fixed"] = os.path.join(dirs["generated"], 'fixed')
    dirs["reconstructed"] = os.path.join(dirs["training"], 'reconstructed')
    dirs["reconstructed_random"] = os.path.join(dirs["reconstructed"], 'random')
    dirs["reconstructed_fixed"] = os.path.join(dirs["reconstructed"], 'fixed')
    dirs["output"] = os.path.join(dirs["experiment"], 'output')
    dirs["output_models"] = os.path.join(dirs["output"], 'models')
    dirs["output_interpolations"] = os.path.join(dirs["output"], 'interpolations')

    for __, i in dirs.items():
        if not os.path.exists(i):
            os.makedirs(i)
    return dirs
