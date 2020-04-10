import os
import glob

from PIL import Image
import numpy as np
from tensorflow.keras.datasets.mnist import load_data
import progressbar
from skimage.color import rgb2hsv, hsv2rgb

def plot_images(ims, num, filename):
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
    full_im[:, :, 3] = 1

    im = Image.fromarray(np.array(full_im * 255, dtype=np.uint8))
    im.save(filename)
    im.close()



def load_image_files(data_dir, reshape=None, flip=True):
    FILEPATHS = glob.glob(os.path.join(data_dir, "*.jpg"))
    FILEPATHS += glob.glob(os.path.join(data_dir, "*.png"))
    FILEPATHS += glob.glob(os.path.join(data_dir, "*.webp"))
    num = len(FILEPATHS) * (2 if flip else 1)
    images = np.zeros((num, *reshape, 3))
    counter = 0
    for image_path in progressbar.progressbar(FILEPATHS):
        im = Image.open(image_path)
        width, height = im.size
        if reshape is not None:
            im = im.resize(reshape)
        im = np.array(im) / 255.0
        if len(im.shape) == 2:  # No channels greyscale
            im = np.reshape((*im.shape, 1))
        if im.shape[2] == 1:  # One channel greyscale
            im = np.tile(im, (1, 1, 3))
        im = im * 2 - 1
        images[counter] = im
        counter += 1
        if flip:
            images[counter] = im[:, ::-1]
            counter += 1
    np.random.shuffle(images)
    return images


def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_train = x_train / 255
    x_train = x_train * 2 - 1
    x_test = x_test.reshape((-1, 28, 28, 1))
    x_test = x_test / 255
    x_test = x_test * 2 - 1
    x_train *= -1
    x_test *= -1
    return ((x_train.astype(np.float32), y_train.astype(np.float32)),
            (x_test.astype(np.float32), y_test.astype(np.float32)))


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
            upscale_times=2):
    if not os.path.exists(directory):
        os.makedirs(directory)
    start_encoded = encoder(np.array([start]))[0]
    finish_encoded = encoder(np.array([finish]))[0]
    if len(start.shape) == 2:
        start = decoder(start)
        finish = decoder(finish)
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
    plot_images([start_upscaled], 1, os.path.join(directory, f"start.png"))
    plot_images([finish_upscaled], 1, os.path.join(directory, f"finish.png"))
    for index, im in enumerate(interpolated):
        this_filename = os.path.join(directory, f"interp.{index:05d}.png")
        plot_images([im], 1, this_filename)


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
            upscale_times=2):
    starts_encoded = encoder(starts)
    finishes_encoded = encoder(finishes)
    ims = []
    for i in range(len(starts)):
        start = starts[i]
        if len(start.shape) != 3:
            start = decoder(start.reshape((1, -1)))[0]
        finish = finishes[i]
        if len(finish.shape) != 3:
            finish = decoder(finish.reshape((1, -1)))[0]
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
    plot_images(ims, steps+2, filename)


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
