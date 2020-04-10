from abc import ABC, abstractmethod
import json
import gc
import sys
import time

import numpy as np
from tensorflow.keras.layers import (Input, Dense, Flatten, Concatenate,
                                     Reshape, UpSampling2D,
                                     Conv2D, Activation, Dropout, LeakyReLU,
                                     GaussianNoise, GaussianDropout, 
                                     LayerNormalization)
from tensorflow.compat.v1.keras.layers import BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

tf.keras.backend.set_floatx('float32')
CLIPNORM = None

class GenerativeModel(ABC):

    def __init__(self,
                 image_shape,
                 latent_dim,
                 parameter_json_path,
                 data_generating_function,
                 noise_generating_function):
        with open(parameter_json_path, "r") as f:
            self.parameters = json.load(f)
        self.image_shape = image_shape
        self.latent_dim = latent_dim
        self.data_generating_function = data_generating_function
        self.noise_generating_function = noise_generating_function

        self.generator = self._build_generator(
                latent_dim,
                **self.parameters["generator"]
                )


    def decode(self, latent_vectors):
        """Decode a set of latent vectors as images."""
        return self.generator.predict(latent_vectors)


    def generate(self, num):
        """Generate `num` random samples."""
        return self.decode(self.noise_generating_function(num))


    def _build_feature_extractor(self,
                                 image_shape,
                                 output_width,
                                 channels=None,
                                 kernel_widths=None,
                                 strides=None,
                                 hidden_activation='relu',
                                 output_activation='linear',
                                 init=RandomNormal(mean=0, stddev=0.02),
                                 add_noise=False):
        """Build a model that maps images to some feature space."""

        if not (len(channels) == len(kernel_widths)
                and len(kernel_widths) == len(strides)):
            raise ValueError("channels, kernel_widths, strides must have equal"
                             f" length; got {len(channels)},"
                             f"{len(kernel_widths)}, {len(strides)}")

        input_layer = Input(image_shape)
        X = input_layer

        if add_noise:
            X = GaussianNoise(0.01)(X)
        for channel, kernel, stride in zip(channels, kernel_widths, strides):
            X = Conv2D(channel, kernel, strides=stride,
                       padding='same', kernel_initializer=init)(X)
            if add_noise:
                X = GaussianDropout(0.005)(X)
            X = LayerNormalization()(X)
            if hidden_activation == 'leaky_relu':
                X = LeakyReLU(0.02)(X)
            else:
                X = Activation(hidden_activation)(X)
        X = Flatten()(X)
        X = Dense(output_width, kernel_initializer=init)(X)
        output_layer = Activation(output_activation)(X)
        model = Model(input_layer, output_layer)
        return model


    def _build_generator(self,
                       latent_dim,
                       starting_shape=None,
                       channels=None,
                       kernel_widths=None,
                       strides=None,
                       upsampling=None,
                       hidden_activation='relu',
                       output_activation='tanh',
                       init=RandomNormal(mean=0, stddev=0.02)):
        """Build a model that maps a latent space to images."""

        if not (len(channels) == len(kernel_widths)
                and len(kernel_widths) == len(strides)):
            raise ValueError("channels, kernel_widths, strides must have equal"
                             f" length; got {len(channels)},"
                             f"{len(kernel_widths)}, {len(strides)}")

        input_layer = Input((latent_dim,))
        X = Dense(np.prod(starting_shape),
                  kernel_initializer=init)(input_layer)
        X = LayerNormalization()(X)
        if hidden_activation == 'leaky_relu':
            X = LeakyReLU(0.02)(X)
        else:
            X = Activation(hidden_activation)(X)
        X = Reshape(starting_shape)(X)

        Y = Dense(64)(input_layer)
        Y = LayerNormalization()(Y)
        if hidden_activation == 'leaky_relu':
            Y = LeakyReLU(0.02)(Y)
        else:
            Y = Activation(hidden_activation)(Y)
        Y = Reshape((1, 1, 64))(Y)
        Y = UpSampling2D(np.array(starting_shape[:2]))(Y)

        for i in range(len(channels)-1):
            X = Concatenate()([X, Y])
            X = UpSampling2D(upsampling[i])(X)
            Y = UpSampling2D(upsampling[i])(Y)
            X = Conv2D(channels[i], kernel_widths[i], strides=strides[i],
                       padding='same', kernel_initializer=init)(X)
            X = LayerNormalization()(X)
            if hidden_activation == 'leaky_relu':
                X = LeakyReLU(0.02)(X)
            else:
                X = Activation(hidden_activation)(X)
        else:
            X = Concatenate()([X, Y])
            X = Conv2D(channels[-1], kernel_widths[-1], strides=strides[-1],
                       padding='same', kernel_initializer=init)(X)
            output_layer = Activation(output_activation)(X)

        model = Model(input_layer, output_layer)
        return model


    @abstractmethod
    def train(self, batch_size=32, batch_num=1, verbose=True, prepend=''):
        pass


class AutoEncoder(GenerativeModel):

    def __init__(self,
            image_shape,
            latent_dim,
            parameter_json_path,
            data_generating_function,
            noise_generating_function,):
        super().__init__(image_shape,
                         latent_dim,
                         parameter_json_path,
                         data_generating_function,
                         noise_generating_function)
        self.encoder = self._build_encoder(self.image_shape,
                                           self.latent_dim,
                                           **self.parameters["encoder"])
        self.autoencoder_image = self._build_ae(
                self.encoder,
                self.generator,
                loss=self.parameters["loss"]["reconstruct_image"],
                lr=self.parameters["lr"]["ae_image"])


    def _build_encoder(self,
                       image_shape,
                       latent_dim,
                       channels=None,
                       kernel_widths=None,
                       strides=None,
                       hidden_activation='relu',
                       output_activation='linear',
                       init=RandomNormal(mean=0, stddev=0.02),
                       add_noise=True):
        """Build a model that maps images to a latent space."""
        model = self._build_feature_extractor(
                image_shape=image_shape,
                output_width=latent_dim,
                channels=channels,
                kernel_widths=kernel_widths,
                strides=strides,
                hidden_activation=hidden_activation,
                output_activation=output_activation,
                init=init,
                add_noise=add_noise)
        return model


    def _build_ae(self, encoder, decoder, loss='mae', lr=1e-3):
        input_shape = encoder.layers[0].input_shape[0][1:]
        input_layer = Input(input_shape)
        encoding = encoder(input_layer)
        reconstruction = decoder(encoding)
        model = Model(input_layer, reconstruction)
        model.compile(Adam(clipnorm=1, lr=lr), loss=loss)
        return model


    def encode(self, images):
        """Encode a set of images as latent vectors."""
        return self.encoder.predict(images)


    def autoencode_images(self, images):
        """Encode then decode a set of images through latent space."""
        return self.autoencoder_image.predict(images)


    def train(self, batch_size=32, batch_num=1, verbose=True, prepend=''):
        running_loss = 0
        for i in range(0, batch_num):
            samples = self.data_generating_function(batch_size)
            loss = self.autoencoder_image.train_on_batch(samples, samples)
            running_loss += loss
            if verbose:
                print(f"{prepend}[{(i+1)*batch_size}/{batch_num*batch_size}]: "
                      f"loss={running_loss/(i+1):.4f}",
                      end='\r')
        print()
        return running_loss / batch_num


class GenerativeAdversarialNetwork(GenerativeModel):

    def __init__(self,
            image_shape,
            latent_dim,
            parameter_json_path,
            data_generating_function,
            noise_generating_function,):
        super().__init__(image_shape,
                         latent_dim,
                         parameter_json_path,
                         data_generating_function,
                         noise_generating_function)
        self.discriminator_image = self._build_discriminator_image(
                image_shape,
                **self.parameters["discriminator_image"])
        self.gan_image = self._build_gan(
                self.generator,
                self.discriminator_image)



    def _build_discriminator_image(self,
                                   image_shape,
                                   channels=None,
                                   kernel_widths=None,
                                   strides=None,
                                   hidden_activation='relu',
                                   init=RandomNormal(mean=0, stddev=0.02)):
        """Build a model that classifies images as real (1) or fake (0)."""

        model = self._build_feature_extractor(image_shape=image_shape,
                                    output_width=1,
                                    channels=channels,
                                    kernel_widths=kernel_widths,
                                    strides=strides,
                                    hidden_activation=hidden_activation,
                                    output_activation='sigmoid',
                                    init=init,
                                    add_noise=True)
        model.compile(optimizer=Adam(clipnorm=1, lr=self.parameters["lr"]["gan_discriminator"], beta_1=0.5),
                      loss=self.parameters["loss"]["adversarial"])
        return model


    def _build_gan(self, generator, discriminator):
        discriminator.trainable = False
        input_shape = generator.layers[0].input_shape[0][1:]
        input_layer = Input(input_shape)
        generated = generator(input_layer)
        prediction = discriminator([generated])
        model = Model(input_layer, prediction)
        model.compile(optimizer=Adam(clipnorm=1, lr=self.parameters["lr"]["gan_generator"],
                           beta_1=0.5),
                      loss=self.parameters["loss"]["adversarial"])
        return model


    def discriminate_images(self, images):
        """Predict whether a set of images are real or not."""
        return self.discriminator_image.predict(images)


    def train(self, batch_size=32, batch_num=1, verbose=True, prepend=''):
        running_loss_d = 0
        running_loss_g = 0
        real_labels_d = np.ones((batch_size, 1))*0.95
        fake_labels_d = np.ones((batch_size, 1))*0.05
        labels_g = np.ones((batch_size, 1))
        for i in range(0, batch_num, 2):
            real = self.data_generating_function(batch_size)
            fake = self.decode(self.noise_generating_function(batch_size))
            loss_d1 = self.discriminator_image.train_on_batch(
                    real,
                    real_labels_d)
            loss_d2 = self.discriminator_image.train_on_batch(
                    fake,
                    fake_labels_d)
            running_loss_d += loss_d1 + loss_d2
            loss_g1 = self.gan_image.train_on_batch(
                    self.noise_generating_function(batch_size),
                    labels_g)
            loss_g2 = self.gan_image.train_on_batch(
                    self.noise_generating_function(batch_size),
                    labels_g)
            running_loss_g += loss_g1 + loss_g2
            if verbose:
                print(f"{prepend}[{(i+1)*batch_size}/{batch_num*batch_size}]: "
                      f"G={running_loss_g/(i+2):.4f}; "
                      f"D={running_loss_d/(i+2):.4f}",
                      end='\r')
        print()
        return running_loss_d / batch_num, running_loss_g / batch_num


class AdversarialAutoEncoder(AutoEncoder):

    def __init__(self,
            image_shape,
            latent_dim,
            parameter_json_path,
            data_generating_function,
            noise_generating_function,):
        super().__init__(image_shape,
                         latent_dim,
                         parameter_json_path,
                         data_generating_function,
                         noise_generating_function)
        self.discriminator_latent = self._build_discriminator_latent(
                latent_dim,
                **self.parameters["discriminator_latent"])
        self.aae = self._build_aae(
                self.encoder,
                self.generator,
                self.discriminator_latent)


    def _build_discriminator_latent(self,
                                    latent_dim,
                                    layers=16,
                                    width=16,
                                    hidden_activation='relu',
                                    init=RandomNormal(mean=0, stddev=0.02),
                                    add_noise=True):
        """Build a model that classifies latent vectors as real or fake."""
        input_layer = Input((latent_dim,))
        F = input_layer
        if add_noise:
            F = GaussianNoise(0.01)(F)
        for i in range(layers):
            X = Dense(width)(F)
            if add_noise:
                X = GaussianDropout(0.005)(X)
            X = LayerNormalization()(X)
            if hidden_activation == 'leaky_relu':
                X = LeakyReLU(0.02)(X)
            else:
                X = Activation(hidden_activation)(X)
            F = Concatenate()([F, X])
        X = Dense(128)(F)
        if hidden_activation == 'leaky_relu':
            X = LeakyReLU(0.02)(X)
        else:
            X = Activation(hidden_activation)(X)
        X = Dense(1)(X)
        output_layer = Activation('sigmoid')(X)
        model = Model(input_layer, output_layer)
        model.compile(Adam(clipnorm=1, lr=self.parameters["lr"]["gan_discriminator"],
                           beta_1=0.5),
                      loss=self.parameters["loss"]["adversarial"])
        return model


    def _build_aae(self, encoder, generator, discriminator):
        discriminator.trainable = False
        input_shape = encoder.layers[0].input_shape[0][1:]
        input_layer = Input(input_shape)
        encoded = encoder(input_layer)
        decoded = generator(encoded)
        prediction = discriminator(encoded)
        model = Model(input_layer, [decoded, prediction])
        model.compile(Adam(clipnorm=1, lr=self.parameters["lr"]["gan_generator"],
                           beta_1=0.5),
                      loss=[self.parameters["loss"]["reconstruct_image"],
                            self.parameters["loss"]["adversarial"]],
                      loss_weights=[self.parameters["alpha"]["reconstruct_image"],
                                    self.parameters["alpha"]["discriminate_latent"]])
        return model


    def discriminate_images(self, images):
        """Predict whether a set of images are real or not."""
        return self.discriminator_image.predict(images)



    def train(self, batch_size=32, batch_num=1, verbose=True, prepend=''):
        running_loss_r = 0
        running_loss_d = 0
        running_loss_g = 0
        real_labels_d = np.ones((batch_size, 1))*0.95
        fake_labels_d = np.ones((batch_size, 1))*0.05
        labels_g = np.ones((batch_size, 1))
        for i in range(0, batch_num, 2):
            images = self.data_generating_function(batch_size)
            real = self.noise_generating_function(batch_size)
            fake = self.encode(images)
            loss_d1 = self.discriminator_latent.train_on_batch(
                    real,
                    real_labels_d)
            loss_d2 = self.discriminator_latent.train_on_batch(
                    fake,
                    fake_labels_d)
            running_loss_d += loss_d1 + loss_d2
            images = self.data_generating_function(batch_size)
            __, loss_r1, loss_g1 = self.aae.train_on_batch(
                    images,
                    [images, labels_g])
            images = self.data_generating_function(batch_size)
            __, loss_r2, loss_g2 = self.aae.train_on_batch(
                    images,
                    [images, labels_g])
            running_loss_g += loss_g1 + loss_g2
            running_loss_r += loss_r1 + loss_r2
            if verbose:
                print(f"{prepend}[{(i+1)*batch_size}/{batch_num*batch_size}]: "
                      f"G={running_loss_g/(i+2):.4f}; "
                      f"D={running_loss_d/(i+2):.4f}; ",
                      f"R={running_loss_r/(i+2):.4f}",
                      end='\r')
        print()
        return (running_loss_d / batch_num,
                running_loss_g / batch_num,
                running_loss_r / batch_num)


class EncodingGenerativeAdversarialNetwork(AutoEncoder, GenerativeAdversarialNetwork):

    def __init__(self,
            image_shape,
            latent_dim,
            parameter_json_path,
            data_generating_function,
            noise_generating_function,):
        super().__init__(image_shape,
                         latent_dim,
                         parameter_json_path,
                         data_generating_function,
                         noise_generating_function)

        self.dae = self._build_dae(
                self.encoder,
                self.generator,
                self.discriminator_image)


    def _build_dae(self, encoder, generator, discriminator):
        discriminator.trainable = False
        input_shape = encoder.layers[0].input_shape[0][1:]
        input_layer = Input(input_shape)
        encoded = encoder(input_layer)
        decoded = generator(encoded)
        prediction = discriminator(decoded)
        model = Model(input_layer, [decoded, prediction])
        model.compile(Adam(clipnorm=1, lr=self.parameters["lr"]["gan_generator"],
                           beta_1=0.5),
                      loss=[self.parameters["loss"]["reconstruct_image"],
                            self.parameters["loss"]["adversarial"]],
                      loss_weights=[self.parameters["alpha"]["reconstruct_image"],
                                    self.parameters["alpha"]["discriminate_image"]])
        return model


    def train(self, batch_size=32, batch_num=1, verbose=True, prepend=''):
        running_loss_r = 0
        running_loss_d = 0
        running_loss_g = 0
        real_labels_d = np.ones((batch_size, 1))*0.95
        fake_labels_d = np.ones((batch_size, 1))*0.05
        labels_g = np.ones((batch_size, 1))
        for i in range(0, batch_num, 4):
            real1 = self.data_generating_function(batch_size)
            real2 = self.data_generating_function(batch_size)
            generated = self.generate(batch_size)
            autoencoded = self.autoencode_images(self.data_generating_function(batch_size))
            running_loss_d += self.discriminator_image.train_on_batch(
                    real1,
                    real_labels_d)
            running_loss_d += self.discriminator_image.train_on_batch(
                    generated,
                    fake_labels_d)
            running_loss_d += self.discriminator_image.train_on_batch(
                    real2,
                    real_labels_d)
            running_loss_d += self.discriminator_image.train_on_batch(
                    autoencoded,
                    fake_labels_d)
            for __ in range(4):
                images = self.data_generating_function(batch_size)
                __, loss_r, loss_g = self.dae.train_on_batch(
                        images,
                        [images, labels_g])
                running_loss_g += loss_g
                running_loss_r += loss_r
            if verbose:
                print(f"{prepend}[{(i+1)*batch_size}/{batch_num*batch_size}]: "
                      f"G={running_loss_g/(i+2):.4f}; "
                      f"D={running_loss_d/(i+2):.4f}; ",
                      f"R={running_loss_r/(i+2):.4f}",
                      end='\r')
        print()
        return (running_loss_d / batch_num,
                running_loss_g / batch_num,
                running_loss_r / batch_num)


class AutoEncodingGenerativeAdversarialNetwork(AdversarialAutoEncoder, EncodingGenerativeAdversarialNetwork):

    def __init__(self,
            image_shape,
            latent_dim,
            parameter_json_path,
            data_generating_function,
            noise_generating_function,):
        super().__init__(image_shape,
                         latent_dim,
                         parameter_json_path,
                         data_generating_function,
                         noise_generating_function)
        self.aegan = self._build_aegan(
                self.encoder,
                self.generator,
                self.discriminator_image,
                self.discriminator_latent)



    def _build_aegan(self, encoder, generator, discriminator_image, discriminator_latent):
        discriminator_image.trainable = False
        discriminator_latent.trainable = False
        input_image_shape = encoder.layers[0].input_shape[0][1:]
        input_latent_shape = generator.layers[0].input_shape[0][1:]
        x_real = Input(input_image_shape)
        z_real = Input(input_latent_shape)
        z_hat = encoder(x_real)
        x_tilde = generator(z_hat)
        x_hat = generator(z_real)
        z_tilde = encoder(x_hat)
        prediction_x_hat = discriminator_image(x_hat)
        prediction_x_tilde = discriminator_image(x_tilde)
        prediction_z_hat = discriminator_latent(z_hat)
        prediction_z_tilde = discriminator_latent(z_tilde)
        model = Model(
                [x_real, z_real],
                [x_tilde,
                    z_tilde,
                    prediction_x_hat,
                    prediction_x_tilde,
                    prediction_z_hat,
                    prediction_z_tilde
                    ])
        model.compile(Adam(clipnorm=1, lr=self.parameters["lr"]["gan_generator"],
                           beta_1=0.5),
                      loss=[self.parameters["loss"]["reconstruct_image"],
                            self.parameters["loss"]["reconstruct_latent"],
                            self.parameters["loss"]["adversarial"],
                            self.parameters["loss"]["adversarial"],
                            self.parameters["loss"]["adversarial"],
                            self.parameters["loss"]["adversarial"],
                            ],
                      loss_weights=[self.parameters["alpha"]["reconstruct_image"],
                                    self.parameters["alpha"]["reconstruct_latent"],
                                    self.parameters["alpha"]["discriminate_image"],
                                    self.parameters["alpha"]["discriminate_image"],
                                    self.parameters["alpha"]["discriminate_latent"],
                                    self.parameters["alpha"]["discriminate_latent"],
                                    ])
        return model


    def train(self, batch_size=32, batch_num=1, verbose=True, prepend=''):
        running_loss_rx = 0
        running_loss_rz = 0
        running_loss_dx = 0
        running_loss_dz = 0
        running_loss_gx = 0
        running_loss_gz = 0
        real_labels_d = np.ones((batch_size, 1))*0.95
        fake_labels_d = np.ones((batch_size, 1))*0.05
        labels_g = np.ones((batch_size, 1))
        start = time.time()
        for i in range(0, batch_num, 4):
            x1 = self.data_generating_function(batch_size)
            x2 = self.data_generating_function(batch_size)
            x_hat = self.generate(batch_size)
            x_tilde = self.autoencode_images(self.data_generating_function(batch_size))
            running_loss_dx += self.discriminator_image.train_on_batch(
                    x1,
                    real_labels_d)
            running_loss_dx += self.discriminator_image.train_on_batch(
                    x_hat,
                    fake_labels_d)
            running_loss_dx += self.discriminator_image.train_on_batch(
                    x2,
                    real_labels_d)
            running_loss_dx += self.discriminator_image.train_on_batch(
                    x_tilde,
                    fake_labels_d)
            del x_hat, x_tilde, x1, x2

            z1 = self.noise_generating_function(batch_size)
            z2 = self.noise_generating_function(batch_size)
            z_hat = self.encode(self.data_generating_function(batch_size))
            z_tilde = self.encode(self.decode(self.noise_generating_function(batch_size)))
            running_loss_dz += self.discriminator_latent.train_on_batch(
                    z1,
                    real_labels_d)
            running_loss_dz += self.discriminator_latent.train_on_batch(
                    z_hat,
                    fake_labels_d)
            running_loss_dz += self.discriminator_latent.train_on_batch(
                    z2,
                    real_labels_d)
            running_loss_dz += self.discriminator_latent.train_on_batch(
                    z_tilde,
                    fake_labels_d)

            for j in range(4):
                images = self.data_generating_function(batch_size)
                latent = self.noise_generating_function(batch_size)
                losses = self.aegan.train_on_batch(
                        [images, latent],
                        [images, latent, labels_g, labels_g, labels_g, labels_g])
                (_, loss_rx, loss_rz, loss_dx_g_z, loss_dx_g_e_x, loss_dz_e_x, loss_dz_e_g_z, ) = losses
                running_loss_rx += loss_rx
                running_loss_rz += loss_rz
                running_loss_gx += (loss_dx_g_e_x + loss_dx_g_z) / 2
                running_loss_gz += (loss_dz_e_g_z + loss_dz_e_x) / 2
            if verbose:
                t = int(time.time() - start)
                print(f"{prepend}[{(i+1)*batch_size}/{batch_num*batch_size}]: "
                      f"Gx={running_loss_gx/(i+4):.4f}; "
                      f"Gz={running_loss_gz/(i+4):.4f}; "
                      f"Dx={running_loss_dx/(i+4):.4f}; "
                      f"Dz={running_loss_dz/(i+4):.4f}; "
                      f"Rx={running_loss_rx/(i+4):.4f}; "
                      f"Rz={running_loss_rz/(i+4):.4f}; "
                      f'({t//(3600):02d}:{(t%3600)//60:02d}:{t%60:02d})',
                      end='\r')
        print()

