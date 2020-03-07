from abc import ABC, abstractmethod
import json

import numpy as np
from tensorflow.keras.layers import (Input, Dense, Flatten, Concatenate,
                                     Reshape, BatchNormalization, UpSampling2D,
                                     Conv2D, Activation, Dropout)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam

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
                                 init=RandomNormal(mean=0, stddev=0.002)):
        """Build a model that maps images to some feature space."""

        if not (len(channels) == len(kernel_widths)
                and len(kernel_widths) == len(strides)):
            raise ValueError("channels, kernel_widths, strides must have equal"
                             f" length; got {len(channels)},"
                             f"{len(kernel_widths)}, {len(strides)}")

        input_layer = Input(image_shape)
        X = input_layer

        for channel, kernel, stride in zip(channels, kernel_widths, strides):
            X = Conv2D(channel, kernel, strides=stride,
                       padding='same', kernel_initializer=init)(X)
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
                       batchnorm_momentum=0.9,
                       hidden_activation='relu',
                       output_activation='tanh',
                       init=RandomNormal(mean=0, stddev=0.002)):
        """Build a model that maps a latent space to images."""

        if not (len(channels) == len(kernel_widths)
                and len(kernel_widths) == len(strides)):
            raise ValueError("channels, kernel_widths, strides must have equal"
                             f" length; got {len(channels)},"
                             f"{len(kernel_widths)}, {len(strides)}")

        input_layer = Input((latent_dim,))
        X = Dense(np.prod(starting_shape),
                  kernel_initializer=init)(input_layer)
        if batchnorm_momentum is not None:
            X = BatchNormalization(momentum=batchnorm_momentum)(X)
        X = Activation(hidden_activation)(X)
        X = Reshape(starting_shape)(X)


        for i in range(len(channels)-1):
            X = UpSampling2D(upsampling[i])(X)
            X = Conv2D(channels[i], kernel_widths[i], strides=strides[i],
                       padding='same', kernel_initializer=init)(X)
            if batchnorm_momentum is not None:
                X = BatchNormalization(momentum=batchnorm_momentum)(X)
            X = Activation(hidden_activation)(X)
        else:
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
                       init=RandomNormal(mean=0, stddev=0.002)):
        """Build a model that maps images to a latent space."""
        model = self._build_feature_extractor(
                image_shape=image_shape,
                output_width=latent_dim,
                channels=channels,
                kernel_widths=kernel_widths,
                strides=strides,
                hidden_activation=hidden_activation,
                output_activation=output_activation,
                init=init)
        return model


    def _build_ae(self, encoder, decoder, loss='mae', lr=1e-3):
        input_shape = encoder.layers[0].input_shape[0][1:]
        input_layer = Input(input_shape)
        encoding = encoder(input_layer)
        reconstruction = decoder(encoding)
        model = Model(input_layer, reconstruction)
        model.compile(Adam(lr=lr), loss=loss)
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
                                    init=init)
        model.compile(optimizer=Adam(lr=self.parameters["lr"]["gan_discriminator"], beta_1=0.5),
                      loss=self.parameters["loss"]["adversarial"])
        return model


    def _build_gan(self, generator, discriminator):
        discriminator.trainable = False
        input_shape = generator.layers[0].input_shape[0][1:]
        input_layer = Input(input_shape)
        generated = generator(input_layer)
        prediction = discriminator([generated])
        model = Model(input_layer, prediction)
        model.compile(optimizer=Adam(lr=self.parameters["lr"]["gan_generator"],
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
                    np.ones((batch_size, 1))*0)
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
                image_shape,
                **self.parameters["discriminator_latent"])
        self.aae = self._build_aae(
                self.encoder,
                self.generator,
                self.discriminator_latent)


    def _build_discriminator_latent(self,
                                    latent_dim,
                                    layers=16,
                                    width=16,
                                    hidden_activation='softplus',
                                    init=RandomNormal(mean=0, stddev=0.02)):
        """Build a model that classifies latent vectors as real or fake."""

        input_layer = Input((latent_dim,))
        F = input_layer
        for i in range(layers):
            X = Dense(width)(F)
            X = Activation(hidden_activation)(X)
            F = Concatenate()([F, X])
        X = Dense(1)(F)
        output_layer = Activation('sigmoid')(X)
        model = Model(input_layer, output_layer)
        model.compile(Adam(lr=self.parameters["lr"]["gan_discriminator"],
                           beta_1=0.5),
                      loss=self.parameters["loss"]["reconstruct_latent"])
        return model


    def _build_aae(encoder, generator, discriminator):
        discriminator.trainable = False
        input_shape = encoder.layers[0].input_shape[0][1:]
        input_layer = Input(input_shape)
        encoded = encoder(input_layer)
        decoded = generator(encoded)
        prediction = discriminator(encoded)
        model = Model(input_layer, [decoded, prediction])
        model.compile(Adam(lr=self.parameters["lr"]["gan_generator"],
                           beta_1=0.5),
                      loss=[self.parameters["loss"]["reconstruct_image"],
                            self.parameters["loss"]["adversarial"]],
                      loss_weights=[self.parameters["alpha"]["reconstruct_image"],
                                    self.parameters["alpha"]["discriminate_latent"]]),
        return model


    def discriminate_images(self, images):
        """Predict whether a set of images are real or not."""
        return self.discriminator_image.predict(images)



    def train(self, batch_size=32):
        raise NotImplementedError
