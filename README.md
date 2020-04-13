# AEGAN-keras
To use, run the script train.py with the following arguments:
```
-t TYPE
  The type of network. Must be one of "GAN", "AE", "AEGAN"
  For Generative Adversarial Network, Autoencoder,
  or Autoencoding Generative Adversarial Network
-n NAME
  The name of the experiment to run (must be unique)
  Files will be saved to "../results/<NAME>_<TYPE>/
-p PLOT_EVERY
  The number of epochs between saving samples
-l LATENT_DIM
  The dimension of the latent space
-b BATCH SIZE
  Size of the minibatch
-d DATA_DIR
  Directory where images are saved. If not specified,
  MNIST is used
-e EPOCHS
  Number of epochs to train for
-s STEPS
  Number of training steps per epoch
-f PARAMETER_FILE
  Parameters for building the network. See
  code/params/params_64.json as an example.
  Images are resized to match the output size
  of the generator.
-x FLIP
  Whether (True) or not (False) to flip images before training
```
