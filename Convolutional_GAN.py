from __future__ import print_function
import numpy as np
from six.moves import range
import matplotlib.pyplot as plt
from PIL import Image
from mpl_toolkits.axes_grid1 import AxesGrid

import keras
import tensorflow as tf

import random

from keras.models import Sequential
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape, UpSampling2D
from keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, GaussianNoise, Lambda
from keras.optimizers import SGD, Adam

from keras.preprocessing.image import ImageDataGenerator
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.core import MaxoutDense

from tensorflow.examples.tutorials.mnist import input_data

def prepare_dataset():
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    train_dataset = mnist.train.images - 0.5
    train_labels = mnist.train.labels
    valid_dataset = mnist.validation.images -0.5
    valid_labels = mnist.validation.labels
    test_dataset = mnist.test.images - 0.5
    test_labels = mnist.test.labels

    def reformat(dataset, labels):
        dataset = dataset.reshape((-1, image_size,image_size,1)).astype(np.float32)
        return dataset, labels

    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)

    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)
    
    return train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels

# tools for showing and saving images
def show_imagelist_as_grid(img_list, nrow, ncol):
    fig = plt.figure(figsize=(5,5))
    grid = AxesGrid(fig, 111, nrows_ncols=(nrow, ncol), axes_pad=0.05, label_mode="1")
    for i in range(nrow*ncol):
        im = grid[i].imshow(img_list[i], interpolation="none", cmap='gray', vmin=-0.5, vmax=0.5)

    plt.draw()
    plt.show()
        
def picture_grid(img_list,nrow,ncol):
    imsx, imsy = img_list.shape[1], img_list.shape[2]
    mosarr=np.zeros([ncol*(imsx+1),nrow*(imsy+1)])+0.25
    for i in range(ncol):
        for j in range(nrow):
            if img_list.shape[0]>j*ncol+i:
                mosarr[i*(imsx+1):(i+1)*(imsx+1)-1, j*(imsy+1) : (j+1)*(imsy+1)-1] = img_list[j*ncol+i,:,:]
    return mosarr

def save_images(img_list,nrow,ncol,label):
    imgtosave = picture_grid(img_list,nrow,ncol)*127.5+127.5
    Image.fromarray(imgtosave.astype(np.uint8)).save('outputimages/' + label + '.jpeg', quality=100)


def normal_init(shape, name=None):
    return keras.initializations.normal(shape, scale=0.04, name=name)

def save_model(fname):
    generator_json = generator.to_json()
    with open("saved_models/" + fname + "_generator.json", "w") as json_file:
        json_file.write(generator_json)

    generator.save_weights("saved_models/" + fname + "_generator.h5")
    print("Saved model to disk")

    discr_json = discriminator.to_json()
    with open("saved_models/" + fname + "_discriminator.json", "w") as json_file:
        json_file.write(discr_json)

    discriminator.save_weights("saved_models/" + fname + "_discriminator.h5")

    print("Saved model to disk")

def generate_picked(pickiness):
    cherry_picked = np.zeros([batch_size, image_size*image_size])
    num_picked = 0
    while num_picked < batch_size:
        batch_noise = np.random.uniform(-0.5, 0.5, size=(batch_size, num_gen_input_size))    
        generated_images = generator.predict(batch_noise, verbose=0)
        preds = discriminator.predict(generated_images)
        print("new batch", np.max(preds))
        for i in range(batch_size):
            if preds[i]>pickiness:
                cherry_picked[num_picked,:] = generated_images[i,:].reshape(784)
                num_picked+=1
                print(num_picked)
                if(num_picked>=batch_size):
                    break

    save_images(cherry_picked.reshape(-1,image_size,image_size), 16,8, "picked" + str(random.random()))



def gen_model():
    model = Sequential()
    model.add(Dense(64*7*7, input_shape=(num_gen_input_size,)))
    model.add(LeakyReLU(alpha=lrelu_alpha))
    model.add(Reshape((7, 7, 64)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(64, 5, 5, border_mode='same'))
    model.add(LeakyReLU(alpha=lrelu_alpha))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(Convolution2D(1, 5, 5, border_mode='same'))
    model.add(Activation('tanh'))
    return model

def discr_model():
    model = Sequential()
    model.add(Convolution2D(64, 7, 7, border_mode='same', subsample=(2,2), input_shape=(28,28,1), init=normal_init))
    model.add(BatchNormalization(mode=2))
    model.add(LeakyReLU(alpha=lrelu_alpha))
    model.add(Convolution2D(128, 5, 5, border_mode='same', subsample=(2,2), init=normal_init))
    model.add(BatchNormalization(mode=2))
    model.add(LeakyReLU(alpha=lrelu_alpha))
    model.add(Flatten())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    return model

def discr_on_gen_model(gen, discr):
    model = Sequential()
    model.add(gen)
    discr.trainable = False
    model.add(discr)
    return model
   


if __name__ == "__main__":

    ## Operation 
    load_old_weights = False 
    old_weights_fname = "saved_models/200000/conv_"
    generate_and_leave = False 
    pickiness = 0.0 # for generated outputs

    ## Parameters

    image_size = 28 # now fixed: some vars in architecture assume 28
    
    batch_size = 128 
    
    dropout_prob_gen = 0.0
    dropout_prob_discr = 0.0
    lrelu_alpha = 0.2
    num_gen_input_size = 100 # the dimension of the random input of generator

    print_step = 400 
    save_step = 1000 
    reshuffle_step = 1100 # how often to reshuffle data set 
    num_steps = 20000000 # max number of batches 
    range_start = 0 # output filenames will start at range_start 

    discr_lr = 0.002  # learning rates
    gen_lr = 0.002
    discr_decay=1e-10  # decays
    gen_decay=1e-10
    discr_momentum=0.5  # for discr SGD

    ## dataset
    train_dataset, train_labels, valid_dataset, valid_labels, test_dataset, test_labels = prepare_dataset()

    train_dataset = np.concatenate((train_dataset, valid_dataset, test_dataset), axis=0)

    dataset_mean = np.mean(train_dataset)
    dataset_std = np.std(train_dataset)
    print("mean and std: ", dataset_mean, dataset_std)

    np.random.shuffle(train_dataset)

    ## models 
    discriminator = discr_model();
    generator = gen_model();

    if load_old_weights:
        generator.load_weights(old_weights_fname + "_generator.h5")
        print("Loaded generator")
        discriminator.load_weights(old_weights_fname + "_discriminator.h5")
        print("Loaded discriminator")
        
    discriminator_on_generator = discr_on_gen_model(generator, discriminator);

    discr_optimizer = SGD(lr=discr_lr, decay=discr_decay, momentum=discr_momentum)
    gen_optimizer = Adam(lr=gen_lr, beta_1=0.5, decay=gen_decay)

    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=gen_optimizer)
    discriminator.trainable = True # we had made it untrainable when we added to the discriminator_on_generator
    discriminator.compile(loss='binary_crossentropy', optimizer=discr_optimizer)


    if generate_and_leave:
        generate_picked(pickiness) 
        exit()
            
    ## training

    gen_loss_total = 0.0
    gen_trained = 0
    discr_loss_total = 0.0
    discr_trained = 0

    batch_1s = np.zeros(batch_size)+1
    batch_0s = np.zeros(batch_size)

    print("Starting training.")
    for step in range(range_start,range_start+num_steps):
        for i in range(1):    
            #train discriminator
            offset = (round(random.uniform(0, 100000)) + step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_noise = np.random.uniform(-0.5, 0.5, size=(batch_size, num_gen_input_size))
            generated_images = generator.predict(batch_noise, verbose=0)
            discriminator.trainable = True # just to make sure it's still trainable
            ld1 = discriminator.train_on_batch(batch_data, batch_1s - np.abs(np.random.normal(0,0.15, batch_size)))
            ld2 = discriminator.train_on_batch(generated_images, batch_0s)
            discr_loss_total += (ld1+ld2)/2
            discr_trained += 1
        
        for i in range(1):
            #train generator
            batch_noise = np.random.uniform(-0.5, 0.5, size=(batch_size, num_gen_input_size))    
            discriminator.trainable = False # make sure we train only the generator
            lg = discriminator_on_generator.train_on_batch(batch_noise, np.zeros(batch_size) + 1)
            discriminator.trainable = True # just to make sure it's still trainable
            gen_loss_total += lg
            gen_trained += 1
            
        if (step % print_step == print_step-1):
            if discr_trained == 0:
                discr_trained = 1
            print('Minibatch loss before step %d: discriminator %f, generator: %f' % (step+1, discr_loss_total/discr_trained, gen_loss_total/gen_trained))
            with open("logg.txt", "a") as myfile:
                myfile.write('Minibatch loss before step %d: discriminator %f, generator: %f\n' % (step+1, discr_loss_total/discr_trained, gen_loss_total/gen_trained))
            gen_loss_total = 0.0
            discr_loss_total = 0.0
            gen_trained = 0
            discr_trained = 0
            save_images(generated_images.reshape(-1,image_size,image_size),4,4,"gen_"+ str(step))
        if step % save_step == save_step-1:
            save_model("conv_"+ str(step))  
        if step % reshuffle_step == reshuffle_step-1:
            np.random.shuffle(train_dataset)
    



# load the models from files:
def load_model(fname):
    generator.load_weights(fname + "_generator.h5")
    print("Loaded generator")

    discriminator.load_weights(fname + "_discriminator.h5")
    print("Loaded discriminator")

    discriminator_on_generator = discr_on_gen_model(generator, discriminator);
    discr_optimizer = Adam(lr=0.0002, decay=1e-8)
    gen_optimizer = SGD(lr=0.0002, decay=1e-8, momentum=0.5)

    generator.compile(loss='binary_crossentropy', optimizer="SGD")
    discriminator_on_generator.compile(loss='binary_crossentropy', optimizer=gen_optimizer)
    discriminator.trainable = True # this will have to be switched on and off
    discriminator.compile(loss='binary_crossentropy', optimizer=discr_optimizer)
    return generator, discriminator, discriminator_on_generator




