import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import gc
import datetime
import numpy as np
import h5py
from keras import callbacks
from keras.layers import Dense, Flatten, Input, Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, load_model, model_from_json
from keras.optimizers import Adam
import chess
import tensorflow as tf
from utils import chess_dict, squares
from batch_generator import batch_generator

# fix TF 2.4 issue
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

stages = ['early','mid','late']

variants = ['from','to']
network = '256-256-1024-1024'
batch_size = 1024

for stage in stages:
    for variant in variants:
        print('Training network '+stage+'-'+variant)
        data_path = 'data\\data2014-2020-staged-'+stage+'.h5'
        validation_data_path = 'data\\data2013-staged-'+stage+'.h5'

        model_path = 'models/'+stage+'-'+variant+'-withTurn-b'+str(batch_size)+'-'+network+'-model.h5'
        # model_path = 'models/to-withTurn-b2048-128-256-1024-1024-model.h5'
        model_path_json = 'models/'+stage+'-'+variant+'-withTurn-b'+str(batch_size)+'-'+network+'-model.json'

        # log_dir = "logs/" + variant + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = 'logs/'+stage+'-'+variant+'-b'+str(batch_size)+'-'+network

        # training data
        with h5py.File(data_path, "r") as h5f:
            num_samples = len(h5f['moved_'+variant+'_'+stage])
            
        steps_per_epoch = np.ceil(num_samples/batch_size)
        training_data_batch_generator = batch_generator(data_path, batch_size, steps_per_epoch, variant, stage)

        # validation data
        with h5py.File(validation_data_path, "r") as h5f:
            validation_num_samples = len(h5f['moved_'+variant+'_'+stage])
            
        validation_steps_per_epoch = np.ceil(validation_num_samples/batch_size)
        validation_data_batch_generator = batch_generator(validation_data_path, batch_size, validation_steps_per_epoch, variant, stage)

        # initalize neural network
        model = Sequential()
        model.add(Input(shape=(9, 8, 12)))
        model.add(Conv2D(256, kernel_size=(2, 2)))
        model.add(Activation('relu'))
        model.add(Conv2D(256, kernel_size=(2, 2)))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1024))
        model.add(Activation('relu'))
        # 64 classes as output (one-hot encoded position on the board)
        model.add(Dense(64, activation='softmax'))

        # in case we need to resume training
        # model = load_model(model_path)

        # decay of learning rate to avoid overfitting
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(0.001, decay_steps=100000, decay_rate=0.96, staircase=True)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        # # save model for later
        model_json = model.to_json()
        with open(model_path_json, 'w') as json_file:
            json_file.write(model_json)

        # print(model.summary())

        # checkpoints in case training stops
        checkpoint = callbacks.ModelCheckpoint(model_path, monitor='val_accuracy', verbose=2, save_best_only=True, mode='max', save_freq='epoch')

        # stops early if loss doesn't improve after 500 epocs
        # es = callbacks.EarlyStopping(monitor='val_accuracy', mode='min', verbose=1, patience=500)

        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        model.fit(training_data_batch_generator,
                steps_per_epoch=steps_per_epoch,
                validation_data=validation_data_batch_generator,
                validation_steps=validation_steps_per_epoch,
                # initial_epoch=10,  # in case we need to resume training
                epochs=10,
                verbose=1,
                shuffle=True,
                # workers=4,
                # max_queue_size=8,
                callbacks=[checkpoint, tensorboard_callback])