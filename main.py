
from keras.optimizers import SGD, Adam
from keras.losses import mean_squared_error, binary_crossentropy
from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout, LeakyReLU
from keras.models import Model
from keras.regularizers import l2
from keras import backend as K

import pandas as pd
import tensorflow as tf
import numpy as np
import random
import seaborn as sns
import matplotlib.pyplot as plt
import os
import datetime


"""
INFO:
- x = real data point
- z = generated noise data
- g = generator
- d = distriminator
- g(z) = output of the generator, given noise z
- d(g(z)) = 0 or 1, representing of the discriminator thinks g(z) is real or fake

- Generator wants d(g(z)) to be 1
- Discriminator want d(g(z)) to be 0
"""


D_INPUT_SIZE = 3
D_OUTPUT_SIZE = 1
G_INPUT_SIZE = 100
G_OUTPUT_SIZE = D_INPUT_SIZE
REAL_LABEL = np.array([1])
FAKE_LABEL = np.array([0])
G_DROPOUT_RATE = 0
D_DROPOUT_RATE = 0
G_REGULARIZATION = None #l2(0.01)
D_REGULARIZATION = None #l2(0.01)


def wasserstein_loss(y_true, y_pred):
    return abs(K.mean(y_true * y_pred))


def __build_generator():
    # Input layer.
    input_x = Input(shape=(G_INPUT_SIZE,))
    x = input_x
    
    # Hidden layers.
    x = Dense(50, kernel_regularizer=G_REGULARIZATION)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    #x = Dropout(rate=G_DROPOUT_RATE)(x)

    x = Dense(50, kernel_regularizer=G_REGULARIZATION)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    #x = Dropout(rate=G_DROPOUT_RATE)(x)
    
    # Output layer. 
    x = Dense(G_OUTPUT_SIZE)(x)
    x = BatchNormalization()(x)
    out_x = Activation("linear")(x)
    
    return Model(input_x, out_x)


def __build_discriminator():
    # Input layer.
    input_x = Input(shape=(D_INPUT_SIZE,))
    x = input_x
    
    # Hidden layers.
    x = Dense(50, kernel_regularizer=D_REGULARIZATION)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    #x = Dropout(rate=D_DROPOUT_RATE)(x)

    x = Dense(50, kernel_regularizer=D_REGULARIZATION)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    #x = Dropout(rate=D_DROPOUT_RATE)(x)
    
    # Output layer. 
    x = Dense(D_OUTPUT_SIZE)(x)
    x = BatchNormalization()(x)
    out_x = Activation("linear")(x)
    #out_x = Activation("sigmoid")(x)
    
    return Model(input_x, out_x)


def get_loss():
    #return mean_squared_error
    #return binary_crossentropy
    return wasserstein_loss


def get_optimizer():
    return Adam()
    #return SGD(lr=0.0001, decay=0.0)


def build_stacked_model():
    # Define Optimizer. 
    optimizer = get_optimizer()
    
    # Build and compile Discriminator. 
    d = __build_discriminator()
    d.compile(
        optimizer=optimizer,
        loss=get_loss()
    )
    
    # Build Generator. 
    g = __build_generator()

    # Initialize the input layer. 
    z = Input(shape=(G_INPUT_SIZE,))
    
    # Prevent the discriminator to be trained.
    # This is required to update accuratly update the Generator. 
    d.trainable = False

    # Define and compile the stacked model. 
    combined = Model(z, d(g(z)))
    combined.compile(
        optimizer=optimizer,
        loss=get_loss()
    )

    return g, d, combined


def load_data():
    return pd.read_csv("my_dataset.csv", sep=',')


def save_plot(df, folder_name, epoch):
    #ax = sns.scatterplot(x="F1", y="F2", style="TheLabel", data=df_o)
    ax = sns.scatterplot(x="F1", y="F2", hue="data_type", style="TheLabel", data=df)
    fig = ax.get_figure()

    path = './plots/' + folder_name
    if not os.path.exists(path):
        os.mkdir(path)
    fig.savefig(path + '/' + str(epoch) + '.png')
    plt.clf()
    plt.close(fig)
    #plt.show()


def sample_from_dataset_new(df, num):
    return df.sample(num, replace=False).values


def sample_random(num):
    z = []
    for _ in range(num):
        z.append([np.random.normal(0, 1) for _ in range(G_INPUT_SIZE)])
    return np.array(z)


def get_real_labels(num):
    return np.array([[1] for _ in range(num)])


def get_fake_labels(num):
    return np.array([[0] for _ in range(num)])


def get_folder_name():
    now = datetime.datetime.now()
    return str(now.day) + "_" + str(now.month) + "_" + str(now.year) + "___" \
        + str(now.hour) + "_" + str(now.minute) + "_" + str(now.second)


def score_g_and_d(g, d, combined, num_comparisons=1000):
    z = sample_random(num=num_comparisons)
    pred = combined.predict(z)
    g_score, d_score = 0, 0
    for p in pred:
        if p[0] > 0.5:
            g_score += 1
        else:
            d_score += 1
    print("g_score =", g_score)
    print("d_score =", d_score)
    return g_score, d_score


# Load the data
df = load_data()
df = df.drop('id', axis=1)
df['TheLabel'] = df['TheLabel'].map(lambda x: 1 if x else 0)
#plot_data(df)

# INIT: Generator and Discriminator
g, d, combined = build_stacked_model()

BATCH_SIZE = 100
FAKE = get_fake_labels(num=BATCH_SIZE)
REAL = get_real_labels(num=BATCH_SIZE)
length = 100000
folder_name = get_folder_name()
save_limit = 250

for i in range(length):

    # ***** TRAINING Discriminator *****
    # Sample real target
    x = sample_from_dataset_new(df, num=BATCH_SIZE)
    # Generate random input z. 
    z = sample_random(num=BATCH_SIZE)
    # Pass z through generator. 
    g_out = g.predict(z)
    # Train the discriminator. 
    #d.fit(x, REAL)
    d_loss1 = d.train_on_batch(x, REAL)
    d_loss2 = d.train_on_batch(g_out, FAKE)
    d_loss = (d_loss1 + d_loss2) * 0.5 / BATCH_SIZE

    # ***** TRAINING Generator *****
    z = sample_random(num=BATCH_SIZE)
    # Pretend that z is real and change the generator accordingly (the discriminator is not trained here). 
    #combined.fit(z, REAL)
    g_loss = combined.train_on_batch(z, REAL)
    g_loss = g_loss / BATCH_SIZE
    print(i / length, g_loss, d_loss)

    
    if (i+1) % save_limit == 0:
        synt_data = []
        z = sample_random(num=1000)
        synt = g.predict(z)
        synt_df = pd.DataFrame({
            'F1':synt[:,0],
            'F2':synt[:,1],
            'TheLabel':synt[:,2]
        })
        synt_df['TheLabel'] = synt_df['TheLabel'].map(lambda x: 0 if x < 0.5 else 1)
        synt_df['data_type'] = 1

        df_copy = df.copy()
        df_copy['data_type'] = 0
        df_copy = pd.concat([df_copy, synt_df])
        save_plot(df_copy, folder_name, i+1)

        # Compare D and G. 
        g_score, d_score = score_g_and_d(g, d, combined)
        relative_score = g_score / (g_score + d_score)
        print('relative_score =', relative_score)


print("")
print("***** DONE *****")



#x = np.array([df.sample(1, replace=False).values[0]])
#z = np.array([[np.random.normal(0, 1) for _ in range(G_INPUT_SIZE)]])
