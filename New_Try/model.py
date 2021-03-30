import keras
import keras_metrics
from keras.callbacks import EarlyStopping
from keras import Input, Model
from keras.layers import Conv1D, Flatten, Dense, Reshape, MaxPooling1D, Dropout, concatenate, Activation


def build_nn(input_shape):
    inputs = Input(shape=input_shape)
    #inputs = Reshape((2500, 3,), input_shape=(3, 2500,))(inputs)

    conv1 = Conv1D(64, kernel_size=12, activation='relu', kernel_initializer='glorot_normal')(inputs)
    pool1 = MaxPooling1D(pool_size=4)(conv1)

    conv2 = Conv1D(32, kernel_size=6, activation='relu', kernel_initializer='glorot_normal')(pool1)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    conv3 = Conv1D(32, kernel_size=6, activation='relu', kernel_initializer='glorot_normal')(pool2)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    conv4 = Conv1D(16, kernel_size=3, activation='relu', kernel_initializer='glorot_normal')(pool3)
    pool4 = MaxPooling1D(pool_size=2)(conv4)

    dropout = Dropout(0.5)(pool4)

    flat = Flatten(name='flatten')(dropout)

    den1 = Dense(20, activation='relu', kernel_initializer='glorot_normal')(flat)
    den2 = Dense(10, activation='relu', kernel_initializer='glorot_normal')(den1)
    den3 = Dense(1)(den2)
    pred = Activation('sigmoid', name='linear')(den3)

    model = Model(inputs=inputs, outputs=pred)
    model.compile(loss='binary_crossentropy', optimizer='adam',metrics=[
        keras_metrics.false_positive(), keras_metrics.true_negative(),
        keras_metrics.false_negative(), keras_metrics.true_positive(),
        keras.metrics.binary_accuracy
    ])
    return model


def get_early_stop():
    return EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto', restore_best_weights=True)
