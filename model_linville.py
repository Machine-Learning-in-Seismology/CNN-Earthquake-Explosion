import keras
import keras_metrics
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Conv1D, Conv2D, Flatten, Dense, Reshape, MaxPooling1D, MaxPooling2D, Dropout, concatenate, Activation, Add

def build_nn(input_shape1):
	input1 = Input(shape=input_shape1,name='input1')
	A1 = Conv2D(18, kernel_size=2, activation='relu', kernel_initializer='glorot_normal')(input1)
	pool11 = MaxPooling2D(pool_size=2)(A1)
	A2 = Conv2D(36, kernel_size=2, activation='relu', kernel_initializer='glorot_normal')(pool11)
	pool12 = MaxPooling2D(pool_size=2)(A2)
	A3 = Conv2D(54, kernel_size=2, activation='relu', kernel_initializer='glorot_normal')(pool12)
	pool13 = MaxPooling2D(pool_size=2)(A3)
	A4 = Conv2D(54, kernel_size=2, activation='relu', kernel_initializer='glorot_normal')(pool13)
	pool14 = MaxPooling2D(pool_size=2)(A4)

	flattened_spec = Flatten()(pool14) 

	den = Dense(1)(flattened_spec)
	pred = Activation('sigmoid', name='linear')(den)

	model = Model(inputs=[input1],outputs=[pred])

	model.compile(loss='binary_crossentropy', optimizer='adam',metrics=[
			keras_metrics.false_positive(), keras_metrics.true_negative(),
			keras_metrics.false_negative(), keras_metrics.true_positive(),
			keras.metrics.binary_accuracy
		])
	return model

def get_early_stop():
	return EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto', restore_best_weights=True)	