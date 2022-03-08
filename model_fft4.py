import keras
import keras_metrics
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Conv1D, Conv2D, Flatten, Dense, Reshape, MaxPooling1D, MaxPooling2D, Dropout, concatenate, Activation, Add

def build_nn(input_shape1,input_shape2):
	input1 = Input(shape=input_shape1,name='input1')
	A1 = Conv1D(64, kernel_size=12, activation='relu', kernel_initializer='glorot_normal')(input1)
	pool11 = MaxPooling1D(pool_size=4)(A1)
	A2 = Conv1D(32, kernel_size=6, activation='relu', kernel_initializer='glorot_normal')(pool11)
	pool12 = MaxPooling1D(pool_size=2)(A2)
	A3 = Conv1D(16, kernel_size=3, activation='relu', kernel_initializer='glorot_normal')(pool12)
	pool13 = MaxPooling1D(pool_size=2)(A3)

	flattened_wf = Flatten()(pool13)

	input2 = Input(shape=input_shape2,name='input2')
	B1 = Conv1D(16, kernel_size=12, activation='relu', kernel_initializer='glorot_normal')(input2)
	pool21 = MaxPooling1D(pool_size=2)(B1)
	B2 = Conv1D(8, kernel_size=6, activation='relu', kernel_initializer='glorot_normal')(pool21)
	pool22 = MaxPooling1D(pool_size=2)(B2)
	B3 = Conv1D(2, kernel_size=3, activation='relu', kernel_initializer='glorot_normal')(pool22)
	pool23 = MaxPooling1D(pool_size=2)(B3)

	flattened_fft = Flatten()(pool23) 

	concatted = concatenate([flattened_wf, flattened_fft], axis = 1)

	den1 = Dense(20, activation='relu', kernel_initializer='glorot_normal')(concatted)
	den2 = Dense(10, activation='relu', kernel_initializer='glorot_normal')(den1)
	den3 = Dense(1)(den2)
	pred = Activation('sigmoid', name='linear')(den3)

	model = Model(inputs=[input1,input2],outputs=[pred])

	model.compile(loss='binary_crossentropy', optimizer='adam',metrics=[
			keras_metrics.false_positive(), keras_metrics.true_negative(),
			keras_metrics.false_negative(), keras_metrics.true_positive(),
			keras.metrics.binary_accuracy
		])
	return model

def get_early_stop():
	return EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto', restore_best_weights=True)	
