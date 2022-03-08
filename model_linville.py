import keras
import keras_metrics
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model
from keras.layers import LSTM,Conv1D, Conv2D, Flatten, Dense, Reshape, MaxPooling1D, MaxPooling2D, Dropout, concatenate, Activation, Add

def build_nn(input_shape1):
	input1 = Input(shape=input_shape1,name='input1')
	A1 = LSTM(40,return_sequences=True)(input1)
	A2 = LSTM(40,return_sequences=True)(A1)
	A3 = LSTM(80,return_sequences=True)(A2)
	A4 = LSTM(80,return_sequences=True)(A3)

	flattened_spec = Flatten()(A4)

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
