import keras
import keras_metrics
from keras.callbacks import EarlyStopping
from keras.layers import Input, Dense
from keras.models import Model
from keras.utils import plot_model
from keras.layers import Flatten, Dense, Reshape, MaxPooling1D, MaxPooling2D, Dropout, concatenate, Activation, Add
from keras import regularizers

def build_nn(input_shape1):
	input1 = Input(shape=input_shape1,name='input1')
	A1 = Dense(units=256,kernel_regularizer=regularizers.l1_l2(l1=0.0, l2=5e-4),activation='relu')(input1)
	A2 = Dropout(rate=0.3)(A1)
	A3 = Dense(units=256,kernel_regularizer=regularizers.l1_l2(l1=0.0, l2=5e-4),activation='relu')(A2)
	A4 = Dropout(rate=0.3)(A3)
	A5 = Dense(units=2,kernel_regularizer=regularizers.l1_l2(l1=0.0, l2=5e-4),activation='relu')(A4)

	pred = Activation('softmax', name='linear')(A5)

	model = Model(inputs=[input1],outputs=[pred])

	model.compile(loss='binary_crossentropy', optimizer='adam',metrics=[
			keras_metrics.false_positive(), keras_metrics.true_negative(),
			keras_metrics.false_negative(), keras_metrics.true_positive(),
			keras.metrics.binary_accuracy
		])
	return model

def get_early_stop():
	return EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto', restore_best_weights=True)	
