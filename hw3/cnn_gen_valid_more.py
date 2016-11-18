import numpy as np
from keras import backend as K
from timeit import default_timer as timer
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import adam, SGD
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import sys
import os
os.environ["THEANO_FLAGS"]="device=gpu0"
K.set_image_dim_ordering("th")
#NOTE restrict GPU loading
#config = tf.ConfigProto()
#config.gpu_options.per_process_gpu_memory_fraction = 0.5
#K.set_session(tf.Session(config=config))

#load all labeled data and classify 
all_label = np.load('label.npy')		#10,500,3072
unlabel = np.load('unlabel.npy')  	#45000, 3072
testing = np.load('test.npy')		 	#10000, 3072

#create training & validation data
train_X = np.zeros((10,400,3072))
valid_X = np.zeros((10,100,3072))
train_Y = np.zeros((4000,10))
valid_Y = np.zeros((1000,10))
for i in xrange(10):
	train_X[i][0:400] = all_label[i][0:400]
	valid_X[i][0:100] = all_label[i][400:500]
	train_Y.T[i][i*400:i*400+400] = 1
	valid_Y.T[i][i*100:i*100+100] = 1
train_X = np.reshape(train_X, (4000,3,32,32))
valid_X = np.reshape(valid_X, (1000,3,32,32))
unlabel = np.reshape(unlabel, (45000,3,32,32))
testing = np.reshape(testing, (10000,3,32,32))
#generate image data
datagen = ImageDataGenerator(
	#shear_range = 0.1,
	zoom_range=0.05,
	rotation_range=1,  
	width_shift_range=0.1,  
	height_shift_range=0.1,  
	horizontal_flip=True, 
	vertical_flip=False,
	dim_ordering="th")

datagen.fit(train_X)
checkpointer1 = ModelCheckpoint(filepath=sys.argv[1],monitor='val_acc', verbose=1,save_best_only=True)
#checkpointer2 = ModelCheckpoint(filepath="model_22a",monitor='val_acc', verbose=1,save_best_only=True)
#checkpointer3 = ModelCheckpoint(filepath="model_22a",monitor='val_acc', verbose=1,save_best_only=True)
#valid_gen = datagen.flow(valid_X, valid_Y, batch_size = 100)

#set model parameters
model = Sequential()
model.add(Convolution2D(32,3,3,border_mode='same',input_shape=(3,32,32), dim_ordering='th'))
model.add(BatchNormalization())
model.add(Activation('elu'))		#1
model.add(Convolution2D(32,3,3))
model.add(BatchNormalization())
model.add(Activation('elu'))		#2
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))

#more layers
model.add(Convolution2D(64, 3, 3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('elu'))		#3
model.add(Convolution2D(64, 3, 3))
model.add(BatchNormalization())
model.add(Activation('elu'))		#4
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

#more
model.add(Convolution2D(64, 3, 3,border_mode='same'))
model.add(BatchNormalization())
model.add(Activation('elu'))		#5
model.add(Convolution2D(64, 3, 3))
model.add(BatchNormalization())
model.add(Activation('elu'))		#6
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Dense(128))		
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Dropout(0.5))
model.add(Dense(10))					
model.add(Activation('softmax'))	

model.summary()
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
ADAM = adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08,decay=0.0)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#model.fit(train_X, train_Y, batch_size=100, nb_epoch=30, validation_data=(valid_X, valid_Y), shuffle=True)
model.fit_generator(datagen.flow(train_X, train_Y, batch_size=64), samples_per_epoch=train_X.shape[0]*10, nb_epoch=30, validation_data=(valid_X, valid_Y),callbacks=[checkpointer1],verbose=1)
#NOTE without validation, the result may not converge everytime.
##### however, the loss hovering around still
##### decrease the dropout rate to let some fitting happen
##### learning rate should less than 0.001 

#self-training process
for i in xrange(6):
	new_train_X = train_X
	new_train_Y = train_Y
	predict = model.predict(unlabel)

	row_max = np.amax(predict, axis=1)
	index = row_max >= 0.99-(0.008*i)
	append_row = unlabel[index]
	append_ans = predict[index]
	bigger_idx = append_ans >= 0.99-(0.008*i)
	smaler_idx = append_ans < 0.99-(0.008*i)
	append_ans[bigger_idx] = 1
	append_ans[smaler_idx] = 0
	new_idx = len(new_train_X) + len(append_row)
	new_train_X = np.append(new_train_X, append_row).reshape(new_idx,3,32,32)
	new_train_Y = np.append(new_train_Y, append_ans).reshape(new_idx,10)
	###############################
	if (i > 2):
		answer  = model.predict(testing)
		row_max = np.amax(answer, axis=1)
		index = row_max >= 0.99-(0.008*i)
		append_row = testing[index]
		append_ans = answer[index]
		bigger_idx = append_ans >= 0.99-(0.008*i)
		smaler_idx = append_ans < 0.99-(0.008*i)
		append_ans[bigger_idx] = 1
		append_ans[smaler_idx] = 0
		new_idx = len(new_train_X) + len(append_row)
		new_train_X = np.append(new_train_X, append_row).reshape(new_idx,3,32,32)
		new_train_Y = np.append(new_train_Y, append_ans).reshape(new_idx,10)
				
	model.fit(new_train_X, new_train_Y, batch_size=64, nb_epoch=16, validation_data=(valid_X, valid_Y),callbacks=[checkpointer1], shuffle=True)
	print ('tuning with label data only ====>') 
	model.fit_generator(datagen.flow(train_X, train_Y, batch_size=64), samples_per_epoch=train_X.shape[0]*5, nb_epoch=30, validation_data=(valid_X, valid_Y),callbacks=[checkpointer1],verbose=1)
	model.fit(train_X, train_Y, batch_size=64, nb_epoch=50, validation_data=(valid_X, valid_Y),callbacks=[checkpointer1], shuffle=True)
