import numpy as np
import sys
import os
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Reshape
from keras.layers import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.optimizers import Adam
from keras import backend as K
K.set_image_dim_ordering('th')
batch_size = 64
#load all labeled data and classify
all_label = np.load('label.npy')
all_label = all_label.astype(np.float) / 255
unlabel = np.load('unlabel.npy')
unlabel = unlabel.astype(np.float) / 255
testing = np.load('test.npy')
#generate training & validation data
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
#####atuoencoder
input_img = Input(shape=(3,32,32))
x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', input_shape=(3,32,32))(input_img)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, output=encoded)
autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy') 
autoencoder.fit(train_X, train_X,
	nb_epoch=10,
	batch_size=128,
	shuffle=True,
	validation_data=(valid_X, valid_X))
encoded_train = encoder.predict(train_X)
encoded_validation= encoder.predict(valid_X)
encoder.save('encoder')

#construct training model
model = Sequential()
            
model.add(Reshape((256,), input_shape=(16,4,4)))
model.add(Dense(256, input_shape=(256,)))
model.add(BatchNormalization())
model.add(Activation('elu'))
            
model.add(Dense(128))
model.add(BatchNormalization())
model.add(Activation('elu'))
            
model.add(Dense(64))
model.add(BatchNormalization())
model.add(Activation('elu'))
            
model.add(Dense(32))
model.add(BatchNormalization())
model.add(Activation('elu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
checkpointer1= ModelCheckpoint(sys.argv[1], monitor='val_acc', verbose=0, save_best_only=True, mode='max')
earlystop= EarlyStopping(monitor='val_acc', patience= 15, mode= 'max')
        
model.fit(encoded_train, train_Y, batch_size= batch_size, shuffle=True, nb_epoch=50,verbose=1, 
validation_data= (encoded_validation, valid_Y), callbacks= [earlystop, checkpointer1])
        
for iteration in range(10):

	if os.path.isfile('auto_model'):
		model= load_model('auto_model')
		print("found current_model !!")
 
	model.fit(encoded_train, train_Y, batch_size= batch_size, shuffle=True, nb_epoch=50,verbose=1, 
	validation_data= (encoded_validation, valid_Y), callbacks= [earlystop, checkpointer1])
        
#print (history.history)
score = model.evaluate(encoded_validation, valid_Y, verbose=0)
print('Validation score:', score[0])
print('Validation accuracy:', score[1])
#model.save('test_model.h5')
#del model
  
for i in xrange(3):     
	encoded_test= encoder.predict(unlabel)
	Y_test= model.predict(encoded_test, batch_size=batch_size, verbose= 0)
	index= np.argwhere(Y_test>0.95)
	X_index= []
	Y_max= []
	for ii in index:
		X_index=np.append(X_index, ii[0])
		Y_max=np.append(Y_max, max(Y_test[ii[0]]))
	X_index= np.array(X_index, dtype=int)
	Y_max= np.array(Y_max, dtype=float)
	X_new= encoded_test[X_index]
	Y_new= np.equal(Y_test[X_index], Y_max.reshape(Y_max.shape[0], 1)).astype(float)
        
	X_new= np.concatenate((encoded_train, X_new), axis=0)
	Y_new= np.concatenate((train_Y, Y_new), axis=0)
	print (encoded_train.shape, X_new.shape)
        
	checkpoint= ModelCheckpoint(sys.argv[1], monitor='val_acc', verbose=0, save_best_only=True, mode='max')
	earlystop= EarlyStopping(monitor='val_acc', patience= 15, mode= 'max')

	model.fit(X_new, Y_new, batch_size= batch_size, shuffle=True, nb_epoch=20,verbose=1, 
                validation_data= (encoded_validation, valid_Y), callbacks= [earlystop, checkpoint])
        
score = model.evaluate(encoded_validation, valid_Y, verbose=0)
print('Validation score:', score[0])
print('Validation accuracy:', score[1])
