import os ,sys
import glob
import numpy as np
import keras
import pandas as pd
import random
from keras.models import Sequential  
from pandas import Series, DataFrame 
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Input, BatchNormalization, Bidirectional, LSTM, Dense, Dropout, Conv2D, MaxPooling2D, Reshape
from keras.layers.advanced_activations import PReLU
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from keras.callbacks import ModelCheckpoint
from six.moves import range
from PIL import Image
from keras.initializers import glorot_uniform

Data = []
Label = []
type1_path  = "../../../projectA/920A/type1/"
type2_path  = "../../../projectA/920A/type2/"
type3_path  = "../../../projectA/920A/type3/"

#Preprocessing

#依序讀取三個資料夾中所有檔案
#將檔案進行處理 1024000筆資料與768000筆資料拆成兩筆檔案
#如果512000筆則直接存入Data這個List當中
#將相對應的Label值(1~3)存入Label這個List當中
feature_ms = 51200
pathList = [type1_path, type2_path, type3_path] 
for num, path in enumerate(pathList):
	for i in range(len(os.listdir(path))):
		data = pd.read_csv(path+os.listdir(path)[i], header = None).values
		if data.shape[0]==512000:
			data = np.pad(data, ((0,512000), (0,0)), 'constant')
			data = data.reshape((-1, feature_ms))
			Data.append(data)
			Label.append(num)
		elif data.shape[0]==1024000:
			data = data.reshape((-1, feature_ms))
			Data.append(data)
			Label.append(num)
		elif data.shape[0]==768000:
			data = np.pad(data, ((0,256000), (0,0)), 'constant')
			data = data.reshape((-1, feature_ms))
			Data.append(data)
			Label.append(num)
		else:
			print("Delete type{}: ".format(num+1) ,'No.' , i ,'Data number: ' ,data.shape[0])
			print(os.listdir(path)[i])
					
		
#將Data與Label綁定並隨機打亂
#再將處理後的List 變回Data_after_random與Label_after_random這兩個Array中
#Data_after_random的格式為(337,512000,1)
#Label_after_random的格式為(337,1)
Data_with_Label = list(zip(Data,Label))
random.shuffle(Data_with_Label)
Data_after_random, Label_after_random = zip(*Data_with_Label)
Data_after_random = np.array(Data_after_random)
Label_after_random = np.array(Label_after_random)

train_data_num = 270 #訓練資料個數
train_data = Data_after_random[ 0 : train_data_num ]
train_labels = Label_after_random[ 0 : train_data_num ]
validation_data = Data_after_random[ train_data_num : len(Data_with_Label) ]
validation_labels = Label_after_random[ train_data_num : len(Data_with_Label) ]

# one-hot
train_labels = np.eye(3)[train_labels]
validation_labels = np.eye(3)[validation_labels]

print('')
print('> Data.shape= ',Data_after_random.shape)
print('> Label.shape= ',Label_after_random.shape)
print('> total:' , len(Data_after_random) ,'train:' , len(train_data) , 'validation:' , len(validation_data) )
print('')
print('> train_data_shape: ',train_data.shape)
print('> train_labels_shape: ',train_labels.shape)
print('> validation_data_shape: ',validation_data.shape)
print('> validation_labels_shape: ',validation_labels.shape)
print('')

#model

model = Sequential()

# LSTM
outputLSTM = int(np.round(np.sqrt(512000 * 3)/2))
model.add(BatchNormalization(input_shape=train_data.shape[1:]))
model.add(Bidirectional(LSTM(outputLSTM,
    return_sequences=False,      # True: output at all steps. False: output as last step.
	activation = 'tanh'
)))

# Dense
model.add(BatchNormalization())
model.add(Dense(3, activation='softmax', kernel_initializer = glorot_uniform(seed=0)))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Checkpoint
filepath="../../home/t125605/saved_weights/weights-{epoch:003d}epochs-{loss:.3f}loss-{val_loss:.3f}val_loss.h5"
checkpoint = ModelCheckpoint(filepath, save_weights_only=True)
callbacks_list = [checkpoint]

# Fitting
numEpoch = 30
#history = model.fit(train_data, train_labels, epochs=numEpoch, callbacks=callbacks_list, validation_data=(validation_data, validation_labels), batch_size=32)
History = model.fit(train_data, train_labels, epochs=numEpoch, validation_data=(validation_data, validation_labels), batch_size=32)
history = History.history
import pickle
with open('../../home/t125605/saved_history/trainHistoryDict', 'wb') as file_pi:
	pickle.dump(history, file_pi)

indexTrainAcc_0p85 = np.argmax(np.array(history['acc'])>=0.85)
print(indexTrainAcc_0p85 + 1, 'epoch reaches train_acc of 85 %')
print('\ttrain_loss:', "%.3f" % np.array(history['loss'])[indexTrainAcc_0p85])
print('\tval_acc:', "%.3f" % np.array(history['val_acc'])[indexTrainAcc_0p85])
print('\tval_loss:', "%.3f" % np.array(history['val_loss'])[indexTrainAcc_0p85])
print('Epoch', numEpoch)
print('\ttrain_acc', "%.3f" % np.array(history['acc'])[-1])
print('\ttrain_loss', "%.3f" % np.array(history['loss'])[-1])
print('\tval_acc', "%.3f" % np.array(history['val_acc'])[-1])
print('\tval_loss', "%.3f" % np.array(history['val_loss'])[-1])