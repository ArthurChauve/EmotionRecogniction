#%%
# Imports
import tensorflow as tf 
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.optimizers import Adam,RMSprop,SGD,Adamax
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout,Activation, Flatten, Conv2D, MaxPooling2D,BatchNormalization
import numpy as np
import seaborn as sn
from google.colab import drive
import pandas 
#%%
drive.mount("/content/gdrive")
data = pandas.read_csv('/content/gdrive/MyDrive/fer2013.csv')
#%%
# We now need to rearrange data. We split it between two set, one that will
# be used for training, and one that will allow us to test the model.

def conv_data():
    Xtrain, ytrain,Xtest,ytest = [], [], [], []
    for i in range (len(data.Usage)):
        if data.Usage[i] == 'Training':
            # Converting image
            current = data.pixels[i]
            img = list(map(int,(str.split(current))))
            # Adding it to Xtrain
            Xtrain.append(np.array(img))
            # Adding the corresponding emotion to ytrain
            ytrain.append(data.emotion[i])
        else : 
            # Converting image
            current = data.pixels[i]
            img = list(map(int,(str.split(current))))
            # Adding it to Xtest
            Xtest.append(np.array(img))
            # Adding the corrresponding emotion to ytest
            ytest.append(data.emotion[i])
    Xtrain,ytrain,Xtest,ytest = np.array(Xtrain), np.array(ytrain), np.array(Xtest), np.array(ytest)
    Xtrain = Xtrain.reshape(28709,48,48,1)
    Xtest = Xtest.reshape(7178,48,48,1)
    Ytrain = []
    Ytest = []
    # Converting the emotions, caracterised by a number k, into a vector where
    #all coordinates are set to 0 exept the kth set to 1
    for i in range (len(ytrain)):
        t = np.zeros(7)
        t[ytrain[i]] = 1 
        Ytrain.append(t)
    for i in range (len(ytest)):
        t = np.zeros(7)
        t[ytest[i]] = 1 
        Ytest.append (t)
    Ytest = np.array(Ytest)
    Ytrain = np.array(Ytrain)
    return Xtrain,ytrain,Xtest,ytest

Xtrain,ytrain,Xtest,ytest = conv_data()
#%%
# We create the function that will compile our models.
def compiler(model):
    model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])
#%%
# Now, we define the function that will train the model

def train(model) : 
    call_list = [keras.callbacks.EarlyStopping(patience = 10)]
    NB_epochs = 100
    BATCH_size = 100
    history = model.fit(Xtrain,
                    ytrain,
                    validation_data=(Xtest, ytest),
                    epochs=NB_epochs,
                    batch_size=BATCH_size,
                    callbacks= call_list
                    )
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
#%%
# We create a first network to see the result.
# Here, we use Conv2D layers because we are working on images.
# We do a max pooling between each Conv2D layer and we end the model by 
# adding a dense layer wich finalise the prediction.
def model1():
    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape = (48,48,1) ,padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(128,(3,3),padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(256,(3,3), padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(512,(3,3),padding = "same", activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add (Dense(64, activation = 'relu'))
    model.add(Dropout(0.3))
    model.add(Dense(7))
    model.add(Activation('softmax'))
    return model 
#%%
# To increase accuracy, we can try data augmentation 
# we need to randomly modify the images without distorting the information.
# That's why I choosed here to allow horizontal_flip, zoom, random rescaling, 
# little rotations and shift.
datagen = keras.preprocessing.image.ImageDataGenerator(rotation_range = 10,
                                         width_shift_range = 0.2,
                                         height_shift_range = 0.2,
                                         horizontal_flip = True,
                                         rescale = 1./255,
                                         zoom_range = 0.2,
                                         )
#%% 
# Now we can train models with it
def train_with_data_augment(model):
    call_list = [keras.callbacks.EarlyStopping(monitor = "val_accuracy",patience = 15)]
    NB_epochs = 100 
    BATCH_size = 64
    history = model.fit(x = datagen.flow(Xtrain,ytrain,batch_size=BATCH_size),
                    steps_per_epoch = 28709//BATCH_size,
                    validation_data=(Xtest/255, ytest),
                    epochs=NB_epochs,
                    callbacks= call_list
                    )
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'])
 #%%
# Here we define another model.
def model2():
    model = Sequential()
    model.add(Conv2D(64, (3,3), input_shape = (48,48,1),padding = "same" ,activation = 'relu'))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(64,(3,3), activation = 'relu',padding = "same" ))
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Conv2D(64,(3,3), activation = 'relu',padding = "same" ))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size = (2,2)))
    model.add(Flatten())
    model.add(Dropout(0.2))
    model.add (Dense(64, activation = 'relu'))
    model.add(Dense(7))
    model.add(Activation('softmax'))
#%%
# We can now also make prediction on our own images
def predict(img,model):
    return np.argmax(model.predict(img.reshape((1,48,48,1))))
#%% 
# Lastly, we can show the confusion matrix to see 
# model's strengths and weaknesses
def matrice(model):
    ypred = model.predict(Xtest)
    matrix = confusion_matrix(ytest.argmax(axis=1), ypred.argmax(axis=1),normalize = 'pred')
    confuse = pd.DataFrame(matrix, ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"] , 
        ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]) 
    sn.set(font_scale=1.4)
    sn.heatmap(confuse, cmap="YlGnBu", annot=True, annot_kws={"size": 10})
    plt.show()
