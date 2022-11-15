"""
n11098279, Xuefei, Li

"""

###import all the necessary libraries for this assignment
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications.mobilenet_v2 import preprocess_input
from keras.models import Model
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image
import random
from keras.models import Model



def task_1():
    """
    task_1:
        download the small flower dataset from blackboard
    """
    print("The small flower dataset has been downloaded successfully from Blackboard.")
    
def task_2():
    """
    task_2:
        Download a pretrained MobileNetV2 network;
        Create the base model from the pre-trained model MobileNet V2 witout the last layer
    
    """
    base_model = tf.keras.applications.MobileNetV2(input_shape=(224, 224,3),
                                                   include_top=False,
                                                   weights='imagenet')
    print(base_model.summary())
    print(len(base_model.layers))
    return base_model
    

def task_3(base_model):
    """
    task_3:
        build customized model:
        Set MobileNetV2 base_model untrainable(2257984 untrainable params).
        Add GlobalAveragePooling2D layer to reduce overfitting(no parameters to train).
        Add Flatten layer to reshape output fit new last layer(no parameters to train).
        Add Dropout layer to prevent overfitting(no parameters to train).
        Add Dense layer as new last layer with 5 units represent 5 classification classes
            and activation="softmax" to calculate actual values to probabilities, the highest
            probability will be considered as the actual prediction  result in classification model(6405 trainable params).
    """
    
    base_model.trainable = False   
    model = tf.keras.Sequential([
      base_model,
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Flatten(),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(5, activation="softmax"),
    ])
    print(model.summary())
    print(len(model.layers))
    return model
    
def task_4():
    """
    task_4:
        prepare training, validation and test sets for non-accelerated version of transfer learning:
        main steps:
            1.get image from flower image dataset and pair with an integer label which represent flower category.
            2.randomly shuffle labelled image dataset.
            3.split image dataset into training_dataset(70%), validation_dataset(15%), test_dataset(15%).
            4.use img_to_array and preprocess_input to format input data as model input layer required..
            5.use keras.utils.to_categorical to format labels to binary class matrices
            6.return x_train,y_train,x_validation,y_validation,x_test,y_test
    
    """
    # all the categories of flowers images in flower dataset
    categories=['daisy','dandelion','roses','sunflowers','tulips']
    # using dictionary to pair categories with integers
    labelsDic={'daisy':1,'dandelion':2,'roses':3,'sunflowers':4,'tulips':5}
    
    # flower dataset directory
    dataset_Dir=r"C:\Users\xuefe\Desktop\master degree study\semester2-2022\IFN680\assessment2\small_flower_dataset"

    img_dataset=[]
    # put all the images into a image dataset with integer labels
    for category in categories:
        path=os.path.join(dataset_Dir,category) # path to five catigories of flowers
        for img in os.listdir(path):            
            image=Image.open(path+'/'+img)
            newsize=(224,224) # standrdize to a fixed image size 224x224
            image=image.resize(newsize) # resize image data to (224x224)    
            img_dataset.append([image,labelsDic[category]]) # pair image with integer label  
    print(len(img_dataset))
    
    # randomly shuffle image dataset
    random.seed(10)
    random.shuffle(img_dataset)

    print(img_dataset[:5])
    
    # split image dataset to training_dataset(70%), validation_dataset(15%), test_dataset(15%)
    training_dataset=img_dataset[:int(len(img_dataset)*0.7)]
    validation_dataset=img_dataset[int(len(img_dataset)*0.7):int(len(img_dataset)*0.85)]
    test_dataset=img_dataset[int(len(img_dataset)*0.85):int(len(img_dataset))]
    print(len(training_dataset))
    print(len(validation_dataset))
    print(len(test_dataset))
    
    # processing training dataset image into np array format as x_train
    x_train = np.empty((len(training_dataset), 224, 224, 3))
    for i in range(0, len(training_dataset)):
        img_array = keras.utils.img_to_array(training_dataset[i][0])
        x_train[i]=preprocess_input(img_array)        
    
    # processing validation dataset image into np array format as x_validation
    x_validation = np.empty((len(validation_dataset), 224, 224, 3))
    for i in range(0, len(validation_dataset)):
        img_array = keras.utils.img_to_array(validation_dataset[i][0])
        x_validation[i]=preprocess_input(img_array)
       
    # processing test dataset image into np array format as x_test
    x_test = np.empty((len(test_dataset), 224, 224, 3))
    for i in range(0, len(test_dataset)):
        img_array = keras.utils.img_to_array(test_dataset[i][0])
        x_test[i]=preprocess_input(img_array)
       
   
    
    # get labels of training dataset for further processing
    training_labels=[]
    for i in range(0, len(training_dataset)):
        training_labels.append(training_dataset[i][1])
    # convert class vectors to binary class matrices of training dataset labels
    y_train = keras.utils.to_categorical(training_labels,dtype ="int32")
    y_train=y_train[:,1:]
    
    # get labels of validation dataset for further processing
    validation_labels=[]
    for i in range(0, len(validation_dataset)):
        validation_labels.append(validation_dataset[i][1])
    # convert class vectors to binary class matrices of validation dataset labels
    y_validation = keras.utils.to_categorical(validation_labels,dtype ="int32")
    y_validation=y_validation[:,1:]
    
    # get labels of test dataset for further processing
    test_labels=[]
    for i in range(0, len(test_dataset)):
        test_labels.append(test_dataset[i][1])
    # convert class vectors to binary class matrices of test dataset labels
    y_test = keras.utils.to_categorical(test_labels,dtype ="int32")
    y_test =y_test[:,1:]
    
    print(x_train[0].shape)
    print(y_train[0].shape)
    print(x_test[0].shape)
    print(y_test[0].shape)
    return x_train,y_train,x_validation,y_validation,x_test,y_test



def task_5_6(model,x_train,y_train,x_validation,y_validation,x_test,y_test):
    """
    task_5:
        Compile and train your model with an SGD3 optimizer using the following parameters:
        learning_rate=0.01, momentum=0.0, nesterov=False.
    task_6:
        Plot the training and validation errors(loss) by epochs as well as the training and validation
        accuracies.
    
    Also, evaluate the model's performance using test dataset.
    """
     
    # complie model with SGD optimizer using task required parameters.
    model.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.0, nesterov=False),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
    
    #train model with batch_size=32 and epochs=35
    history=model.fit(
        x=x_train,
        y=y_train,
        shuffle=True,
        batch_size=32,
        epochs=35,
        verbose=2,
        validation_data=(x_validation, y_validation),
    )
    
    print(type(history))
    print(type(history.history))
    print(history.history.keys())
    
    #plot accuracy and loss of both taining dataset and validation dataset
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('model training analysis')

    
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('accuracy')
    ax1.legend(['training_set','validation_set'])
    
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('loss')
    ax2.legend(['training_set','validation_set'])
    
    plt.show()
    
    # evaluate model on test dataset
    score = model.evaluate(x_test,y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])
    
    
    
def task_7(base_model,x_train,y_train,x_validation,y_validation,x_test,y_test):
    """
    task_7:
       Experiment with 3 different orders of magnitude for the learning rate. Plot the
       results, draw conclusions.
       build 3 clean model and complie with learning rate: 1, 0.1, 0.0001
       train them with the same training dataset and validation dataset as task_5&6
       Also, evaluate each model in test dataset.
    """
     
    ###build another three models to test different learning rate (1,0.1,0.0001)

    #learning rate=1
    learningRate_model1 = tf.keras.Sequential([
      base_model,
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Flatten(),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(5, activation="softmax"),
    ])
    
    learningRate_model1.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate=1, momentum=0.0, nesterov=False),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    #learning rate=0.1
    learningRate_model2 = tf.keras.Sequential([
      base_model,
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Flatten(),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(5, activation="softmax"),
    ])
    
    learningRate_model2.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.0, nesterov=False),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    #learning rate=0.0001
    learningRate_model3 = tf.keras.Sequential([
      base_model,
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Flatten(),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(5, activation="softmax"),
    ])
    
    learningRate_model3.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.0001, momentum=0.0, nesterov=False),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    
    
    
    #train model(learning rate=1) 
    history1=learningRate_model1.fit(
        x=x_train,
        y=y_train,
        shuffle=True,
        batch_size=32,
        epochs=35,
        verbose=2,
        validation_data=(x_validation, y_validation),
    )
        
    #plot accuracy and loss of model(learning rate=1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('model training analysis with learning_rate=1')

    
    ax1.plot(history1.history['accuracy'])
    ax1.plot(history1.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('accuracy')
    ax1.legend(['training_set','validation_set'])
    
    ax2.plot(history1.history['loss'])
    ax2.plot(history1.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('loss')
    ax2.legend(['training_set','validation_set'])
    
    plt.show()
    
    # evaluate model(learning rate=1) on test dataset
    score1 = learningRate_model1.evaluate(x_test,y_test, verbose=0)
    print("Test loss:", score1[0])
    print("Test accuracy:", score1[1])
 
    
    #train model(learning rate=0.1)
    history2=learningRate_model2.fit(
        x=x_train,
        y=y_train,
        shuffle=True,
        batch_size=32,
        epochs=35,
        verbose=2,
        validation_data=(x_validation, y_validation),
    )
       
    #plot accuracy and loss of model(learning rate=0.1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('model training analysis with learning_rate=0.1')

    
    ax1.plot(history2.history['accuracy'])
    ax1.plot(history2.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('accuracy')
    ax1.legend(['training_set','validation_set'])
    
    ax2.plot(history2.history['loss'])
    ax2.plot(history2.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('loss')
    ax2.legend(['training_set','validation_set'])
    
    plt.show()
    
    # evaluate model(learning rate=0.1) on test dataset
    score2 = learningRate_model2.evaluate(x_test,y_test, verbose=0)
    print("Test loss:", score2[0])
    print("Test accuracy:", score2[1])
    
    
    #train model(learning rate=0.0001)
    history3=learningRate_model3.fit(
        x=x_train,
        y=y_train,
        shuffle=True,
        batch_size=32,
        epochs=35,
        verbose=2,
        validation_data=(x_validation, y_validation),
    )
        
    #plot accuracy and loss of model(learning rate=0.0001)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('model training analysis with learning_rate=0.0001')

    
    ax1.plot(history3.history['accuracy'])
    ax1.plot(history3.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('accuracy')
    ax1.legend(['training_set','validation_set'])
    
    ax2.plot(history3.history['loss'])
    ax2.plot(history3.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('loss')
    ax2.legend(['training_set','validation_set'])
    
    plt.show()
    
    # evaluate model(learning rate=0.0001) on test dataset
    score3 = learningRate_model3.evaluate(x_test,y_test, verbose=0)
    print("Test loss:", score3[0])
    print("Test accuracy:", score3[1])


def task_8(base_model,x_train,y_train,x_validation,y_validation,x_test,y_test):
    """
    task_8:
       According the conclusion from task_7, the best performance is when learning rate=0.1 
       Add a non zero momentum to the training with the SGD optimizer (consider 3 values for the momentum) and analyse the results.
       build 3 clean model and complie with learning rate=0.1, momentum is 0.3, 0.6, 0.9
       train them with the same training dataset and validation dataset as task_5&6
       Also, evaluate each model in test dataset.
    """
    
    ###build another three models to test different momentum value (0.3,0.6,0.9) with learning rete=0.1

    #momentum=0.3
    momentum_model1 = tf.keras.Sequential([
      base_model,
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Flatten(),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(5, activation="softmax"),
    ])
    
    momentum_model1.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.3, nesterov=False),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    #momentum=0.6
    momentum_model2 = tf.keras.Sequential([
      base_model,
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Flatten(),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(5, activation="softmax"),
    ])
    
    momentum_model2.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.6, nesterov=False),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    #momentum=0.9
    momentum_model3 = tf.keras.Sequential([
      base_model,
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Flatten(),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(5, activation="softmax"),
    ])
    
    momentum_model3.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=False),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    
    #train model(momentum=0.3)
    history1=momentum_model1.fit(
        x=x_train,
        y=y_train,
        shuffle=True,
        batch_size=32,
        epochs=35,
        verbose=2,
        validation_data=(x_validation, y_validation),
    )
    
    #plot accuracy and loss of model(momentum=0.3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('model training analysis with learning_rate=0.1, momentum=0.3')

    
    ax1.plot(history1.history['accuracy'])
    ax1.plot(history1.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('accuracy')
    ax1.legend(['training_set','validation_set'])
    
    ax2.plot(history1.history['loss'])
    ax2.plot(history1.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('loss')
    ax2.legend(['training_set','validation_set'])
    
    plt.show()
    
    # evaluate model(momentum=0.3) on test dataset
    score1 = momentum_model1.evaluate(x_test,y_test, verbose=0)
    print("Test loss:", score1[0])
    print("Test accuracy:", score1[1])

    
    #train model(momentum=0.6)
    history2=momentum_model2.fit(
        x=x_train,
        y=y_train,
        shuffle=True,
        batch_size=32,
        epochs=35,
        verbose=2,
        validation_data=(x_validation, y_validation),
    )
    
    #plot accuracy and loss of model(momentum=0.6)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('model training analysis with learning_rate=0.1, momentum=0.6')

    
    ax1.plot(history2.history['accuracy'])
    ax1.plot(history2.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('accuracy')
    ax1.legend(['training_set','validation_set'])
    
    ax2.plot(history2.history['loss'])
    ax2.plot(history2.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('loss')
    ax2.legend(['training_set','validation_set'])
    
    plt.show()
    
    # evaluate model(momentum=0.6) on test dataset
    score2 = momentum_model2.evaluate(x_test,y_test, verbose=0)
    print("Test loss:", score2[0])
    print("Test accuracy:", score2[1])
    
    
    #train model(momentum=0.9)
    history3=momentum_model3.fit(
        x=x_train,
        y=y_train,
        shuffle=True,
        batch_size=32,
        epochs=35,
        verbose=2,
        validation_data=(x_validation, y_validation),
    )
    
    #plot accuracy and loss of model(momentum=0.9)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('model training analysis with learning_rate=0.1, momentum=0.9')

    
    ax1.plot(history3.history['accuracy'])
    ax1.plot(history3.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('accuracy')
    ax1.legend(['training_set','validation_set'])
    
    ax2.plot(history3.history['loss'])
    ax2.plot(history3.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('loss')
    ax2.legend(['training_set','validation_set'])
    
    plt.show()
    
    # evaluate #train model(momentum=0.9) on test dataset
    score3 = momentum_model3.evaluate(x_test,y_test, verbose=0)
    print("Test loss:", score3[0])
    print("Test accuracy:", score3[1])

  
def task_9(base_model,x_train,y_train,x_validation,y_validation,x_test,y_test):
    """
    task_9:
        Re-prepare your training, validation and test sets to avoid duplicate computation in untrainable base_model layer. 
        Using base_model predict to get new input data(x_train, x_validation, x_test)
        The labels is the same as original lables.
    
    """
    x_train_acc=base_model.predict(x_train)
    x_validation_acc=base_model.predict(x_validation)
    x_test_acc=base_model.predict(x_test)
    
    
    y_train_acc=y_train
    y_validation_acc=y_validation
    y_test_acc=y_test
    
    return x_train_acc,y_train_acc,x_validation_acc,y_validation_acc,x_test_acc,y_test_acc


def task_10(x_train_acc,y_train_acc,x_validation_acc,y_validation_acc,x_test_acc,y_test_acc):
    """
    task_10:
        Perform Task 8 on the new dataset created in Task 9.
    """
    
    
    ###build another three models to test different momentum value (0.3,0.6,0.9) with learning rete=0.1
    ###new models are accelerated version transfer learning model without untrainable layer
    #momentum=0.3
    acc_model1 = tf.keras.Sequential([
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Flatten(),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(5, activation="softmax"),
    ])
    
    acc_model1.compile(
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.3, nesterov=False),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])
    
    #momentum=0.6
    acc_model2 = tf.keras.Sequential([
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Flatten(),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(5, activation="softmax"),
    ])
    
    acc_model2.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.6, nesterov=False),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    #momentum=0.9
    acc_model3 = tf.keras.Sequential([   
      keras.layers.GlobalAveragePooling2D(),
      keras.layers.Flatten(),
      keras.layers.Dropout(0.2),
      keras.layers.Dense(5, activation="softmax"),
    ])
    
    acc_model3.compile(
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9, nesterov=False),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    
    
    #train model(momentum=0.3)
    history1=acc_model1.fit(
        x=x_train_acc,
        y=y_train_acc,
        shuffle=True,
        batch_size=32,
        epochs=35,
        verbose=2,
        validation_data=(x_validation_acc, y_validation_acc),
    )
    
    #plot accuracy and loss of model(momentum=0.3)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('accelerated version model training analysis with learning_rate=0.1, momentum=0.3')

    
    ax1.plot(history1.history['accuracy'])
    ax1.plot(history1.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('accuracy')
    ax1.legend(['training_set','validation_set'])
    
    ax2.plot(history1.history['loss'])
    ax2.plot(history1.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('loss')
    ax2.legend(['training_set','validation_set'])
    
    plt.show()
    
    # evaluate model(momentum=0.3) on test dataset
    score1 = acc_model1.evaluate(x_test_acc,y_test_acc, verbose=0)
    print("Test loss:", score1[0])
    print("Test accuracy:", score1[1])
    
    #train model(momentum=0.6)
    history2=acc_model2.fit(
        x=x_train_acc,
        y=y_train_acc,
        shuffle=True,
        batch_size=32,
        epochs=35,
        verbose=2,
        validation_data=(x_validation_acc, y_validation_acc),
    )
    
    #plot accuracy and loss of model(momentum=0.6)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('accelerated version model training analysis with learning_rate=0.1, momentum=0.6')

    
    ax1.plot(history2.history['accuracy'])
    ax1.plot(history2.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('accuracy')
    ax1.legend(['training_set','validation_set'])
    
    ax2.plot(history2.history['loss'])
    ax2.plot(history2.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('loss')
    ax2.legend(['training_set','validation_set'])
    
    plt.show()
    
    # evaluate model(momentum=0.6) on test dataset
    score2 = acc_model2.evaluate(x_test_acc,y_test_acc, verbose=0)
    print("Test loss:", score2[0])
    print("Test accuracy:", score2[1])
    
    
    #train model(momentum=0.9)
    history3=acc_model3.fit(
        x=x_train_acc,
        y=y_train_acc,
        shuffle=True,
        batch_size=32,
        epochs=35,
        verbose=2,
        validation_data=(x_validation_acc, y_validation_acc),
    )
    
    #plot accuracy and loss of model(momentum=0.9)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('accelerated version model training analysis with learning_rate=0.1, momentum=0.9')

    
    ax1.plot(history3.history['accuracy'])
    ax1.plot(history3.history['val_accuracy'])
    ax1.set_title('model accuracy')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('accuracy')
    ax1.legend(['training_set','validation_set'])
    
    ax2.plot(history3.history['loss'])
    ax2.plot(history3.history['val_loss'])
    ax2.set_title('model loss')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('loss')
    ax2.legend(['training_set','validation_set'])
    
    plt.show()
    
    # evaluate model(momentum=0.9) on test dataset
    score3 = acc_model3.evaluate(x_test_acc,y_test_acc, verbose=0)
    print("Test loss:", score3[0])
    print("Test accuracy:", score3[1])



    
#%%
if __name__ == "__main__":
    pass
    #%% task 1
    #task_1()
    
    #%% task 2
    base_model=task_2()
      
    #%% task 3 (need to run with task_2 together)
    model=task_3(base_model)
    
    #%% task 4 (need to run with task_2, task_3 together)
    x_train,y_train,x_validation,y_validation,x_test,y_test=task_4()
    
    #%% task 5&6 (need to run with task_2, task_3, task_4 together)
    #task_5_6(model,x_train,y_train,x_validation,y_validation,x_test,y_test)
    
    #%% task 7 (need to run with task_2, task_3, task_4 together)
    #task_7(base_model,x_train,y_train,x_validation,y_validation,x_test,y_test)
    
    #%% task 8 (need to run with task_2, task_3, task_4 together)
    #task_8(base_model,x_train,y_train,x_validation,y_validation,x_test,y_test)
    
    #%% task 9 (need to run with task_2, task_3, task_4 together)
    x_train_acc,y_train_acc,x_validation_acc,y_validation_acc,x_test_acc,y_test_acc=task_9(base_model,x_train,y_train,x_validation,y_validation,x_test,y_test)

    #%% task 10 (need to run with task_2, task_3, task_4, task_9 together)
    task_10(x_train_acc,y_train_acc,x_validation_acc,y_validation_acc,x_test_acc,y_test_acc)