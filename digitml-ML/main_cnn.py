from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from numpy import argmax

import numpy as np
import cv2
import os

# # load train and test dataset
# def load_dataset():
#     (trainX, trainY), (testX, testY) = mnist.load_data()
#     trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
#     testX = testX.reshape((testX.shape[0], 28, 28, 1))
#     trainY = to_categorical(trainY)
#     testY = to_categorical(testY)
#     return trainX, trainY, testX, testY

# # scale pixels
# def prep_pixels(train, test):
#     train_norm = train.astype('float32') / 255.0
#     test_norm = test.astype('float32') / 255.0
#     return train_norm, test_norm

# def define_data_augmentation():
#     # Crea un generatore di immagini con augmentation
#     datagen = ImageDataGenerator(
#         rotation_range=10,          # rotazione casuale dell'immagine
#         width_shift_range=0.1,      # traslazione orizzontale
#         height_shift_range=0.1,     # traslazione verticale
#         shear_range=0.2,            # distorsione geometrica
#         zoom_range=0.2,             # zoom casuale
#         horizontal_flip=False,      # non fare flip orizzontale
#         fill_mode='nearest'         # metodo per riempire i pixel mancanti
#     )
#     return datagen
# # define cnn model
# def define_model():
#     model = Sequential()
#     model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
#     model.add(BatchNormalization())
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
#     model.add(BatchNormalization())
#     model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform'))
#     model.add(MaxPooling2D((2, 2)))
#     model.add(Flatten())
#     model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
#     model.add(BatchNormalization())
#     model.add(Dense(10, activation='softmax'))
#     opt = SGD(learning_rate=0.01, momentum=0.9)
#     model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# # # evaluate a model using k-fold cross-validation

# def evaluate_model(dataX, dataY, n_folds=5):
#     print("Avvio della valutazione del modello")
#     scores, histories = list(), list()
#     kfold = KFold(n_folds, shuffle=True, random_state=1)
#     datagen = define_data_augmentation()
#     for train_ix, test_ix in kfold.split(dataX):
#         print("Allenamento del fold")
#         model = define_model()
#         trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
#         datagen.fit(trainX)  # Adatta il generatore ai dati di addestramento

#         # EarlyStopping per fermare l'allenamento se la validazione non migliora
#         early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

#         # Allenamento con il generatore di dati
#         history = model.fit(datagen.flow(trainX, trainY, batch_size=32), epochs=10, validation_data=(testX, testY), callbacks=[early_stopping], verbose=0)
        
#         _, acc = model.evaluate(testX, testY, verbose=0)
#         print('> %.3f' % (acc * 100.0))  # Stampa l'accuratezza per il fold
#         scores.append(acc)
#         histories.append(history)

#     print("Valutazione completata")
#     return scores, histories


# # plot diagnostic learning curves
# def summarize_diagnostics(histories):
#     for i in range(len(histories)):
#         plt.subplot(2, 1, 1)
#         plt.title('Perdita di Entropia Cross')
#         plt.plot(histories[i].history['loss'], color='blue', label='allenamento')
#         plt.plot(histories[i].history['val_loss'], color='orange', label='test')
#         plt.subplot(2, 1, 2)
#         plt.title('Accuratezza di Classificazione')
#         plt.plot(histories[i].history['accuracy'], color='blue', label='allenamento')
#         plt.plot(histories[i].history['val_accuracy'], color='orange', label='test')
#     plt.show()

# # summarize model performance
# def summarize_performance(scores):
#     print('Accuratezza: media=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
#     plt.boxplot(scores)
#     plt.show()

# def run_test_harness():
#     try:
#         trainX, trainY, testX, testY = load_dataset()
#         trainX, testX = prep_pixels(trainX, testX)
#         scores, histories = evaluate_model(trainX, trainY)
#         datagen = define_data_augmentation()
#         summarize_diagnostics(histories)
#         summarize_performance(scores)
#         print("Preparazione per salvare il modello finale")
#         # Addestramento sul dataset completo prima di salvare
#         model = define_model()
#         # allenamento
#         early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
#         model.fit(datagen.flow(trainX, trainY, batch_size=32), epochs=10, validation_data=(testX, testY), callbacks=[early_stopping], verbose=0)
#         # salva il modello
#         model.save('final_model.h5')
#         print("Modello salvato correttamente in 'final_model.h5'")
#     except Exception as e:
#         print("Si è verificato un errore:", e)

# run_test_harness()



# Carica e prepara l'immagine
def load_image(filename):
    # Carica l'immagine a colori
    img = load_img(filename, target_size=(28, 28))
    img = img_to_array(img)
    
    # Converti l'immagine in scala di grigi usando la formula percepita: 0.299*R + 0.587*G + 0.114*B
    img_gray = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    
    # Riformatta per avere il canale come richiesto dal modello
    img_gray = np.expand_dims(img_gray, axis=-1)  # Aggiungi il canale
    
    # Controlla se il testo è bianco su sfondo scuro o viceversa
    # Se lo sfondo è chiaro, inverti i colori
    if np.mean(img_gray) > 127:
        img_gray = 255 - img_gray
    
    # Riformatta in un singolo campione con 1 canale (28x28x1)
    img_gray = img_gray.reshape(1, 28, 28, 1)
    
    # Normalizza i pixel
    img_gray = img_gray.astype('float32') / 255.0
    
    return img_gray



# Carica il modello
model = load_model('final_model.h5')

# Inizializza il contatore delle immagini
image_number = 0

# Ciclo su tutte le immagini nella cartella digits_miei
while os.path.isfile('digit_paolo/digit{}.jpg'.format(image_number)):
    try:
        # Carica e prepara l'immagine
        img = load_image('digit_paolo/digit{}.jpg'.format(image_number))
        
        # Predici la classe dell'immagine
        prediction = model.predict(img)
        
        # Stampa la previsione
        predicted_class = np.argmax(prediction)
        print(f"The number in digit{image_number}.png is probably a {predicted_class}")
        print("Prediction probabilities:", prediction)
        
        # Mostra l'immagine
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        
        # Incrementa il numero per passare alla prossima immagine
        image_number += 1
    except Exception as e:
        print(f"Error reading image digit{image_number}.png: {e}")
        image_number += 1