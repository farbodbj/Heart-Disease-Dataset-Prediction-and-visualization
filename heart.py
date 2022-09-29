#importing packages
import pandas as pd
from matplotlib import pyplot as plt
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
import numpy as np

#Just for ignoring tensorflow's warnings and info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


#produce dataset handle
#remember to set the index_col=0 otherwise you will see an unnamed: 0 column in your data
HeartData=pd.read_csv('heart.csv',index_col=0)


                                                        ###########       DATA VISUALIZATION PART          ###########

#Set figure size
plt.figure(1,figsize=[8,7.5])

#Set subplots layout
plt.subplots_adjust(left=0.1,
                    bottom=0.08,
                    right=0.9,
                    top=0.93,
                    wspace=0.7,
                    hspace=0.86)

#Normalizes a list of (R,G,B,A)-like tuples where every number in tuple is between 0 and 255                  
def to_RGB(list1):
    tmp1=[]
    RGB=[]
    for tuple1 in list1:
        tmp1.clear()
        for item in tuple1:
            tmp1.append(item/255)
        RGB.append(tuple(tmp1))
    return RGB

#Units and attributes for histogram chart
hists=['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
units=['year','mm HG','mm/dl','Heart beat','ST']

#Puts every subplot in its place
for counter in range(5):
    #Specifies the place of each subplot
    plt.subplot(4,3,counter+1)
    #Specifies histogram features
    plt.hist(HeartData[hists[counter]],bins=20,color=None,histtype='barstacked',edgecolor='black',linewidth=0.5)
    #Specifies the label features
    plt.ylabel(units[counter],fontdict={'fontsize':10})
    #Specifies the title and its features
    plt.title(hists[counter],fontdict={'fontsize':10})

#Units and attributes for pie chart
pies=['Sex','ChestPainType','RestingECG','ST_Slope','FastingBS','ExerciseAngina','HeartDisease']

#Colors for the pie chart
colors=to_RGB([(99, 62, 187),(190, 97, 202),(242, 188, 94),(241, 60, 89)])

#Puts every subplot in its place
for counter in range(7):
    #Preats a list of pie chart features
    key=list(HeartData[pies[counter]].value_counts().to_dict().keys())
    #A dict for the abbreviations used in the pie charts
    kv={'M':'Male','F':'Female','TA': 'Typical Angina', 'ATA': 'Atypical Angina', 'NAP': 'Non-Anginal Pain', 'ASY': 'Asymptomatic',
    'Normal': 'Normal', 'ST': 'having ST-T wave abnormality', 'LVH': 'probable or definite left ventricular hypertrophy',
    'Up': 'upsloping', 'Flat': 'flat', 'Down': 'downsloping','Y': 'Has heart condition', 'N': 'No heart condition',1:'heart disease', 0:'non heart disease',2:'Fasting Blood Sugar < 120 mg/dl',4:'Fasting Blood Sugar > 120 mg/dl'}
    #Specifies the place of each subplot    
    plt.subplot(4,3,counter+6)
    #creates and specifies the features of the pie chaer
    plt.pie(HeartData[pies[counter]].value_counts(),radius=1.5,textprops={'fontsize':8},autopct='%1.1f%%',colors=colors,wedgeprops={"edgecolor" : "black",'linewidth': 0.5,'antialiased': True})
    #creats the legend for the pie charts
    plt.legend(loc='best', bbox_to_anchor=(0.9, -0.1, 0.5, -0.05),fontsize='xx-small',labels=[kv[item] for item in key],facecolor=to_RGB([(211,211,211)])[0])

#Shows the plots in one figure
plt.show()


                                                ###########               THE NEURAL NETWORK PART                     ###########

#Removing NaN values from the dataset
HeartData=HeartData.dropna()

#Converting label characters to numbers for the NN to use
HeartData['Sex']=HeartData['Sex'].replace({'M':0,'F':1})
HeartData['ChestPainType']=HeartData['ChestPainType'].replace({'TA':0,'ATA':1,'NAP':2,'ASY':3})
HeartData['RestingECG']=HeartData['RestingECG'].replace({'Normal':0,'ST':1,'LVH':2})
HeartData['ST_Slope']=HeartData['ST_Slope'].replace({'Up':0,'Flat':1,'Down':2})
HeartData['ExerciseAngina']=HeartData['ExerciseAngina'].replace({'Y':0,'N':1})


#Spliting train and test data -----> 85% train and 15% test. Note that this also could be done by sklearn. 
x_train=HeartData.sample(frac=0.85,random_state=0)
x_test=HeartData.drop(x_train.index)

x_train.reset_index()

#Making two pandas dataframes that do not contain the 'HeartDisease' column
train_labels=x_train.copy()
train_labels.pop('HeartDisease')

test_labels=x_test.copy()
test_labels.pop('HeartDisease')

#Prints some statistics about the dataframe
print(x_train.describe().transpose()[['mean', 'std']])

#Model builder function, note that the hyper parameters have been set using keras_tuner
def MyModel():
    #Activation function for all layers is set to elu
    hp_act='elu'
    #Learning rate (obtained by keras_tuner)
    hp_lr=tf.constant(0.0504, name='learning_rate')
    #Making the normalizing layer
    normalizer=preprocessing.Normalization()
    normalizer.adapt(np.array(train_labels))
    #Empty model object is created
    model=keras.Sequential()
    model.add(normalizer)                                        #Normalizer layer
    model.add(layers.Dense(units=15))                            #First hidden layer units=15
    model.add(layers.Dense(units=20,activation=hp_act))          #Second hiddent layer units=20
    model.add(layers.Dense(units=75,activation=hp_act))          #Third hidden layer units=75
    model.add(layers.Dense(units=75,activation=hp_act))          #Fourht hidden layer units=75
    model.add(layers.Dense(units=1,activation=hp_act))           #Output layer (a number between 0 and 1 s.t 0=no heart disease 1=heart disease)

    #Defining the loss function to be optimized (also obtained from tuner)
    loss=keras.losses.MeanAbsoluteError()
    #By trial and Error I understood that Adadelta had better results that Adam, Nadam, SGD etc. note that 'rho' is NOT set to default=(0.95)
    optim=keras.optimizers.Adadelta(learning_rate=hp_lr,rho=0.87, epsilon=1e-07)
    #The model is compiled
    model.compile(optimizer=optim, loss=loss,metrics=['accuracy'])

    return model

model=MyModel()

#Setting an early stop callback on val_loss
earlyStopper = EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True,min_delta=0.004)
#Fitting and Training the model to our train data (85% of the dataset)
print("Processing")
model.fit(
    train_labels,
    x_train['HeartDisease'],
    epochs=200,
    batch_size=16, #Although this value for batchsize significantly increases the calculation time it seems to be good for val_accuracy and test_accuracy
    verbose=1,
    shuffle=True,
    # Calculate validation results on 10% of the training data
    validation_split = 0.10,
    callbacks=earlyStopper)

model.summary()

#Testing the data to our test data (15% of the whole dataset)
model.evaluate(test_labels,x_test['HeartDisease'],verbose=1)
model.summary()
print('\n\n')
#A random row is selected from the dataframe and fed into the NN to show it is working
random_data=HeartData.sample(1)
print('selecting a random row from the data frame...\n\n')
print(random_data,sep='\n')
random_data=random_data.drop('HeartDisease',axis=1)
print("Prediction in progress...\n\n")
prob=float(model.predict(random_data))
print('PREDICTIION READY!\n\n')
print(f"Deep Neural Network output: {prob}")

#Output explanation
if prob>0.8:
    print("This person will most probably be diagnosed with some kind of heart disease.")
elif prob<0.5:
    print("This person will most probably NOT be diagnosed with some kind of heart disease.")
else:
    print("The model couldn't predict for sure if the person will or will not be diagnosed with heart disease.")


