# Keras Heart Disease predictor
This code first visualizes all 11 features of kaggle's public [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) and then creates a Neural Network model that trains and predicts based on that dataset to predict the possibility of a person having a heart disease. 
The accuracy of the model on test data is about 84-87%. 
I learned most of the syntax and techniques in this project from official keras docs, [this](https://www.youtube.com/watch?v=hvgnX1gbsLA&list=PLqnslRFeH2Uqfv1Vz3DqeQfy0w20ldbaV) and [this](https://www.youtube.com/watch?v=s3kH7_6xF-4) tutorial.
Needed libraries:
- Pandas
- Matplotlib
- Tensorflow
- Numpy
- Keras_tuner (optional)

## heart.csv

This dataset was created by combining different datasets already available independently but not combined before. In this dataset, 5 heart datasets are combined over 11 common features which makes it the largest heart disease dataset available so far for research purposes. The five datasets used for its curation are:

- Cleveland: 303 observations
- Hungarian: 294 observations
- Switzerland: 123 observations
- Long Beach VA: 200 observations
- Stalog (Heart) Data Set: 270 observations

Total: 1190 observations
Duplicated: 272 observations

Final dataset: 918 observations

Every dataset used can be found under the Index of heart disease datasets from UCI Machine Learning Repository on the following link: https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/

**note:** Most features in the dataset are represented with abbreviations which might be unclear, for further clarification refer to dataset description in [this](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) link.
## heart.py

The code in this file is consisted of 2 major parts:
1. Data Visualization using matplotlib module
2. Creating, Training and Testing a Neural Network model using Keras module

### Data Visualization:
The data was first loaded using pandas module and then plotted in one figure and 12 subplots (5 of which were histograms and others were pie charts). 


The columns that contained classification data were plotted using pie charts including: 
```
pies=['Sex','ChestPainType','RestingECG','ST_Slope','FastingBS','ExerciseAngina','HeartDisease']
```
 and for the others histogram was used:
 ```
 hists=['Age','RestingBP','Cholesterol','MaxHR','Oldpeak']
```
The colors, fonts and other plot decorations were obtained by playing around the parameters. Any change in that makes them look better is very much appreciated. :)

**note:** While the plot figure is open the code will NOT move on to the next lines so make sure to close it after looking at the data.

#### **Example visualization ouput: **


![Figure_1](https://user-images.githubusercontent.com/110523279/193018221-a1abfd56-d656-48ed-bcb0-5f0e30fe4440.png)

### Neural Network Model:
**Preparing the data:**
Before getting to creating and training the model the dataset should undergo some minor changes: 
1. Deleting any NaN values using ```DataFrame.dropna()```
2. The alphabetical labels were changed into numbers (e.g M-->0 and F-->1)
3. The dataset was split into train (85%) and test (15%)
4. 
**Creating the model and setting hyper parameters:**
First the model function was created as ```def MyModel(hp)``` and the hyper parameters were tuned using Keras_tuner's Hyperband class. The part for tuning is provided in the code but it is commented out which can be used for finding even better hyperparameters.

RandomSearch and BayesianSearch classes for finding hyperparams were also tested but didn't return such high of a```val_accuracy``` as Hyperband did.

After playing around with hyper parameters, optimizers and loss function about 89-91% of val_accuracy and 83-84% of test_data accuracy was achieved in early tests. By reducing batch_size (which significantly increases calculation time) and also playing around with Adamdelta's parameters I could increase the accuracy on test data to 84-87% percent.

For preventing overfitting and resource overuse an earlystopper was also set and tested multiple times (I changed the patience and min_delta parameters several times to achieve good results) and epochs were set to 200 but that number of epochs is not usually reached. 

After the model summaries are returned a random sample of size 1 is taken from the dataset and is fed into the model for the user to see the actual outputs of the model.
I also set a conditional for explaining the output as follows:

```
if prob>0.8:
    print("This person will most probably be diagnosed with some kind of heart disease.")
elif prob<0.5:
    print("This person will most probably NOT be diagnosed with some kind of heart disease.")
else:
    print("The model couldn't predict for sure if the person will or will not be diagnosed with heart disease.")
```
**IMPORTANT NOTE:** The conditions set here are just for returning a more user-friendly  and understandable output and are NOT MEANT TO BE USED AS MEDICAL ADVICE OR PREDICTION. Needless to say such advice and predictions can only be supplied by qualified enough people. 

## Example output
```                       
                    mean         std
Age              53.512821    9.638716
Sex               0.212821    0.409564
ChestPainType     2.253846    0.929987
RestingBP       132.189744   18.435048
Cholesterol     197.938462  110.933471
FastingBS         2.466667    0.846448
RestingECG        0.601282    0.806742
MaxHR           136.506410   25.774398
ExerciseAngina    0.594872    0.491232
Oldpeak           0.895769    1.046552
ST_Slope          0.653846    0.608602
HeartDisease      0.560256    0.496674
Processing...
Epoch 1/200
702/702 [==============================] - 2s 2ms/step - loss: 0.5197 - accuracy: 0.5413 - val_loss: 0.2468 - val_accuracy: 0.8205
Epoch 2/200
702/702 [==============================] - 1s 2ms/step - loss: 0.2837 - accuracy: 0.8034 - val_loss: 0.2361 - val_accuracy: 0.8590
Epoch 3/200
702/702 [==============================] - 1s 2ms/step - loss: 0.2626 - accuracy: 0.8419 - val_loss: 0.2437 - val_accuracy: 0.8718
Epoch 4/200
702/702 [==============================] - 1s 2ms/step - loss: 0.2529 - accuracy: 0.8433 - val_loss: 0.2298 - val_accuracy: 0.8846
Epoch 5/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2477 - accuracy: 0.8533 - val_loss: 0.2284 - val_accuracy: 0.8846
Epoch 6/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2424 - accuracy: 0.8533 - val_loss: 0.2173 - val_accuracy: 0.8846
Epoch 7/200
702/702 [==============================] - 1s 2ms/step - loss: 0.2401 - accuracy: 0.8561 - val_loss: 0.2182 - val_accuracy: 0.8846
Epoch 8/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2354 - accuracy: 0.8604 - val_loss: 0.2043 - val_accuracy: 0.8846
Epoch 9/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2349 - accuracy: 0.8561 - val_loss: 0.2053 - val_accuracy: 0.8718
Epoch 10/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2317 - accuracy: 0.8604 - val_loss: 0.2093 - val_accuracy: 0.8718
Epoch 11/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2277 - accuracy: 0.8647 - val_loss: 0.2158 - val_accuracy: 0.8590
Epoch 12/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2281 - accuracy: 0.8561 - val_loss: 0.1986 - val_accuracy: 0.8846
Epoch 13/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2256 - accuracy: 0.8632 - val_loss: 0.1966 - val_accuracy: 0.8846
Epoch 14/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2240 - accuracy: 0.8632 - val_loss: 0.1953 - val_accuracy: 0.8718
Epoch 15/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2242 - accuracy: 0.8618 - val_loss: 0.1941 - val_accuracy: 0.8846
Epoch 16/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2209 - accuracy: 0.8661 - val_loss: 0.1891 - val_accuracy: 0.8846
Epoch 17/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2188 - accuracy: 0.8647 - val_loss: 0.1876 - val_accuracy: 0.8846
Epoch 18/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2181 - accuracy: 0.8604 - val_loss: 0.1869 - val_accuracy: 0.8846
Epoch 19/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2154 - accuracy: 0.8590 - val_loss: 0.1796 - val_accuracy: 0.8846
Epoch 20/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2149 - accuracy: 0.8675 - val_loss: 0.1863 - val_accuracy: 0.8846
Epoch 21/200
702/702 [==============================] - 1s 979us/step - loss: 0.2128 - accuracy: 0.8647 - val_loss: 0.1803 - val_accuracy: 0.8846
Epoch 22/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2118 - accuracy: 0.8632 - val_loss: 0.1850 - val_accuracy: 0.8974
Epoch 23/200
702/702 [==============================] - 1s 957us/step - loss: 0.2098 - accuracy: 0.8675 - val_loss: 0.1905 - val_accuracy: 0.8590
Epoch 24/200
702/702 [==============================] - 1s 979us/step - loss: 0.2084 - accuracy: 0.8604 - val_loss: 0.1817 - val_accuracy: 0.8718
Epoch 25/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2069 - accuracy: 0.8632 - val_loss: 0.1885 - val_accuracy: 0.8846
Epoch 26/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2062 - accuracy: 0.8718 - val_loss: 0.1756 - val_accuracy: 0.8846
Epoch 27/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2055 - accuracy: 0.8661 - val_loss: 0.1820 - val_accuracy: 0.8846
Epoch 28/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2036 - accuracy: 0.8661 - val_loss: 0.1808 - val_accuracy: 0.8974
Epoch 29/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2013 - accuracy: 0.8661 - val_loss: 0.1870 - val_accuracy: 0.8846
Epoch 30/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2021 - accuracy: 0.8632 - val_loss: 0.1865 - val_accuracy: 0.8846
Epoch 31/200
702/702 [==============================] - 1s 1ms/step - loss: 0.2008 - accuracy: 0.8689 - val_loss: 0.1784 - val_accuracy: 0.8718
Epoch 32/200
702/702 [==============================] - 1s 1ms/step - loss: 0.1992 - accuracy: 0.8689 - val_loss: 0.1749 - val_accuracy: 0.8846
Epoch 33/200
702/702 [==============================] - 1s 1ms/step - loss: 0.1984 - accuracy: 0.8661 - val_loss: 0.1765 - val_accuracy: 0.8846
Epoch 34/200
702/702 [==============================] - 1s 1ms/step - loss: 0.1973 - accuracy: 0.8661 - val_loss: 0.1762 - val_accuracy: 0.8974
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 normalization (Normalizatio  (None, 11)               23
 n)

 dense (Dense)               (None, 15)                180

 dense_1 (Dense)             (None, 20)                320

 dense_2 (Dense)             (None, 75)                1575

 dense_3 (Dense)             (None, 75)                5700

 dense_4 (Dense)             (None, 1)                 76

=================================================================
Total params: 7,874
Trainable params: 7,851
Non-trainable params: 23
_________________________________________________________________
5/5 [==============================] - 0s 4ms/step - loss: 0.2271 - accuracy: 0.8478
Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #
=================================================================
 normalization (Normalizatio  (None, 11)               23
 n)

 dense (Dense)               (None, 15)                180

 dense_1 (Dense)             (None, 20)                320

 dense_2 (Dense)             (None, 75)                1575

 dense_3 (Dense)             (None, 75)                5700

 dense_4 (Dense)             (None, 1)                 76

=================================================================
Total params: 7,874
Trainable params: 7,851
Non-trainable params: 23
_________________________________________________________________



selecting a random row from the data frame...


     Age  Sex  ChestPainType  RestingBP  Cholesterol  FastingBS  RestingECG  MaxHR  ExerciseAngina  Oldpeak  ST_Slope  HeartDisease
910   41    0              1        120          157          2           0    182               1      0.0         0             0
Prediction in progress...


1/1 [==============================] - 0s 100ms/step
RESULT IS READY!


Deep Neural Network output: 0.001690749078989029
This person will most probably NOT be diagnosed with some kind of heart disease.                    

```



