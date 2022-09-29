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
**note:**While the plot figure is open the code will NOT move on to the next lines so make sure to close it after looking at the data.

### Neural Network Model:
**Preparing the data:**
Before getting to creating and training the model the dataset should undergo some minor changes: 
1. Deleting any NaN values using ```DataFrame.dropna()```
2. The alphabetical labels were changed into numbers (e.g M-->0 and F-->1)
3. The dataset was split into train (85%) and test (15%)
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



