import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rdatasets
from sklearn.model_selection import train_test_split

###############
#if the predicted classes are mostly wrong even with good learning rate and good test-train split, reverse the step function
#to adapt to general dataset, many changes need to be made by removing iris specific commands
###############

#training set
training=rdatasets.data('iris')
training=training[training.Species.isin (['versicolor','virginica'])]
training=training.reset_index().drop(['index'],axis=1)
label=lambda x: 1 if x=='versicolor' else -1
training['Species']=training['Species'].apply(label)
training['bias']=[1]*len(training.index)
training=training[['bias','Sepal.Length','Sepal.Width','Petal.Length','Petal.Width','Species']]

#activation function
def step(x,jump):
    if x<=jump:
        return(1)
    else:
        return(-1)

#perceptron
def perceptron(training,rate=0.5,epochs=10):
    weights=np.zeros(len(training.columns)-1)
    epoch=0
    while epoch<epochs:
        for i in training.index:
            x_i=np.delete(np.matrix(training.iloc[[i]]),-1,1)
            pred=step(x_i.dot(weights.T),0)
            diff=training['Species'][i]-pred
            weights=weights-(rate*diff*(x_i))
        epoch=epoch+1
    return(weights)

#test-train split
X=training[['bias','Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']]
Y=training[['Species']]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.30)
#train data in a new dataframe
df_train=X_train
df_train['Species']=Y_train['Species']
df_train=df_train.reset_index()
#train data in a new dataframe
df_test=X_test
df_test['Species']=Y_test['Species']
df_test=df_test.reset_index()


#model training
model_weights=perceptron(df_train,0.01,100)
predictions=[]
for i in range(len(df_test.index)):
    x_i=np.delete(np.matrix(df_test.iloc[[i]]),-1,1)
    pred=step(x_i.dot(model_weights.T),0)
    predictions.append(pred)

data={"Species":df_test['Species'],"Predictions":predictions}
data=pd.DataFrame(data)
data['match']=data.Species==data.Predictions
#print(list((zz['Species']==zz['Predictions'])))
groups=data.groupby('match')
print('True :',len(groups.get_group(True)),'\nFalse :',len(groups.get_group(False)),'\nTotal :',len(data))
