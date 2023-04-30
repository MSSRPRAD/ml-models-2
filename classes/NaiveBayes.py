import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

class NaiveBayes:
  def __init__(self):
    self.priors = None
    self.cond_prob_of_feature = None
    self.class1 = None
    self.class2 = None
    self.y_pred = None
    self.con = None
    self.acc = None
    self.prec = None
    self.rec= None
    self.truepositives = None
    self.falsepositives = None
    self.truenegatives = None
    self.falsenegatives = None
    pass

  def calc_prior_prob(self, y):
      self.priors = y.groupby(y).count().add(250).div(len(y))
      self.class1 = y.groupby(y).count()[0]
      self.class2 = y.groupby(y).count()[1]
      return  
    
  def calc_conditional_prob(self, X, y):
      self.cond_prob_of_feature = {}
      train = pd.concat([X,y], axis=1)
      i=0
      for col in train.columns:
        if i%1000==0 and i!=0:
           print(i)
        self.cond_prob_of_feature[col] = train.groupby(['salary', col]).size().add(0)
        self.cond_prob_of_feature[col]/=len(train)
        self.cond_prob_of_feature[col]/=self.priors
      return
  
  
  def fit(self, X, y):
    print("calculating prior probabilites.....")
    self.calc_prior_prob(y)
    print("calculating conditional probabilites.....")
    self.calc_conditional_prob(X, y)
    return

  def classify(self, x):
    prob_class1 = 1
    prob_class2 = 1
    count=0
    for i in x.index:
      if(x[count] not in self.cond_prob_of_feature[i]['-1']):
          prob_class1 *= 0.001  
      else:
          prob_class1 *= self.cond_prob_of_feature[i]['-1'][x[count]]
      if(x[count] not in self.cond_prob_of_feature[i]['1']):
          prob_class2 *= 0.001  
      else:
          prob_class2 *= self.cond_prob_of_feature[i]['1'][x[count]]
      count+=1
    if prob_class1*self.priors[0] >= prob_class2*self.priors[1]:
      return -1 
    elif prob_class1*self.priors[0] > prob_class2*self.priors[1]:
      return 1
    else:
       return 1

  
  def predict(self, X):
    predictions = []
    for i in range(X.shape[0]):
      if i%1000==0:
        print("Percentage Predicted:")
        print(i/10000)
      predictions.append(str(self.classify(X.iloc[i, :])))
    return predictions
  
  def test(self, X, y):
    self.y_pred = self.predict(X)
    self.acc = self.accuracy(y)
    self.prec = self.precision(y)
    # self.rec = self.recall(y)
    self.con = self.confusion(y)

  def precision(self, y):
        precision = 0
        count = 0
        for i in range(len(self.y_pred)):
            if self.y_pred[i]=="-1":
                if y.iloc[i]=="-1":
                    precision+=1
                count+=1
        return precision/count

  def confusion(self, y):
    return confusion_matrix(y_true=y, y_pred=self.y_pred)
  
  def recall(self, y):
    recall = 0
    count = 0
    for i in range(len(self.y_pred)):
      if y.iloc[i]=="-1":
        if self.y_pred[i]=="-1":
          recall+=1
          count+=1
        else:
          count+=1
      return recall/count

  def accuracy(self, y):
    misclassifications = 0
    for i in range(len(self.y_pred)):
      if str(self.y_pred[i]) != str(y.iloc[i]):
        misclassifications += 1  

    return 1-misclassifications/len(self.y_pred)
    
