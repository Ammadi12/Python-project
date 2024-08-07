#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)

df = pd.read_csv(r'D:\DLD\data.csv', delimiter='\t')


# In[10]:


df.head()


# In[12]:


df.shape


# In[15]:


df.columns


# In[14]:


press = df.drop("screensize", axis=1)
press.head()


# In[16]:


press = press.drop("source", axis=1)


# In[17]:


press.head()


# In[19]:


press = press.drop("introelapse", axis=1)
press = press.drop("testelapse",axis=1)
press = press.drop("surveyelapse", axis=1)


# In[20]:


press = press.drop('VCL1', axis=1)
press = press.drop('VCL2', axis=1)
press = press.drop('VCL3', axis=1)
press = press.drop('VCL4', axis=1)
press = press.drop('VCL5', axis=1)
press = press.drop('VCL6', axis=1)
press = press.drop('VCL7', axis=1)
press = press.drop('VCL8', axis=1)
press = press.drop('VCL9', axis=1)
press = press.drop('VCL10', axis=1)
press = press.drop('VCL11', axis=1)
press = press.drop('VCL12', axis=1)
press = press.drop('VCL13', axis=1)
press = press.drop('VCL14', axis=1)
press = press.drop('VCL15', axis=1)


# In[21]:


press.head()


# In[22]:


press = press.drop("engnat",axis=1)
press = press.drop("uniquenetworklocation", axis=1)
press = press.drop("hand", axis=1)
press = press.drop("orientation",axis=1)
press = press.drop("voted",axis=1)


# In[23]:


press.head()


# In[32]:


removed= [f'Q{i}E' for i in range (1,43)]
#removed1= [f'Q{i}I' for i in range(1,43)]

press = press.drop(removed, axis=1)
#press = press.drop(removed1, axis=1)


# In[33]:


press.head()


# In[34]:


removed1= [f'Q{i}I' for i in range(1,43)]
press = press.drop(removed1, axis=1)


# In[35]:


press.head()


# In[36]:





# In[43]:


press['major'].value_counts()


# In[44]:


press.isnull().sum()


# In[50]:


press['education']=press['education'].map({0:1,1:1,2:2,3:3,4:4})

def changeed(title) -> str:
    if title == 0 or title == 1:
        return 'Less Than High school'
    if title == 2:
        return 'Highschool'
    if title == 3:
        return "Bachelor's degree"
    if title == 4:
        return 'Graduate Degree'
    return title
education_string = press['education'].apply(changeed)

plt.figure(figsize=(14,7))
sns.countplot(x=press['education'], hue=education_string)


# In[51]:


press.drop('major',axis=1)
press.head()


# In[55]:


col = press.columns
a = df.pop('orientation')
print (a)


# In[56]:


print(df)


# In[57]:


print(press)


# In[58]:


press = df.join(a).reindex(columns=col)


# In[59]:


press.head()


# In[61]:


press = press.drop('major',axis=1)


# In[62]:


press.head()


# In[63]:


press = press.drop('VCL16', axis=1)


# In[68]:


press['urban']= press['urban'].map({0:3, 1:1,2:2,3:3})

def changeur(value):
    if value == 1:
        return 'Rural'
    if value == 2:
        return 'Suburban'
    if value == 3:
        return 'Urban'
    return value

urban = press['urban'].apply(changeur)
plt.figure(figsize=(18,9))
sns.countplot(x=press['urban'], hue = urban)


# In[69]:


press['gender']=press['gender'].map({0:2,1:1,2:2,3:3})

def genderval(value):
    if value == 1:
        return 'Male'
    if value == 2 or value == 0:
        
        return 'Female'
    return 'Other'
gender = press['gender'].apply(genderval)
plt.figure(figsize=(19,6))
sns.countplot(x=press['gender'],hue=gender)


# In[71]:


def updateEducationValue(value):
    if value == 0: 
        return 12
    return value

press['religion'] = press['religion'].apply(updateEducationValue)

def changeReliginValues(value) -> str:
    if value == 0:
        return 'Other'
    if value == 1:
        return 'Agnostic'
    if value == 2:
        return 'Atheist'
    if value == 3:
        return 'Buddhist'
    if value == 4:
        return 'Christian (Catholic)'
    if value == 5:
        return 'Christian (Mormon)'
    if value == 6:
        return '=Christian (Protestant)'
    if value == 7: 
        return 'Christian (Other)'
    if value == 8:
        return 'Hindu'
    if value == 9:
        return 'Jewish'
    if value == 10:
        return 'Muslim'
    if value == 11:
        return 'Sikh'
    if value == 12:
        return 'Other'
    return value

religin = press['religion'].apply(changeReliginValues)

# show value counts of religin depression
display(press['religion'].value_counts())

plt.figure(figsize=(18, 7))
sns.countplot(x=press['religion'], hue= religin)


# In[73]:


display(press['race'].value_counts())


# In[78]:


press['race'] = press['race'].apply(lambda x : x/10)
display(press['race'].value_counts())


# In[79]:


press.head()


# In[82]:


def changeFromToinTIPI(value, From, to):
    if value == From:
        return to
    return value

press['TIPI1'] = press['TIPI1'].apply(lambda value: changeFromToinTIPI(value, 0, 5))
press['TIPI2'] = press['TIPI2'].apply(lambda value: changeFromToinTIPI(value, 0, 5))
press['TIPI3'] = press['TIPI3'].apply(lambda value: changeFromToinTIPI(value, 0, 5))
press['TIPI4'] = press['TIPI4'].apply(lambda value: changeFromToinTIPI(value, 0, 5))
press['TIPI5'] = press['TIPI5'].apply(lambda value: changeFromToinTIPI(value, 0, 5))
press['TIPI6'] = press['TIPI6'].apply(lambda value: changeFromToinTIPI(value, 0, 5))
press['TIPI7'] = press['TIPI7'].apply(lambda value: changeFromToinTIPI(value, 0, 5))
press['TIPI8'] = press['TIPI8'].apply(lambda value: changeFromToinTIPI(value, 0, 5))
press['TIPI9'] = press['TIPI9'].apply(lambda value: changeFromToinTIPI(value, 0, 5))
press['TIPI10'] = press['TIPI10'].apply(lambda value: changeFromToinTIPI(value, 0, 5))


# In[83]:


press['TIPI1'].value_counts()


# In[84]:


press['familysize'].value_counts()


# In[85]:


familyre = press[press['familysize']> 13].index
press = press.drop(familyre, axis=0)


# In[86]:


press['familysize'].value_counts()


# In[87]:


press['married'].value_counts()


# In[88]:


def changemarried(value):
    if value == 0:
        return 1
    return value
press['married']=press['married'].apply(changemarried)


# In[89]:


press['married'].value_counts()


# In[90]:


press['age'].value_counts()


# In[93]:


age_remov= press[press['age']> 80]['age'].index
press.drop(age_remov,axis=0,inplace=True)


# In[94]:


press['age'].value_counts()


# In[95]:


def Agegroup(value):
    if value<= 10:
        return 1
    if 11 <= value <= 16:
        return 2
    if 17 <= value <= 21:
        return 3
    if 22 <= value <= 35:
        return 4
    if 36 <= value <= 48:
        return 5
    if value >= 49:
        return 6
    
press['agegroup'] = press['age'].apply(Agegroup)
press.head()


# In[96]:


press.drop('age',axis=1)


# In[97]:


press= press.drop('age',axis=1)


# In[98]:


press.head()


# In[99]:


press['total']=press.sum(axis=1)


# In[100]:


press.head()


# In[101]:


press['total'].describe()


# In[102]:


press[press['total'] < 170]['total'].describe()


# In[103]:


press[press['total'] < 147]['total'].describe()


# In[104]:


press[press['total'] > 170]['total'].describe()


# In[106]:


press[press['total'] > 194]['total'].describe()


# In[107]:


press['total'].value_counts()


# In[110]:


def target(value):
    if value <= 143:
        return 'Normal'
    if 143 < value <= 157:
        return 'Mild'
    if 157 < value <= 180:
        return 'Moderate'
    if 180 < value <= 204:
        return 'Severe'
    if value > 204:
        return 'Extremely Sever'
    
press['result'] = press['total'].apply(target)    


# In[111]:


sns.countplot(x=press['result'])


# In[112]:


press.head()


# In[114]:


result = press['result']
press.drop(['result', 'total'], axis=1,inplace=True)


# In[117]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(press,result,test_size=.2)

print(f'x_train:{x_train.shape},y_train:{y_train.shape}')
print(f'x_test: {x_test.shape},y_test:{y_test.shape}')


# In[121]:


press.head()


# In[122]:


press['country'].value_counts()


# In[126]:


press =press.drop('country',axis=1)


# In[127]:


press.head()


# In[129]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(press,result,test_size=.2)

print(f'x_train:{x_train.shape},y_train:{y_train.shape}')
print(f'x_test: {x_test.shape},y_test:{y_test.shape}')


# In[130]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = MinMaxScaler()

x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


# In[131]:


press.head()


# In[132]:


from sklearn.svm import SVC

clf=SVC()

clf.fit(x_train_scaled, y_train)


# In[133]:


from sklearn.model_selection import cross_val_score

cross_score = cross_val_score(clf,x_train_scaled, y_train, cv=5)
print(f'Mean Score{np.mean(cross_score)}')


# In[136]:


from sklearn.metrics import classification_report

y_pred_svc = clf.predict(x_test_scaled)

print(classification_report(y_test,y_pred_svc))


# In[137]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg = LogisticRegression()

logreg.fit(x_train_scaled, y_train)


# In[138]:


y_pred = logreg.predict(x_test_scaled)

print(classification_report(y_test,y_pred))


# In[139]:


from sklearn.metrics import confusion_matrix

y_pred_lf = clf.predict(x_test_scaled)

confusion_matrix(y_test,y_pred_lf)


# In[140]:


from sklearn.metrics import confusion_matrix

y_pred_l = logreg.predict(x_test_scaled)

confusion_matrix(y_test,y_pred_l)


# In[ ]:




