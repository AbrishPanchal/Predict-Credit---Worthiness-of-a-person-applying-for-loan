import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv(r'C:\Users\HARSHIT\Documents\Python project\XYZCorp_LendingData.csv',  delimiter= ',', engine = 'python')
df.head()
df.shape
df.isnull().sum()
pd.set_option('display.max_column', None)
pd.set_option('display.max_rows', None)
df_rev= pd.DataFrame.copy(df)
df_rev=df_rev.drop(['id','member_id','mths_since_last_major_derog','desc','mths_since_last_record','emp_title', 'sub_grade','title','zip_code',
                    'addr_state','annual_inc_joint','pymnt_plan','dti_joint','verification_status_joint','open_acc_6m','open_il_6m',
                    'open_il_12m','open_il_24m', 'mths_since_rcnt_il','total_bal_il','il_util','open_rv_12m','open_rv_24m','max_bal_bc', 
                    'all_util','inq_fi','total_cu_tl','inq_last_12m','next_pymnt_d', 'last_pymnt_d', 'last_credit_pull_d', 'earliest_cr_line', 'out_prncp_inv', 'total_pymnt_inv' ], axis=1)
df_rev.isnull().sum()
df.isnull().sum()
df_rev.shape

for x in ['emp_length', 'tot_coll_amt', 'tot_cur_bal', 'total_rev_hi_lim','revol_util']:
    if df_rev[x].dtype=='object':
        df_rev[x].fillna(df_rev[x].mode()[0], inplace=True)
    elif df_rev[x].dtype== 'float64':
        df_rev[x].fillna(round(df_rev[x].mean(),2), inplace=True)
    else:
        pass

df_rev.describe()
df_rev['mths_since_last_delinq'].fillna(0, inplace = True)
df_rev.collections_12_mths_ex_med.value_counts()
df_rev['collections_12_mths_ex_med'].fillna(df_rev['collections_12_mths_ex_med'].mode()[0], inplace = True)

df_rev.isnull().sum()
#%%
#encoding

df_rev.emp_length.value_counts()
C1={'emp_length':{'< 1 year':0,'1 year':1,'2 years':2,'3 years':3,'4 years':4,'5 years':5,'6 years':6,'7 years':7,'8 years':8,'9 years':9,'10+ years':10}}
df_rev.replace(C1,inplace=True)

df_rev.head()
df_rev.dtypes


colname =['term', 'grade', 'home_ownership', 'verification_status', 'purpose', 'application_type','initial_list_status' ]

from sklearn import preprocessing
le= preprocessing.LabelEncoder()

for item in colname:
    df_rev[item]=le.fit_transform(df_rev[item])

df_rev.head()
#%%
#Data Visusalization

df_rev.boxplot(column='annual_inc')
plt.show()
sns.boxplot(data =df_rev['annual_inc'])
sns.distplot(df_rev.annual_inc)
df_rev.boxplot(column='total_pymnt')
sns.distplot(df_rev.total_pymnt)
sns.distplot(df_rev.installment)
#%%
#Data Partition
df_rev.dtypes
df_rev.head()
df_rev['issue_d'] = pd.to_datetime(df_rev['issue_d'])
df_rev=df_rev.set_index(df_rev['issue_d'])
df_rev=df_rev.drop(['issue_d'], axis=1)


train = df_rev['2007-01-06': '2015-01-05']
train.head()
test =df_rev['2015-01-06':'2015-01-12']



X_train =train.values[:,:-1]
Y_train=train.values[:,-1]

X_test = test.values[:,:-1]
Y_test = test.values[:,-1]


df_rev.boxplot()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
scaler.transform(X_train)
scaler.transform(X_test)
Y_train.astype(int)
Y_train.dtype



train.dtypes
test.dtypes
#%%
#Model Building
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,Y_train)
Y_pred= classifier.predict(X_test)


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm= confusion_matrix(Y_test, Y_pred)
print(cfm)

accuracy = accuracy_score(Y_test, Y_pred)
print('accuracy of model:', accuracy)

CR= classification_report(Y_test, Y_pred)
print(CR)

y_pred_prob =classifier.predict_proba(X_test)

for a in np.arange(0,1,0.01):
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(Y_test, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :", cfm[1,0]," , type 1 error:", cfm[0,1])

y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value>0.98:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)

#Performance of logistic Regression:
cfm= confusion_matrix(Y_test, y_pred_class)
print(cfm)
accuracy = accuracy_score(Y_test, y_pred_class)
print('accuracy of model:', accuracy)
CR = classification_report(Y_test, y_pred_class)
print(CR)


from sklearn import metrics

fpr,tpr,z=metrics.roc_curve(Y_test, y_pred_prob[:,1])
auc=metrics.auc(fpr, tpr)
print(auc)
print(z)

import matplotlib.pyplot as plt
#%matplotlib inline not required in spyder
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
#%%
#performing kfold_cross_validation
from sklearn.model_selection import KFold
kfold_cv=KFold(n_splits=10)
print(kfold_cv)

from sklearn.model_selection import cross_val_score
#running the model using scoring metric as accuracy
kfold_cv_result=cross_val_score(estimator=classifier,X=X_train,y=Y_train, cv=kfold_cv)
print(kfold_cv_result)
#finding the mean
print(kfold_cv_result.mean())

for train_value, test_value in kfold_cv.split(X_train):
    classifier.fit(X_train[train_value], Y_train[train_value]).predict(X_train[test_value])

##additive model-- learn from each iteration in model building
Y_pred=classifier.predict(X_test)
#print(list(zip(Y_test,Y_pred)))

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm=confusion_matrix(Y_test, Y_pred)
print(cfm)
acc=accuracy_score(Y_test,Y_pred)
print('Accuracy of the model:',acc)
print('Classification Report:')
print(classification_report(Y_test, Y_pred))


y_pred_prob =classifier.predict_proba(X_test)

#%%
from sklearn.tree import DecisionTreeClassifier
#model_DecisionTree=DecisionTreeClassifier(random_state=10) ##default model
model_DecisionTree=DecisionTreeClassifier(random_state=10, min_samples_leaf=300, max_depth =20, criterion='entropy')
model_DecisionTree.fit(X_train, Y_train)

Y_pred=model_DecisionTree.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#performance of Decision Tree
cfm=confusion_matrix(Y_test, Y_pred)
print(cfm)
acc= accuracy_score(Y_test, Y_pred)
print('accuracy of model:', acc)
print('Classification Report:')
print(classification_report(Y_test, Y_pred))


print(list(zip(colname, model_DecisionTree.feature_importances_)))


#%%
##random forest
from sklearn.ensemble import RandomForestClassifier
model=(RandomForestClassifier(100, random_state=10))
model=model.fit(X_train, Y_train)
Y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#performance of Random forest
cfm=confusion_matrix(Y_test, Y_pred)
print(cfm)
acc= accuracy_score(Y_test, Y_pred)
print(acc)
print('Classification Report:')
print(classification_report(Y_test, Y_pred))

#%%
##Boosting
from sklearn.ensemble import AdaBoostClassifier

model_AdaBoost =AdaBoostClassifier(base_estimator=DecisionTreeClassifier(), random_state=10)
model_AdaBoost.fit(X_train,Y_train)
Y_pred=model_AdaBoost.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
#performance of AdaBoostClassifier
cfm=confusion_matrix(Y_test, Y_pred)
print(cfm)
acc= accuracy_score(Y_test, Y_pred)
print(acc)
print('Classification Report:')
print(classification_report(Y_test, Y_pred))






