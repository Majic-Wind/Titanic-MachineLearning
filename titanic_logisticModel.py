import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import warnings
warnings.filterwarnings("ignore")

def process_age(df,fields):
    age_df = df[fields]
    age_df = pd.get_dummies(age_df,dummy_na=True , columns=["Embarked","Title"])
    X_age = age_df.loc[ (df.Age.notnull()) ].values[:, 1::]
    y_age = age_df.loc[ (df.Age.notnull()) ].values[:, 0]
    from sklearn.ensemble import RandomForestRegressor
    rtr = RandomForestRegressor()
    rtr.fit(X_age,y_age)
    predictedAges = rtr.predict(age_df.loc[ (df.Age.isnull()) ].values[:, 1::])
    return predictedAges
def features_standard(df,fields):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[fields] =scaler.fit_transform(df[fields])
    return df

def search_best_clf(param_grid,clf,cv,scoring,X,y):
    from sklearn.model_selection import GridSearchCV
    clf_param = GridSearchCV(clf,param_grid,cv=cv,scoring = scoring)
    clf_param.fit(X,y)
    #print(clf_param.best_params_)
    clf_best=clf_param.best_estimator_
    #pd.DataFrame(clf_param.cv_results_)  可查看各个得分
    return clf_best

def main():
#读取数据集
    train_data = pd.read_csv("train.csv")
    test_data = pd.read_csv("test.csv")
    train_data["ID"] = "train"
    test_data['ID'] = "test"
    df = pd.concat([train_data,test_data],axis=0,ignore_index=True)
    #姓名中提取titile字段
    df["Title"] = df["Name"].map(lambda t :re.compile(",(.*?)\.").findall(t)[0].strip()) 
    df['Title'][df.Title == 'Jonkheer'] = 'Master'
    df['Title'][df.Title.isin(['Ms','Mlle'])] = 'Miss'
    df['Title'][df.Title == 'Mme'] = 'Mrs'
    df['Title'][df.Title.isin(['Capt', 'Don', 'Major', 'Col', 'Sir'])] = 'Sir'
    df['Title'][df.Title.isin(['Dona', 'Lady', 'the Countess'])] = 'Lady'
    #填补Fare中缺失值
    df['Fare'][df['Fare'].isnull()] =df['Fare'].median()
    #对缺失年龄利用随机森林填补

    predict_age_fields = ['Age','Embarked','Fare', 'Parch', 'SibSp', 'Title','Pclass']
    predictedAges = process_age(df,predict_age_fields)
    df.loc[ (df.Age.isnull()), 'Age' ] = predictedAges
    #将年龄按段划分
    df.loc[df["Age"] < 14 , "Adult/Child"] = "Child"
    df.loc[(df["Age"] < 35) & (df['Age'] >=14), "Adult/Child"] = "Adult1"
    df.loc[(df["Age"] < 60) & (df['Age'] >=35), "Adult/Child"] = "Adult2"
    df.loc[df["Age"] >= 60 , "Adult/Child"] = "Old"
    #特征无量纲化

    standard_fields = ['Fare','Parch','SibSp']
    df = features_standard(df,standard_fields)
    #分类变量转换
    df= pd.get_dummies(df,dummy_na=False , columns=["Title","Sex","Adult/Child",'Pclass'])
    #拆分训练集和测试集
    train = df.loc[df['ID'] == 'train']
    test = df.loc[df['ID'] == 'test']
    train.drop(['ID','Age','Fare','Name','Cabin','Embarked','PassengerId','Ticket'],inplace=True,axis=1)
    test.drop(['ID','Age','Fare','Survived','Name','Cabin','Embarked','PassengerId','Ticket'],inplace=True,axis=1)
    
    #交叉验证
    X= train.drop(["Survived"],axis=1)
    y= train["Survived"]
    from sklearn.cross_validation import KFold
    KF = KFold(len(X),5)


    #建立模型并找最佳参数
    from sklearn import linear_model
    clf_log = linear_model.LogisticRegression(penalty='l2')
    param_grid = {
         'C':[1,2,3,1.5],
    'tol':[1e-3,1e-2,1e-1,1e-0]
          }
    clf_logistic = search_best_clf(param_grid=param_grid,clf = clf_log,cv = KF,scoring='f1',X=X,y=y)

    from sklearn.cross_validation import cross_val_score,cross_val_predict
    scores = cross_val_score(clf_logistic,X,y,cv = 5)
    scores_mean = np.mean(scores)
    print("validation_mean_score: %f"%(scores_mean))

    #在测试集上预测并保存至csv
    final_predictions = clf_logistic.predict(test)
    result = pd.DataFrame({'PassengerId':test_data['PassengerId'].as_matrix(), 'Survived':final_predictions.astype(np.int32)})
    result.to_csv("logistics_predictions.csv", index=False)
if __name__ == '__main__':
	main()