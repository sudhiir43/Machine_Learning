import pandas as pd                            # dataframe operations
import lightgbm as lgbm                        # lightgbm 
import matplotlib.pyplot as plt                # plotting
from sklearn.metrics import roc_auc_score      # to measure training auc score

path = 'D:\\my_projects\\monsoon\\'

X_train = pd.read_csv(path + 'X_train.csv')           # read train data
y_train = pd.read_csv(path+ 'y_train.csv')           # read test data

X_test = pd.read_csv(path + 'X_test.csv')             # read test data

X_train.drop(['N6'], axis=1, inplace = True)   # for train 
X_test.drop (['N6'], axis=1, inplace = True)   # fro evaluation

X_train = pd.merge(X_train, y_train, on = 'Unique_ID')   # merge X_train and y_train


# Probability encoding for categorical variables 

# lets create empty dictionaries to store probabilities
probs_C1 = {}
probs_C2 = {}
probs_C3 = {}
probs_C4 = {}
probs_C5 = {}
probs_C6 = {}
probs_C7 = {}
probs_C8 = {}


#store the probabilities in dictonaries
for a in X_train.C1.unique():
    probs_C1[a] = X_train[X_train.C1 == a].Dependent_Variable.mean()
for a in X_train.C2.unique():
    probs_C2[a] = X_train[X_train.C2 == a].Dependent_Variable.mean()
for a in X_train.C3.unique():
    probs_C3[a] = X_train[X_train.C3 == a].Dependent_Variable.mean()
for a in X_train.C4.unique():
    probs_C4[a] = X_train[X_train.C4 == a].Dependent_Variable.mean()    
for a in X_train.C5.unique():
    probs_C5[a] = X_train[X_train.C5 == a].Dependent_Variable.mean()
for a in X_train.C6.unique():
    probs_C6[a] = X_train[X_train.C6 == a].Dependent_Variable.mean()
for a in X_train.C7.unique():
    probs_C7[a] = X_train[X_train.C7 == a].Dependent_Variable.mean()
for a in X_train.C8.unique():
    probs_C8[a] = X_train[X_train.C8 == a].Dependent_Variable.mean()
    
print("train probability encoding")

X_train['C1_prob'] = [probs_C1[a] for a in X_train['C1']]
X_train['C2_prob'] = [probs_C2[a] for a in X_train['C2']]
X_train['C3_prob'] = [probs_C3[a] for a in X_train['C3']]
X_train['C4_prob'] = [probs_C4[a] for a in X_train['C4']]
X_train['C5_prob'] = [probs_C5[a] for a in X_train['C5']]
X_train['C6_prob'] = [probs_C6[a] for a in X_train['C6']]
X_train['C7_prob'] = [probs_C7[a] for a in X_train['C7']]
X_train['C8_prob'] = [probs_C8[a] for a in X_train['C8']]

print("test probability encoding")

X_test['C1_prob'] = [probs_C1[a] for a in X_test['C1']]
X_test['C2_prob'] = [probs_C2[a] for a in X_test['C2']]
X_test['C3_prob'] = [probs_C3[a] for a in X_test['C3']]
X_test['C4_prob'] = [probs_C4[a] for a in X_test['C4']]
X_test['C5_prob'] = [probs_C5[a] for a in X_test['C5']]
X_test['C6_prob'] = [probs_C6[a] for a in X_test['C6']]
X_test['C7_prob'] = [probs_C7[a] for a in X_test['C7']]
X_test['C8_prob'] = [probs_C8[a] for a in X_test['C8']]

# as lightgbm handles missing values we are not imputing anything
# Just adding another column to data which tells us if a row has missing value 
#X_train['missing'] = X_train.isnull().any(axis=1)   # Missing values 
#X_test['missing'] = X_test.isnull().any(axis=1)     # Misiing values



label = X_train.Dependent_Variable                  #store Dependent variable in separate variable for ease of use

#inplace = True works same as below
X_train = X_train.drop(['Unique_ID', 'Dependent_Variable'], axis = 1)  # drop unique id and Dependent variable from X_train

test_UniqueID = X_test.Unique_ID                    # stoew test id in a variable

X_test = X_test.drop(['Unique_ID'],axis = 1)        # drop unique id from X_test


# store training data as a lightgbm dataset
train_data=lgbm.Dataset(X_train,label = label)

# few parameters to try
num_leaves_choices = [24, 31, 56]
learning_rate_choices = [0.05, 0.01]

# We will store the cross validation results in a simple list,
# with tuples in the form of (hyperparam dict, cv score):
cv_results = []      # to store the cv results

for num_lv in num_leaves_choices:
    for lr in learning_rate_choices:
        hyperparams = {"objective": 'binary',
                        "num_leaves": num_lv,
                        "learning_rate": lr,
                       'feature_fraction': 0.7,
                       'bagging_fraction': 0.6,
                       'bagging_freq': 10,
                                 }
        validation_summary = lgbm.cv(hyperparams,train_data, num_boost_round=4096, nfold=3, metrics=["l2","auc"],
                                    early_stopping_rounds=50, verbose_eval=50)
        
        optimal_num_trees = len(validation_summary["auc-mean"])
        # Let's just add the optimal number of trees (chosen by early stopping)
        # to the hyperparameter dictionary:
        optimal_number_of_trees = optimal_num_trees

        # And we append results to cv_results:
        cv_results.append((hyperparams, optimal_number_of_trees, validation_summary["auc-mean"][-1]))

print("printing cv results")
print(pd.DataFrame(cv_results))

# parameters which gave the best auc
params = {
		 'bagging_fraction': 0.6,
		 'bagging_freq': 10,
 		 'feature_fraction': 0.7,
 		 'learning_rate': 0.01,
 		 'metric': ['l2', 'auc'],
 		 'num_leaves': 56,
 		 'objective': 'binary'
          }
    
model = lgbm.train(params,train_data,num_boost_round=757)
X_predictions = model.predict(X_train)
print("train auc score: {} ".format(roc_auc_score(label,X_predictions)))

#plot feature importances
print('Plot feature importances...')
ax = lgbm.plot_importance(model)
plt.show()


# Predict on X_test data
print("predicting on X_test")
Class_1_Probability = model.predict(X_test)


# write the results to a csv file
print("writing results to submission.csv")
d = pd.DataFrame({'Unique_ID':test_UniqueID,
                 'Class_1_Probability':Class_1_Probability})
d.to_csv(path+'submission.csv',index=None)
