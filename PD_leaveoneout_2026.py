
import glob
import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

## external data, modified to external data used for your own experiment

df_pd_xx = pd.read_csv(*data_directory*) #reading data after feature extraction
df_control_xx = pd.read_csv(*data_directory*)

df_external = pd.concat([pddata+controldata]).dropna(axis=0) #combined PD and healthy data
df_external = df_external.drop(['stdevF0Hz'],axis=1)

df_external[df_external.isna().any(axis=1)]
df_external['voiceName'] = np.array(df_external['voiceName'].str.replace(r'DATASET_FULL/Dataset1_half_toserver/', '', regex=True).tolist())

external_subjects = [*subject_name*] #add data tag such as 'PD_01', 'Control_01'

########################################################################################

UCI = pd.read_fwf(*UCI_parkinsons_data_directory*)

UCI['split'] = UCI['name,MDVP:Fo(Hz),MDVP:Fhi(Hz),MDVP:Flo(Hz),MDVP:Jitter(%),MDVP:Jitter(Abs),MDVP:RAP,MDVP:PPQ,Jitter:DDP,MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA,NHR,HNR,status,RPDE,DFA,spread1,spread2,D2,PPE'].str.split(',')

UCI = pd.DataFrame(UCI['split'].tolist())

UCI.columns = ['voiceName','meanF0Hz','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','localJitter','localabsoluteJitter','rapJitter','ppq5Jitter','ddpJitter','localShimmer','localdbShimmer','apq3Shimmer','apq5Shimmer','apq11Shimmer','ddaShimmer','NHR','HNR','class','RPDE','DFA','spread1','spread2','D2','PPE']

UCI_trim = UCI.drop(['spread1','spread2','D2','PPE','MDVP:Fhi(Hz)','MDVP:Flo(Hz)','NHR'], axis =1)

x_UCI = UCI_trim.drop(['voiceName','class'], axis = 1)
y_UCI = UCI_trim['class']
name = UCI_trim['voiceName']

x_UCI = x_UCI.apply(pd.to_numeric, errors = 'coerce')
y_UCI = y_UCI.apply(pd.to_numeric, errors = 'coerce')

UCI_new = pd.concat([name,x_UCI,y_UCI],axis = 1)
UCI_new.shape

UCI_subjects = ['S01','S34','S44','S20','S24','S26','S08','S39',
            'S33','S32','S02','S22','S37','S21','S04','S19',
            'S35','S05','S18','S16','S27','S25','S06','S10',
            'S07','S13','S43','S17','S42','S50','S49']

#################################################################

df_full = pd.concat([df_external, UCI_new])
df_full.shape

all_subjects = external_subjects + UCI_subjects

#################################################################

from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
import re

svm_rbf = SVC(kernel='rbf', probability= True)
svm_linear = SVC(kernel='linear',probability = True)

C_range1 = [1] #range1 for testing the model with param_grid1
C_range2 = np.logspace(0,2,4,endpoint=True)

gamma_range1 = [0.009] #range1 for testing the model with param_grid1
gamma_range2 = np.linspace(0.01,1,10,endpoint=True)

param_grid1 = dict(SVC__C=C_range1, SVC__gamma = gamma_range1)
param_grid2 = dict(SVC__C=C_range2, SVC__gamma = gamma_range2)

###########################################################################

def drop(df):
  x_label = df['voiceName']
  x_drop = df.drop(['voiceName','class','group'],axis=1)
  group = df['group']
  y_drop = df['class']
  return x_drop, y_drop, x_label, group

############################################################################

import collections
from matplotlib import pyplot as plt

from sklearn.model_selection import PredefinedSplit, GridSearchCV
from sklearn.metrics import (roc_auc_score, precision_score, recall_score, f1_score, accuracy_score)

#############################################################################

def classifierSVM_PD_LOSO(groups,classifier_kernel,param_grid,X,y):

  pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("SVC", SVC(kernel = classifier_kernel, probability = True)),
  ])

  logo = LeaveOneGroupOut()

  gs = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid,
    cv=logo,                
    scoring="f1_macro",         
    n_jobs=-1,
    verbose = 1)

  gs.fit(X, y, groups=groups)
  return gs.best_score_, gs.best_params_

########################################################################################

def clf_test(c, g, x_train, y_train, x_test, y_test,threshold,classifier_kernel):

  clf_for_test = SVC(C = c, gamma = g, kernel = classifier_kernel, probability = True)
  test_pipe = Pipeline(steps=[
    ("scaler", StandardScaler()),
    ("SVC", clf_for_test),
  ])

  test_pipe.fit(x_train, y_train)
  y_hat = test_pipe.predict_proba(x_test)[:,1]
  y_pred = (y_hat >= threshold).astype(bool)

  return y_pred, y_test

#####################################################################################

def l1norm_selection(data,subjects,test_subject_list,param_grid):
  df_best_val_score_l1 = pd.DataFrame()
  df_best_c_l1 = pd.DataFrame()
  df_best_gamma_l1 = pd.DataFrame()

  l1_sel_feat_list = []
  df_all_feature_count = pd.DataFrame()

  test_subject_name = []

  all_y_test = []
  all_y_pred = []

  for j in test_subject_list:

    print(j)
    test_subject_name.append(j)

    features_list = []
    sel_feat_list = []
    xtrain_list = []
    xtest_list = []

    best_score_list = []
    best_params_list = []

    #group data for Leave One Subject out
    subjects = sorted(subjects, key=len, reverse=True)  # avoid A vs AA collisions
    pattern = "(" + "|".join(map(re.escape, all_subjects)) + ")"

    data["group"] = (data["voiceName"].astype(str).str.extract(pattern, expand=False))
    data = data.dropna(subset=["group"])

    s = data["voiceName"].str.contains(j, regex=False)
    test_notdrop = data[s]
    train_val_notdrop = data[~s]

    test = drop(test_notdrop)
    train_val = drop(train_val_notdrop)

    groups = train_val[3].to_numpy()

    num_features = train_val[0].shape[1]

    for n in range(1,num_features + 1):
      df = pd.DataFrame(train_val[0], columns = ['meanF0Hz', 'HNR', 'localJitter', 'localabsoluteJitter', 'rapJitter',
                                        'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer',
                                        'apq11Shimmer', 'ddaShimmer', 'RPDE', 'DFA']) #add if you have more audio features

      sel = SelectFromModel(LinearSVC(penalty = 'l1', max_iter = 10000, dual = False), threshold=-np.inf, max_features = n)
      sel.fit(train_val[0], train_val[1])

      sel_feat = df.columns[(sel.get_support())]
      sel_feat_list.append(sel_feat)

      print(f'Results when using {n} features')
      print(f'Selected features are {sel_feat}.')

      x_train_selected = sel.transform(train_val[0])
      xtrain_list.append(x_train_selected)

      x_test_selected = sel.transform(test[0])
      xtest_list.append(x_test_selected)

      cc = classifierSVM_PD_LOSO(groups,'linear',param_grid,x_train_selected,train_val[1])
      print(' ')
      print(f'The parameters are {cc[1]} with f1-macro score at {cc[0]}')
      print('-------------------------------------------------------------------------- ')

      #caching best_score, best_param of each iteration
      best_score_list.append(cc[0])
      best_params_list.append(cc[1])

      #caching independent features for counting occurrences at the end of the program
      for i in range(0,len(sel_feat)):
        features_list.append(sel_feat[i])


    #counting feature occurrences
    feature_dict = collections.Counter(features_list)
    df_feature_count = pd.DataFrame.from_dict(feature_dict, orient='index', columns=[j]).transpose()
    df_rename = df_feature_count.rename(index={0:name})

    print(' ')

    best_index = np.argmax(best_score_list)
    print(f'The best model use {best_index+1} features.')

    b_params = best_params_list[best_index]
    b_score = best_score_list[best_index]
    print(f'The parameter of this model are {b_params} with f1-macro score at {b_score}.')

    best_sel_feat = sel_feat_list[best_index]
    print(f'The features of this model consist of {best_sel_feat}')

    print('---------------------------------------------------------')

    #selecting datasets that create the best classifier
    x_train_best = xtrain_list[best_index]
    x_test_best = xtest_list[best_index]

    if x_train_best.shape[1] != (best_index +1):
      print('Error: Mismatch in number of features for x_train_best')

    print('---------------------------------------------------------')

    pred_test = clf_test(b_params['SVC__C'], b_params['SVC__gamma'], x_train_best, train_val[1], x_test_best, test[1],0.5,'linear')

    summary_columns =  ['1 feature','2 features','3 features','4 features','5 features',
                                    '6 features','7 features','8 features','9 features','10 features',
                                    '11 features','12 features','13 features','14 features','15 features'] #add if you have more audio features

    df_best_score = pd.DataFrame(
      data = best_score_list,
      columns = [j]
    ).T

    df_best_score.columns = summary_columns

    df_best_params = pd.DataFrame.from_dict(
      data =best_params_list
    )

    df_best_c = pd.DataFrame(df_best_params['SVC__C'])
    df_best_gamma = pd.DataFrame(df_best_params['SVC__gamma'])

    df_best_c.columns = [j]
    df_best_gamma.columns = [j]

    df_best_c = df_best_c.T
    df_best_gamma = df_best_gamma.T

    df_best_c.columns = summary_columns
    df_best_gamma.columns = summary_columns

    all_y_pred.extend(pred_test[0])
    all_y_test.extend(pred_test[1])
    
    l1_sel_feat_list.append(best_sel_feat)

    df_all_feature_count = pd.concat([df_all_feature_count, df_rename])
    df_best_val_score_l1 = pd.concat([df_best_val_score_l1, df_best_score])
    df_best_c_l1 = pd.concat([df_best_c_l1, df_best_c])
    df_best_gamma_l1 = pd.concat([df_best_gamma_l1, df_best_gamma])

  all_y_test = np.asarray(all_y_test).ravel()
  all_y_pred = np.asarray(all_y_pred).ravel()

  l1norm_recall = precision_score(all_y_test, all_y_pred, zero_division=0)
  l1norm_precision = recall_score(all_y_test, all_y_pred, zero_division=0)
  l1norm_f1 = f1_score(all_y_test, all_y_pred, average="macro", zero_division=0)
  l1norm_accuracy = accuracy_score(all_y_test, all_y_pred)
  final_auc = roc_auc_score(all_y_test, all_y_pred)

  print("Pooled AUC:", final_auc)

  return(l1norm_recall, l1norm_f1, l1_sel_feat_list, df_all_feature_count,
         df_best_val_score_l1, df_best_c_l1, df_best_gamma_l1,test_subject_name,
         final_auc, l1norm_precision, l1norm_accuracy, all_y_test, all_y_pred)

########################################################################################

def rfe_selection(data,subjects,test_subject_list,param_grid):
  df_best_val_score_rfe = pd.DataFrame()
  df_best_c_rfe = pd.DataFrame()
  df_best_gamma_rfe = pd.DataFrame()

  rfe_sel_feat_list = []
  df_all_feature_count = pd.DataFrame()

  test_subject_name = []

  all_y_test = []
  all_y_pred = []

  for j in test_subject_list:
    print(j)
    test_subject_name.append(j)

    features_list = []
    sel_feat_list = []
    xtrain_list = []
    xtest_list = []

    best_score_list = []
    best_params_list = []

    #group data for Leave One Subject out
    subjects = sorted(subjects, key=len, reverse=True)  # avoid A vs AA collisions
    pattern = "(" + "|".join(map(re.escape, all_subjects)) + ")"

    data["group"] = (data["voiceName"].astype(str).str.extract(pattern, expand=False))
    data = data.dropna(subset=["group"])

    s = data["voiceName"].str.contains(j, regex=False)
    test_notdrop = data[s]
    train_val_notdrop = data[~s]

    test = drop(test_notdrop)
    train_val = drop(train_val_notdrop)

    groups = train_val[3].to_numpy()

    num_features = train_val[0].shape[1]

    for n in range(1,num_features + 1):
      df = pd.DataFrame(train_val[0], columns = ['meanF0Hz', 'HNR', 'localJitter', 'localabsoluteJitter', 'rapJitter',
                                        'ppq5Jitter', 'ddpJitter', 'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer',
                                        'apq11Shimmer', 'ddaShimmer', 'RPDE', 'DFA']) #add if you have more audio features

      sel = RFE(estimator= LinearSVC(penalty = 'l2', max_iter = 10000, dual = False), n_features_to_select = n, step = 1)
      sel.fit(train_val[0], train_val[1])

      sel_feat = df.columns[(sel.get_support())]
      sel_feat_list.append(sel_feat)

      print(f'Results when using {n} features')
      print(f'Selected features are {sel_feat}.')

      x_train_selected = sel.transform(train_val[0])
      xtrain_list.append(x_train_selected)

      x_test_selected = sel.transform(test[0])
      xtest_list.append(x_test_selected)

      cc = classifierSVM_PD_LOSO(groups,'rbf',param_grid,x_train_selected,train_val[1])
      print(' ')
      print(f'The parameters are {cc[1]} with f1-macro score at {cc[0]}')
      print('-------------------------------------------------------------------------- ')

      #caching best_score, best_param of each iteration
      best_score_list.append(cc[0])
      best_params_list.append(cc[1])

      #caching independent features for counting occurrences at the end of the program
      for i in range(0,len(sel_feat)):
        features_list.append(sel_feat[i])


    #counting feature occurrences
    feature_dict = collections.Counter(features_list)
    df_feature_count = pd.DataFrame.from_dict(feature_dict, orient='index', columns=[j]).transpose()
    df_rename = df_feature_count.rename(index={0:name})

    print(' ')

    best_index = np.argmax(best_score_list)
    print(f'The best model use {best_index+1} features.')

    b_params = best_params_list[best_index]
    b_score = best_score_list[best_index]
    print(f'The parameter of this model are {b_params} with f1-macro score at {b_score}.')

    best_sel_feat = sel_feat_list[best_index]
    print(f'The features of this model consist of {best_sel_feat}')

    print('---------------------------------------------------------')

    #selecting datasets that create the best classifier
    x_train_best = xtrain_list[best_index]
    x_test_best = xtest_list[best_index]

    if x_train_best.shape[1] != (best_index +1):
      print('Error: Mismatch in number of features for x_train_best')

    print('---------------------------------------------------------')

    pred_test = clf_test(b_params['SVC__C'], b_params['SVC__gamma'], x_train_best, train_val[1], x_test_best, test[1],0.5,'rbf')

    summary_columns =  ['1 feature','2 features','3 features','4 features','5 features',
                                    '6 features','7 features','8 features','9 features','10 features',
                                    '11 features','12 features','13 features','14 features','15 features'] #add if you have more audio features

    df_best_score = pd.DataFrame(
      data = best_score_list,
      columns = [j]
    ).T

    df_best_score.columns = summary_columns

    df_best_params = pd.DataFrame.from_dict(
      data =best_params_list
    )

    df_best_c = pd.DataFrame(df_best_params['SVC__C'])
    df_best_gamma = pd.DataFrame(df_best_params['SVC__gamma'])

    df_best_c.columns = [j]
    df_best_gamma.columns = [j]

    df_best_c = df_best_c.T
    df_best_gamma = df_best_gamma.T

    all_y_test.extend(pred_test[1])
    all_y_pred.extend(pred_test[0])

    df_best_c.columns = summary_columns
    df_best_gamma.columns = summary_columns

    rfe_sel_feat_list.append(best_sel_feat)

    df_all_feature_count = pd.concat([df_all_feature_count, df_rename])
    df_best_val_score_rfe = pd.concat([df_best_val_score_rfe, df_best_score])
    df_best_c_rfe = pd.concat([df_best_c_rfe, df_best_c])
    df_best_gamma_rfe = pd.concat([df_best_gamma_rfe, df_best_gamma])

  all_y_test = np.asarray(all_y_test).ravel()
  all_y_pred = np.asarray(all_y_pred).ravel()

  rfe_recall = precision_score(all_y_test, all_y_pred, zero_division=0)
  rfe_precision = recall_score(all_y_test, all_y_pred, zero_division=0)
  rfe_f1 = f1_score(all_y_test, all_y_pred, average="macro", zero_division=0)
  rfe_accuracy = accuracy_score(all_y_test, all_y_pred)
  final_auc = roc_auc_score(all_y_test, all_y_pred)

  print("Pooled AUC:", final_auc)

  return(rfe_recall, rfe_f1, rfe_sel_feat_list, df_all_feature_count,
         df_best_val_score_rfe, df_best_c_rfe, df_best_gamma_rfe,test_subject_name,
         final_auc, rfe_precision, rfe_accuracy, all_y_test, all_y_pred)

######################################################################################

l1norm_result = l1norm_selection(df_full,all_subjects,external_subjects_subjects,param_grid2)

rfe_result = rfe_selection(df_full,all_subjects,external_subjects,param_grid2)

######################################################################################

def plot(x,y,label1):
  plt.plot(x,y, label = label1)
  plt.xlabel('Number of features used')
  plt.ylabel('Score')
  plt.legend(loc='best')
  plt.grid(axis = 'x')
  plt.show()

def f1_allplot(df1,df2):
  x = np.arange(1,16)
  mean1, mean2 = df1.mean(axis=0), df2.mean(axis=0)
  std1, std2 = df1.std(axis=0), df2.std(axis=0)

  y_min = min(min(mean1-std1), min(mean2-std2))
  y_max = max(max(mean1+std1), max(mean2+std2))

  # Plot 1
  plt.figure()
  plt.plot(x, mean1)
  plt.fill_between(x, mean1 - std1, mean1 + std1, alpha=0.2)
  plt.ylim(y_min, y_max)
  plt.xlabel("Number of Features")
  plt.ylabel("Accuracy")
  plt.title("F1-Macro Score with Standard Deviation (L1-Norm)")
  plt.savefig('l1norm_f1_macro_score.png') # Save the plot
  plt.show()

  # Plot 2
  plt.figure()
  plt.plot(x, mean2)
  plt.fill_between(x, mean2 - std2, mean2 + std2, alpha=0.2)
  plt.ylim(y_min, y_max)
  plt.xlabel("Number of Features")
  plt.ylabel("Accuracy")
  plt.title("F1-Macro Score with Standard Deviation (RFE)")
  plt.savefig('rfe_f1_macro_score.png') # Save the plot
  plt.show()

f1_allplot(l1norm_result[4],rfe_result[4])

df_best_val_score_l1 = l1norm_result[4].style.highlight_max(color = 'pink',axis =1)
df_best_val_score_rfe = rfe_result[4].style.highlight_max(color = 'pink',axis =1)

df_l1norm_feature_selected = l1norm_result[3].transpose()
df_rfe_feature_selected = rfe_result[3].transpose()

df_feature_selected_mean = pd.concat([df_l1norm_feature_selected.mean(axis=1), df_rfe_feature_selected.mean(axis=1)], axis=1)
df_feature_selected_mean.columns = ["L1-Norm","RFE"]

df_feature_selected_mean.to_excel('feature_selected.xlsx')

from collections import Counter

def sel_feat_count(sel_feat_list):
    counter_list = []
    for i in sel_feat_list:
        counter_list.append(Counter(i))

    return sum(counter_list, Counter())

def sel_feat_count_df(l1,rfe):
    l1_count = sel_feat_count(l1)
    rfe_count = sel_feat_count(rfe)
    one = pd.DataFrame.from_dict(l1_count, columns = ['l1_count'], orient='index').transpose()
    two = pd.DataFrame.from_dict(rfe_count,columns = ['rfe_count'], orient='index').transpose()
    one = pd.concat([one,two]).fillna(int(0))

    return one

selfeatcount = sel_feat_count_df(l1norm_result[2], rfe_result[2])
selfeatcount.to_excel('selfeatcount.xlsx')

l1norm_ytrue = l1norm_result[11]
l1norm_ypred = l1norm_result[12]

rfe_ytrue = rfe_result[11]
rfe_ypred = rfe_result[12]

df_true_pred_score = pd.DataFrame(
    [l1norm_ytrue, l1norm_ypred, rfe_ytrue, rfe_ypred],
    index=['L1-Norm-ytrue', 'L1-Norm-ypred', 'RFE-ytrue','RFE-ypred']
)

df_true_pred_score.to_excel('true_pred_score.xlsx')

######################################################
l1_precision = l1norm_result[9]
l1_recall = l1norm_result[0]
l1_f1 = l1norm_result[1]
l1_accuracy = l1norm_result[10]
l1_auc = l1norm_result[8]

rfe_precision = rfe_result[9]
rfe_recall = rfe_result[0]
rfe_f1 = rfe_result[1]
rfe_accuracy = rfe_result[10]
rfe_auc = rfe_result[8]

dataset_col = ['Precision', 'Recall', 'F1-Macro','Accuracy','AUC-ROC']

data_l1_row = [l1_precision, l1_recall, l1_f1, l1_accuracy, l1_auc]
data_rfe_row = [rfe_precision, rfe_recall, rfe_f1, rfe_accuracy, rfe_auc]

df_metric_score_raw = pd.DataFrame(
    [data_l1_row, data_rfe_row],
    columns=dataset_col,
    index=['L1-Norm', 'RFE']
)
####################################################################
# Define a custom function to bold the maximum value in each column
def bold_max(s):
    is_max = s == s.max()
    return ['font-weight: bold' if v else '' for v in is_max]
 
#####################################################################
df_metric_score = df_metric_score_raw.style.apply(bold_max, axis=0)
df_metric_score.to_excel('df_metric_score.xlsx')

df_l1norm_result = l1norm_result[4].style.highlight_max(color = 'pink',axis =1)
df_rfe_result = rfe_result[4].style.highlight_max(color = 'pink',axis =1)

df_l1norm_result.to_excel('df_l1norm_f1score.xlsx')
df_rfe_result.to_excel('df_rfe_f1score.xlsx')
