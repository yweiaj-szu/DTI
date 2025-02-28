import colorsys
import os
from sklearn.feature_selection  import SelectKBest, chi2
from sklearn.linear_model  import LogisticRegression
import numpy
import numpy as np
from sklearn.feature_selection  import RFE
import sklearn.model_selection 
from matplotlib import pyplot as plt
from sklearn.ensemble  import RandomForestClassifier
from sklearn.multiclass  import OneVsOneClassifier
from sklearn.multioutput  import MultiOutputClassifier
from sklearn import svm
from sklearn.model_selection  import train_test_split, cross_val_score, KFold, cross_val_predict, LeaveOneOut
import scipy
from sklearn.model_selection  import GridSearchCV
from sklearn.decomposition  import PCA
from sklearn.preprocessing  import StandardScaler
from sklearn.svm  import SVC
from sklearn.metrics  import roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.metrics  import accuracy_score, roc_auc_score
import joblib
import statsmodels.api  as sm
import argparse

parse = argparse.ArgumentParser(description="parameter")
parse.add_argument('--F',  default=1, type=int, help="number of feature")  # Number of features
parse.add_argument('--output_path',  default="../output", type=str, help="Output Path")  # Output directory
parse.add_argument('--data_path',  nargs='+')  # Input feature file path(s)
parse.add_argument('--co_path',  nargs='+')  # Path to significance matrix
label_path = "./zreadBav.mat"   # Label file path
parse.add_argument('--o',  type=str, help="chinese or speed")  # Task selection: Chinese or speed
parse.add_argument('--c',  type=int, help="0 for dichotomy, 1 for three categories, 2 for binary with middle values removed")  # Classification type
parse.add_argument('--s',  type=int, help="feature size")  # Feature size selection
parse.add_argument('--model',  type=str, default="svm", help="model of predict (svm or logistic regression)")  # Model selection
parse.add_argument('--cv',  type=int, default=10, help="Cross-validation folds; cv=0 for LOOCV")  # Cross-validation settings
data_name = 'alldata'
label_name = 'zz'
co_name = 'NetworkMatrix'
arg = parse.parse_args() 

np.random.seed(77) 

def data_process(data_path, label_path):
    # Load target data
    target = scipy.io.loadmat(label_path) 
    target = target[label_name][:, :9]
    # Extract features
    datas = []
    indexs = []
    new_datas = []
    for i in arg.data_path: 
        data = scipy.io.loadmat(i) 
        data = data[data_name]
        data = np.triu(data) 
        datas.append(data) 
    for i in arg.co_path: 
        index = scipy.io.loadmat(i) 
        index = index[co_name]
        index = np.triu(index) 
        indexs.append(index) 
    if arg.o == "chinese":
        target = numpy.mean(target[:,  :2], axis=1)
        print(target)
    elif arg.o == "speed":
        target = numpy.mean(target[:,  6:9], axis=1)
    if arg.c == 0:
        mask = numpy.zeros_like(target) 
        mean = numpy.mean(target,  axis=0)
        mask = (target >= mean).astype(int)
        target = mask
        new_datas = data
    # Divide labels into three parts based on percentiles
    if arg.c == 1 or arg.c == 2:
        percentiles = np.percentile(target,  [40, 60, 100], axis=0)
        new_matrix = np.zeros_like(target) 
        new_matrix[(target >= percentiles[0]) & (target < percentiles[1])] = 2
        new_matrix[(target >= percentiles[1]) & (target <= percentiles[2])] = 1
        new_label = new_matrix
    if arg.c == 2:
        delete_index = numpy.where(new_label  == 2)[0]
        for sample in datas:
            data = numpy.delete(sample,  delete_index, axis=0)
            new_datas.append(data) 
        new_label = numpy.delete(new_label,  delete_index, axis=0)
        target = new_label
    print(target.shape) 
    brain = new_datas[0][:, :90, :90]
    all_data = new_datas[0]
    cerebellum = new_datas[0][:, 90:116, 90:116]
    joint = new_datas[0][:, :90, 90:116]
    return target, cerebellum, brain, joint, all_data, indexs

def feature_select(feature, feature_size, index, option):
    # Select values from the matrix with absolute values exceeding a threshold
    extracted_values = []
    abs_index = numpy.abs(index) 
    if option == "cerebellum":
        or_row = 90
        or_col = 90
    elif option == "brain":
        or_row = 0
        or_col = 0
    elif option == "joint":
        or_row = 0
        or_col = 90
    else:
        or_row = 0
        or_col = 0
    for sample_matrix in feature:
        indices = np.argpartition(abs_index,  -feature_size, axis=None)[-feature_size:]
        indices = np.unravel_index(indices,  abs_index.shape) 
        tuple_indices = (indices[0], indices[1])
        top_values = sample_matrix[tuple_indices]
        extracted_values.append(top_values) 
    max_length = max(len(values) for values in extracted_values)
    result = np.zeros((len(extracted_values),  feature_size))
    for i, values in enumerate(extracted_values):
        result[i, :len(values)] = values
    select_feature = result
    tuple_indices = np.array(tuple_indices) 
    tuple_indices[0] += or_row
    tuple_indices[1] += or_col
    np.savetxt(f"{arg.o}_{option}_id",  tuple_indices)
    return select_feature, tuple_indices

def feature_cal(feature, target):
    # Perform cross-validation using the selected model
    log_score = cross_val_score(model, feature, target, cv=cv, scoring='accuracy')
    return log_score

def cal_metric(select_feature, target):
    # Define cross-validation metrics
    scoring = ['accuracy', 'precision', 'recall']
    results = cross_val_score(model, select_feature, target, cv=cv, scoring='accuracy')
    accuracy = results.mean() 
    sensitivity_results = []
    specificity_results = []
    ppv_results = []
    npv_results = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,  1, 100)
    for train_index, test_index in cv.split(select_feature,  target):
        X_train, X_test = select_feature[train_index], select_feature[test_index]
        y_train, y_test = target[train_index], target[test_index]
        model.fit(X_train,  y_train)
        y_pred = model.predict(X_test) 
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        sensitivity = tp / (tp + fn)
        specificity = tn / (tn + fp)
        ppv = tp / (tp + fp)
        npv = tn / (tn + fn)
        sensitivity_results.append(sensitivity) 
        specificity_results.append(specificity) 
        ppv_results.append(ppv) 
        npv_results.append(npv) 
        y_score = model.predict_proba(select_feature[test_index])[:,  1]
        fpr, tpr, thresholds = roc_curve(target[test_index], y_score)
        tprs.append(np.interp(mean_fpr,  fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc) 
        plt.plot(fpr,  tpr, lw=1, alpha=0.3)
    return accuracy, sensitivity_results, specificity_results, ppv_results, npv_results, aucs, results

def save_txt(accuracy, sensitivity_results, specificity_results, ppv_results, npv_results, option, aucs, accs, folder_path):
    # Save metrics to text file
    with open(folder_path + f'/feature_{option}_cross_validation_metrics.txt',  'w') as file:
        file.write(f'Accuracy:  {accuracy}\n')
        file.write(f'Mean  Sensitivity: {np.mean(sensitivity_results)}\n') 
        file.write(f'Mean  Specificity: {np.mean(specificity_results)}\n') 
        file.write(f'Mean  PPV: {np.mean(ppv_results)}\n') 
        file.write(f'Mean  NPV: {np.mean(npv_results)}\n') 
        file.write(f'AUC  Cross Validation: {aucs}\n')
        file.write(f'ACC  Cross Validation: {accs}\n')

def save_metric(target):
    num_colors = arg.F
    colors = []
    folder_path = os.path.join(arg.output_path,  "metric")
    if not os.path.exists(folder_path): 
        os.makedirs(folder_path) 
    for i in range(num_colors):
        hue = (i * 0.618033988749895) % 1
        saturation = 0.6
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue,  saturation, value)
        colors.append(rgb) 
    for i in range(arg.F):
        select_feature = feature_select(datas, arg.s, indexs[i], "")
        print(select_feature.shape) 
        accuracy, sensitivity_results, specificity_results, ppv_results, npv_results, aucs, accs = cal_metric(select_feature[i], target)
        save_txt(accuracy, sensitivity_results, specificity_results, ppv_results, npv_results, i, aucs, accs, folder_path)
    plt.savefig(os.path.join(folder_path,  "ROC.jpg")) 

def draw_feature_size_map():
    cerebellum_scores = []
    brain_scores = []
    joint_scores = []
    together_scores = []
    for i in s:
        cerebellum_feature, cere_indices = feature_select(cerebellum, i, cere_R, "cerebellum")
        brain_feature, brain_indices = feature_select(brain, i, brain_R, "brain")
        joint_feature, joint_indices = feature_select(joint, i, joint_R, "joint")
        together_feature, together_indices = feature_select(together_data, i, ch_R, "together")
        cerebellum_score = feature_cal(cerebellum_feature, target)
        brain_score = feature_cal(brain_feature, target)
        joint_score = feature_cal(joint_feature, target)
        together_score = feature_cal(together_feature, target)
        cerebellum_scores.append(np.mean(cerebellum_score)) 
        brain_scores.append(np.mean(brain_score)) 
        joint_scores.append(np.mean(joint_score)) 
        together_scores.append(np.mean(together_score)) 
        print(i, np.mean(together_score)) 
    plt.plot(s,  cerebellum_scores, marker='o', label='Cerebellum', color='blue')
    plt.plot(s,  brain_scores, marker='s', label='Brain', color='green')
    plt.plot(s,  joint_scores, marker='^', label='Joint', color='red')
    plt.plot(s,  together_scores, label='Together', color='black')
    plt.legend() 
    plt.savefig(f"{arg.o}.jpg") 
    plt.show() 

def draw_roc(select_feature, target, option):
    model = SVC(probability=True, kernel='rbf')
    cv2 = cv
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0,  1, 100)
    for train, test in cv2.split(select_feature,  target):
        model.fit(select_feature[train],  target[train])
        y_score = model.predict_proba(select_feature[test])[:,  1]
        fpr, tpr, thresholds = roc_curve(target[test], y_score)
        tprs.append(np.interp(mean_fpr,  fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc) 
        plt.plot(fpr,  tpr, lw=1, alpha=0.3)
    with open(f'AUC_Cross_validation.txt',  'a') as file:
        file.write(f'{option}  AUC Cross Validation: {aucs}\n')
    mean_tpr = np.mean(tprs,  axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs) 
    color = 'blue'
    if option == 'brain':
        color = 'green'
    elif option == 'joint':
        color = 'red'
    elif option == 'together':
        color = 'black'
    plt.plot(mean_fpr,  mean_tpr, color=color, label=f'{option} Mean ROC (AUC = %0.2f)' % mean_auc, lw=2)
    plt.plot([0,  1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0,  1.0])
    plt.ylim([0.0,  1.05])
    plt.xlabel('False  Positive Rate')
    plt.ylabel('True  Positive Rate')
    plt.title(f'ROC  Curve with Cross-Validation in {arg.o}')
    plt.legend(loc="lower  right")

def draw_all_roc(feature_size):
    cerebellum_feature, cere_indices = feature_select(cerebellum_data, feature_size, cere_R, "cerebellum")
    brain_feature
