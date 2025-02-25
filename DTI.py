import colorsys
import os
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
import numpy
import numpy as np
from sklearn.feature_selection import RFE
import sklearn.model_selection
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict, LeaveOneOut
import scipy
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import statsmodels.api as sm
import argparse

parse = argparse.ArgumentParser(description="parameter")
parse.add_argument('--F', default=1, type=int, help="Number of feature types")
parse.add_argument('--output_path', default="../output", type=str, help="Output directory path")
parse.add_argument('--data_path', nargs='+')  # Input feature file path, e.g., 'zz'
parse.add_argument('--co_path', nargs='+')  # Significance matrix path
label_path = "./zreadBav.mat"  # Label file path
parse.add_argument('--o', type=str, help="Task selection: 'chinese' or 'speed'")
parse.add_argument('--c', type=int,
                   help="0 for binary classification, 1 for three categories, 2 for binary classification with intermediate values removed")
parse.add_argument('--s', type=int, help="Feature size")
parse.add_argument('--model', type=str, default="svm", help="Model choice: 'svm' or 'logistic regression'")
parse.add_argument('--cv', type=int, default=10, help="Cross-validation method: cv=0 for LOOCV")

arg = parse.parse_args()


def data_process(data_path, label_path):
    # Load target labels
    target = scipy.io.loadmat(label_path)
    target = target[label_name][:, :9]

    # Load feature data
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
        target = numpy.mean(target[:, :2], axis=1)
    elif arg.o == "speed":
        target = numpy.mean(target[:, 6:9], axis=1)

    if arg.c == 0:
        mask = numpy.zeros_like(target)
        mean = numpy.mean(target, axis=0)
        mask = (target >= mean).astype(int)
        target = mask
        new_datas = data

    if arg.c == 1 or arg.c == 2:
        percentiles = np.percentile(target, [40, 60, 100], axis=0)
        new_matrix = np.zeros_like(target)
        new_matrix[(target >= percentiles[0]) & (target < percentiles[1])] = 2
        new_matrix[(target >= percentiles[1]) & (target <= percentiles[2])] = 1
        new_label = new_matrix

    if arg.c == 2:
        delete_index = numpy.where_label(new == 2)[0]
        for sample in datas:
            data = numpy.delete(sample, delete_index, axis=0)
            new_datas.append(data)
        new_label = numpy.delete(new_label, delete_index, axis=0)
        target = new_label

    print(target.shape)
    return target, new_datas, indexs


def feature_select(feature, feature_size, index):
    # Select features based on absolute values greater than a threshold
    abs_index = numpy.abs(index)
    selected_result = []
    for i in range(0, arg.F):
        extracted_values = []
        for sample_matrix in feature[i]:
            indices = np.argpartition(abs_index[i], -feature_size, axis=None)[-feature_size:]
            indices = np.unravel_index(indices, abs_index[i].shape)
            tuple_indices = (indices[0], indices[1])
            top_values = sample_matrix[tuple_indices]
            extracted_values.append(top_values)

        result = np.zeros((len(extracted_values), feature_size))
        for i, values in enumerate(extracted_values):
            result[i, :len(values)] = values

        select_feature = np.expand_dims(result, axis=0)
        selected_result.append(select_feature)

    selected_result = np.concatenate(selected_result, axis=0)
    return selected_result

def permutation_Test(feature1,feature2,target,sp_target,task1,task2):
    num_replacements = 5000
    model1 = SVC(probability=True,kernel='linear')

    target1 = np.zeros(len(target))
    target2 = np.ones(len(sp_target))
    result_1 = np.column_stack((target,target1))
    permutation_matrix = result_1
    shuffle1 = target.copy()
    results = np.zeros((num_replacements,2))
    for i in range(num_replacements):
        # shuffle label
        indices= np.random.choice(permutation_matrix.shape[0],2,replace=False)
        # permutation_matrix[indices[0],1] ,permutation_matrix[indices[1],1]= permutation_matrix[indices[1],1] , permutation_matrix[indices[0],1]
        zero_indices = np.where(shuffle1 == 0)[0]


        one_indices = np.where(shuffle1 == 1)[0]


        zero_choice = np.random.choice(zero_indices)


        one_choice = np.random.choice(one_indices)


        shuffle1[zero_choice], shuffle1[one_choice] = shuffle1[one_choice], shuffle1[zero_choice]

        #计算ACC.ROC
        # shuffle1 = permutation_matrix[permutation_matrix[:,1] == 0][:,0]
        model1.fit(feature1,shuffle1)
        output1 = model1.predict(feature1)
        # shuffle2 = permutation_matrix[permutation_matrix[:,1] == 1][:,0]
        acc_scores1 = accuracy_score(target,output1)
        # acc_scores2 = accuracy_score(sp_target,shuffle2)
        # acc_scores = acc_scores1 - acc_scores2
        roc_scores1 = roc_auc_score(target,output1)
        # roc_scores2 = roc_auc_score(sp_target,shuffle2)
        # roc_scores = roc_scores1 - roc_scores2
        results[i, 0] = acc_scores1
        results[i, 1] = roc_scores1
    sorted_matrix = np.sort(results[:,0])[::-1]
    value_5 = sorted_matrix[2]
    value_25 = sorted_matrix[24]
    value_50 = sorted_matrix[125]
    roc_sorted_matrix = np.sort(results[:,1])[::-1]
    roc_value_5 = roc_sorted_matrix[2]
    roc_value_25 = roc_sorted_matrix[24]
    roc_value_50 = roc_sorted_matrix[125]
    plt.hist(sorted_matrix, bins=15, edgecolor='black')
    plt.title(f"{task1}")
    plt.savefig(f"acc map {task1}")
    plt.close()
    plt.hist(roc_sorted_matrix, bins=15, edgecolor='black')
    plt.title(f"{task2}")
    plt.savefig(f"roc map {task1}")
    plt.close()
    with open(f"top_values_CandS_{task1}_{task2}.txt",'w') as file:
        file.write("ACC \t ROC\n")
        file.write(f"{value_5} , {roc_value_5}\n")
        file.write(f"{value_25}, {roc_value_25}\n")
        file.write(f"{value_50}, {roc_value_50}\n")

def feature_cal(feature, target):
    # Perform logistic regression and calculate cross-validation scores
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
    mean_fpr = np.linspace(0, 1, 100)

    for train_index, test_index in cv.split(select_feature, target):
        X_train, X_test = select_feature[train_index], select_feature[test_index]
        y_train, y_test = target[train_index], target[test_index]

        model.fit(X_train, y_train)
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

        y_score = model.predict_proba(select_feature[test_index])[:, 1]
        fpr, tpr, thresholds = roc_curve(target[test_index], y_score)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3)

    return accuracy, sensitivity_results, specificity_results, ppv_results, npv_results, aucs, results


def save_txt(accuracy, sensitivity_results, specificity_results, ppv_results, npv_results, option, aucs, accs,
             folder_path):
    with open(folder_path + f'/feature_{option}_cross_validation_metrics.txt', 'w') as file:
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
    folder_path = os.path.join(output_path, "metric")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    for i in range(num_colors):
        hue = (i * 0.618033988749895) % 1
        saturation = 0.6
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)

    for i in range(0, arg.F):
        select_feature = feature_select(datas, arg.s, indexs)
        print(select_feature.shape)

        accuracy, sensitivity_results, specificity_results, ppv_results, npv_results, aucs, accs = cal_metric(
            select_feature[i], target)

        save_txt(accuracy, sensitivity_results, specificity_results, ppv_results, npv_results, i, aucs, accs,
                 folder_path)

        draw_roc(select_feature[i], target, i, colors[i])

    plt.savefig(os.path.join(folder_path, "ROC.jpg"))


def draw_feature_size_map():
    num_colors = arg.F
    colors = []
    for i in range(num_colors):
        hue = (i * 0.618033988749895) % 1
        saturation = 0.6
        value = 0.9
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        colors.append(rgb)

    markers = ['o', 's', '^', 'x', 'D', 'v', '*', 'P', 'h', '+']

    for count in range(arg.F):
        scores = []
        for i in s:
            selected_data = feature_select(datas, i, indexs)
            score = feature_cal(selected_data[count], target)
            scores.append(np.mean(score))

        plt.plot(s, scores, marker=markers[count], label=f'Feature {count}', color=colors[count])

    plt.legend()
    folder_path = os.path.join(output_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    plt.savefig(os.path.join(output_path, "feature_size_map"))
    plt.show()


def draw_roc(select_feature, target, option, color):
    cv2 = cv

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for train, test in cv2.split(select_feature, target):
        model.fit(select_feature[train], target[train])
        y_score = model.predict_proba(select_feature[test])[:, 1]

        fpr, tpr, thresholds = roc_curve(target[test], y_score)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        plt.plot(fpr, tpr, lw=1, alpha=0.3)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)

    plt.plot(mean_fpr, mean_tpr, color=color, label=f'Feature {option} Mean ROC (AUC = %0.2f)' % mean_auc, lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False  Positive Rate')
    plt.ylabel('True  Positive Rate')
    plt.title('ROC  Curve with Cross-Validation')
    plt.legend(loc="lower  right")


def select_best_aic(feature, ch_R):
    best_aics = []
    best_bics = []
    best_features = []

    for i in range(0, arg.F):
        best_aic = np.inf
        best_bic = np.inf
        best_feature = None

        for num_features in s:
            to_select_feature = feature_select(datas, num_features, ch_R)
            to_select_feature = to_select_feature[i, :, :num_features]

            logit_model = sm.Logit(target, sm.add_constant(to_select_feature)).fit_regularized(alpha=0.1, L1_wt=0.1)
            aic = logit_model.aic
            bic = logit_model.bic

            print(f"Feature {i}'s feature_size: {num_features}, AIC: {aic}, BIC: {bic}")

            if aic < best_aic:
                best_aic = aic
                best_feature = num_features

            if bic < best_bic:
                best_bic = bic
                best_feature = num_features

        best_aics.append(best_aic)
        best_bics.append(best_bic)
        best_features.append(best_feature)

    folder_path = os.path.join(output_path, "aic")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(os.path.join(folder_path, "best_aic_bic.txt"), "w") as file:
        for i in range(0, arg.F):
            file.write(f"Feature  {i}'s Best AIC: {best_aics[i]} with {best_features[i]} features\n")
            file.write(f"Feature  {i}'s Best BIC: {best_bics[i]} with {best_features[i]} features\n")


if __name__ == "__main__":
    feature_size = arg.s
    output_path = arg.output_path

    if arg.cv == 0:
        cv = LeaveOneOut()
    else:
        cv = KFold(n_splits=arg.cv, shuffle=True, random_state=42)

    if arg.model == 'log':
        model = LogisticRegression()
    elif arg.model == 'svm':
        model = SVC(probability=True)
    elif arg.model == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)

    target, datas, indexs = data_process(arg.data_path, label_path)

    if arg.F == 1:
        datas = np.expand_dims(datas, axis=0)

    s = range(1, 100)

    draw_feature_size_map()
    # select_best_aic(datas, indexs)
    save_metric(target)
