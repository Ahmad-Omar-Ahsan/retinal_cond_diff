import pickle
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, balanced_accuracy_score,multilabel_confusion_matrix
import os
import shutil
import pandas as pd
import seaborn as sns
from argparse import ArgumentParser

def misc_measures(confusion_matrix):
    
    acc = []
    sensitivity = []
    specificity = []
    precision = []
    G = []
    F1_score_2 = []
    mcc_ = []
    
    for i in range(0, confusion_matrix.shape[0]):
        cm1=confusion_matrix[i]
        acc.append(1.*(cm1[0,0]+cm1[1,1])/np.sum(cm1))
        sensitivity_ = 1.*cm1[1,1]/(cm1[1,0]+cm1[1,1])
        sensitivity.append(sensitivity_)
        specificity_ = 1.*cm1[0,0]/(cm1[0,1]+cm1[0,0])
        specificity.append(specificity_)
        precision_ = 1.*cm1[1,1]/(cm1[1,1]+cm1[0,1])
        precision.append(precision_)
        G.append(np.sqrt(sensitivity_*specificity_))
        F1_score_2.append(2*precision_*sensitivity_/(precision_+sensitivity_))
        mcc = (cm1[0,0]*cm1[1,1]-cm1[0,1]*cm1[1,0])/np.sqrt((cm1[0,0]+cm1[0,1])*(cm1[0,0]+cm1[1,0])*(cm1[1,1]+cm1[1,0])*(cm1[1,1]+cm1[0,1]))
        mcc_.append(mcc)
        
    acc = np.array(acc).mean()
    sensitivity = np.array(sensitivity).mean()
    specificity = np.array(specificity).mean()
    precision = np.array(precision).mean()
    G = np.array(G).mean()
    F1_score_2 = np.array(F1_score_2).mean()
    mcc_ = np.array(mcc_).mean()
    
    return acc, sensitivity, specificity, precision, G, F1_score_2, mcc_


def main(args):
    score_dir = args.dir

    test_pkls_dir = [
        os.path.join(score_dir, "Test_trial_250_seed_0.pkl"),
        os.path.join(score_dir, "Test_trial_250_seed_1.pkl"),
        os.path.join(score_dir, "Test_trial_250_seed_2.pkl"),
        os.path.join(score_dir, "Test_trial_250_seed_3.pkl"),
    ]

    predict_pkls_dir = [
        os.path.join(score_dir, "Predict_trial_500_seed_0.pkl"),
        os.path.join(score_dir, "Predict_trial_500_seed_1.pkl"),
    ]

    test_pkls = []
    predict_pkls = []

    for i in range(len(test_pkls_dir)):
        with open(test_pkls_dir[i], "rb") as pickle_file:
            pkl_file = pickle.load(pickle_file)
        test_pkls.append(pkl_file)
    
    for i in range(len(predict_pkls_dir)):
        with open(predict_pkls_dir[i], "rb") as pickle_file:
            pkl_file = pickle.load(pickle_file)
        predict_pkls.append(pkl_file)

    predict_pkl_1000 = dict(predict_pkls[0])
    pkl_1000_trial = dict(test_pkls[0])

    for filename, file_dict in pkl_1000_trial.items():
        for i in range(1, len(test_pkls_dir)):
            pkl_1000_trial[filename]['class_errors_each_trial'].extend(test_pkls[i][filename]['class_errors_each_trial'])
            pkl_1000_trial[filename]['timestep'].extend(test_pkls[i][filename]['timestep'])
            pkl_1000_trial[filename]['predicted_label'] = np.argmin(np.mean(np.array(pkl_1000_trial[filename]['class_errors_each_trial']),axis=0),axis=0)

    journal_data_test = {}
    for filepath, values in pkl_1000_trial.items():
        filepath = filepath.replace("/home/ahmad99/projects/def-mwilms/ahmad99/journal_data/", "/home/ahmad/ahmad_experiments/retinal_data/journal_data/")
        # print(filepath)
        journal_data_test[filepath] = values

    with open(f"{score_dir}/test_1000.pkl", 'wb') as pickle_file:
        pickle.dump(journal_data_test, pickle_file)
    
    for filename, file_dict in predict_pkl_1000.items():
        for i in range(1, len(predict_pkls_dir)):
            predict_pkl_1000[filename]['class_errors_each_trial'].extend(predict_pkls[i][filename]['class_errors_each_trial'])
            predict_pkl_1000[filename]['timestep'].extend(predict_pkls[i][filename]['timestep'])
            predict_pkl_1000[filename]['predicted_label'] = np.argmin(np.mean(np.array(predict_pkl_1000[filename]['class_errors_each_trial']),axis=0),axis=0)

    with open(f"{score_dir}/predict_1000.pkl", 'wb') as pickle_file:
        pickle.dump(predict_pkl_1000, pickle_file)

    prediction = []
    label = []
    voting_predicted_labels = []
    #NV test
    print(f"Non-voting Test\n")
    for name, value in pkl_1000_trial.items():
        prediction.append(pkl_1000_trial[name]['predicted_label'])
        label.append(pkl_1000_trial[name]['test_label'])
        min_indexes = np.argmin(value['class_errors_each_trial'], axis=1)
        unique, counts = np.unique(min_indexes, return_counts=True)
        most_common_index = np.argmax(counts)
        most_common = unique[most_common_index]
        voting_predicted_labels.append(most_common)

    confusion_matrix = metrics.confusion_matrix(label, prediction)
    target_names = ['AMD','Cataract','DR','Glaucoma','Myopia','Normal']
    per_classes_acc = {name: accuracy_score(np.array(label) == i, np.array(prediction) == i) for i, name in enumerate(target_names)}
    



    for condition, value in per_classes_acc.items():
        print(f"{condition} accuracy: {value:.4f}")

    total_acc = 0
    for key,value in per_classes_acc.items():
        total_acc += value

    print(f"\nAverage accuracy: {total_acc/6:.4f}")
    per_classes_acc['Average Accuracy'] = total_acc/6

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["AMD", "Cataract", "DR", "Glaucoma", "Myopia", "Normal"])

    cm_display.plot(cmap=plt.cm.Blues)
    plt.title(f'Test')
    plt.savefig(f'{os.path.join(score_dir, score_dir.split("/")[-1])}_test.pdf')

    confusion_matrix = multilabel_confusion_matrix(y_true=label, y_pred=prediction,labels=[i for i in range(6)])
    acc, sensitivity, specificity, precision, G, F1, mcc = misc_measures(confusion_matrix)
    print('Sklearn Metrics -\n Acc: {:.4f}\n F1-score: {:.4f}\n MCC: {:.4f}\n Sensitivity: {:.4f}\n Specificity: {:.4f}\n Precision: {:.4f}\n G: {:.4f}\n'.format(acc, F1, mcc, sensitivity, specificity, precision, G))

    nv_metrics = {
        "Accuracy": acc,
        "F1": F1,
        "mcc": mcc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "G": G,
    }
    non_voting_scheme = [per_classes_acc, nv_metrics]
    nv_df = pd.DataFrame(non_voting_scheme)
    # Transpose the DataFrame
    nv_df = nv_df.transpose()

    # Reset the index to make the transposed DataFrame more readable
    nv_df = nv_df.reset_index()
    nv_df.to_csv(f"{score_dir}/nv_metrics.csv")

    # V Test
    print(f"Voting Test\n")
    confusion_matrix = metrics.confusion_matrix(label, voting_predicted_labels)
    target_names = ['AMD','Cataract','DR','Glaucoma','Myopia','Normal']
    voting_per_classes_acc = {name: accuracy_score(np.array(label) == i, np.array(voting_predicted_labels) == i) for i, name in enumerate(target_names)}
    

    for condition, value in voting_per_classes_acc.items():
        print(f"{condition} accuracy: {value:.4f}")

    voting_total_acc = 0
    for key,value in voting_per_classes_acc.items():
        voting_total_acc += value

    print(f"\nAverage accuracy: {voting_total_acc/6:.4f}")
    voting_per_classes_acc['Average Accuracy'] = voting_total_acc/6
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["AMD", "Cataract", "DR", "Glaucoma", "Myopia", "Normal"])

    cm_display.plot(cmap=plt.cm.Blues)
    plt.title(f'Test (Voting scheme)')
    plt.savefig(f'{os.path.join(score_dir, score_dir.split("/")[-1])}_test_voting.pdf')

    confusion_matrix = multilabel_confusion_matrix(y_true=label, y_pred=voting_predicted_labels,labels=[i for i in range(6)])
    acc, sensitivity, specificity, precision, G, F1, mcc = misc_measures(confusion_matrix)
    print('Sklearn Metrics -\n Acc: {:.4f}\n F1-score: {:.4f}\n MCC: {:.4f}\n Sensitivity: {:.4f}\n Specificity: {:.4f}\n Precision: {:.4f}\n G: {:.4f}\n'.format(acc, F1, mcc, sensitivity, specificity, precision, G))

    voting_metrics = {
        "Accuracy": acc,
        "F1": F1,
        "mcc": mcc,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "G": G,
    }
    voting_scheme = [voting_per_classes_acc, voting_metrics]
    v_df = pd.DataFrame(voting_scheme)
    # Transpose the DataFrame
    v_df = v_df.transpose()

    # Reset the index to make the transposed DataFrame more readable
    v_df = v_df.reset_index()
    v_df.to_csv(f"{score_dir}/v_metrics.csv")

    predicted_labels = []
    true_labels = []
    voting_predicted_labels = []
    for filename, value in predict_pkl_1000.items():
        predicted_labels.append(predict_pkl_1000[filename]['predicted_label'])
        true_labels.append(2)
        min_indexes = np.argmin(value['class_errors_each_trial'], axis=1)
        unique, counts = np.unique(min_indexes, return_counts=True)
        most_common_index = np.argmax(counts)
        most_common = unique[most_common_index]
        voting_predicted_labels.append(most_common)

    confusion_matrix = metrics.confusion_matrix(true_labels, predicted_labels, labels=[0,1,2,3,4,5])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["AMD", "Cataract", "DR", "Glaucoma", "Myopia", "Normal"])

    cm_display.plot(cmap=plt.cm.Blues)
    plt.title(f'Predict')
    plt.savefig(f'{os.path.join(score_dir, score_dir.split("/")[-1])}_predict.pdf')

    confusion_matrix = metrics.confusion_matrix(true_labels, voting_predicted_labels, labels=[0,1,2,3,4,5])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["AMD", "Cataract", "DR", "Glaucoma", "Myopia", "Normal"])

    cm_display.plot(cmap=plt.cm.Blues)
    plt.title(f'Predict (Voting scheme)')
    plt.savefig(f'{os.path.join(score_dir, score_dir.split("/")[-1])}_predict_voting.pdf')

if __name__=="__main__":
    parser = ArgumentParser("Driver code.")
    parser.add_argument(
        "--dir", type=str, required=True, help="Path to directory containing pkl files"
    )
    args = parser.parse_args()

    main(args)