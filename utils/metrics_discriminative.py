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

    test_csv_dir = os.path.join(score_dir, "Test_seed_0.csv")
    predict_csv_dir = os.path.join(score_dir, "Predict_seed_0.csv")

    test_df = pd.read_csv(test_csv_dir)
    predict_df = pd.read_csv(predict_csv_dir)


    prediction = test_df['Predicted Label'].tolist()
    label = test_df['Actual Label'].tolist()     
            
    
    confusion_matrix = metrics.confusion_matrix(label, prediction)
    target_names = ['AMD','Cataract','DR','Glaucoma','Myopia','Normal']
    per_classes_acc = {name: accuracy_score(np.array(label) == i, np.array(prediction) == i) for i, name in enumerate(target_names)}
    # Extract true positives, false positives, true negatives, and false negatives

    for condition, value in per_classes_acc.items():
        print(f"{condition} accuracy: {value:.4f}")

    total_acc = 0
    for key,value in per_classes_acc.items():
        total_acc += value

    print(f"\nAverage accuracy: {total_acc/6:.4f}")
    per_classes_acc['average_accuracy'] = total_acc/6
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["AMD", "Cataract", "DR", "Glaucoma", "Myopia", "Normal"])

    cm_display.plot(cmap=plt.cm.Blues)
    plt.title(f"Test")
    plt.savefig(f'{os.path.join(score_dir, score_dir.split("/")[-1])}_test.pdf')
    # plt.show() 

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
    nv_df.to_csv(f"{score_dir}/metrics.csv")


    predicted_labels = predict_df['Predicted Label'].tolist()
    true_labels = [2 for i in range(len(predicted_labels))]
    # Map true labels: 2 as correct (1) and others as incorrect (0)
    # Calculate confusion matrix for multi-class classification
    confusion_matrix = metrics.confusion_matrix(true_labels, predicted_labels, labels=[0,1,2,3,4,5])
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ["AMD", "Cataract", "DR", "Glaucoma", "Myopia", "Normal"])

    cm_display.plot(cmap=plt.cm.Blues)
    plt.title(f'Predict')
    plt.savefig(f'{os.path.join(score_dir, score_dir.split("/")[-1])}_predict.pdf')




if __name__=="__main__":
    parser = ArgumentParser("Driver code.")
    parser.add_argument(
        "--dir", type=str, required=True, help="Path to directory containing pkl files"
    )
    args = parser.parse_args()

    main(args)