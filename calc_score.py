import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    f1_score
)
from sklearn.preprocessing import label_binarize


def compute_metrics(y_true, y_pred, metrics_to_compute=None, average='macro'):
    if metrics_to_compute is None:
        metrics_to_compute = ['accuracy', 'specificity', 'sensitivity', 'precision', 'recall', 'F1-score', 'AUC']

    results = {}


    if 'accuracy' in metrics_to_compute:
        # 计算准确率
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        y_pred_bin = label_binarize(y_pred, classes=np.unique(y_pred))

        accuracy = {}
        if y_true_bin.shape[1] == 1:
            # 二分类问题
            accuracy['0'] = round(accuracy_score(y_true, y_pred), 4)
            accuracy['1'] = round(accuracy_score(y_true, y_pred), 4)
            accuracy['macro'] = accuracy['0']
        else:
            # 多分类问题
            for i in range(y_true_bin.shape[1]):
                accuracy[str(i)] = round(accuracy_score(y_true_bin[:, i], y_pred_bin[:, i]), 4)
            accuracy['macro'] = round(accuracy_score(y_true, y_pred), 4)
        results['accuracy'] = accuracy

        # accuracy = accuracy_score(y_true, y_pred)
        # accuracy_dict = {str(i): accuracy for i in np.unique(y_true)}
        # accuracy_dict['macro'] = accuracy
        # results['accuracy'] = accuracy_dict

    if any(metric in metrics_to_compute for metric in ['specificity', 'sensitivity']):
        # 计算混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        # print("Confusion Matrix:\n", cm)
        results['confusion_matrix'] = cm
        if 'specificity' in metrics_to_compute:
            specificity = {}
            for i in range(len(cm)):
                tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
                fp = cm[:, i].sum() - cm[i, i]
                specificity[str(i)] = round(tn / (tn + fp), 4)
            specificity['macro'] = round(np.mean(list(specificity.values())), 4)
            results['specificity'] = specificity

        if 'sensitivity' in metrics_to_compute:
            sensitivity = {}
            for i in range(len(cm)):
                fn = cm[i, :].sum() - cm[i, i]
                tp = cm[i, i]
                sensitivity[str(i)] = round(tp / (tp + fn), 4)
            sensitivity['macro'] = round(np.mean(list(sensitivity.values())), 4)
            results['sensitivity'] = sensitivity


    if any(metric in metrics_to_compute for metric in ['precision', 'recall', 'F1-score']):
        # 计算精确率、召回率和F1-score
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

        if 'precision' in metrics_to_compute:
            precision = {str(i): round(report[str(i)]['precision'], 4) for i in np.unique(y_true)}
            precision['macro'] = round(report['macro avg']['precision'], 4)
            results['precision'] = precision

        if 'recall' in metrics_to_compute:
            recall = {str(i): round(report[str(i)]['recall'], 4) for i in np.unique(y_true)}
            recall['macro'] = round(report['macro avg']['recall'], 4)
            results['recall'] = recall

        if 'F1-score' in metrics_to_compute:
            f1 = {str(i): round(report[str(i)]['f1-score'], 4) for i in np.unique(y_true)}
            f1['macro'] = round(report['macro avg']['f1-score'], 4)
            results['F1-score'] = f1

    if 'AUC' in metrics_to_compute:
        # 计算AUC
        y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        y_pred_bin = label_binarize(y_pred, classes=np.unique(y_pred))

        auc = {}
        try:
            if y_true_bin.shape[1] == 1:
                # 二分类问题
                auc['0'] = round(roc_auc_score(y_true, y_pred), 4)
                auc['1'] = round(roc_auc_score(y_true, y_pred), 4)
                auc['macro'] = auc['0']
            else:
                # 多分类问题
                for i in range(y_true_bin.shape[1]):
                    auc[str(i)] = round(roc_auc_score(y_true_bin[:, i], y_pred_bin[:, i]), 4)
                auc['macro'] = round(roc_auc_score(y_true_bin, y_pred_bin, average=average), 4)
        except:
            auc['macro'] = 0
        results['AUC'] = auc

    return results


# 示例使用
if __name__ == "__main__":
    # 示例多分类真实标签和预测标签
    y_true_multiclass = [0, 1, 2, 2, 0, 1, 1, 2, 0, 1]
    y_pred_multiclass = [0, 2, 2, 2, 0, 0, 1, 2, 0, 2]

    metrics_to_compute = ['accuracy', 'precision', 'recall', 'F1-score', 'specificity', 'sensitivity', 'AUC']
    metrics_multiclass = compute_metrics(y_true_multiclass, y_pred_multiclass, metrics_to_compute)

    print("多分类结果：")
    for metric, value in metrics_multiclass.items():
        print(f"{metric}: {value}")

    # 示例二分类真实标签和预测标签
    y_true_binary = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    y_pred_binary = [0, 1, 0, 1, 0, 1, 1, 1, 0, 1]

    metrics_binary = compute_metrics(y_true_binary, y_pred_binary, metrics_to_compute)

    print("\n二分类结果：")
    for metric, value in metrics_binary.items():
        print(f"{metric}: {value}")
