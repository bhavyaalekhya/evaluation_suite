from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, recall_score, precision_score, \
    accuracy_score
import numpy as np

def read_file(error_file):
    with open(error_file, 'r') as file:
        lines = file.readlines()

    gt_line = lines[0].strip()
    predicted_line = lines[1].strip()

    gt_data = gt_line.split('Ground Truth: ')[1].strip('[]').split(',')
    predicted_data = predicted_line.split('Predicted: ')[1].strip('[]').split(',')

    gt_data = [int(x) for x in gt_data]
    predicted_data = [int(x) for x in predicted_data]

    return gt_data, predicted_data


def metrics(ground_truth, predicted):
    modified_gt = np.array(ground_truth)
    modified_pred = np.array(predicted)

    recall = recall_score(modified_gt, modified_pred)

    precision = precision_score(modified_gt, modified_pred)

    f1 = f1_score(modified_gt, modified_pred)

    acc = accuracy_score(modified_gt, modified_pred)

    return recall, precision, f1, acc


def cf_matrix(ground_truth, predicted):
    mat = ConfusionMatrixDisplay(confusion_matrix(ground_truth, predicted))
    mat.plot()


def task_verification():
    v1_gt, v1_pred = read_file('/content/variant_1.txt')

    v1_r, v1_p, v1_f1, v1_acc = metrics(v1_gt, v1_pred)
    print(f"Task Verification: Recall: {v1_r} \nPrecision: {v1_p} \nF1: {v1_f1} \nAccuracy: {v1_acc}")

    cf_matrix(v1_gt, v1_pred)

    return v1_gt, v1_pred

def error_category(error_type, file):
    gt, predict = read_file(file)
    r, p, f1, acc = metrics(gt, predict)

    print(f"{error_type}: Recall: {r} \nPrecision: {p} \nF1: {f1} \nAccuracy: {acc}")

    cf_matrix(gt, predict)

    return predict

def combined_error(pred):
    gt, _ = task_verification()

    v2_r, v2_p, v2_f1, v2_acc = metrics(gt, pred)
    print(f"Combined Error: Recall: {v2_r} \nPrecision: {v2_p} \nF1: {v2_f1} \nAccuracy: {v2_acc}")

    cf_matrix(gt, pred)

def main():
    #technique error
    tech_pred = error_category('Technique Error', '/timechat_metrics/technique_error.txt')

    #measurement error
    me_pred = error_category('Measurement Error', '/timechat_metrics/measurement_error.txt')

    #missing error
    mi_pred = error_category('Missing Error', '/timechat_metrics/missing_error.txt')

    #order error
    o_pred = error_category('Order Error', '/timechat_metrics/order_error.txt')

    #preparation error
    p_pred = error_category('Preparation Error', '/timechat_metrics/preparation_error.txt')

    #temperature error
    temp_pred = error_category('Temperature Error', '/timechat_metrics/temperature_error.txt')

    #timing error
    ti_error = error_category('Timing Error', '/timechat_metrics/timing_error.txt')

    #combined metrics
    pred = np.logical_or.reduce((tech_pred, mi_pred, p_pred, me_pred, o_pred, temp_pred, ti_error))
    combined_error(pred)

if __name__ == '__main__':
    main()
