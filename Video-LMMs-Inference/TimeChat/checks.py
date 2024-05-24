import pandas as pd

accuracy_file = './accuracy.csv'
f1_score_file = './f1_score.csv'
precision_file = './precision.csv'
recall_file = './recall.csv'

df = pd.read_csv(accuracy_file)

df['timechat - accuracy'] = pd.to_numeric(df['timechat - accuracy'], errors='coerce')

df = df.dropna(subset=['timechat - accuracy'])

# Calculate the mean of the accuracy values, ignoring NaN values
overall_accuracy = df['timechat - accuracy'].mean()

df1 = pd.read_csv(f1_score_file)

df1['timechat - f1_score'] = pd.to_numeric(df1['timechat - f1_score'], errors='coerce')


df2 = pd.read_csv(precision_file)

df2['timechat - precision'] = pd.to_numeric(df2['timechat - precision'], errors='coerce')


df3 = pd.read_csv(recall_file)

df3['timechat - recall'] = pd.to_numeric(df3['timechat - recall'], errors='coerce')

df4 = pd.merge(df3, df2, on='Step')
df4 = pd.merge(df4, df1, on='Step')

df_selected = df4[['Step', 'timechat - recall', 'timechat - precision', 'timechat - f1_score']]
df_selected.columns = ['Step', 'r', 'p', 'f1']

tp = 0
fp = 0
fn = 0

for idx, row in df_selected.iterrows():
    if row['p'] == 0 or row['r'] == 0:
        continue
    
    tp_i = (row['f1'] * (row['p'] + row['r'])) / (2 * row['p'] * row['r'])
    fp_i = tp_i * ((1 / row['p']) - 1)
    fn_i = tp_i * ((1 / row['r']) - 1)
    
    tp += tp_i
    fp += fp_i
    fn += fn_i

# Calculate overall precision, recall, and F1 score
if (fp + tp) == 0 or (tp + fn) == 0:
    overall_precision = 0
    overall_recall = 0
    overall_f1 = 0
else:
    overall_precision = tp / (fp + tp)
    overall_recall = tp / (tp + fn)
    if (overall_precision + overall_recall) == 0:
        overall_f1 = 0
    else:
        overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall)

print(f'Overall Accuracy: {overall_accuracy:.2f}')
print(f'Overall F1: {overall_f1:.2f}')
print(f'Overall Precision: {overall_precision:.2f}')
print(f'Overall Recall: {overall_recall:.2f}')