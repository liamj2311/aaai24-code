from sklearn.metrics import recall_score
import numpy as np
import pandas as pd

# Recall below 25 percentile and above 75 percentile (top level vs rest of levels))
def compute_recall(data, categories, top_level):
    recall_dfs = []
    for category in categories:
        recall_results = {}
        for i in [1, 4]:
            condition_top = (data['score_MATq'] == i) & (data[category] == top_level)
            condition_low = (data['score_MATq'] == i) & (data[category] != top_level)
            scores_MATq_top = data.loc[condition_top, 'score_MATq']
            scores_MATq_low = data.loc[condition_low, 'score_MATq']

            for variable in ["pred1", "pred2", "pred3", "pred_C", "pred_X"]:
                scores_MAT_variable_top = data.loc[condition_top, f'score_MAT_{variable}']
                scores_MAT_variable_low = data.loc[condition_low, f'score_MAT_{variable}']

                recall_top = recall_score(scores_MATq_top, scores_MAT_variable_top, average='micro')
                recall_low = recall_score(scores_MATq_low, scores_MAT_variable_low, average='micro')

                recall_results[f'recall_top_{variable}_{i}'] = recall_top
                recall_results[f'recall_low_{variable}_{i}'] = recall_low

        recall_df = pd.DataFrame(list(recall_results.items()), columns=['Metric', 'Recall'])
        recall_df[['Group', 'Model', 'Percentile']] = recall_df['Metric'].str.extract(r'recall_(top|low)_(\w+)_(\d+)')
        recall_df['Recall'] = recall_df['Recall'].round(4)
        recall_df['Variable'] = category
        recall_df['Prediction'] = '25_75'
        recall_dfs.append(recall_df[['Group', 'Model', 'Percentile', 'Recall', 'Variable', 'Prediction']])
    return recall_dfs

# Recall between 25 percentile and 75 percentile (top level vs rest of levels))
def compute_recall_terciles(data, categories, top_level):
    recall_dfs = []
    for category in categories:
        recall_results = {}
        i = 2
        condition_top = (data['score_MATq'] == i) & (data[category] == top_level)
        condition_low = (data['score_MATq'] == i) & (data[category] != top_level)
        scores_MATq_top = data.loc[condition_top, 'score_MATq']
        scores_MATq_low = data.loc[condition_low, 'score_MATq']

        for variable in ["pred1_t", "pred2_t", "pred3_t", "pred_C_t", "pred_X_t"]:
            scores_MAT_variable_top = data.loc[condition_top, f'score_MAT_{variable}']
            scores_MAT_variable_low = data.loc[condition_low, f'score_MAT_{variable}']

            recall_top = recall_score(scores_MATq_top, scores_MAT_variable_top, average='micro')
            recall_low = recall_score(scores_MATq_low, scores_MAT_variable_low, average='micro')

            recall_results[f'recall_top_{variable}_{i}'] = recall_top
            recall_results[f'recall_low_{variable}_{i}'] = recall_low

        recall_df = pd.DataFrame(list(recall_results.items()), columns=['Metric', 'Recall'])
        recall_df[['Group', 'Model', 'Tercile']] = recall_df['Metric'].str.extract(r'recall_(top|low)_(\w+)_(\d+)')
        recall_df['Recall'] = recall_df['Recall'].round(4)
        recall_df['Variable'] = category
        recall_df['Prediction'] = 'between25_75'
        recall_dfs.append(recall_df[['Group', 'Model', 'Tercile', 'Recall', 'Variable', 'Prediction']])
    return recall_dfs

# Recall below and above the median (top level vs rest of levels))
def compute_recall_median(data, categories, top_level):
    recall_dfs = []
    score_pairs = [(1, 2), (3, 4)]
    for category in categories:
        recall_results = {}
        for pair in score_pairs:
            condition_top = ((data['score_MATq'] == pair[0]) | (data['score_MATq'] == pair[1])) & (data[category] == top_level)
            condition_low = ((data['score_MATq'] == pair[0]) | (data['score_MATq'] == pair[1])) & (data[category] != top_level)
            scores_MATq_top = data.loc[condition_top, 'score_MATq']
            scores_MATq_low = data.loc[condition_low, 'score_MATq']
            scores_MATq_top_binary = scores_MATq_top.apply(lambda x: 1 if x in pair else 0)
            scores_MATq_low_binary = scores_MATq_low.apply(lambda x: 1 if x in pair else 0)

            for variable in ["pred1", "pred2", "pred3", "pred_C", "pred_X"]:
                scores_MAT_variable_top = data.loc[condition_top, f'score_MAT_{variable}']
                scores_MAT_variable_low = data.loc[condition_low, f'score_MAT_{variable}']
                scores_MAT_variable_top_binary = scores_MAT_variable_top.apply(lambda x: 1 if x in pair else 0)
                scores_MAT_variable_low_binary = scores_MAT_variable_low.apply(lambda x: 1 if x in pair else 0)
                
                recall_top = recall_score(scores_MATq_top_binary, scores_MAT_variable_top_binary, average='binary')
                recall_low = recall_score(scores_MATq_low_binary, scores_MAT_variable_low_binary, average='binary')
                
                recall_results[f'recall_top_{variable}_{pair[0]}_{pair[1]}'] = recall_top
                recall_results[f'recall_low_{variable}_{pair[0]}_{pair[1]}'] = recall_low

        recall_df = pd.DataFrame(list(recall_results.items()), columns=['Metric', 'Recall'])
        recall_df[['Group', 'Model', 'Pair1', 'Pair2']] = recall_df['Metric'].str.extract(r'recall_(top|low)_(\w+)_(\d+)_(\d+)')
        recall_df['Recall'] = recall_df['Recall'].round(4)
        recall_df['Variable'] = category
        recall_df['Prediction'] = 'median'
        recall_dfs.append(recall_df[['Group', 'Model', 'Pair1', 'Pair2', 'Recall', 'Variable', 'Prediction']])
    return recall_dfs

# Calculate equalized odds for each variable
def calculate_odds(row_value, top_value):
    return row_value / top_value if top_value != 0 else None

#############

def mld(vals):
    sum = 0
    avg = np.mean(vals)
    for val in vals:
        if val == 0.0:
            sum += 0
        else:
            sum += np.log(avg/val)
    return sum/len(vals)

def gini(vals):
    # this function is based on the third equation in
    # https://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    sorted_vals = sorted(vals, reverse=False)
    # values cannot be 0
    sorted_vals = [x + 0.0000001 for x in sorted_vals]
    # multiply by 100 each probability to have numerical stability
    # with values between 0 and 1 and n > 2 negative values were being produced
    sorted_vals = [x * 100 for x in sorted_vals]
    den = len(sorted_vals) * sum(sorted_vals)
    num = 0
    for i, v in enumerate(sorted_vals):
        num += (2*(i+1) - len(sorted_vals) - 1) * v
    return num/den

def expected_value(probs, labels):
    return np.sum(np.dot(np.array(probs), np.array(labels))) / len(probs)

def compute_moment(moment: str, probs, labels):
    if moment == "mean":
        return expected_value(probs=probs, labels=labels)
    
def iop(df: pd.DataFrame, sensitive_attr: str, labels = [0, 1], moment="mean", ineq_index="mld"):
    sensitive_vals = np.unique(df[sensitive_attr].values[~np.isnan(df[sensitive_attr].values)])
    val_to_count = {val: df.loc[df[sensitive_attr] == val].shape[0] for val in sensitive_vals}
    moments = []
    for val in sensitive_vals:
        probs = []
        for label in labels:
            probs.append(df.loc[(df[sensitive_attr] == val) & (df["label"] == label)].shape[0] / df.shape[0])
        w = val_to_count[val] / df.shape[0]
        moments.append(w * compute_moment(moment=moment, probs=probs, labels=labels))
    if ineq_index == "mld":
        return mld(moments)
    elif ineq_index == "gini":
        return gini(moments)
    else:
        return None
    
def binarise_predictions(preds: pd.Series, percentile_range: str):
    percentiles = pd.qcut(preds, 4, labels=[x for x in range(1, 5)])
    if percentile_range == "below-25": 
        return pd.Series((percentiles <= 1).astype(int))
    elif percentile_range == "above-75":
        return pd.Series((percentiles > 3).astype(int))
    elif percentile_range == "between-25-75":
        res = pd.Series((percentiles > 1) & (percentiles < 4)).astype(int)
        return res
    
def print_iop(iop_val, percentile_range, sensitive_attr):
    pr_verbose = {
        "below-25": "below the 25th percentile",
        "above-75": "above the 75th percentile",
        "between-25-75": "between the 25th and the 75th percentile"
    }

    print(f"IOP value for predictions {pr_verbose[percentile_range]} for groups identified by values of sensitive attribute {sensitive_attr}: {iop_val}")