import os, pickle, pybedtools
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression

def train(feature_file, model_file):
    """
    Description:
        Function for training of the sklearn logistic classification.

    Arguments:
        feature_file str: Filename of the feature file for training.
        model_file str: Filename of the output model.
    """
    feature_df = pd.read_csv(feature_file, sep="\t")

    model = LogisticRegression(solver="newton-cg", penalty=None, max_iter=10000)
    # Remove feature vectors with label -1
    feature_df = feature_df[feature_df['label'] != -1]
    labels = feature_df['label']
    data = feature_df.drop(
        columns=['chrom', 'start', 'end', 'sample', 'label']).values

    model.fit(data, labels.astype(int))

    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    pickle.dump(model, open(model_file, "wb"))
    
 
def infer(feature_file, model_file, output):
    """
    Description:
        Function for inference using the sklearn logistic classifciation.

    Arguments:
        feature_file str: Filename of the feature file for inference.
        model_file str: Filename of the trained logistic regression model.
        prediction_file str: Filename of the output predictions.
    """
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

    feature_df = pd.read_csv(feature_file, sep="\t")
    data = feature_df.drop(columns=['chrom', 'start', 'end', 'sample']).values

    predictions = model.predict_proba(data)

    classes = model.classes_
    columns = ['chrom', 'start', 'end', 'sample']
    for i in range(len(classes)):
        feature_df[f'class_{classes[i]}_prob'] = predictions[:,i]
        columns.append(f'class_{classes[i]}_prob')

    feature_df = feature_df[columns]
    feature_df.sort_values(
        by=['sample', 'chrom', 'start', 'end']
    ).to_csv(output, sep="\t", index=False)
    
    
def get_inferred_tracts(prediction_file, cutoff, output):
    """
    Description:
        Obtains inferred introgressed fragments from predictions.

    Arguments:
        prediction_file str: Name of the file containing predictions from
                             a model.
        cutoff float: Probability threshold to determine whether a fragment
                      is introgressed or not.
        output str: Name of the output file storing the inferred introgressed
                    fragments.
    """
    prediction_df = pd.read_csv(prediction_file, sep="\t")
    inferred_tracts = prediction_df[prediction_df['class_1_prob'] > cutoff]
    open(output, 'w').close()
    for s in inferred_tracts['sample'].unique():
        sample_tracts = inferred_tracts[inferred_tracts['sample'] == s]
        sample_tracts = pybedtools.BedTool.from_dataframe(
                sample_tracts
            ).sort().merge().to_dataframe()
        sample_tracts['sample'] = s
        sample_tracts.to_csv(output, sep="\t", mode='a',
                             header=False, index=False)
                             
                             
def cal_pr(ntruth_tracts, ninferred_tracts, ntrue_positives):
    """
    Description:
        Calculates precision and recall.

    Arguments:
        ntruth_tracts int: Length of true introgressed fragments.
        ninferred_tracts int: Length of inferred introgressed fragments.
        ntrue_positives int: Length of fragments belonging to true positives.

    Returns:
        precision float: Estimated precision.
        recall float: Estimated recall.
    """
    if float(ninferred_tracts) == 0: precision = np.nan
    else: precision = ntrue_positives / float(ninferred_tracts) * 100
    if float(ntruth_tracts) == 0: recall = np.nan
    else: recall = ntrue_positives / float(ntruth_tracts) * 100

    return precision, recall
    
    
def evaluate(truth_tract_file, inferred_tract_file, cutoff, output):
    """
    Description:
        Evaluates model performance with precision and recall.

    Arguments:
        truth_tract_file str: Name of the file containing true fragments.
        inferred_tract_file str: Name of the file containing inferred fragments
        cutoff float: Probability threshold to determine whether a fragment
                      is introgressed or not.
        output str: Name of the output file storing the model performance.
    """
    try:
        truth_tracts = pd.read_csv(truth_tract_file, sep="\t", header=None)
    except pd.errors.EmptyDataError:
        truth_tracts_samples = []
    else:
        truth_tracts.columns = ['chrom', 'start', 'end', 'sample']
        truth_tracts_samples = truth_tracts['sample'].unique()

    try:
        inferred_tracts = pd.read_csv(inferred_tract_file, sep="\t",
                                      header=None)
    except pd.errors.EmptyDataError:
        inferred_tracts_samples = []
    else:
        inferred_tracts.columns = ['chrom', 'start', 'end', 'sample']
        inferred_tracts_samples = inferred_tracts['sample'].unique()

    res = pd.DataFrame(columns=['precision', 'recall', 'cutoff'])

    sum_ntruth_tracts = 0
    sum_ninferred_tracts = 0
    sum_ntrue_positives = 0

    for s in np.intersect1d(truth_tracts_samples, inferred_tracts_samples):
        ind_truth_tracts = truth_tracts[
            truth_tracts['sample'] == s
        ][['chrom', 'start', 'end']]
        ind_inferred_tracts = inferred_tracts[
            inferred_tracts['sample'] == s
        ][['chrom', 'start', 'end']]

        ind_truth_tracts = pybedtools.BedTool.from_dataframe(
            ind_truth_tracts
        ).sort().merge()
        ind_inferred_tracts = pybedtools.BedTool.from_dataframe(
            ind_inferred_tracts
        ).sort().merge()

        ntruth_tracts = sum([x.stop - x.start for x in (ind_truth_tracts)])
        ninferred_tracts = sum([x.stop - x.start for x in (
            ind_inferred_tracts
        )])
        ntrue_positives = sum([
            x.stop - x.start for x in ind_inferred_tracts.intersect(
                ind_truth_tracts
            )
        ])

        sum_ntruth_tracts += ntruth_tracts
        sum_ninferred_tracts += ninferred_tracts
        sum_ntrue_positives += ntrue_positives

    for s in np.setdiff1d(truth_tracts_samples, inferred_tracts_samples):
        # ninferred_tracts = 0
        ind_truth_tracts = truth_tracts[
            truth_tracts['sample'] == s
        ][['chrom', 'start', 'end']]
        ind_truth_tracts = pybedtools.BedTool.from_dataframe(
            ind_truth_tracts
        ).sort().merge()
        ntruth_tracts = sum([x.stop - x.start for x in (ind_truth_tracts)])

        sum_ntruth_tracts += ntruth_tracts

    for s in np.setdiff1d(inferred_tracts_samples, truth_tracts_samples):
        # ntruth_tracts = 0
        ind_inferred_tracts = inferred_tracts[
            inferred_tracts['sample'] == s
        ][['chrom', 'start', 'end']]
        ind_inferred_tracts = pybedtools.BedTool.from_dataframe(
            ind_inferred_tracts
        ).sort().merge()
        ninferred_tracts = sum([
            x.stop - x.start for x in (ind_inferred_tracts)
        ])

        sum_ninferred_tracts += ninferred_tracts

    total_precision, total_recall = cal_pr(
        sum_ntruth_tracts, sum_ninferred_tracts, sum_ntrue_positives
    )
    res.loc[len(res.index)] = [total_precision, total_recall, cutoff]

    res.fillna('NaN').to_csv(output, sep="\t", index=False)
    
    
def plot(summary_file):
    """
    Description:
        Plots the precision-recall curve.

    Arguments:
        summary_file str: Name of the file storing the performance of a model.
    """
    df = pd.read_csv(summary_file, sep="\t")
    plt.plot(df['recall'], df['precision'], marker='o', label='PR curve')
    plt.plot([0,100],[2,2], label='baseline', linestyle='dashed')
    plt.xlim([0,100])
    plt.ylim([0,100])
    plt.title('Performance')
    plt.xlabel('Recall (%)')
    plt.ylabel('Precision (%)')
    plt.legend()
    plt.show()
    
    
def q1(Power):
    if Power == '': return 'Please choose a metric'
    if Power == 'Precision': return 'Correct'
    else: return 'Not correct'


def q2(FDR):
    if FDR == '': return 'Please choose a metric'
    if FDR == 'Recall': return 'Correct'
    else: return 'Not correct'