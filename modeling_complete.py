from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score
from sklearn import metrics
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import os

# constants
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_DIR, 'data'))
HOSPITAL_DIR = os.path.join(DATA_DIR, 'heart_disease')

def load_hospital_data():
    with open(os.path.join(HOSPITAL_DIR, 'processed.cleveland.data')) as infile:
        hospital_data = pd.read_csv(infile, header=None, na_values=['?'])

    hospital_data = hospital_data.dropna()
    X = hospital_data.iloc[:13]
    Y = hospital_data.iloc[:,13]
    return X, Y

def k_value_test(X, Y, ks=range(1,50), scoring=['accuracy','f1_macro']):

    metrics = get_metrics_across_ks(X, Y, ks, scoring)

    plot_df = pd.DataFrame(ks, columns=['k'])
    print "====== K-value testing report ======"
    for metric, test_results in metrics.iteritems():
        best = sorted(test_results, key=lambda p: p[1], reverse=True)
        report_data = [metric]
        report_data.extend(best[0])
        print "Best K for %s: %i (%f)" % tuple(report_data)

        metric_df = pd.DataFrame(test_results, columns=['k',metric])
        plot_df = plot_df.merge(metric_df, on="k")

    plot_df.plot(x='k')
    #plt.show()

def get_metrics_across_ks(X, Y, ks=range(1,50), scoring=['accuracy','f1_macro']):
    metrics = defaultdict(list)

    for k in ks:
        model = KNeighborsClassifier(k)
        for metric in scoring:
            results = cross_val_score(model, X, Y, cv=10, scoring=metric)
            metrics[metric].append((k, results.mean()))

    return metrics

if __name__ == "__main__":
    X, Y = load_hospital_data()
    k_value_test(X, Y)
