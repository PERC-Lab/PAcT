import json

import matplotlib
import matplotlib.pyplot as plt

from torch.nn import functional as F

def replot_conf_matrix(experiment):

    matplotlib.rc('xtick', labelsize=12)
    matplotlib.rc('ytick', labelsize=12)
    matplotlib.rc('axes', titlesize=12)
    matplotlib.rc('axes', labelsize=12)
    matplotlib.rc('legend', fontsize=12)
    matplotlib.rc('font', weight='bold')
    matplotlib.rc('font', size=24)
    matplotlib.rc('font', family='monospace')

    experiment.load_model()

    test_results = experiment.evaluate(test=True)

    conf_matrix = experiment.plot_confusion_matrix(test_results.pred,
                                                   test_results.labels)

    # Remove colorbar
    axes = conf_matrix.axes
    cb = axes[-1]  # Get colorbar
    cb.remove()

    dirpath = experiment.dirpath
    experiment_name = experiment.name
    cm_filepath = dirpath.joinpath(experiment_name + ".jpg")
    results_filepath = dirpath.joinpath(experiment_name + ".json")

    metric_scores = {}

    metric_scores['accuracy'] = test_results.accuracy
    metric_scores['f1'] = test_results.f1_score

    with open(results_filepath, 'w') as f:
        json.dump(metric_scores, f)

    plt.savefig(cm_filepath)
    print(f"Confusion Matrix for experiment: \"{experiment_name}\" saved!")
