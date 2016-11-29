"""
"""
import glob

import pandas as pd
from post_hoc_analysis import (_categorize_data, _generate_formula,
                               analyze_effects)

categories = {}
categories['atlas'] = ['aal_spm12', 'basc_scale122', 'ho_cort_symm_split',
                       'dictlearn', 'ica', 'kmeans', 'ward']
categories['measure'] = ['correlation', 'partial correlation', 'tangent']
categories['classifier'] = ['svc_l1', 'svc_l2', 'ridge']

path_to_csvs = sorted(glob.glob('results_csv/results_*.csv'))
betas_list = []
conf_int_list = []
p_values_list = []
labels = []

for csv in path_to_csvs:
    label = csv.split('/')[1].split('_')[1].split('.csv')[0]
    print(" === Running ANOVA analysis on %s dataset scores === " % label)
    labels.append(label)
    data = pd.read_csv(csv)

    betas = {}
    pvalues = {}
    conf_int = {}

    ##########################################################################
    # Data preparation
    scores = data['scores'].str.strip('[ ]')
    del data['scores']
    data = data.join(scores)
    data.scores = data.scores.astype(float)

    ##########################################################################
    # Categorize data with columns to pandas data frame
    columns = ['scores', 'atlas', 'measure', 'classifier']
    data = _categorize_data(data, columns=columns)

    #########################################################################
    # Generate the formula for model
    k = ['atlas', 'measure', 'classifier']
    # atlas models
    atlases = ['aal_spm12', 'basc_scale122', 'ho_cort_symm_split',
               'dictlearn', 'ica', 'kmeans', 'ward']
    this_betas = {}
    this_pvalues = {}
    this_conf_int = {}

    for target in ['aal_spm12', 'basc_scale122']:
        formula = _generate_formula(dep_variable='scores', effect1=k[0],
                                    effect2=k[1], effect3=k[2],
                                    target_in_effect1=target)
        print(formula)
        # Fit with first target
        model = analyze_effects(data, formula=formula, model='ols')
        print(model.summary())
        for at in atlases:
            if at == target:
                continue
            key = "C(%s, Sum('%s'))[S.%s]" % (k[0], target, at)
            print(key)
            this_betas[at] = model.params[key]
            this_pvalues[at] = model.pvalues[key]
            this_conf_int[at] = model.conf_int()[1][key] - model.params[key]

    betas['atlas'] = this_betas
    pvalues['atlas'] = this_pvalues
    conf_int['atlas'] = this_conf_int

    ##########################################################################
    # Measure

    measures = ['correlation', 'partial correlation', 'tangent']
    this_betas = {}
    this_pvalues = {}
    this_conf_int = {}

    for target in ['correlation', 'partial correlation']:
        formula = _generate_formula(dep_variable='scores', effect1=k[1],
                                    effect2=k[0], effect3=k[2],
                                    target_in_effect1=target)
        print(formula)
        # Fit with first target
        model = analyze_effects(data, formula=formula, model='ols')
        print(model.summary())
        for me in measures:
            if me == target:
                continue
            key = "C(%s, Sum('%s'))[S.%s]" % (k[1], target, me)
            print(key)
            this_betas[me] = model.params[key]
            this_pvalues[me] = model.pvalues[key]
            this_conf_int[me] = model.conf_int()[1][key] - model.params[key]

    betas['measure'] = this_betas
    pvalues['measure'] = this_pvalues
    conf_int['measure'] = this_conf_int

    ##########################################################################
    # Classifier

    classifiers = ['svc_l1', 'svc_l2', 'ridge']
    this_betas = {}
    this_pvalues = {}
    this_conf_int = {}

    for target in ['svc_l1', 'svc_l2']:
        formula = _generate_formula(dep_variable='scores', effect1=k[2],
                                    effect2=k[0], effect3=k[1],
                                    target_in_effect1=target)
        print(formula)
        # Fit with first target
        model = analyze_effects(data, formula=formula, model='ols')
        print(model.summary())
        for cl in classifiers:
            if cl == target:
                continue
            key = "C(%s, Sum('%s'))[S.%s]" % (k[2], target, cl)
            print(key)
            this_betas[cl] = model.params[key]
            this_pvalues[cl] = model.pvalues[key]
            this_conf_int[cl] = model.conf_int()[1][key] - model.params[key]

    betas['classifier'] = this_betas
    pvalues['classifier'] = this_pvalues
    conf_int['classifier'] = this_conf_int

    betas_list.append(betas)
    p_values_list.append(pvalues)
    conf_int_list.append(conf_int)

from post_hoc_analysis2 import plot_multiple_data
import matplotlib.pyplot as plt
from matplotlib import cm

colormapper = cm.Set3([0.27, 0.36, 0.81])
margin = 0.01
space = 0.06
bs = 0.05

alias = {
    'aal_spm12': 'AAL',
    'basc_scale122': 'BASC',
    'ho_cort_symm_split': 'Harv. Oxf.',
    'kmeans': 'KMeans',
    'ward': "Ward",
    'ica': 'Group ICA',
    'dictlearn': 'DictLearn',
    'correlation': 'Correlation',
    'partial correlation': 'Partial corr.',
    'tangent': 'Tangent',
    'svc_l2': 'SVC-$\ell_2$',
    'svc_l1': 'SVC-$\ell_1$',
    'ridge': 'Ridge',
}


categories = [('atlas', ['aal_spm12', 'basc_scale122',
                         'ho_cort_symm_split',
                         'kmeans', 'ward', 'ica',
                         'dictlearn']),
              ('measure', ['correlation', 'partial correlation', 'tangent']),
              ('classifier', ['svc_l2', 'svc_l1', 'ridge'])]
bars = plot_multiple_data(categories, betas_list, conf_int_list,
                          colormapper, mode='barplot', alias=alias, bg=True)
plt.legend([i[0] for i in bars],
           [alias.get(i, i) for i in labels],
           loc=(0.05, 0.7), )
plt.show()
