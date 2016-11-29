# Author: Alexandre Abraham

from statsmodels.formula.api import ols, mixedlm
import numpy as np
import pandas
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm

class ValueMapper(object):

    def __init__(self, mapper):
        self.mapper = mapper

    def __call__(self, value):
        return self.mapper.get(value, value)


class RangeMapper(object):

    def __init__(self, mapper):
        self.mapper = mapper

    def __call__(self, value):
        i = 0
        if value > self.mapper[-1]:
            i = len(self.mapper)
        else:
            for i, n in enumerate(self.mapper):
                if value <= n:
                    break
        cat = '' if i == 0 else str(self.mapper[i - 1])
        cat += '-'
        cat += '' if i == len(self.mapper) else str(self.mapper[i])
        return cat


def _categorize_data(data, columns, **kwargs):
    data_dict = {}
    for column in columns:
        if column in kwargs:
            convert = kwargs[column]
            # Mapping simple si dict
            if isinstance(convert, dict):
                v = ValueMapper(convert)
            # List = range
            else:
                v = RangeMapper(convert)
            v = np.vectorize(v)
            data_dict[column] = v(data[column])
        else:
            data_dict[column] = data[column]
    data = pandas.DataFrame(data_dict)
    return data


def analyze_single_effects(data, effects, data_name='data', **kwargs):
    betas = {}
    pvalues = {}
    conf_int = {}
    effects = list(effects)  # copy
    # Build data
    data = _categorize_data(data, [data_name] + effects, **kwargs)
    for effect in effects:
        # Move axis to put effect in first position
        effects.remove(effect)
        effects.insert(0, effect)

        # Take 2 different values as ref.
        targets = np.unique(data[effect].values)

        model = ols(
            "%s ~ C(%s, Sum('%s')) + " % (data_name, effect, targets[0]) +
            "+".join([" C(%s, Sum) "] * (len(effects) - 1)) %
            tuple(effects[1:]), data).fit()

        # Retrieve the corresponding estimates
        this_betas = {}
        betas[effect] = this_betas
        this_pvalues = {}
        pvalues[effect] = this_pvalues
        this_conf_int = {}
        conf_int[effect] = this_conf_int
        for k in model.params.keys():
            # Remove "C("
            k_ = k[2:]
            if k_.startswith(effect):
                ename = k.split('[')[1][2:-1]
                this_betas[ename] = model.params[k]
                this_pvalues[ename] = model.pvalues[k]
                this_conf_int[ename] = \
                    model.conf_int()[1][k] - model.params[k]

        # Refit to get last target
        model = ols(
            "%s ~ C(%s, Sum('%s')) " % (data_name, effect, targets[1]) +
            "+".join([''] + [" C(%s, Sum) "] * (len(effects) - 1)) %
            tuple(effects[1:]), data).fit()
        key = "C(%s, Sum('%s'))[S.%s]" % (effect, targets[1], targets[0])
        this_betas[targets[0]] = model.params[key]
        this_pvalues[targets[0]] = model.pvalues[key]
        this_conf_int[targets[0]] = (model.conf_int()[1][key] -
            model.params[key])

    return betas, pvalues, conf_int


def analyze_mixed_effects(data, effects, cov_name='group', data_name='data',
                          **kwargs):
    betas = {}
    pvalues = {}
    conf_int = {}
    effects = list(effects)  # copy
    # Build data
    data = _categorize_data(data, [data_name] + effects, **kwargs)
    for effect in effects:
        # Move axis to put effect in first position
        effects.remove(effect)
        effects.insert(0, effect)

        # Take 2 different values as ref.
        targets = np.unique(data[effect].values)

        model = mixedlm(
            "%s ~ C(%s, Sum('%s')) + " % (data_name, effect, targets[0]) +
            "+".join([" C(%s, Sum) "] * (len(effects) - 1)) %
            tuple(effects[1:]), data, groups=data[cov_name]).fit()

        # Retrieve the corresponding estimates
        this_betas = {}
        betas[effect] = this_betas
        this_pvalues = {}
        pvalues[effect] = this_pvalues
        this_conf_int = {}
        conf_int[effect] = this_conf_int
        for k in model.params.keys():
            # Remove "C("
            k_ = k[2:]
            if k_.startswith(effect):
                ename = k.split('[')[1][2:-1]
                this_betas[ename] = model.params[k]
                this_pvalues[ename] = model.pvalues[k]
                this_conf_int[ename] = \
                    model.conf_int()[1][k] - model.params[k]

        # Refit to get last target
        model = ols(
            "%s ~ C(%s, Sum('%s')) " % (data_name, effect, targets[1]) +
            "+".join([''] + [" C(%s, Sum) "] * (len(effects) - 1)) %
            tuple(effects[1:]), data).fit()
        key = "C(%s, Sum('%s'))[S.%s]" % (effect, targets[1], targets[0])
        this_betas[targets[0]] = model.params[key]
        this_pvalues[targets[0]] = model.pvalues[key]
        this_conf_int[targets[0]] = (model.conf_int()[1][key] -
            model.params[key])

    stop
    return betas, pvalues, conf_int


def plot_single_effects(categories, betas, conf_int, colormapper, axis=None,
                        alias={}, offset=None, y_lim=None, legend=True):
    x = []
    y = []
    group = []
    cpt = 0
    ci = []
    colors = []

    for i, (category, items) in enumerate(categories):
        for item in items:
            x.append(cpt)
            y.append(betas[category][item])
            ci.append(conf_int[category][item])
            group.append(i)
            colors.append(colormapper[i])
            cpt += 1

    x = np.asarray(x)
    y = np.asarray(y)
    group = np.asarray(group)
    ci = np.asarray(ci)

    # Display variables
    n_blocks = len(categories)
    space_bar_ratio = 1.
    bar_size = 1.
    space = bar_size * space_bar_ratio

    # Add space to coordinates
    x = x + group * space

    # Draw
    if axis is None:
        plt.figure(figsize=(np.round(n_blocks * 1.5), 4))
        axis = plt.gca()
    bars = axis.bar(x, y, width=bar_size, color=colors, ecolor='k', yerr=ci)
    if y_lim is not None:
        plt.gca().set_ylim(plt.gca().get_ylim()[0], y_lim)
    if offset is not None:
        plt.gca().set_ylim(offset, plt.gca().get_ylim()[1])

    return bars


def plot_multiple_data(categories, betas_list, conf_int_list, colors,
                       axis=None, alias={}, offset=None, y_lim=None,
                       mode='barplot',
                       legend=True, labels=None, bar_width=1., bg=False):

    n_data = len(betas_list)
    # Space between bars of a group
    bar_space = 0.5
    # Internal data, space between bar of a group
    bar_gap = n_data * bar_width + bar_space
    space_bar_ratio = 2.

    bars_list = []

    for j, betas, conf_int, color in zip(range(n_data), betas_list,
                                         conf_int_list, colors):

        x = []
        y = []
        group = []
        cpt = j * bar_width
        ci = []

        for i, (category, items) in enumerate(categories):
            for item in items:
                x.append(cpt)
                y.append(betas[category][item])
                ci.append(conf_int[category][item])
                group.append(i)
                cpt += bar_gap

        x = np.asarray(x)
        y = np.asarray(y)
        group = np.asarray(group)
        ci = np.asarray(ci)

        # Display variables
        n_blocks = len(categories)
        space = bar_width * n_data * space_bar_ratio

        # Add space to coordinates
        x = x + group * space

        # Draw
        if axis is None:
            plt.figure(figsize=(np.round(n_blocks * 1.5), 4))
            axis = plt.gca()
        if mode == 'barplot':
            bars = axis.barh(x, y, height=bar_width, color=color, ecolor='k',
                             xerr=ci)
        elif mode == 'boxplot':
            bars = axis.boxplot(y, positions=x, widths=bar_width,
                                conf_intervals=ci)
            plt.setp(bars['boxes'], color=color)
        xmin = -0.11
        xmax = 0.06
        if j == 0:
            # Background patches
            if bg:
                [axis.add_patch(Rectangle(
                    (xmin, xi - bar_space / 2),
                    xmax - xmin, bar_gap,
                    facecolor=cm.Set2(gi / float(n_blocks)),
                    alpha=.1 + (i % 2) * .05, zorder=-1)) for i, xi, gi in
                    zip(range(len(x)), x, group)]
            # Labels
            plt.yticks(x + (n_data / 2.) * bar_width,
                       [alias.get(name, name)
                        for _, namel in categories
                        for name in namel])
        bars_list.append(bars)
        if y_lim is not None:
            axis.set_ylim(axis.get_ylim()[0], y_lim)
        if offset is not None:
            axis.set_ylim(offset, axis.get_ylim()[1])

    axis.axvline(0, color='black', lw=1)
    axis.invert_yaxis()

    if labels is not None:
        axis.legend([b[0] for b in bars_list], labels)

    return bars_list
