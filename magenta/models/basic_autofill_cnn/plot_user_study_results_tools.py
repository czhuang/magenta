import os

import numpy as np
import pylab as plt

POSTER_PLOT = False


def plot_csv():
    import csv
    from collections import defaultdict
    #fname = 'four_cols.csv'
    fname = 'sample_results.csv'
    no_pref_count = 0
    #ordering = ['d', 'rm', 'b', 'o']
    ordering = ['i', 'c', 'n', 'o']
    scores = defaultdict(int)
    with open(fname, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                continue
            label, pref_strength = row
            if pref_strength == 'no_preference':
              no_pref_count += 1
            else:
              scores[label] += 1
    #        label, score = row
    #        if int(score) > 0:
    #            scores[label] += 1
    assert set(scores.keys()) == set(ordering)
    assert np.sum(scores.values()) == 192 - no_pref_count	
    print scores
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    width = 0.7
    alpha = 1
    HORIZONTAL = True
    #plt.figure()

    ordered_scores = [scores[label] for label in ordering]
    colors = np.array([[  4.17642000e-01,   5.64000000e-04,   6.58390000e-01,
          1.00000000e+00],
       [  6.92840000e-01,   1.65141000e-01,   5.64522000e-01,
          1.00000000e+00],
       [  8.81443000e-01,   3.92529000e-01,   3.83229000e-01,
          1.00000000e+00],
       [  9.88260000e-01,   6.52325000e-01,   2.11364000e-01,
          1.00000000e+00]])
    #plt.grid(True, linestyle='solid', alpha=0.1)
    ax.xaxis.grid(True, linestyle='solid', alpha=0.1)
    gridlines = ax.xaxis.get_gridlines()
    for line in gridlines:
        line.set_zorder(-1)
    if HORIZONTAL:
        barlist = plt.barh(np.arange(len(scores.keys()))+0.5-width/2, ordered_scores, height=width, alpha=alpha)
    else:
        barlist = plt.barh(np.arange(len(scores.keys()))+0.5-width/2, ordered_scores, height=width, alpha=alpha)
    for i, bar in enumerate(barlist):
        bar.set_color(colors[i])
        #bar.set_zorder(10)

    label_lookup = {'o':'Bach', 'b':'Balanced\nby sampling', 'rm':'Bernoulli\n(0.5)', 'd':'Denoising'}
    label_lookup = {'o':'Bach', 'n':'NADE', 'i':'Independent Gibbs', 'c':'Ancestral Gibbs\ncontiguous(0.50)'}
    xticks = [label_lookup[key] for key in ordering]
    if POSTER_PLOT:
      title_fs = 'xx-large'
      label_fs = 'xx-large'
      ticks_fs = 'x-large'
    else:
      # Title not shown.
      #title_fs = 'x-large'
      label_fs = 'large'
      ticks_fs = 'large'

    padding = 0.15

    if HORIZONTAL:
        plt.yticks(np.asarray(range(len(xticks)))+0.5, xticks)#, rotation=45)#,  fontweight='bold')
        plt.ylim(0-padding, 4+padding)
        plt.xlim(0, 55)
        plt.xlabel('# of wins', fontsize=label_fs)
        plt.ylabel('Sampling scheme', fontsize=label_fs)
    else:
        plt.xticks(np.asarray(range(len(xticks)))+0.5, xticks)#, rotation=45)#,  fontweight='bold')
        plt.xlim(0-padding, 4+padding)
        plt.ylim(0, 55)
        plt.ylabel('# of wins', fontsize=label_fs)
        plt.xlabel('Generator of music fragment', fontsize=label_fs)

    ax.tick_params(axis='both', which='major', labelsize=ticks_fs)
    ax.tick_params(axis='both', which='minor', labelsize=ticks_fs)

    #plt.title('Aggregates from human evaluations on \nwhich musical fragment is more preferable', fontsize=title_fs)
    #plt.title('Human evaluations', fontsize=title_fs)
    plt.tight_layout()
    print 'Plotted and saving to', os.getcwd()
    #plt.savefig('unconditioned-eval-first.png', bbox_inches='tight')
    #plt.savefig('unconditioned-eval-first.pdf', bbox_inches='tight')
    plt.savefig('unconditioned-sample-eval.png', bbox_inches='tight')
    plt.savefig('unconditioned-sample-eval.pdf', bbox_inches='tight')


if __name__ == '__main__':
    #plot_satisfaction()
    #plot_novelty()
    plot_csv()
