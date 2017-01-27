import os
from collections import defaultdict
import csv
import numpy as np
import pylab as plt

POSTER_PLOT = False


def plot_csv():
    fname = 'sample_results_with_which_compared.csv'
    ordering = ['i', 'c', 'n', 'o']
    
    first_col_count = defaultdict(int)
    second_col_count = defaultdict(int)
    
    # The third dimension for storing # of wins, ties, and losses
    matrix = np.zeros((4, 4, 3))
    with open(fname, 'rb') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      for i, row in enumerate(reader):
        if i == 0:
            continue
        cand1, cand2, winner, pref_strength = row
        first_col_count[ordering.index(cand1)] += 1
        second_col_count[ordering.index(cand2)] += 1
        cands = [cand1, cand2]
        assert winner in cands, 'Winning class should be one of the candidate classes.'
        cands.remove(winner)
        loser = cands[0]
        if pref_strength == 'no_preference':
          matrix[ordering.index(winner), ordering.index(loser), 1] += 1 
          matrix[ordering.index(loser), ordering.index(winner), 1] += 1 
        else:
          matrix[ordering.index(winner), ordering.index(loser), 0] += 1 
          matrix[ordering.index(loser), ordering.index(winner), 2] += 1
    
    # Checks if the numbers add up correctly.
    scores = dict((ordering[i], summed) for i, summed in enumerate(np.sum(matrix[:, :, 0], axis=1)))
    
    num_sets = 4
    num_samples_per_set = 4
    total_comparisons = 4*3 * 4*4  # 192
    comp_between_two_sets = 4*4*2  # 32, 6*32 = 192, 6 = 4*3/2
    set_one_side_count = 192 / 4  # 48, the number of times it appears on one side of comparison
    set_total_count = 192 / 2
    assert total_comparisons == 192
    no_pref_count = np.sum(matrix[:, :, 1]) / 2
    print 'no_pref_count', no_pref_count
    assert np.sum(scores.values()) == total_comparisons - no_pref_count	

    assert np.allclose(first_col_count.values(), set_one_side_count)
    assert np.allclose(second_col_count.values(), set_one_side_count)
    # Sets are not compared to itself.
    assert np.allclose(np.unique(np.sum(matrix, axis=2)), np.array([0, comp_between_two_sets]))
    assert np.allclose(np.sum(matrix, axis=(1,2)), set_total_count)
   
    print ordering
    for i in range(3):
      print matrix[:, :, i] 

    # Plot preference "confusion" matrices.
    fig, axes = plt.subplots(1, 3)
    cmaps = ["viridis", "bone", "viridis_r"]
    cmaps = ["inferno", "gray_r", "inferno"]
    title_strs = ['row WINS col', 'TIE', 'row LOSSES col']
    for i, ax in enumerate(axes.ravel()):
      ax.imshow(matrix[:, :, i], cmap="gray_r", interpolation="nearest", 
                aspect="equal", vmin=0, vmax=np.max(matrix))
      ax.set_xticks(np.arange(len(ordering)))
      ax.set_yticks(np.arange(len(ordering)))
      ax.set_xticklabels(ordering)  #fontweight='bold')
      ax.set_yticklabels(ordering)  #fontweight='bold')
      ax.set_title(title_strs[i])
    fpath_pdf = os.path.join(os.getcwd(), 'confusion.pdf')     
    fpath_png = os.path.join(os.getcwd(), 'confusion.png')     
    print 'Plotting to', fpath_pdf
    plt.savefig(fpath_pdf)
    print 'Plotting to', fpath_png
    plt.savefig(fpath_png)
    
    # Plot aggregated scores.
    plt.figure(figsize=(7, 5))
    ax = plt.gca()
    width = 0.7
    alpha = 1
    HORIZONTAL = True

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
