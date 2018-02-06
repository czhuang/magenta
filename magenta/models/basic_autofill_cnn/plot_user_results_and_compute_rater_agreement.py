import os
from collections import defaultdict
import csv
import numpy as np
import pylab as plt

POSTER_PLOT = False


def plot_csv():
    fname = 'data/user_study/study3_take2.csv'
#    plt_order = ['i', 'c', 'n', 'o']
    plt_order = ['n', 'i', 'o']
    nsets = len(plt_order)
    num_samples_per_source = 4
    
    first_col_count = defaultdict(int)
    second_col_count = defaultdict(int)
    
    # In option 1's perspective.
    pref_map = {'strongly_preferred_option1': 5,
                'weakly_preferred_option1': 4,
                'no_preference': 3,
                'weakly_preferred_option2': 2, 
                'strongly_preferred_option2': 1} 
    M = defaultdict(list)
    # The third dimension for storing # of wins, ties, and losses
    matrix = np.zeros((nsets, nsets, 3))
    with open(fname, 'rb') as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      for i, row in enumerate(reader):
        if i == 0:
          continue
        winner, pref_strength, cand1_and_ind, cand2_and_ind = row

        cand1, cand1_ind = cand1_and_ind.split('_')
        cand2, cand2_ind = cand2_and_ind.split('_')
        assert winner in [cand1, cand2], 'Winning class should be one of the candidate classes.'
        winner_ind = int(cand1_ind) if winner == cand1 else int(cand2_ind)
        loser, loser_ind = (cand2, int(cand2_ind)) if winner == cand1 else (cand1, int(cand1_ind))
  
        # Only update matrix if it's not the duplicates.
        if cand1 != 'c' and cand2 != 'c':
          rank = pref_map[pref_strength]
          if rank > 3:
            M[winner].append(rank)
            M[loser].append(6-rank)
          else:
            M[winner].append(6-rank)
            M[loser].append(rank)  
 
          first_col_count[plt_order.index(cand1)] += 1
          second_col_count[plt_order.index(cand2)] += 1
        
          if pref_strength == 'no_preference':
            matrix[plt_order.index(winner), plt_order.index(loser), 1] += 1 
            matrix[plt_order.index(loser), plt_order.index(winner), 1] += 1 
          else:
            matrix[plt_order.index(winner), plt_order.index(loser), 0] += 1 
            matrix[plt_order.index(loser), plt_order.index(winner), 2] += 1
    print pref_map
    # Checks if the numbers add up correctly.
    scores = dict((plt_order[i], summed) for i, summed in enumerate(np.sum(matrix[:, :, 0], axis=1)))
    print scores    

    total_comparisons = nsets*(nsets-1) * 4*4  # 192
    print 'total_comparisons', total_comparisons
    comp_between_two_sets = 4*4*2  # 32, 6*32 = 192, 6 = 4*3/2
    set_one_side_count = total_comparisons / nsets  # 48, the number of times it appears on one side of comparison
    # The # of times a set is involved in a comparison.
    set_total_count = total_comparisons * 2 / 3
    print 'set_one_side_count, set_total_count', set_one_side_count, set_total_count
    #assert total_comparisons == 192
    no_pref_count = np.sum(matrix[:, :, 1]) / 2
    print 'no_pref_count', no_pref_count
    assert np.sum(scores.values()) == total_comparisons - no_pref_count	

    assert np.allclose(first_col_count.values(), set_one_side_count)
    assert np.allclose(second_col_count.values(), set_one_side_count)
    # Sets are not compared to itself.
    assert np.allclose(np.unique(np.sum(matrix, axis=2)), np.array([0, comp_between_two_sets]))
    assert np.allclose(np.sum(matrix, axis=(1,2)), set_total_count)

    import scipy.stats as sps
    # assumes only 3 ranks, win, no pref, loss
    if False:
      measures = []
      for i in range(nsets):
         print i
         counts = np.sum(matrix[i, :, :], axis=0) 
         print counts
         ranks = []
         # Loop loss, no preference, win
         for rank_idx in range(3):
           ranks.extend([rank_idx] * int(counts[rank_idx]))
         measures.append(ranks)
      lens = [len(row) for row in measures]
      assert len(np.unique(lens)) == 1
  
      overall = sps.kruskal(measures[0], measures[1], measures[2])
    
    assert len(np.unique([len(vs) for vs in M.values()])) == 1
    overall = sps.kruskal(M.values()[0], M.values()[1], M.values()[2])
    print overall
    for key, values in M.iteritems():
      print key, values
   
    for key1 in M.keys():
      for key2 in M.keys():
        if key1 == key2:
          continue
        result = sps.wilcoxon(M[key1], y=M[key2])
        print key1, key2, ':', result

#    scipy.stats.wilcoxon(x, y=None, zero_method='wilcox', correction=False)[source]

    if False:
      assert 'Not making plots now.'    
    # ========== PLOTS ==============
    # Plot preference "confusion" matrices.
    fig, axes = plt.subplots(1, 3)
    cmaps = ["viridis", "bone", "viridis_r"]
    cmaps = ["inferno", "gray_r", "inferno"]
    title_strs = ['row WINS col', 'TIE', 'row LOSSES col']
    for i, ax in enumerate(axes.ravel()):
      ax.imshow(matrix[:, :, i], cmap="gray_r", interpolation="nearest", 
                aspect="equal", vmin=0, vmax=np.max(matrix))
      ax.set_xticks(np.arange(len(plt_order)))
      ax.set_yticks(np.arange(len(plt_order)))
      ax.set_xticklabels(plt_order)  #fontweight='bold')
      ax.set_yticklabels(plt_order)  #fontweight='bold')
      ax.set_title(title_strs[i])
    fpath_pdf = os.path.join(os.getcwd(), 'confusion.pdf')     
    fpath_png = os.path.join(os.getcwd(), 'confusion.png')     
    print 'Plotting to', fpath_pdf
    plt.savefig(fpath_pdf)
    print 'Plotting to', fpath_png
    plt.savefig(fpath_png)
    
    # Plot bar chart for aggregated scores.
    plt.figure(figsize=(6, 4))
    ax = plt.gca()
    width = 0.7
    alpha = 1
    HORIZONTAL = True

    ordered_scores = [scores[label] for label in plt_order]
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

    # Error bars (standard error of means)
    # variance of binomial distribution: np(1-p)
    n = set_total_count
    p = np.asarray(ordered_scores) / float(n)
    var = n * p * (1-p)
    print 
    std = np.sqrt(var)
    print std
    sem = std / np.sqrt(n) 
    errorbars = sem  
    print 'SEM'
    print n
    print ordered_scores
    print sem
    print errorbars
    assert np.sum(np.isnan(sem)) == 0 

    black = [0, 0, 0, 0.5]

    if HORIZONTAL:
        barlist = plt.barh(np.arange(len(scores.keys()))+0.5-width/2, ordered_scores, xerr=std, ecolor=black, height=width, alpha=alpha)
    else:
        barlist = plt.barh(np.arange(len(scores.keys()))+0.5-width/2, ordered_scores, height=width, alpha=alpha)
    for i, bar in enumerate(barlist):
        bar.set_color(colors[i])
        #bar.set_zorder(10)

    label_lookup = {'o':'Bach', 'b':'Balanced\nby sampling', 'rm':'Bernoulli\n(0.5)', 'd':'Denoising'}
    label_lookup = {'o':'Bach', 'n':'NADE', 'i':'Independent\nGibbs', 'c':'Ancestral Gibbs\ncontiguous(0.50)'}
    xticks = [label_lookup[key] for key in plt_order]
    if POSTER_PLOT:
      title_fs = 'xx-large'
      label_fs = 'xx-large'
      ticks_fs = 'x-large'
    else:
      # Title not shown.
      #title_fs = 'x-large'
      label_fs = 'large'
      ticks_fs = 'medium'

    padding = 0.15
    #max_count = np.max(ordered_scores) + 3
    max_count = 36
    if HORIZONTAL:
        plt.yticks(np.asarray(range(len(xticks)))+0.5, xticks)#, rotation=45)#,  fontweight='bold')
        plt.ylim(0-padding, nsets+padding)
        plt.xlim(0, max_count)
        plt.xlabel('# of wins', fontsize=label_fs)
        plt.ylabel('Sampling scheme', fontsize=label_fs)
    else:
        plt.xticks(np.asarray(range(len(xticks)))+0.5, xticks)#, rotation=45)#,  fontweight='bold')
        plt.xlim(0-padding, nsets+padding)
        plt.ylim(0, max_count)
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
