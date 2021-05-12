from typing import Dict, Any

import os
import json
import argparse
from functools import reduce

import openml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler

from piasbenchmark import Benchmark

def process(df: pd.DataFrame) -> Dict[str, Any]:
    classifier_results = df[df.apply(
        lambda row: 'classifier' in row.name, axis=1
    )]
    selector_results = df[df.apply(
        lambda row: 'selector' in row.name, axis=1
    )]
    baseline_results = df[df.apply(
        lambda row: 'baseline' in row.name and 'static' not in row.name, axis=1
    )]

    classifier_best_names = classifier_results.idxmax()
    classifier_best = classifier_results.max()

    selector_best_names = selector_results.idxmax()
    selector_best = selector_results.max()

    baseline_best_names = baseline_results.idxmax()
    baseline_best = baseline_results.max()

    # shape (n_task, 3), the best result of each category
    best_values = pd.concat(
        [classifier_best, selector_best, baseline_best], axis=1)
    best_names = pd.concat(
        [classifier_best_names, selector_best_names, baseline_best_names],
        axis=1)

    rename_values = {0:'classifiers', 1:'selectors', 2:'baselines'}
    best_values.rename(columns=rename_values, inplace=True)
    best_names.rename(columns=rename_values, inplace=True)

    return {
        'classifier_results': classifier_results,
        'selector_results': selector_results,
        'baseline_results': baseline_results,
        'classifier_best': classifier_best,
        'selector_best': selector_best,
        'baseline_best': baseline_best,
        'classifier_best_names': classifier_best_names,
        'selector_best_names': selector_best_names,
        'baseline_best_names': baseline_best_names,
        'best_values': best_values,
        'best_names': best_names,
    }

def centered_plot(best_values: pd.DataFrame, best_names: pd.DataFrame):
    # Things are normalized so classifiers are centered at normscore 0
    # Hence we only have to plot best_selector vs best_baseline
    best_values = best_values.drop(columns='classifiers')
    best_names = best_names.drop(columns='classifiers')

    selector_normscores = best_values['selectors'].values
    baseline_normscores = best_values['baselines'].values

    # We choose the label name of which performed better
    labels = [
        best_names.loc[task, category]
        for task, category
        in best_values.idxmax(axis=1).items()
    ]

    # Assign colors so selectors and baselines are visually distinct
    selector_names = set(filter(lambda name: 'selector' in name, labels))
    baseline_names = set(filter(lambda name: 'baseline' in name, labels))

    selector_colors = sns.color_palette('viridis_r', len(selector_names) * 2)
    baseline_colors = sns.color_palette('rocket', len(baseline_names))

    cmap = {
        ** { name: selector_colors[i] for i, name in enumerate(selector_names)},
        ** { name: baseline_colors[i] for i, name in enumerate(baseline_names)},
    }
    colors = [cmap[label] for label in labels]

    figsize = (8,10)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1)
    xlims = (-1.3, 1.05)
    ylims = (-1.3, 1.8)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    # Axis lines within box of radius 1
    ax.plot((xlims[0], 1), (0, 0), c='black',
            linestyle=':', linewidth=0.5)
    ax.plot((0, 0), (ylims[0], 1), c='black',
            linestyle=':', linewidth=0.5)

    # Horizontal Oracle line
    ax.plot((xlims[0], 1), (1, 1), c='black',
            linestyle=':', linewidth=1.0)

    # Vertical line for outisde box
    ax.plot((1, 1), (ylims[0], ylims[1]), c='black',
            linestyle=':', linewidth=0.5)

    # Diagonal line for marking which side is better
    ax.plot((xlims[0],1), (ylims[0], 1), c='grey', linestyle='--', linewidth=0.2)

    # Text indicating the regions
    offsets = (0.1, 0.05)
    ax.text(0 + 0.3 + offsets[0], ylims[0] + offsets[1],
            "best baseline < single best\nbest selector > single best",
            fontsize=8)
    ax.text(xlims[0] + offsets[0], ylims[0] + offsets[1],
            "best baseline < single best\nbest selector < single best",
            fontsize=8)
    ax.text(xlims[0] + offsets[0], 0 + offsets[1],
            "best baseline > single best\nbest selector < single best",
            fontsize=8)
    ax.text(0 + 0.3 + offsets[0], 0 + offsets[1],
            "best baseline > single best\nbest selector > single best",
            fontsize=8)
    ax.text(0 + offsets[0], 1 + offsets[1], "baseline > oracle",
            fontsize=8)

    legend_lines = [
        Line2D([0], [0], color='w', marker='o', markerfacecolor=col,
               label=name.replace('_', ' '))
        for name, col in cmap.items()
    ]
    ax.legend(handles=legend_lines)


    ax.scatter(x=selector_normscores, y=baseline_normscores, c=colors)
    ax.set_xlabel('Selector normalized score')
    ax.set_ylabel('Baseline normalized score')
    #ax.axes.set_aspect('equal')
    ax.set_title('Selector/Baseline performances for 62 Datasets')
    return fig

def umap_dataset_properties(best_values: pd.DataFrame, best_names: pd.DataFrame,
                            cached_metaprops: str, random_state=5):
    if os.path.exists(cached_metaprops):
        df_metaprops = pd.read_csv(cached_metaprops, index_col=0)
    else:
        tasks = list(map(int, best_values.index))
        dataset_ids = [openml.tasks.get_task(task).dataset_id for task in tasks]

        # This will take a while to get
        # Hence the caching
        dataset_metaprops = [openml.datasets.get_dataset(dataset_id).qualities
                             for dataset_id in dataset_ids]

        available_keys = reduce(
            lambda acc, metaprops: acc.intersection(metaprops.keys()),
            dataset_metaprops, set(dataset_metaprops[0].keys())
        )
        dict_metaprops = {
            k : [ metaprop[k] for metaprop in dataset_metaprops]
            for k in available_keys
        }
        df_metaprops = pd.DataFrame.from_dict(dict_metaprops, orient='index',
                                              columns=tasks)
        df_metaprops.to_csv(cached_metaprops)

    # Drop features that have more than 30% missing
    cut_percentage = 0.00 # Most features have 0%, 12% or 67% missing
    for row in df_metaprops.index:
        missing = sum(df_metaprops.loc[row].isnull()) / len(df_metaprops.loc[row])
        if missing > cut_percentage:
            df_metaprops.drop(index=row, inplace=True)

    # Convert the rest of the nans to the mean (8/62 had 24/48 missing features)
    df_metaprops = df_metaprops.apply(lambda row: row.fillna(row.mean()),
                                      axis=1)

    df_metaprops = df_metaprops.T # Make the tasks be on the index

    # Scale Data according to UMAPS recommendation
    df_scaled_metaprops = StandardScaler().fit_transform(df_metaprops)

    # Use UMAP to produce embedding
    umapper = UMAP(n_neighbors=10, random_state=random_state)
    embeddings = umapper.fit_transform(df_scaled_metaprops)

    figsize = (10, 12)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)

    # We choose the label name of which performed better
    labels = [
        best_names.loc[task, category]
        for task, category
        in best_values.idxmax(axis=1).items()
    ]

    # Assign colors so selectors and baselines are visually distinct
    classifier_names = set(filter(lambda name: 'classifier' in name, labels))
    selector_names = set(filter(lambda name: 'selector' in name, labels))
    baseline_names = set(filter(lambda name: 'baseline' in name, labels))

    classifier_colors = sns.dark_palette("#69d", reverse=True, n_colors=len(classifier_names)*2)
    selector_colors = sns.color_palette('viridis_r', len(selector_names) * 2)
    baseline_colors = sns.color_palette('rocket', len(baseline_names))

    cmap = {
        ** { name: selector_colors[i] for i, name in enumerate(selector_names)},
        ** { name: baseline_colors[i] for i, name in enumerate(baseline_names)},
        ** { name: classifier_colors[i] for i, name in enumerate(classifier_names)},
    }
    colors = [cmap[label] for label in labels]

    ax.scatter(embeddings[:,0], embeddings[:,1], c=colors)

    legend_lines = [
        Line2D([0], [0], color='w', marker='o', markerfacecolor=col,
               label=name.replace('_', ' '))
        for name, col in cmap.items()
    ]
    ax.legend(handles=legend_lines)

    ax.set_xlabel('UMAP axis 1')
    ax.set_ylabel('UMAP axis 2')
    ax.set_title('UMAP projection of dataset meta-features')

    return fig

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots of the benchmark")
    parser.add_argument('-c', '--config', type=str, required=True,
                       help='Path to benchmark config.')
    parser.add_argument('-r', '--resultsdir', type=str, required=True,
                       help='Path to the directory where results are stored.')
    parser.add_argument('-p', '--plotsdir', type=str,
                       help='Path to the directory to generate figures to')
    args = parser.parse_args()

    if not args.plotsdir:
        args.plotsdir = args.resultsdir
    sns.set(style='white')

    bench = Benchmark(args.config)

    full_path = os.path.abspath(args.resultsdir)
    results_path = os.path.join(full_path, f'{bench.id}_results.json')
    summary_path = os.path.join(full_path, f'{bench.id}_summary.json')
    accuracies_path = os.path.join(full_path, f'{bench.id}_accuracies.csv')
    normscores_path = os.path.join(full_path, f'{bench.id}_normscores.csv')
    cached_metaprops = os.path.join(full_path, f'{bench.id}_cached_metaprops.csv')

    plot_path = os.path.abspath(args.plotsdir)
    plot_centered_path = os.path.join(plot_path, f'{bench.id}_centered')
    plot_umap_projection = os.path.join(plot_path, f'{bench.id}_umap_projection')

    df_accs = pd.read_csv(accuracies_path, index_col=0)
    pr_accs = process(df_accs)

    df_nrms = pd.read_csv(normscores_path, index_col=0)
    pr_nrms = process(df_nrms)

    results = json.load(open(results_path, 'r'))
    summary = json.load(open(summary_path, 'r'))

    # Create centered plot
    fig = centered_plot(pr_nrms['best_values'], pr_nrms['best_names'])
    plt.savefig(f'{plot_centered_path}.svg', dpi=300, format='svg')
    plt.savefig(f'{plot_centered_path}.jpg', dpi=300, format='jpg')

    fig = umap_dataset_properties(pr_accs['best_values'], pr_accs['best_names'], 
                                  cached_metaprops)
    plt.savefig(f'{plot_umap_projection}.svg', dpi=300, format='svg',
                bbox_inches='tight')
    plt.savefig(f'{plot_umap_projection}.jpg', dpi=300, format='jpg',
                bbox_inches='tight')
