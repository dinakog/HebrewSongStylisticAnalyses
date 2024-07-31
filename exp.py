from sklearn.manifold import TSNE
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns
from lexical_diversity import lex_div as ld
from collections import defaultdict
import matplotlib.colors as mcolors


def fit_tsne(embedding_matrix, target_path):
    # Fit t-SNE model if an no existing model available
    if os.path.exists(target_path):
        with open(target_path, 'rb') as file:
            embedding_tsne = pickle.load(file)
    else:
        tsne = TSNE(n_components=2, random_state=42)
        embedding_tsne = tsne.fit_transform(embedding_matrix)
        with open(target_path, 'wb') as file:
            pickle.dump(embedding_tsne, file, protocol=pickle.HIGHEST_PROTOCOL)

    return embedding_tsne


def visualize_pairs(style_labels, style_mapping, embedding_tsne, save_dir, setup):
    # Get unique style labels
    style_labels = np.array(style_labels)
    unique_styles = np.unique(style_labels)

    # Create color mapping for each style
    num_styles = len(unique_styles)
    palette = sns.color_palette("husl", num_styles)
    color_mapping = {style: palette[i] for i, style in enumerate(unique_styles)}

    # Visualize unique pairs of styles
    style_pairs = list(combinations(unique_styles, 2))

    for i, (style1, style2) in enumerate(style_pairs):
        pair_mask = np.isin(style_labels, [style1, style2])
        pair_style_labels = style_labels[pair_mask]
        pair_embedding_tsne = embedding_tsne[pair_mask]
        pair_style_mapping = {style1: style_mapping[style1], style2: style_mapping[style2]}

        # Create and save Visualization
        pair_save_path = f"{save_dir}/tsne_pair_{i+1}_{style_mapping[style1]}_vs_{style_mapping[style2]}_{setup}.png"
        visualize_tsne_with_clusters(pair_style_labels, pair_style_mapping, pair_embedding_tsne, pair_save_path, color_mapping)


def visualize_tsne_with_clusters(style_labels, style_mapping, embedding_tsne, save_path, color_mapping=None):
    unique_styles = np.unique(style_labels)

    if color_mapping is None:
        colors = list(mcolors.TABLEAU_COLORS.values())
        color_mapping = {style: colors[i % len(colors)] for i, style in enumerate(unique_styles)}

    # Create plot of the t-SNE reduced data
    plt.figure(figsize=(10, 8))
    for style in unique_styles:
        style_mask = (style_labels == style)
        plt.scatter(embedding_tsne[style_mask, 0], embedding_tsne[style_mask, 1],
                    c=[color_mapping[style]], label=style_mapping[style], alpha=0.6, edgecolors='w', linewidth=0.5, s=30)

    plt.title('t-SNE Visualization of Styles')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.grid(True)

    # Save figure
    plt.savefig(save_path)
    plt.close()

def calc_textual_diversity(texts, styles, style_mapping):
    results = defaultdict(lambda: {'ttr': [], 'mtld': []})

    for text, style in zip(texts, styles):
        ttr = calculate_ttr(text)
        mtld = calculate_mtld(text)

        results[style]['ttr'].append(ttr)
        results[style]['mtld'].append(mtld)

    # Calculate averages for each style
    for style, metrics in results.items():
        avg_ttr = sum(metrics['ttr']) / len(metrics['ttr'])
        avg_mtld = sum(metrics['mtld']) / len(metrics['mtld'])

        print(f"Style: {style_mapping[style]}")
        print(f"Average TTR: {avg_ttr:.4f}")
        print(f"Average MTLD: {avg_mtld:.4f}")


def calculate_ttr(text):
    flt = ld.flemmatize(text)
    return ld.ttr(flt)


def calculate_mtld(text):
    flt = ld.flemmatize(text)
    return ld.mtld(flt)

