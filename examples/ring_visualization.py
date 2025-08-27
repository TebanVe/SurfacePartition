#!/usr/bin/env python3
"""
Ring visualization script.

Usage:
  python examples/ring_visualization.py --solution <path_to_solution.h5> [--use-initial] [--level 0.5] [--save out.png] [--no-fill] [--no-mesh]
"""

import os
import sys
import argparse
import h5py
import yaml

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from find_contours import ContourAnalyzer
from plot_utils import plot_partitions_with_contours, plot_contours_on_ring, build_plot_title


def main():
    parser = argparse.ArgumentParser(description='Visualize ring partitions from solution file')
    parser.add_argument('--solution', type=str, required=True, help='Path to ring solution .h5 file')
    parser.add_argument('--use-initial', action='store_true', help='Use x0 instead of x_opt')
    parser.add_argument('--level', type=float, default=0.5, help='Level-set threshold for contours (default: 0.5)')
    parser.add_argument('--save', type=str, help='Path to save image (e.g., results/partition.png)')
    parser.add_argument('--no-fill', action='store_true', help='Disable filled partition rendering')
    parser.add_argument('--no-mesh', action='store_true', help='Disable mesh overlay when contours only')
    args = parser.parse_args()

    analyzer = ContourAnalyzer(args.solution)
    analyzer.load_results(use_initial_condition=args.use_initial)

    # Compose title from H5 attrs or metadata fallback
    surface = None
    label1 = 'v1'
    label2 = 'v2'
    var1_val = None
    var2_val = None
    lam = None
    seed = None
    optimizer = 'PySLSQP'
    try:
        with h5py.File(args.solution, 'r') as f:
            surface = f.attrs.get('surface')
            labels = f.attrs.get('resolution_labels')
            if labels is not None and len(labels) >= 2:
                label1 = labels[0]
                label2 = labels[1]
            var1_val = f.attrs.get('var1')
            var2_val = f.attrs.get('var2')
            lam = f.attrs.get('lambda_penalty')
            seed = f.attrs.get('seed')
            opt_attr = f.attrs.get('optimizer')
            optimizer = opt_attr if opt_attr is not None else optimizer
    except Exception:
        pass
    # Fallback: try metadata.yaml in parent dir
    if var1_val is None or var2_val is None or surface is None:
        run_dir = os.path.dirname(args.solution)
        meta_path = os.path.join(run_dir, 'metadata.yaml')
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as mf:
                meta = yaml.safe_load(mf)
            surface = surface or meta.get('input_parameters', {}).get('surface')
            labels = meta.get('input_parameters', {}).get('resolution_labels')
            if labels and len(labels) >= 2:
                label1, label2 = labels[0], labels[1]
            levels = meta.get('levels') or []
            if levels:
                last = levels[-1]
                var1_val = var1_val or last.get(label1)
                var2_val = var2_val or last.get(label2)
            lam = lam or meta.get('input_parameters', {}).get('lambda_penalty')
            seed = seed or meta.get('input_parameters', {}).get('seed')
            optimizer = meta.get('optimizer') or optimizer
    # Final fallback: parse from filename
    if var1_val is None or var2_val is None:
        name = os.path.basename(args.solution)
        import re
        m1 = re.search(r"_v1([a-zA-Z]+)?(\d+)", name)
        m2 = re.search(r"_v2([a-zA-Z]+)?(\d+)", name)
        if m1:
            if m1.group(1):
                label1 = m1.group(1)
            var1_val = int(m1.group(2))
        if m2:
            if m2.group(1):
                label2 = m2.group(1)
            var2_val = int(m2.group(2))
    # Default numeric values if still None
    var1_val = int(var1_val) if var1_val is not None else 0
    var2_val = int(var2_val) if var2_val is not None else 0
    title_str = build_plot_title(optimizer, surface, label1, var1_val, label2, var2_val, lam, seed, None, prefix='Partition')

    contours = analyzer.extract_contours(level=args.level)

    if args.no_fill:
        from plot_utils import plot_contours_on_ring
        fig, ax = plot_contours_on_ring(analyzer.vertices, analyzer.faces, contours,
                                        show_mesh=not args.no_mesh, show_vertices=False,
                                        title=title_str)
    else:
        triangle_labels = analyzer.label_triangles_from_indicator()
        fig, ax = plot_partitions_with_contours(analyzer.vertices, analyzer.faces,
                                                triangle_labels, contours,
                                                title=title_str)

    if args.save:
        import matplotlib.pyplot as plt
        os.makedirs(os.path.dirname(args.save) or '.', exist_ok=True)
        fig.savefig(args.save, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {args.save}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


if __name__ == '__main__':
    main() 