#!/usr/bin/env python3
"""
Ring visualization script.

Usage:
  python examples/ring_visualization.py --solution <path_to_solution.h5> [--use-initial] [--level 0.5] [--save out.png] [--no-fill] [--no-mesh]
"""

import os
import sys
import argparse

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from find_contours import ContourAnalyzer
from plot_utils import plot_partitions_with_contours, plot_contours_on_ring


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

    contours = analyzer.extract_contours(level=args.level)

    if args.no_fill:
        from plot_utils import plot_contours_on_ring
        fig, ax = plot_contours_on_ring(analyzer.vertices, analyzer.faces, contours,
                                        show_mesh=not args.no_mesh, show_vertices=False,
                                        title='Ring Partitions - Contours')
    else:
        triangle_labels = analyzer.label_triangles_from_indicator()
        fig, ax = plot_partitions_with_contours(analyzer.vertices, analyzer.faces,
                                                triangle_labels, contours,
                                                title='Ring Partitions')

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