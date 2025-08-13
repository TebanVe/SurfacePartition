#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from find_contours import ContourAnalyzer
from plot_utils import plot_vertex_field, overlay_zero_isoline, plot_histogram
from island_analysis import (
    compute_delta, compute_confidence, compute_entropy,
    ambiguity_mask, summarize_fields, extract_zero_isolines,
    polyline_length, polyline_area
)


def main():
    parser = argparse.ArgumentParser(description='Analyze ambiguity (islands) in ring partitions')
    parser.add_argument('--solution', type=str, required=True, help='Path to solution .h5 file')
    parser.add_argument('--tau', type=float, default=0.1, help='Ambiguity threshold for |Δ| < tau')
    parser.add_argument('--save-dir', type=str, default='results', help='Directory to save analysis outputs')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    analyzer = ContourAnalyzer(args.solution)
    analyzer.load_results(use_initial_condition=False)
    vertices = analyzer.vertices
    faces = analyzer.faces
    densities = analyzer.densities

    # Compute fields
    delta = compute_delta(densities)
    conf = compute_confidence(densities)
    ent = compute_entropy(densities)

    # Summary stats
    summary = summarize_fields(densities)

    # Plots: delta map with zero isoline
    fig, ax = plot_vertex_field(vertices, faces, delta, cmap='seismic', title='Delta (u1 - u2)')
    overlay_zero_isoline(vertices, faces, delta, ax=ax, color='k')
    fig.savefig(os.path.join(args.save_dir, 'delta_map.png'), dpi=300, bbox_inches='tight')

    # Plots: confidence map
    fig, ax = plot_vertex_field(vertices, faces, conf, cmap='viridis', title='Confidence max(u_i)')
    fig.savefig(os.path.join(args.save_dir, 'confidence_map.png'), dpi=300, bbox_inches='tight')

    # Plots: entropy map
    fig, ax = plot_vertex_field(vertices, faces, ent, cmap='magma', title='Entropy H(u)')
    fig.savefig(os.path.join(args.save_dir, 'entropy_map.png'), dpi=300, bbox_inches='tight')

    # Histograms
    fig, ax = plot_histogram(densities[:, 0], bins=50, title='Histogram u1', xlabel='u1')
    fig.savefig(os.path.join(args.save_dir, 'hist_u1.png'), dpi=300, bbox_inches='tight')

    fig, ax = plot_histogram(delta, bins=50, title='Histogram Delta', xlabel='u1 - u2')
    fig.savefig(os.path.join(args.save_dir, 'hist_delta.png'), dpi=300, bbox_inches='tight')

    # Ambiguity mask stats
    mask = ambiguity_mask(delta, tau=args.tau)
    ambiguity_fraction = float(np.mean(mask))
    summary['ambiguity_fraction_tau'] = args.tau
    summary['ambiguity_fraction'] = ambiguity_fraction

    # Zero-isolines (potential island boundaries) lengths and areas
    isolines = extract_zero_isolines(vertices, faces, delta)
    loops_info = []
    for poly in isolines:
        loops_info.append({
            'num_points': int(poly.shape[0]),
            'length': polyline_length(poly),
            'area': polyline_area(poly)
        })
    loops_info_sorted = sorted(loops_info, key=lambda x: x['area'], reverse=True)
    summary['num_isolines'] = len(isolines)
    summary['top_loops'] = loops_info_sorted[:5]

    with open(os.path.join(args.save_dir, 'island_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Saved analysis to {args.save_dir}")


if __name__ == '__main__':
    main() 