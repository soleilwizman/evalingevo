#!/usr/bin/env python3
"""Minimal Evo 2 benchmark for four-haplotype regulatory epistasis."""

import argparse
import json

from benchmark_data import CONTRAST as CONTRAST
from benchmark_data import KMER_VOCAB as KMER_VOCAB
from benchmark_data import SOURCE as SOURCE
from benchmark_data import STATES as STATES
from benchmark_data import differences as differences
from benchmark_data import file_hash as file_hash
from benchmark_data import load_quartets as load_quartets
from benchmark_data import mutate as mutate
from benchmark_data import parse_variant as parse_variant
from benchmark_data import prepare_siraj as prepare_siraj
from benchmark_data import read_oligos as read_oligos
from benchmark_data import reconstruct as reconstruct
from benchmark_data import reverse_complement as reverse_complement
from benchmark_data import seq_id as seq_id
from benchmark_plots import make_element_plot as make_element_plot
from benchmark_plots import make_plot as make_plot
from benchmark_stats import auroc as auroc
from benchmark_stats import correlations as correlations
from benchmark_stats import element_kmers as element_kmers
from benchmark_stats import gc_fraction as gc_fraction
from benchmark_stats import group_boot as group_boot
from benchmark_stats import metrics as metrics
from benchmark_stats import noise_ceiling as noise_ceiling
from benchmark_stats import rank_feature_contrasts as rank_feature_contrasts
from epistasis_evaluation import add_flip as add_flip
from epistasis_evaluation import add_predictions as add_predictions
from epistasis_evaluation import audit_analyses as audit_analyses
from epistasis_evaluation import cluster_intervals as cluster_intervals
from epistasis_evaluation import detection_report as detection_report
from epistasis_evaluation import element_report as element_report
from epistasis_evaluation import evaluate as evaluate
from epistasis_evaluation import join_audit as join_audit
from epistasis_evaluation import out_of_fold_linear as out_of_fold_linear
from epistasis_evaluation import precision_strata as precision_strata
from epistasis_evaluation import reproduction_check as reproduction_check
from epistasis_evaluation import select_cases as select_cases
from epistasis_evaluation import sequence_only_features as sequence_only_features
from epistasis_evaluation import single_variant_report as single_variant_report
from evo_scoring import EvoScorer as EvoScorer
from evo_scoring import score_quartets as score_quartets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare-siraj")
    p.add_argument("--windows", required=True)
    p.add_argument("--code-zip", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--cell", default="K562")
    p.add_argument("--min-dna", type=float, default=20)
    p.add_argument("--max-se", type=float, default=0.5)
    p = commands.add_parser("score")
    p.add_argument("--quartets", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--backend", choices=("evo", "gc"), default="evo")
    p.add_argument("--checkpoint", default="evo2_7b_base")
    p.add_argument("--revision", default="UNRECORDED")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--weights")
    p = commands.add_parser("evaluate")
    p.add_argument("--quartets", required=True)
    p.add_argument("--scores", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--label", default="Evo 2")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--bootstrap", type=int, default=1000)
    p.add_argument("--sign-threshold", type=float, default=0.25)
    p.add_argument("--audit", default="data/audit.csv.gz")
    args = parser.parse_args()
    if args.command == "prepare-siraj":
        result = prepare_siraj(
            args.windows, args.code_zip, args.out, args.cell, args.min_dna, args.max_se
        )
    elif args.command == "score":
        result = score_quartets(
            args.quartets,
            args.output,
            args.backend,
            args.checkpoint,
            args.revision,
            args.batch_size,
            args.weights,
        )
    else:
        result = evaluate(
            args.quartets,
            args.scores,
            args.out,
            args.label,
            args.folds,
            args.seed,
            args.bootstrap,
            args.sign_threshold,
            args.audit,
        )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
