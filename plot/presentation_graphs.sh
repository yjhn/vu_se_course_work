#!/bin/bash

# Generuoja grafikus kursinio pristatymui.
# Pašalina senus grafikus jeigu tokie yra.

PLOT_DIR=../../kursinis/pristatymas/plot
BENCH_RESULTS_DIR=../all_bench_results

rm -Ir $PLOT_DIR

# Jeigu kas nors nepavyksta, baigiam.
set -e

# Palyginimas tarp skirtingų populiacijos dydžių (cga2opt).
# Grupuojama P = 2 4 8 16 į vieną ir P = 32 64 128 256 į kitą.

# python plot.py -d $BENCH_RESULTS_DIR -a cga2opt -t gr666 -c 1 2 4 8 --population-sizes 2 4 8 16 -p $PLOT_DIR -f pgf -k cores_diff_popsizes -s 0.45 -e 4

python plot.py -d $BENCH_RESULTS_DIR -a cga2opt -t gr666 -c 1 2 4 8 --population-sizes 32 64 128 256 -p $PLOT_DIR -f pgf -k cores_diff_popsizes -s 0.45 -e 4


# Kartų skaičiaus įtaka skirtingiems atvejams.

python plot.py -d $BENCH_RESULTS_DIR -a cga2opt cga3opt -t gr666 -c 1 2 4 8 --population-sizes 128 -e 4 -p $PLOT_DIR -f pgf -k cores_diff_gens -s 0.45
