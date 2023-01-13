#!/bin/bash

# Generuoja grafikus kursiniam.
# Pašalina senus grafikus jeigu tokie yra.

PLOT_DIR=../../kursinis/plot

rm -rI $PLOT_DIR

# Jeigu kas nors nepavyksta, baigiam.
set -e

# Grafikai tokie patys kaip straipsnyje (D_m palyginimas).
YTOP3OPT=3.0
YBOTTOM3OPT=0.75
YTOP2OPT=8.9
YBOTTOM2OPT=5.2

python plot.py -d ../all_bench_results -a cga3opt -t att532 gr666 rat783 pr1002 -c 1 2 4 8 --population-sizes 128 -p $PLOT_DIR -f pgf -k cores_diff_test_cases -s 0.45 --y-top $YTOP3OPT --y-bottom $YBOTTOM3OPT -e 4

python plot.py -d ../all_bench_results -a cga3opt -t att532 gr666 rat783 pr1002 -c 1 2 4 8 --population-sizes 128 -p $PLOT_DIR -f pgf -k cores_diff_test_cases -s 0.45 --y-top $YTOP3OPT --y-bottom $YBOTTOM3OPT -e 8

python plot.py -d ../all_bench_results -a cga3opt -t att532 gr666 rat783 pr1002 -c 1 2 4 8 --population-sizes 128 -p $PLOT_DIR -f pgf -k cores_diff_test_cases -s 0.45 --y-top $YTOP3OPT --y-bottom $YBOTTOM3OPT -e 16

python plot.py -d ../all_bench_results -a cga3opt -t att532 gr666 rat783 pr1002 -c 1 2 4 8 --population-sizes 128 -p $PLOT_DIR -f pgf -k cores_diff_test_cases -s 0.45 --y-top $YTOP3OPT --y-bottom $YBOTTOM3OPT -e 32

python plot.py -d ../all_bench_results -a cga2opt -t att532 gr666 rat783 pr1002 -c 1 2 4 8 --population-sizes 128 -p $PLOT_DIR -f pgf -k cores_diff_test_cases -s 0.45 --y-top $YTOP2OPT --y-bottom $YBOTTOM2OPT -e 4

python plot.py -d ../all_bench_results -a cga2opt -t att532 gr666 rat783 pr1002 -c 1 2 4 8 --population-sizes 128 -p $PLOT_DIR -f pgf -k cores_diff_test_cases -s 0.45 --y-top $YTOP2OPT --y-bottom $YBOTTOM2OPT -e 8

python plot.py -d ../all_bench_results -a cga2opt -t att532 gr666 rat783 pr1002 -c 1 2 4 8 --population-sizes 128 -p $PLOT_DIR -f pgf -k cores_diff_test_cases -s 0.45 --y-top $YTOP2OPT --y-bottom $YBOTTOM2OPT -e 16

python plot.py -d ../all_bench_results -a cga2opt -t att532 gr666 rat783 pr1002 -c 1 2 4 8 --population-sizes 128 -p $PLOT_DIR -f pgf -k cores_diff_test_cases -s 0.45 --y-top $YTOP2OPT --y-bottom $YBOTTOM2OPT -e 32


# Grafikai, parodantys kaip einant kartoms trumpėja maršrutas.
# Palygina skirtingus D_m
# Tik cga2opt att532 ir cga3opt rat783, nes kitaip per daug grafikų.

python plot.py -d ../all_bench_results -a cga2opt -t att532 -c 1 2 4 8 --population-sizes 128 -e 4 8 16 32 -p $PLOT_DIR -f pgf -k gens_diff_excg -s 0.45

python plot.py -d ../all_bench_results -a cga3opt -t rat783 -c 1 2 4 8 --population-sizes 128 -e 4 8 16 32 -p $PLOT_DIR -f pgf -k gens_diff_excg -s 0.45


# Palyginimas tarp skirtingų populiacijos dydžių (cga2opt).
# Grupuojama P = 2 4 8 16 į vieną ir P = 32 64 128 256 į kitą.
# TODO: cga3opt

python plot.py -d ../all_bench_results -a cga2opt -t att532 gr666 rat783 pr1002 -c 1 2 4 8 --population-sizes 2 4 8 16 -p $PLOT_DIR -f pgf -k cores_diff_popsizes -s 0.45 -e 4

python plot.py -d ../all_bench_results -a cga2opt -t att532 gr666 rat783 pr1002 -c 1 2 4 8 --population-sizes 32 64 128 256 -p $PLOT_DIR -f pgf -k cores_diff_popsizes -s 0.45 -e 4


# Geriausių individų apsikeitimo tarp branduolių santykinis laikas.

python plot.py -d ../all_bench_results -a cga2opt -t att532 pr1002 -c 8 --population-sizes 2 4 8 16 32 64 128 256 -e 4 -p $PLOT_DIR -k gens_diff_popsizes -f pgf -k relative_times -s 1.0
