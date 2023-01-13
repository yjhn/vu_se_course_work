import os
import argparse
import matplotlib as mpl
import numpy as np

# For number decimal separator formatting.
import locale
locale.setlocale(locale.LC_ALL, 'lt_LT.utf8')

# Run:
# python plot.py -d ../all_bench_results -t att532 gr666 rat783 pr1002 -p pgf -f pgf

# test case = tsp problem file

## Results file format:
## header (not relevant to this script):
# \n\n\n\n\n\n (6 newlines)
## Record groups:
## Record group:
## Record:
# algorithm,exc_gen,repeat_time,gen_reached,is_optimum_reached,reached_length
# \n
# best tour length for each generation, separated by ','
# \n
# tour generation time, separated by ','
# \n
# tour optimization time, separated by ',' (always 0 for pure Cga)
# \n\n
## end record
## another record, but repeat_time + 1
## repeat n times
## end record group
# \n\n\n (effectively 5 because these come after end record)
## another record group, different exc_gen
## n record groups
## end record groups

ALGOS = ["cga", "cga2opt", "cga3opt"]
CORE_COUNTS = [1, 2, 4, 8]
TEST_CASES = ["att532", "gr666", "rat783", "pr1002"]
EXCHANGE_GENERATIONS = [4, 8, 16, 32]
PLOT_KINDS = [
    "gens_diff_excg",
    "cores_diff_test_cases",
    "cores_diff_gens",
    "cores_diff_algos",
    "gens_diff_popsizes",
    "cores_diff_popsizes",
    "relative_times"
]
MAX_GENERATIONS = 500
POPULATION_SIZES = [2, 4, 8, 16, 32, 64, 128, 256]

# Optimal problem (test case) lengths
OPTIMAL_LENGTHS = {
    "att532": 27686,
    "gr666": 294358,
    "rat783": 8806,
    "pr1002": 259045
}

ALGO_DISPLAY_NAMES = {
    "cga": "LKGA",
    "cga2opt": "LKGA + 2-opt",
    "cga3opt": "LKGA + 3-opt"
}

ALGO_TO_FILE_NAME_PART = {
    "cga": "cga",
    "cga2opt": "cga2opt",
    "cga3opt": "cga3opt"
}

# controls where in the plot the legend is placed
PLOT_LEGEND_LOCATION = "upper right"

CORE_COUNT_AXIS_LABEL = "branduolių skaičius"
DIFF_FROM_OPTIMAL_AXIS_LABEL = "skirtumas nuo optimalaus, $\%$"
GENERATIONS_AXIS_LABEL = "genetinio algoritmo karta"

# controls plot image resolution (png)
PLOT_DPI = 220

PLOT_FORMAT = "pgf"
# PLOT_FORMAT = "png"

# pgf plot scale
PLOT_SCALE = 1.0

# For controlling Y axis range (ylim)
Y_TOP = None
Y_BOTTOM = None

# height / width
PLOT_ASPECT_RATIO = 0.8

# Got it with '\showthe\textwidth' in Latex
# (stops comilation and shows the number)
DOCUMENT_WIDTH_PT = 469.47049

class RecordMetaInfo:
    def __init__(self, meta_info_line):
        # alg,exc_gen,repeat_time,gens,is_optimum_reached,reached_length
        fields = meta_info_line.split(',')
        self.algorithm = fields[0]
        self.exchange_generations = int(fields[1])
        self.repeat_time = int(fields[2])
        self.generations = int(fields[3])
        self.reached_optimal_length = fields[4]
        if self.reached_optimal_length == "true":
            raise RuntimeError("Reached optimal length")
        self.reached_length = int(fields[5])
        # If average exchange duration is recorded.
        if len(meta_info_line) == 7:
            self.average_best_tour_echange_duration = int(fields[6])

class Record:
    def __init__(self, record):
        # record is made up of meta info line, lengths, gen times, opt times
        parts = record.split("\n")
        assert(len(parts) == 4)
        self.meta = RecordMetaInfo(parts[0])
        self.lengths = parse_int_list(parts[1])
        self.gen_times = parse_int_list(parts[2])
        self.opt_times = parse_int_list(parts[3])

# All the runs with the same exchange_generations
class RecordGroup:
    def __init__(self, record_group):
        self.records = []
        for rec in separate_repeat_runs(record_group):
            self.records.append(Record(rec))
        self.record_count = len(self.records)
        self.exc_gens = self.records[0].meta.exchange_generations

class FileMetaInfo:
    def __init__(self, header, file_name):
        # format:
        # <dir>/bm_<test case>_<algo>_<cpu count>_cpus_p<population size>.out
        self.file_name = file_name.split("/")[-1]
        header_lines = header.split('\n')
        assert(len(header_lines) == 4)
        self.problem_name = header_lines[1].split(": ")[1]
        self.optimal_length = OPTIMAL_LENGTHS[self.problem_name]
        self.cpu_count = int(header_lines[2].split(": ")[1])
        self.population_size = int(header_lines[3].split(": ")[1])
        self.algorithm = self.file_name.split("_")[2]

def parse_int_list(text, separator=','):
    parts = text.split(separator)
    assert(len(parts[-1]) == 0)
    parts = parts[:-1]
    ints = list(map(int, parts))
    return ints

# Parse file content.
def parse_benchmark_results(file_name):
    with open(file_name, "r") as file:
        full_content = file.read()

    (header, content) = separate_header(full_content)
    file_meta_info = FileMetaInfo(header, file_name)
    
    record_groups_text = separate_by_exchange_gens(content)
    record_groups = []
    for r in record_groups_text:
        record_groups.append(RecordGroup(r))
    
    return (file_meta_info, record_groups)

def separate_header(text):
    parts = text.split("\n\n\n\n\n\n")
    return (parts[0], parts[1])

# Separates main content into record groups.
def separate_by_exchange_gens(text):
    # In theory record group separator is "\n\n\n", but in practice
    # after last record in record group "\n\n" is written,
    # so true record group separator is "\n\n\n\n\n".
    parts = text.split("\n\n\n\n\n")
    assert(len(parts[-1]) == 0)
    return parts[:-1]

def separate_repeat_runs(text):
    parts = text.split("\n\n")
    return parts


def main():
    parser = argparse.ArgumentParser(prog="plot")
    # directory is where benchmark results files are stored
    # the program itself will decide which files it needs
    parser.add_argument("-d", "--directory", required=True)
    parser.add_argument("-p", "--plot-directory",
                        required=False,
                        default="./")
    parser.add_argument("-a", "--algorithms",
                        choices=ALGOS,
                        nargs="+",
                        required=False,
                        default=ALGOS)
    parser.add_argument("-c", "--core-counts",
                        type=int,
                        choices=CORE_COUNTS,
                        nargs="+",
                        required=False,
                        default=CORE_COUNTS)
    parser.add_argument("-t", "--test-cases",
                        choices=TEST_CASES,
                        nargs="+",
                        required=False,
                        default=TEST_CASES)
    # Averages over given exchange generations.
    parser.add_argument("-e", "--exchange-generations",
                        choices=EXCHANGE_GENERATIONS,
                        type=int,
                        nargs="+",
                        required=False,
                        default=EXCHANGE_GENERATIONS)
    # show results after this many generations
    parser.add_argument("-g", "--generation-count",
                        type=int,
                        required=False,
                        default=MAX_GENERATIONS)
    # Population size
    parser.add_argument("--population-sizes",
                        choices=POPULATION_SIZES,
                        type=int,
                        nargs="+",
                        required=False,
                        default=[128])
    # what kind of plots to generate
    parser.add_argument("-k", "--plot-kinds",
                        choices=PLOT_KINDS,
                        nargs="+",
                        required=False,
                        default=PLOT_KINDS)
    global PLOT_FORMAT
    parser.add_argument("-f", "--plot-format",
                        choices=["pgf", "png"],
                        required=False,
                        default=PLOT_FORMAT)
    global PLOT_SCALE
    # for pgf only, means width proportion of textwidth
    parser.add_argument("-s", "--plot-scale",
                        type=float,
                        required=False,
                        default=PLOT_SCALE)
    global Y_TOP
    global Y_BOTTOM
    parser.add_argument("--y-top",
                        type=float,
                        required=False,
                        default=Y_TOP)
    parser.add_argument("--y-bottom",
                        type=float,
                        required=False,
                        default=Y_BOTTOM)
    # Whether to add diagram title to the specified plot kind.
    # Currently only affects gens_diff_excg.
    parser.add_argument("--no-titles",
                        type=bool,
                        required=False,
                        default=False)
    args = parser.parse_args()
    directory = canonicalize_dir(args.directory)
    results_dir = canonicalize_dir(args.plot_directory)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)

    PLOT_FORMAT = args.plot_format
    PLOT_SCALE = args.plot_scale
    Y_TOP = args.y_top
    Y_BOTTOM = args.y_bottom

    algos = list(map(lambda x: ALGO_TO_FILE_NAME_PART[x], args.algorithms))
    core_counts = args.core_counts
    test_cases = args.test_cases
    max_generations = args.generation_count
    exc_gens = args.exchange_generations
    population_sizes = args.population_sizes
    plot_kinds = args.plot_kinds
    add_title = not args.no_titles

    if "cores_diff_test_cases" in plot_kinds:
        for a in algos:
            for p in population_sizes:
                plot_cores_diff_from_opt_test_cases(
                    directory=directory,
                    core_counts=core_counts,
                    test_cases=test_cases,
                    algo=a,
                    exc_gens=exc_gens,
                    max_gens=max_generations,
                    pop_size=p,
                    results_dir=results_dir,
                    add_title=add_title
                    )

    if "cores_diff_gens" in plot_kinds:
        for a in algos:
            for t in test_cases:
                for p in population_sizes:
                    plot_cores_diff_from_opt_generations(
                        directory=directory,
                        test_case=t,
                        algo=a,
                        core_counts=core_counts,
                        exc_gens=exc_gens,
                        max_gens=max_generations,
                        pop_size=p,
                        results_dir=results_dir,
                        add_title=add_title
                        )

    if "gens_diff_popsizes" in plot_kinds:
        for a in algos:
            for t in test_cases:
                for c in core_counts:
                    plot_generations_diff_from_opt_pop_sizes(
                        directory=directory,
                        test_case=t,
                        algo=a,
                        core_count=c,
                        exc_gens=exc_gens,
                        max_gens=max_generations,
                        pop_sizes=population_sizes,
                        results_dir=results_dir,
                        add_title=add_title
                        )

    if "cores_diff_popsizes" in plot_kinds:
        for a in algos:
            for t in test_cases:
                plot_cores_diff_from_opt_pop_sizes(
                    directory=directory,
                    test_case=t,
                    algo=a,
                    core_counts=core_counts,
                    exc_gens=exc_gens,
                    max_gens=max_generations,
                    pop_sizes=population_sizes,
                    results_dir=results_dir,
                    add_title=add_title
                    )


    if "cores_diff_algos" in plot_kinds:
        for t in test_cases:
            for p in population_sizes:
                plot_cores_diff_from_opt_algos(
                    directory=directory,
                    test_case=t,
                    algos=algos,
                    core_counts=core_counts,
                    exc_gens=exc_gens,
                    max_gens=max_generations,
                    pop_size=p,
                    results_dir=results_dir,
                    add_title=add_title
                    )

    if "gens_diff_excg" in plot_kinds:
        for p in population_sizes:
            plot_basic(
                directory=directory,
                results_dir=results_dir,
                algos=algos,
                test_cases=test_cases,
                core_counts=core_counts,
                pop_size=p,
                add_title=add_title
                )

    if "relative_times" in plot_kinds:
        plot_relative_times_for_tour_exchange(add_title)


def canonicalize_dir(directory):
    if not directory.endswith("/"):
        return directory + "/"
    else:
        return directory

def percent_diff_from_optimal(x, optimal):
    diff = x - optimal
    return (diff / optimal) * 100.0

def one_exchange_gen_avg(record_group):
    records_meta_info = []
    records_gen_lengths = []
    for record in record_group.records:
        records_meta_info.append(record.meta)
        records_gen_lengths.append(record.lengths)
    
    # Average the generation lengths
    rec_gen_len_len = len(records_gen_lengths)
    avg_gen_lengths = []
    for i in range(0, MAX_GENERATIONS):
        sum_total = records_gen_lengths[0][i]
        for j in range(1, rec_gen_len_len):
            sum_total += records_gen_lengths[j][i]
        avg = sum_total / rec_gen_len_len
        avg_gen_lengths.append(avg)
    
    return (records_meta_info, avg_gen_lengths)

# keyword args should be used
# x_values = array
# y_values = array of arrays
# labels = array of labels, len(labels) == len(y_values)
def plot_and_save(x_values, y_values, labels, title, xlabel, ylabel, file_name, add_title, xticks=None, style={"marker": ".", "linestyle": "dashed", "linewidth": 0.75}):
    if PLOT_FORMAT == "pgf":
        # mpl.use() must be called before importing pyplot
        mpl.use("pgf")
        from matplotlib import pyplot as plt
        plt.rcParams.update({
            "font.family": "serif",  # use serif/main font for text elements
            "text.usetex": True,     # use inline math for ticks
            "pgf.rcfonts": False,    # don't setup fonts from rc parameters
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.labelspacing": 0.1,
            "legend.handlelength": 1.2,
            "legend.frameon": False,
            "legend.shadow": False,
            "legend.borderpad": 0.4,
            "legend.borderaxespad": 0.0,
            "axes.formatter.use_locale": True # use decimal separator ','
        })
        fig = plt.figure(figsize=set_size(fraction=PLOT_SCALE))
    else:
        from matplotlib import pyplot as plt
    
    for (y, l) in zip(y_values, labels):
        plt.plot(x_values, y, label=l, **style)
    if xticks is not None:
        plt.xticks(xticks)
    plt.legend(loc=PLOT_LEGEND_LOCATION)
    if add_title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(top=Y_TOP, bottom=Y_BOTTOM)
    final_file_name = f"{file_name}.{PLOT_FORMAT}"
    if os.path.exists(final_file_name):
        raise FileExistsError(f"Will not overwrite existing plot file:\n{final_file_name}")
    print(f"saving plot: {final_file_name}")
    plt.tight_layout(pad=0.0)
    # dpi is ignored when using pgf
    plt.savefig(final_file_name, format=PLOT_FORMAT, dpi=PLOT_DPI)
    plt.clf()
    plt.close()

# dir must end with '/'
def make_file_name(directory, test_case, algo, cpus, pop_size):
    return f"{directory}bm_{test_case}_{algo}_{cpus}_cpus_p{pop_size}.out"

# Plots the difference from the optimal length.
# x axis - generations
# y axis - diff from optimal
# in one plot: one test case, one thread count, all F_mig (exchange gens)
def plot_basic(
    directory,
    results_dir,
    algos,
    test_cases,
    core_counts,
    pop_size,
    add_title):
    
    x_axis_values = np.arange(1, 501)
    
    for a in algos:
        for t in test_cases:
            for c in core_counts:
                file_name = make_file_name(directory, t, a, c, pop_size)
                (meta, data) = parse_benchmark_results(file_name)
                
                problem_name = meta.problem_name
                optimal_length = meta.optimal_length
                algorithm = meta.algorithm
                cpu_count = meta.cpu_count
                plot_file_base_name = file_name.split('/')[-1].split('.')[0]
                y_values = []
                labels = []
                title = f"{ALGO_DISPLAY_NAMES[algorithm]}, \\texttt{{{problem_name}}}, $B = {cpu_count}$, $P = {pop_size}$"
                xlabel = GENERATIONS_AXIS_LABEL
                ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
                file_name = results_dir + plot_file_base_name
                for exc in data:
                    (meta_info, exc_gen_avg) = one_exchange_gen_avg(exc)
                    # Plot the percentage difference from the optimal tour.
                    diff =  map(lambda x: (x - optimal_length) / optimal_length * 100.0, exc_gen_avg)
                    y_values.append(list(diff))
                    labels.append(f"$D_m={exc.exc_gens}$")

                plot_and_save(x_values=x_axis_values,
                    y_values=y_values,
                    labels=labels,
                    title=title,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    file_name=file_name,
                    add_title=add_title,
                    style={"linewidth": 1}
                    )


# Core count on X axis, difference from optimal on Y,
# different test cases in one plot.
def plot_cores_diff_from_opt_test_cases(
    directory,
    core_counts,
    test_cases,
    algo,
    exc_gens,
    max_gens,
    pop_size,
    results_dir,
    add_title):
    
    if len(exc_gens) == 1:
        title = f"{ALGO_DISPLAY_NAMES[algo]}, $D_m = {exc_gens[0]}$, $K = {max_gens}$ kart\\~{{ų}}, $P = {pop_size}$"
        plot_file_name = f"cores_diff_from_opt_test_cases_{algo}_egen_{exc_gens[0]}_p{pop_size}"
    else:
        title = f"{ALGO_DISPLAY_NAMES[algo]}, $K = {max_gens}$ kart\\~{{ų}}, $P = {pop_size}$"
        plot_file_name = f"cores_diff_from_opt_test_cases_{algo}_p{pop_size}"
    x_values = core_counts
    xlabel = CORE_COUNT_AXIS_LABEL
    ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    parsed_files = []
    labels_all_test_cases = []
    diffs_all_test_cases = []
    for t in test_cases:
        labels_all_test_cases.append(f"\\texttt{{{t}}}")
        diffs_all_core_counts = []
        for c in core_counts:
            file_name = make_file_name(directory, t, algo, c, pop_size)
            (meta, data) = parse_benchmark_results(file_name)
            parsed_files.append((meta, data))
            total = 0
            # We are only interested in exc_gens specified.
            required_exc_gens = filter(lambda rg: rg.exc_gens in exc_gens, data)
            required_exc_gens_count = 0
            for r_group in required_exc_gens:
                required_exc_gens_count += 1
                for rec in r_group.records:
                    total += rec.lengths[max_gens - 1]
            avg = total / (r_group.record_count * required_exc_gens_count)
            diff = percent_diff_from_optimal(avg, meta.optimal_length)
            diffs_all_core_counts.append(diff)
        diffs_all_test_cases.append(diffs_all_core_counts)

    plot_and_save(x_values=x_values,
            y_values=diffs_all_test_cases,
            labels=labels_all_test_cases,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xticks=core_counts,
            file_name=results_dir + plot_file_name,
            add_title=add_title
            )

# Core count on X axis, difference from optimal on Y,
# plots single test case, varies generations count.
# plots every 100 gens, up to and including max_gens
def plot_cores_diff_from_opt_generations(
    directory,
    test_case,
    algo,
    core_counts,
    exc_gens,
    max_gens,
    pop_size,
    results_dir,
    add_title):
    
    if len(exc_gens) == 1:
        title = f"{ALGO_DISPLAY_NAMES[algo]}, \\texttt{{{test_case}}}, $D_m = {exc_gens[0]}$, $P = {pop_size}$"
        plot_file_name = f"cores_diff_from_opt_gens_{test_case}_{algo}_egen_{exc_gens[0]}_p{pop_size}"
    else:
        title = f"{ALGO_DISPLAY_NAMES[algo]}, \\texttt{{{test_case}}}, $P = {pop_size}$"
        plot_file_name = f"cores_diff_from_opt_gens_{test_case}_{algo}_p{pop_size}"
    x_values = core_counts
    xlabel = CORE_COUNT_AXIS_LABEL
    ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    parsed_files = []
    labels_all_gens_counts = []
    diffs_all_gens_counts = []
    # TODO: include generation 0
    for g in range(99, max_gens, 100):
        labels_all_gens_counts.append(f"$K = {str(g + 1)}$")
        diffs_single_gens_count = []
        for c in core_counts:
            file_name = make_file_name(directory, test_case, algo, c, pop_size)
            (meta, data) = parse_benchmark_results(file_name)
            parsed_files.append((meta, data))
            total = 0
            # We are only interested in exc_gens specified.
            required_exc_gens = filter(lambda rg: rg.exc_gens in exc_gens, data)
            required_exc_gens_count = 0
            for r_group in required_exc_gens:
                required_exc_gens_count += 1
                for rec in r_group.records:
                    total += rec.lengths[g]
            avg = total / (r_group.record_count * required_exc_gens_count)
            diff = percent_diff_from_optimal(avg, meta.optimal_length)
            diffs_single_gens_count.append(diff)
        diffs_all_gens_counts.append(diffs_single_gens_count)
    
    plot_and_save(x_values=x_values,
            y_values=diffs_all_gens_counts,
            labels=labels_all_gens_counts,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xticks=core_counts,
            file_name=results_dir + plot_file_name,
            add_title=add_title
            )


# Core count on X axis, difference from optimal on Y,
# plots multiple algorithms and a single test case.
def plot_cores_diff_from_opt_algos(
    directory,
    test_case,
    algos,
    core_counts,
    exc_gens,
    max_gens,
    pop_size,
    results_dir,
    add_title):
    
    if len(exc_gens) == 1:
        title = f"\\texttt{{{test_case}}}, $D_m = {exc_gens[0]}$, $K = {max_gens}$ kart\\~{{ų}} $P = {pop_size}$"
        plot_file_name = f"cores_diff_from_opt_algos_{test_case}_mgen_{max_gens}_egen_{exc_gens[0]}_p{pop_size}"
    else:
        title = f"\\texttt{{{test_case}}}, $K = {max_gens}$ kart\\~{{ų}} $P = {pop_size}$"
        plot_file_name = f"cores_diff_from_opt_algos_{test_case}_mgen_{max_gens}_p{pop_size}"
    x_values = core_counts
    xlabel = CORE_COUNT_AXIS_LABEL
    ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    parsed_files = []
    labels_all_algos = []
    diffs_all_algos = []
    for a in algos:
        labels_all_algos.append(ALGO_DISPLAY_NAMES[a])
        diffs_single_algo = []
        for c in core_counts:
            file_name = make_file_name(directory, test_case, a, c, pop_size)
            (meta, data) = parse_benchmark_results(file_name)
            parsed_files.append((meta, data))
            total = 0
            # We are only interested in exc_gens specified.
            required_exc_gens = filter(lambda rg: rg.exc_gens in exc_gens, data)
            required_exc_gens_count = 0
            for r_group in required_exc_gens:
                required_exc_gens_count += 1
                for rec in r_group.records:
                    total += rec.lengths[max_gens - 1]
            avg = total / (r_group.record_count * required_exc_gens_count)
            diff = percent_diff_from_optimal(avg, meta.optimal_length)
            diffs_single_algo.append(diff)
        diffs_all_algos.append(diffs_single_algo)
    
    plot_and_save(x_values=x_values,
            y_values=diffs_all_algos,
            labels=labels_all_algos,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xticks=core_counts,
            file_name=results_dir + plot_file_name,
            add_title=add_title
            )


# Generations on X axis, difference from optimal on Y,
# plots multiple population sizes and a single test case.
def plot_generations_diff_from_opt_pop_sizes(
    directory,
    test_case,
    algo,
    core_count,
    exc_gens,
    max_gens,
    pop_sizes,
    results_dir,
    add_title):
    
    if len(exc_gens) == 1:
        title = f"{ALGO_DISPLAY_NAMES[algo]} \\texttt{{{test_case}}}, $D_m = {exc_gens[0]}$, $B = {core_count}$"
        plot_file_name = f"gens_diff_from_opt_pop_sizes_{test_case}_{algo}_cpus_{core_count}_mgen_{max_gens}_egen_{exc_gens[0]}"
    else:
        title = f"{ALGO_DISPLAY_NAMES[algo]} \\texttt{{{test_case}}}, $B = {core_count}$"
        plot_file_name = f"gens_diff_from_opt_pop_sizes_{test_case}_{algo}_cpus_{core_count}_mgen_{max_gens}"
    x_values = np.arange(1, max_gens + 1)
    xlabel = GENERATIONS_AXIS_LABEL
    ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    parsed_files = []
    labels_all_pop_sizes = []
    diffs_all_pop_sizes = []
    for p in pop_sizes:
        diffs_single_pop_size = []
        labels_all_pop_sizes.append(f"$P = {p}$")
        for g in range(0, max_gens):
            file_name = make_file_name(directory, test_case, algo, core_count, p)
            (meta, data) = parse_benchmark_results(file_name)
            parsed_files.append((meta, data))
            total = 0
            # We are only interested in exc_gens specified.
            required_exc_gens = filter(lambda rg: rg.exc_gens in exc_gens, data)
            required_exc_gens_count = 0
            for r_group in required_exc_gens:
                required_exc_gens_count += 1
                for rec in r_group.records:
                    total += rec.lengths[g]
            avg = total / (r_group.record_count * required_exc_gens_count)
            diff = percent_diff_from_optimal(avg, meta.optimal_length)
            diffs_single_pop_size.append(diff)
        diffs_all_pop_sizes.append(diffs_single_pop_size)
    
    plot_and_save(x_values=x_values,
            y_values=diffs_all_pop_sizes,
            labels=labels_all_pop_sizes,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            file_name=results_dir + plot_file_name,
            add_title=add_title,
            style={"linewidth": 1}
            )


# Core count on X axis, difference from optimal on Y,
# plots multiple population sizes and a single test case.
def plot_cores_diff_from_opt_pop_sizes(
    directory,
    test_case,
    algo,
    core_counts,
    exc_gens,
    max_gens,
    pop_sizes,
    results_dir,
    add_title):
    
    popstring = '_'.join(map(str, pop_sizes))
    if len(exc_gens) == 1:
        title = f"{ALGO_DISPLAY_NAMES[algo]} \\texttt{{{test_case}}}, $D_m = {exc_gens[0]}$, $K = {max_gens}$"
        plot_file_name = f"cores_diff_from_opt_pop_sizes_{test_case}_{algo}_mgen_{max_gens}_egen_{exc_gens[0]}_p{popstring}"
    else:
        title = f"{ALGO_DISPLAY_NAMES[algo]} \\texttt{{{test_case}}}, $K = {max_gens}$"
        plot_file_name = f"cores_diff_from_opt_pop_sizes_{test_case}_{algo}_mgen_{max_gens}_p{popstring}"
    x_values = core_counts
    xlabel = CORE_COUNT_AXIS_LABEL
    ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    parsed_files = []
    labels_all_pop_sizes = []
    diffs_all_pop_sizes = []
    for p in pop_sizes:
        diffs_single_pop_size = []
        labels_all_pop_sizes.append(f"$P = {p}$")
        for c in core_counts:
            file_name = make_file_name(directory, test_case, algo, c, p)
            (meta, data) = parse_benchmark_results(file_name)
            parsed_files.append((meta, data))
            total = 0
            # We are only interested in exc_gens specified.
            required_exc_gens = filter(lambda rg: rg.exc_gens in exc_gens, data)
            required_exc_gens_count = 0
            for r_group in required_exc_gens:
                required_exc_gens_count += 1
                for rec in r_group.records:
                    total += rec.lengths[max_gens - 1]
            avg = total / (r_group.record_count * required_exc_gens_count)
            diff = percent_diff_from_optimal(avg, meta.optimal_length)
            diffs_single_pop_size.append(diff)
        diffs_all_pop_sizes.append(diffs_single_pop_size)
    
    plot_and_save(x_values=x_values,
            y_values=diffs_all_pop_sizes,
            labels=labels_all_pop_sizes,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xticks=core_counts,
            file_name=results_dir + plot_file_name,
            add_title=add_title
            )


def average(a):
    sum(a) / len(a)

def plot_relative_times_for_tour_exchange(add_title):
    TOUR_EXC_TIMES_ATT532_2_CPUS = [3678, 4319, 3437, 3765, 4101]
    TOUR_EXC_TIMES_ATT532_4_CPUS = [8300, 7115, 8022, 7440, 8434]
    TOUR_EXC_TIMES_ATT532_8_CPUS = [11761, 10933, 11863, 10495, 12262]
    TOUR_EXC_TIMES_GR666_2_CPUS = [4754, 4767, 6550, 5304, 5613]
    TOUR_EXC_TIMES_GR666_4_CPUS = [9629, 11163, 8007, 9457, 10083]
    TOUR_EXC_TIMES_GR666_8_CPUS = [13266, 14978, 14262, 11976, 13933]
    TOUR_EXC_TIMES_RAT783_2_CPUS = [8533, 6837, 6868, 11350, 7365]
    TOUR_EXC_TIMES_RAT783_4_CPUS = [14783, 16501, 19037, 18595, 18171]
    TOUR_EXC_TIMES_RAT783_8_CPUS = [22227, 23529, 25757, 25914, 24239]
    TOUR_EXC_TIMES_PR1002_2_CPUS = [12823, 11792, 11032, 11902, 13026]
    TOUR_EXC_TIMES_PR1002_4_CPUS = [24973, 25967, 23896, 27698, 21294]
    TOUR_EXC_TIMES_PR1002_8_CPUS = [38182, 35785, 36430, 35394, 36911]
    
    labels = [
        "\\texttt{att532}",
        "\\texttt{gr666}",
        "\\texttt{rata783}",
        "\\texttt{pr1002}"
    ]
    
    baseline = average(TOUR_EXC_TIMES_ATT532_2_CPUS)
    
    averages = [
        [
            1,
            average(TOUR_EXC_TIMES_ATT532_4_CPUS) / baseline,
            average(TOUR_EXC_TIMES_ATT532_8_CPUS) / baseline
        ],
        [
            average(TOUR_EXC_TIMES_GR666_2_CPUS) / baseline,
            average(TOUR_EXC_TIMES_GR666_4_CPUS) / baseline,
            average(TOUR_EXC_TIMES_GR666_8_CPUS) / baseline
        ],
        [
            average(TOUR_EXC_TIMES_RAT783_2_CPUS) / baseline,
            average(TOUR_EXC_TIMES_RAT783_4_CPUS) / baseline,
            average(TOUR_EXC_TIMES_RAT783_8_CPUS) / baseline
        ],
        [
            average(TOUR_EXC_TIMES_PR1002_2_CPUS) / baseline,
            average(TOUR_EXC_TIMES_PR1002_4_CPUS) / baseline,
            average(TOUR_EXC_TIMES_PR1002_8_CPUS) / baseline
        ]
    ]
    
    title = "LKGA individų apsikeitimo trukmė"
    xlabel = CORE_COUNT_AXIS_LABEL
    ylabel = "santykinis laikas"
    xvalues = [2, 4, 8]
    xticks = [2, 4, 8]
    
    plot_and_save(x_values=xvalues,
            y_values=averages,
            labels=labels,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            xticks=xticks,
            file_name="relative_time_to_exchange_individuals",
            add_title=add_title
            )

# for pgf
def set_size(fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = DOCUMENT_WIDTH_PT * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * PLOT_ASPECT_RATIO * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

if __name__ == "__main__":
    main()
