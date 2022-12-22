import os
import argparse
from matplotlib import pyplot as plt
import numpy as np

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

# Optimal problem (test case) lengths
OPTIMAL_LENGTHS = {
    "att532": 27686,
    "gr666": 294358,
    "rat783": 8806,
    "pr1002": 259045
}

DIAGRAM_TITLES = {\
    "Cga": "Cga algoritmo trumpiausias kelias",
    "Cga + 2-opt": "Cga 2-opt algoritmo trumpiausias kelias",
    "Cga + 3-opt": "Cga 3-opt algoritmo trumpiausias kelias"
}

ALGO_TO_FILE_NAME_PART = {
    "cga": "Cga",
    "cga2opt": "Cga + 2-opt",
    "cga3opt": "Cga + 3-opt"
}

# controls plot image resolution
PLOT_DPI = 220

# controls where in the plot the legend is placed
PLOT_LEGEND_LOCATION = "upper right"

class RecordMetaInfo:
    def __init__(self, meta_info_line):
        # alg,exc_gen,repeat_time,gens,is_optimum_reached,reached_length
        fields = meta_info_line.split(',')
        self.algorithm = fields[0]
        self.exchange_generations = int(fields[1])
        self.repeat_time = int(fields[2])
        self.generations = int(fields[3])
        self.reached_optimal_length = fields[4]
        self.reached_length = int(fields[5])

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
        self.file_name = file_name
        header_lines = header.split('\n')
        assert(len(header_lines) == 3)
        self.problem_name = header_lines[1].split(": ")[1]
        self.optimal_length = OPTIMAL_LENGTHS[self.problem_name]
        self.cpu_count = header_lines[2].split(": ")[1]
        self.algorithm = file_name.split("alg_")[1].split("_")[0]

def parse_int_list(text, separator=','):
    parts = text.split(separator)
    assert(len(parts[-1]) == 0)
    parts = parts[:-1]
    ints = list(map(int, parts))
    return ints

# Parse file content.
def parse(file_name):
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
    # In theory record group sepearator is "\n\n\n", but in practice
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
                        choices=["cga", "cga2opt", "cga3opt"],
                        nargs="+",
                        required=False,
                        default=["cga", "cga2opt", "cga3opt"])
    parser.add_argument("-c", "--core-counts",
                        type=int,
                        choices=[1, 2, 4, 8],
                        nargs="+",
                        required=False,
                        default=[1, 2, 4, 8])
    parser.add_argument("-t", "--test-cases",
                        choices=["att532", "gr666", "rat783", "pr1002"],
                        nargs="+",
                        required=False,
                        default=["att532", "gr666", "rat783", "pr1002"])
    # will average over the given exchange generations as they
    # do not make any meaningful difference
    parser.add_argument("-e", "--exchange-generations",
                        choices=[4, 8, 16, 32],
                        type=int,
                        nargs="+",
                        required=False,
                        default=[4, 8, 16, 32])
    # show results after this many generations
    parser.add_argument("-g", "--generation-count",
                        type=int,
                        required=False,
                        default=500)
    args = parser.parse_args()
    directory = canonicalize_dir(args.directory)
    results_dir = canonicalize_dir(args.plot_directory)
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    
    algos = list(map(lambda x: ALGO_TO_FILE_NAME_PART[x], args.algorithms))
    core_counts = args.core_counts
    test_cases = args.test_cases
    max_generations = args.generation_count
    exc_gens = args.exchange_generations
    
    for a in algos:
        print("Processing algorithm: " + a)
        for e in exc_gens:
            plot_cores_diff_from_opt_test_cases(
                directory=directory,
                core_counts=core_counts,
                test_cases=test_cases,
                algo=a,
                exc_gens=e,
                max_gens=max_generations,
                results_dir=results_dir
                )
    
    # plot_basic(directory, results_dir)

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
    for i in range(0, 500):
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
def plot_and_save(x_values, y_values, labels, title, xlabel, ylabel, file_name):
    for (y, l) in zip(y_values, labels):
        plt.plot(x_values, y, label=l, marker='o', linestyle='dashed')
    plt.legend(loc=PLOT_LEGEND_LOCATION)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    final_file_name = f"{file_name}.png"
    if os.path.exists(final_file_name):
        raise FileExistsError(f"Will not overwrite existing plot file:\n{final_file_name}")
    print(f"saving plot: {final_file_name}")
    plt.savefig(final_file_name, format="png", dpi=PLOT_DPI)
    plt.clf()

# dir must end with '/'
def make_file_name(directory, test_case, algo, cpus):
    return f"{directory}bm_{test_case}_alg_{algo}_{cpus}_cpus.out"

# Plots the difference from the optimal length.
# x axis - generations
# y axis - diff from optimal
# in one plot: one test case, one thread count, all F_mig (exchange gens)
def plot_basic(directory, results_dir):
    x_axis_values = np.arange(1, 501)
    for name in os.listdir(directory):
        file_name = directory + name
        print(f"Processing file '{file_name}'")
        
        (meta_info, exchange_gens) = parse(file_name)
        
        problem_name = meta_info.problem_name
        optimal_length = meta_info.optimal_length
        algorithm = meta_info.algorithm
        cpu_count = meta.cpu_count
        plot_file_base_name = file_name.split('.')[0].split('/')[-1]
        y_values = []
        labels = []
        if cpu_count == 1:
            cpus_name = "branduolys"
        else:
            cpus_name = "branduoliai"
        title = f"{DIAGRAM_TITLES[algorithm]}, {cpu_count} {cpus_name}"
        xlabel = "genetinio algoritmo karta"
        ylabel = "skirtumas nuo optimalaus kelio, %"
        file_name = results_dir + plot_file_base_name
        for exc in exchange_gens:
            (meta_info, exc_gen_avg) = one_exchange_gen_avg(exc)
            # Plot the percentage difference from the optimal tour.
            diff =  map(lambda x: (x - optimal_length) / optimal_length * 100.0, exc_gen_avg)
            y_values.append(list(diff))
            labels.append(f"{problem_name}, F_mig = {exc.exc_gens}")

        plot_and_save(x_values=x_axis_values,
             y_values=y_values,
             labels=labels,
             title=title,
             xlabel=xlabel,
             ylabel=ylabel,
             file_name=file_name
             )


# Core count on X axis, difference from optimal on Y,
# different test cases in one plot.
def plot_cores_diff_from_opt_test_cases(directory, core_counts, test_cases, algo, exc_gens, max_gens, results_dir):
    title = f"{algo} skirtumas nuo optimalaus po {max_gens} kartų, F_mig = {exc_gens}"
    x_values = [1, 2, 4, 8]
    xlabel = "branduolių skaičius"
    ylabel = "skirtumas nuo optimalaus, %"
    plot_file_name = f"cores_diff_from_opt_test_cases_mgen_{max_gens}_egen_{exc_gens}_{algo}"
    parsed_files = []
    labels_all_test_cases = []
    diffs_all_test_cases = []
    for t in test_cases:
        labels_all_test_cases.append(t)
        diffs_all_core_counts = []
        for c in core_counts:
            file_name = make_file_name(directory, t, algo, c)
            (meta, data) = parse(file_name)
            parsed_files.append((meta, data))
            # we only care about the record group that has the desired
            # exc_gens (F_mig)
            for r_group in data:
                if r_group.exc_gens == exc_gens:
                    # average difference from optimal after max_gens generations
                    total = 0
                    for rec in r_group.records:
                        total += rec.lengths[max_gens - 1]
                    avg = total / r_group.record_count
                    diff = percent_diff_from_optimal(avg, meta.optimal_length)
                    diffs_all_core_counts.append(diff)
        diffs_all_test_cases.append(diffs_all_core_counts)

    plot_and_save(x_values=x_values,
            y_values=diffs_all_test_cases,
            labels=labels_all_test_cases,
            title=title,
            xlabel=xlabel,
            ylabel=ylabel,
            file_name=results_dir + plot_file_name
            )

# Core count on X axis, difference from optimal on Y,
# plots single test case, varies generations count.
def plot_cores_diff_from_opt_generations(test_case, algo):
    TODO

# Core count on X axis, difference from optimal on Y,
# plots multiple algorithms ans a single test case.
def plot_cores_diff_from_opt_multi_alg(test_case, algos):
    TODO

if __name__ == "__main__":
    main()
