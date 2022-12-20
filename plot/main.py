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

# Optimal problem lengths
OPTIMAL_LENGTHS = {\
    "att532": 27686,\
    "gr666": 294358,\
    "rat783": 8806,\
    "pr1002": 259045\
}

DIAGRAM_TITLES = {\
    "Cga": "Cga algoritmo trumpiausias kelias",\
    "Cga + 2-opt": "Cga 2-opt algoritmo trumpiausias kelias",\
    "Cga + 3-opt": "Cga 3-opt algoritmo trumpiausias kelias"\
}

ALGO_TO_FILE_NAME_PART = {
    "cga": "Cga",
    "cga2opt": "Cga + 2-opt",
    "cga3opt": "Cga + 3-opt"
}

PLOT_PREFIX = ""

class RecordMetaInfo:
    def __init__(self, meta_info_line):
        # alg,exc_gen,repeat_time,gens,is_optimum_reached,reached_length
        fields = meta_info_line.split(',')
        self.algorithm = fields[0]
        self.exchange_generations = fields[1]
        self.repeat_time = fields[2]
        self.generations = fields[3]
        self.reached_optimal_length = fields[4]
        self.reached_length = fields[5]

class Record:
    def __init__(self, record):
        # record is made up of meta info line, lengths, gen times, opt times
        parts = record.split("\n")
        self.meta = RecordMetaInfo(parts[0])
        self.lengths = parts[1]
        self.gen_times = parts[2]
        self.opt_times = parts[3] TODO: parse as arrays


def main():
    parser = argparse.ArgumentParser(prog = "plot")
    # directory is where benchmark results files are stored
    # the program itself will decide which files it needs
    parser.add_argument("-d", "--directory", required=True)
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
    # TODO: use this argument
    parser.add_argument("-g", "--generation-count",
                        type=int,
                        required=False,
                        default=500)
    args = parser.parse_args()
    directory = args.directory
    if not directory.endswith("/"):
        directory += "/"
    algos = list(map(lambda x: ALGO_TO_FILE_NAME_PART[x], args.algorithms))
    core_counts = args.core_counts
    test_cases = args.test_cases
    max_generations = args.generation_count
    exc_gens = args.exchange_generations
    # for t in test_cases:
    #     for a in algos:
    #         for c in core_counts:
    #             print(make_file_name(directory, t, a, c))
    
    for a in algos:
        print("Processing algorithm: " + a)
        plot_cores_diff_from_opt_test_cases(directory, core_counts, test_cases, a, exc_gens, max_generations)
        
    
    return
    
    x_axis_values = np.arange(1, 501)
    for name in os.listdir(directory):
        file_name = directory + name
        print("Processing file '" + file_name + "'")
        
        with open(file_name, "r") as file:
            full_content = file.read()
        
        content = full_content.split("\n\n\n\n\n\n")
        header = content[0]
        header_lines = header.split('\n')
        problem_name = header_lines[1].split(": ")[1]
        optimal_length = OPTIMAL_LENGTHS[problem_name]
        cpu_count = header_lines[2].split(": ")[1]
        algorithm = file_name.split("alg_")[1].split("_")[0]
        print("problem name: " + problem_name)
        print("optimal length: " + str(optimal_length))
        print("algorithm: " + algorithm)
        print("cpu count: " + cpu_count)
        content = content[1]
        # In theory recor group sepearator is "\n\n\n", but in practice
        # after last record in record group "\n\n" is written,
        # so true record group separator is "\n\n\n\n\ns".
        exchange_gens_split = content.split("\n\n\n\n\n")
        assert(len(exchange_gens_split[-1]) == 0)
        # Discard empty end.
        exchange_gens = exchange_gens_split[:-1]
        plot_file_base_name = PLOT_PREFIX + file_name.split('.')[0].split('/')[-1] + "_"
        for exc in exchange_gens:
            (meta_info, exc_gen_avg) = one_exchange_gen_avg(exc)
            exc_gen_number = meta_info[0].exchange_generations
            # Plot the percentage difference from the optimal tour.
            diff =  map(lambda x: (x - optimal_length) / optimal_length * 100.0, exc_gen_avg)
            plt.plot(x_axis_values, list(diff), label=problem_name + ", " + "F_mig = " + exc_gen_number)
            plt.legend(loc="upper right")
            # plt.title(DIAGRAM_TITLES[algorithm] + ", F_mig = " + exc_gen_number)
            plt.title(DIAGRAM_TITLES[algorithm])
            plt.xlabel("genetinio algoritmo karta")
            plt.ylabel("skirtumas nuo optimalaus kelio, %")
            # plt.show()
            # plt.savefig(plot_file_base_name + exc_gen_number + ".png", format="png", dpi=220)
            # Clears current plot, otherwise they stack from different loop iterations.
            # plt.clf()
        plt.savefig(plot_file_base_name + exc_gen_number + ".png", format="png", dpi=220)
        plt.clf()

def percent_diff_from_optimal(x, optimal):
    diff = x - optimal
    return (diff / optimal) * 100.0

def one_exchange_gen_avg(record_group):
    # Argument: record group as defined above
    records = record_group.split("\n\n")
    records_meta_info = []
    records_gen_lengths = []
    for record in records:
        # record as defined above
        # for now we don't care about the timings
        lines = record.split("\n")
        # first line is meta info
        records_meta_info.append(RecordMetaInfo(lines[0]))
        
        # second line is best length after each generation
        gen_lengths_text_split = lines[1].split(',')
        assert(len(gen_lengths_text_split[-1]) == 0)
        # Discard empty end.
        gen_lengths_text = gen_lengths_text_split[:-1]
        gen_lengths = list(map(int, gen_lengths_text))
        records_gen_lengths.append(gen_lengths)
    
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

def separate_header(text):
    parts = text.split("\n\n\n\n\n\n")
    return (parts[0], parts[1])

def separate_by_exchange_gens(text):
    parts = text.split("\n\n\n\n\n")
    assert(len(exchange_gens_split[-1]) == 0)
    return parts[:-1]

def separate_repeat_runs(text):
    parts = text.split("\n\n")
    return parts



# dir must end with '/'
def make_file_name(dir, test_case, algo, cpus):
    return dir + "bm_" + test_case + "_alg_" + algo + "_" + str(cpus) + "_cpus.out"

# Core count on X axis, difference from optimal on Y,
# different test cases in one plot.
def plot_cores_diff_from_opt_test_cases(dir, core_counts, test_cases, algo, exc_gens, max_gens):
    # Find out which files we need.
    for t in test_cases:
        for c in core_counts:
            print(make_file_name(dir, t, algo, c))

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
