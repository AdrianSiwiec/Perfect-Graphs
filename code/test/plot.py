# coding: utf-8

usePgf = True
# usePgf = False

import matplotlib as mpl

if usePgf:
    mpl.use("pgf")
else:
    mpl.use("Agg")


from cycler import cycler
import re
import io
import os
import sys
import csv
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.ticker as ticker
import matplotlib.patches as patches
import numpy as np
import ntpath

plt.rcParams.update(
    {
        "pgf.texsystem": "pdflatex",
        "font.family": "serif",
        "pgf.rcfonts": False,
    }
)

csv_filename = "Filename"
csv_N = "n"
csv_result = "result"
csv_algo = "algorithm"
csv_overall = "Overall"
csv_numRuns = "num_runs"
csv_simpleStructures = "Simple Structures"
csv_getNearCleaners = "Get Near Cleaners"
csv_testNCShortestPaths = "Test NC Shortest Paths"
csv_gpuCopyR = "GPU Test NC Copy"
csv_gpuWork = "GPU Test NC work"
csv_testNCRest = "Test NC Rest"
csv_cpuPreparation = "CPU Preparation"
csv_gpuCalculation = "GPU Calculation"
csv_color_res = "Color rest"
csv_color_CSDP = "Color CSDP"


csv_ints = []
csv_ints.extend([csv_N, csv_result, csv_numRuns])
csv_floats = []
csv_floats.extend(
    [
        csv_overall,
        csv_simpleStructures,
        csv_getNearCleaners,
        csv_testNCShortestPaths,
        csv_gpuCopyR,
        csv_gpuWork,
        csv_testNCRest,
        csv_color_res,
        csv_color_CSDP,
    ]
)

csv_detailed_fields = [
    csv_simpleStructures,
    csv_getNearCleaners,
    csv_testNCShortestPaths,
    csv_gpuCopyR,
    csv_gpuWork,
    csv_testNCRest,
    csv_color_res,
    csv_color_CSDP,
]

# csv_detailed_fields = [csv_BFS, csv_spanning_tree,
#                        csv_list_rank, csv_distance_parent, csv_find_bridges, csv_naive_bridges]

hatches = ["++", "**", "oo", "----", "||", ".."]
algo_colors = {
    "Cuda Perfect": "#377eb8",
    "Perfect": "#ff7f00",
    "CUDA Naive": "#ff0000",
    "Naive": "#00ff00",
    "CSDP Color": "#377eb8",
}
algo_markers = {
    "Cuda Perfect": ".",
    "Perfect": "x",
    "CUDA Naive": "o",
    "Naive": "+",
    "CSDP Color": ".",
}

algo_labels = {
    "Cuda Perfect": "GPU Perfect",
    "Perfect": "Perfect",
    "CUDA Naive": "GPU Naive",
    "Naive": "Naive",
    "CSDP Color": "CSDP Color",
}

algo_field_labels = {
    csv_simpleStructures: "Simple Structures",
    csv_testNCRest: "Test NC Rest",
    csv_getNearCleaners: "Get Near Cleaners",
    csv_testNCShortestPaths: "Test NC Shortest Paths",
    csv_gpuCopyR: "GPU Test NC Copy R",
    csv_gpuWork: "GPU test NC work",
    csv_color_res: "Rest",
    csv_color_CSDP: "CSDP",
}
field_colors = {
    "Perfect": {
        csv_simpleStructures: "#0f2231",
        csv_getNearCleaners: "#2c6593",
        csv_testNCShortestPaths: "#6ca5d3",
        csv_testNCRest: "#aaaaaa",
    },
    "Cuda Perfect": {
        csv_simpleStructures: "#1a1a1a",
        csv_getNearCleaners: "#4d4d4d",
        csv_testNCShortestPaths: "#808080",
        csv_gpuCopyR: "#b3b3b3",
        csv_gpuWork: "#d2d2d2",
    },
    "CUDA Naive": {csv_cpuPreparation: "#ff0000", csv_gpuCalculation: "#aa0000"},
    "CSDP Color": {csv_color_res: "#0f2231", csv_color_CSDP: "#6ca5d3"},
}

sizes_in_inches = {
    "regular": (4.77, 3),
    "wide": (6.8, 2.4),
    "huge": (6.8, 8),
    "allGraphs": (10, 25),
    "wideDetailed": (4.77, 3.5),
    "regular_small": (2.38, 1.5),
    "wideDetailed_small": (2.38, 1.5),
}
# sizes_in_inches = {"regular": (5, 3), "wide": (8, 3)}

func_formatter = ticker.FuncFormatter(lambda y, _: "{:g}".format(y))

# labels for plot axes
label_lca_N = "$|V|$"
label_time_percentage = "Time ratio"
label_lca_preprocessing_over_s = "Nodes preprocessed per second"
label_lca_queries_over_s = "Queries answered per second"
label_lca_queries_over_N = "Queries-to-nodes ratio"
label_lca_avg_depth = "Average node depth"
label_lca_batch_size = "Batch size"
label_time_overall_ms = "Total time [ms]"
label_time_overall_s = "Total time [s]"

# labels for algorithms


# Here we declare what tests consist an experiment.
experiments = [
    {
        "restrictions": [(csv_result, 1), (csv_algo, "Perfect|Naive")],
        "x_param": csv_N,
        "x_show": csv_N,
        "type": "lines",
        "size": sizes_in_inches["regular"],
        "y_param": csv_overall,
        "ylog": False,
        "x_label": label_lca_N,
        "y_label": label_time_overall_s,
    },
    {
        "restrictions": [
            (csv_result, 1),
            (csv_algo, "Perfect|Naive"),
            (csv_filename, "grid"),
            (csv_N, "42|48|54|60|66|72|78|84|90|96|102"),
        ],
        "x_param": csv_N,
        "x_show": csv_N,
        "type": "lines",
        "size": sizes_in_inches["regular"],
        "y_param": csv_overall,
        "ylog": True,
        "yformatter": func_formatter,
        "x_label": label_lca_N,
        "y_label": label_time_overall_s,
        "x_ticks": [42, 54, 66, 78, 90, 102],
        "x_ticks_labels": [
            # "$6 \\times 5$",
            "$6 \\times 7$",
            "$6 \\times 9$",
            "$6 \\times 11$",
            "$6 \\times 13$",
            "$6 \\times 15$",
            "$6 \\times 17$",
        ],
        "ylim": (0.5, 250),
    },
    {
        "restrictions": [
            (csv_result, 1),
            (csv_algo, "Perfect|Naive"),
            (csv_filename, "rook"),
            # (csv_N, "42|48|54|60|66|72|78|84|90|96|102"),
        ],
        "x_param": csv_N,
        "x_show": csv_N,
        "type": "lines",
        "size": sizes_in_inches["regular"],
        "y_param": csv_overall,
        "ylog": True,
        "yformatter": func_formatter,
        "x_label": label_lca_N,
        "y_label": label_time_overall_s,
        "x_ticks": [20, 25, 30, 35, 40, 45, 50, 55, 60],
        "x_ticks_labels": [
            "$5 \\times 4$",
            "$5 \\times 5$",
            "$5 \\times 6$",
            "$5 \\times 7$",
            "$5 \\times 8$",
            "$5 \\times 9$",
            "$5 \\times 10$",
            "$5 \\times 11$",
            "$5 \\times 12$",
        ],
        "ylim": (0.5, 200),
    },
    {
        "restrictions": [
            (csv_result, 1),
            (csv_algo, "Perfect|Naive"),
            (csv_filename, "knight"),
            # (csv_N, "42|48|54|60|66|72|78|84|90|96|102"),
        ],
        "x_param": csv_N,
        "x_show": csv_N,
        "type": "lines",
        "size": sizes_in_inches["regular"],
        "y_param": csv_overall,
        "ylog": True,
        "yformatter": func_formatter,
        "x_label": label_lca_N,
        "y_label": label_time_overall_s,
        "x_ticks": [32, 40, 48, 56, 64],
        "x_ticks_labels": [
            "$8 \\times 4$",
            "$8 \\times 5$",
            "$8 \\times 6$",
            "$8 \\times 7$",
            "$8 \\times 8$",
        ],
        "ylim": (0.5, 200),
    },
    {
        "restrictions": [
            (csv_result, 1),
            (csv_algo, "Perfect|Naive"),
            (csv_filename, "hypercube"),
            # (csv_N, "42|48|54|60|66|72|78|84|90|96|102"),
        ],
        "x_param": csv_N,
        "x_show": csv_N,
        "type": "lines",
        "size": sizes_in_inches["regular"],
        "y_param": csv_overall,
        "ylog": True,
        "yformatter": func_formatter,
        "x_label": label_lca_N,
        "y_label": label_time_overall_s,
        "x_ticks": [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75],
        # "x_ticks_labels": [
        #     "$8 \\times 4$",
        #     "$8 \\times 5$",
        #     "$8 \\times 6$",
        #     "$8 \\times 7$",
        #     "$8 \\times 8$",
        # ],
        "ylim": (0.5, 200),
    },
    {
        "restrictions": [
            (csv_result, 1),
            (csv_algo, "Perfect|Naive"),
            (csv_filename, "split"),
            # (csv_N, "42|48|54|60|66|72|78|84|90|96|102"),
        ],
        "x_param": csv_N,
        "x_show": csv_N,
        "type": "lines",
        "size": sizes_in_inches["regular"],
        "y_param": csv_overall,
        "ylog": True,
        "yformatter": func_formatter,
        "x_label": label_lca_N,
        "y_label": label_time_overall_s,
        "x_ticks": [20, 26, 32, 38, 44, 50],
        # "x_ticks_labels": [
        #     "$8 \\times 4$",
        #     "$8 \\times 5$",
        #     "$8 \\times 6$",
        #     "$8 \\times 7$",
        #     "$8 \\times 8$",
        # ],
        # "ylim": (0.5, 200),
    },
    {
        "restrictions": [
            (csv_result, 1),
            (csv_filename, "perf.t.in"),
            # (csv_N, "20|25|30|35|40|45|50|55"),
            (csv_algo, "GPU Perfect|Perfect"),
        ],
        "x_param": csv_N,
        "x_show": csv_N,
        "type": "detailed",
        "size": sizes_in_inches["wideDetailed"],
        "x_label": "N",
        "y_label": label_time_percentage,
        "ylim": (0, 1),
    },
    {
        "restrictions": [
            (csv_result, 1),
            (csv_filename, "perf2.t.in"),
            (csv_N, "15|16|17|18|19"),
            (csv_algo, "GPU Perfect|Perfect"),
        ],
        "x_param": csv_N,
        "x_show": csv_N,
        "type": "detailed",
        "size": sizes_in_inches["wideDetailed"],
        "x_label": "N",
        "y_label": label_time_percentage,
        "ylim": (0, 1),
    },
    {
        "restrictions": [
            (csv_result, 1),
            (csv_filename, "fullBinary"),
            (csv_N, "20|25|30|35|40|45|50|55"),
            (csv_algo, "GPU Perfect|Perfect"),
        ],
        "x_param": csv_N,
        "x_show": csv_N,
        "type": "detailed",
        "size": sizes_in_inches["wideDetailed"],
        "x_label": "N",
        "y_label": label_time_percentage,
        "ylim": (0, 1),
    },
    {
        "restrictions": [
            (csv_result, 1),
            (csv_filename, "grid"),
            (csv_algo, "GPU Perfect|Perfect"),
            (csv_N, "42|48|54|60|66|72|78"),
        ],
        "x_param": csv_N,
        "x_show": csv_N,
        "type": "detailed",
        "size": sizes_in_inches["wideDetailed"],
        "x_label": "N",
        "y_label": label_time_percentage,
        "ylim": (0, 1),
        "x_ticks": [42, 48, 54, 60, 66, 72, 78],
        "x_ticks_labels": [
            # "$6 \\times 5$",
            "$6 \\times 7$",
            "$6 \\times 8$",
            "$6 \\times 9$",
            "$6 \\times 10$",
            "$6 \\times 11$",
            "$6 \\times 12$",
            "$6 \\times 13$",
            "$6 \\times 14$",
        ],
    },
    {
        "restrictions": [
            (csv_result, 1),
            (csv_filename, "hypercube"),
            (csv_N, "16|20|24|28|32|36|40|44|48|50"),
            (csv_algo, "GPU Perfect|Perfect"),
        ],
        "x_param": csv_N,
        "x_show": csv_N,
        "type": "detailed",
        "size": sizes_in_inches["wideDetailed"],
        "x_label": "N",
        "y_label": label_time_percentage,
        "ylim": (0, 1),
    },
    {
        "restrictions": [
            (csv_result, 1),
            (csv_filename, "knight"),
            (csv_algo, "GPU Perfect|Perfect"),
        ],
        "x_param": csv_N,
        "x_show": csv_N,
        "type": "detailed",
        "size": sizes_in_inches["wideDetailed"],
        "x_label": "N",
        "y_label": label_time_percentage,
        "ylim": (0, 1),
    },
    {
        "restrictions": [
            (csv_result, 1),
            (csv_filename, "rook"),
            (csv_algo, "GPU Perfect|Perfect"),
        ],
        "x_param": csv_N,
        "x_show": csv_N,
        "type": "detailed",
        "size": sizes_in_inches["wideDetailed"],
        "x_label": "N",
        "y_label": label_time_percentage,
        "ylim": (0, 1),
    },
    {
        "restrictions": [
            (csv_result, 1),
            (csv_filename, "split"),
            (csv_algo, "GPU Perfect|Perfect"),
        ],
        "x_param": csv_N,
        "x_show": csv_N,
        "type": "detailed",
        "size": sizes_in_inches["wideDetailed"],
        "x_label": "N",
        "y_label": label_time_percentage,
        "ylim": (0, 1),
    },
    #
    # COLOR
    #
    {
        "restrictions": [(csv_algo, "CSDP")],
        "x_param": csv_N,
        "x_show": csv_N,
        "type": "lines",
        "size": sizes_in_inches["regular_small"],
        "y_param": csv_overall,
        "ylog": False,
        "x_label": "N",
        "y_label": label_time_overall_s,
        "out_prefix": "color",
    },
    {
        "restrictions": [(csv_algo, "CSDP")],
        "x_param": csv_N,
        "x_show": csv_N,
        "type": "detailed",
        "size": sizes_in_inches["wideDetailed_small"],
        "x_label": "N",
        "y_label": label_time_percentage,
        "ylim": (0, 1),
        "out_prefix": "color",
    },
]

if len(sys.argv) < 2:
    print(
        "Usage:\n\
        \t./python2 plot.py csv_to_parse.csv [dir_to_save_plots]\n"
    )
    exit()

# Read input and do preprocessing
csv_input = []
with open(sys.argv[1], "r") as input:
    for line in input:
        if not line or line.isspace():
            continue
        if line.split(",")[0] == "algorithm":
            csv_input.append("")
        csv_input[-1] += line.replace("\t", "")

csv_file = []
for input in csv_input:
    csv_reader = csv.DictReader(input.splitlines())
    for line in csv_reader:
        if line[csv_algo] == "file":  # headers outside first line
            continue
        for param in csv_ints:
            if param in line and line[param]:
                line[param] = int(float(line[param]))
        for param in csv_floats:
            if param in line and line[param]:
                line[param] = float(line[param])
        csv_file.append(line)

for i_exp, experiment in enumerate(experiments):

    fig, ax = plt.subplots()
    xSize, ySize = experiment["size"]
    fig.set_size_inches(xSize, ySize, forward=True)
    font_size = 8
    plt.rcParams.update({"font.size": font_size})

    current_rows = []
    # Filter data we should plot in current image
    for row in csv_file:
        isOk = True
        for res_name, res_val in experiment[
            "restrictions"
        ]:  # process restrictions to plot only what we want
            if isinstance(
                res_val, str
            ):  # for a string, restriction should be a substring of a field in a row
                if res_name == csv_filename:
                    if not re.findall(res_val, ntpath.basename(sys.argv[1])):
                        isOk = False
                        break
                else:
                    if not re.findall(res_val, str(row[res_name])):
                        # if not res_val == row[res_name]:
                        isOk = False
                        break
            else:  # for a non-string restriction should equal a field in a row
                if not res_val == row[res_name]:
                    isOk = False
                    break
        if isOk:
            current_rows.append(row)

    if not current_rows:
        continue

    # Scan for algos to plot
    algos = set()
    for row in current_rows:
        if row[csv_algo] not in algos:
            algos.add(row[csv_algo])

    for i_algo, algo in reversed(list(enumerate(algos))):
        # Filter data of a given algorithm
        algo_rows = list(
            filter(lambda row: True if row[csv_algo] == algo else False, current_rows)
        )

        plot_by_values = set()

        # Get all valid x-values
        for row in algo_rows:
            if experiment["x_param"] in row:
                val = row[experiment["x_param"]]
                if val not in plot_by_values:
                    plot_by_values.add(val)

        x = sorted(plot_by_values)

        if experiment["type"] == "detailed":
            y = [[] for _ in range(len(x))]
        else:
            y = [""] * len(x)

        # Prepare data values for current plot
        for row in algo_rows:
            for i_xval, xval in enumerate(x):
                if row[experiment["x_param"]] == xval:
                    if experiment["type"] == "detailed":
                        for field in csv_detailed_fields:
                            if field in row:
                                y[i_xval].append(row[field])
                            else:
                                y[i_xval].append(
                                    0
                                )  # if our field is not in a current algo, use value of 0 which we will skip in plotting
                    else:
                        y[i_xval] = row[experiment["y_param"]]
                    x[i_xval] = row[experiment["x_show"]]

        print(y)

        # Plot data
        if experiment["type"] == "detailed":
            # ax.set_ylabel(experiment["x_show"])

            height = 0.35 / len(x)
            apparent_x = np.arange(0, 1, 1.0 / len(x))
            y_sum = [sum(i) for i in y]
            old_y_sum = y_sum.copy()
            bar_shift = (i_algo + 0.5 - (len(algos) / 2.0)) * height
            ax.invert_yaxis()
            for i_field, field in reversed(list(enumerate(csv_detailed_fields))):
                if y[0][i_field] == 0:
                    alg_label = None
                else:
                    alg_label = algo_labels[algo] + " " + algo_field_labels[field]
                    # ax.barh(apparent_x + bar_shift, y_sum, height, label=alg_label,
                    # color='white', edgecolor=algo_colors[algo], hatch=hatches[i_field], zorder=3)
                    ax.bar(
                        apparent_x + bar_shift,
                        [y_sum[i] / old_y_sum[i] for i in range(len(y_sum))],
                        height,
                        label=alg_label,
                        color=field_colors[algo][field],
                        zorder=2,
                    )

                for i_xval, xval in enumerate(x):
                    y_sum[i_xval] -= y[i_xval][i_field]

            ax.set_xticks(apparent_x)
            # ax.set_xticklabels(x)
            # if "x_ticks" in experiment:
                # ax.set_xticks(experiment["x_ticks"])
            if "x_ticks_labels" in experiment:
                ax.set_xticklabels(experiment["x_ticks_labels"])
        else:
            ax.plot(
                x,
                y,
                label=algo_labels[algo],
                color=algo_colors[algo],
                marker=algo_markers[algo],
            )

            ax.set_xticks(x)

            if "x_ticks" in experiment:
                ax.set_xticks(experiment["x_ticks"])
            if "x_ticks_labels" in experiment:
                ax.set_xticklabels(experiment["x_ticks_labels"])

        if "x_label" in experiment:
            ax.set_xlabel(experiment["x_label"])
        if "y_label" in experiment:
            ax.set_ylabel(experiment["y_label"])

    # Format plot
    if experiment["type"] == "bars":
        plt.xticks(rotation=30, ha="right")
        # fig.subplots_adjust(bottom=0.2)
        fig.tight_layout()
        ax.set_yscale("log")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], ncol=3)
    elif experiment["type"] == "detailed":
        fig.tight_layout(rect=[0, 0.1, 1, 1])

        handles, labels = ax.get_legend_handles_labels()
        legend = ax.legend(handles, labels)

        # rect = patches.Rectangle((-.2,-.3),1.15,.15,linewidth=1,edgecolor='black',facecolor='white', clip_on=False)
        # shadow = patches.Rectangle((-.19,-.31),1.15,.15,linewidth=1,edgecolor='grey',facecolor='grey', clip_on=False)
        # ax.add_patch(shadow)
        # ax.add_patch(rect)

        # handles_and_labels = zip(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1])
        # hl_gpu = [hl for hl in handles_and_labels if "GPU" in hl[1]]
        # handles_gpu, labels_gpu = zip(*hl_gpu)

        # hl_cpu = [hl for hl in handles_and_labels if "GPU" not in hl[1]]
        # handles_cpu, labels_cpu = zip(*hl_cpu)

        # r = mpl.patches.Rectangle((0,0), 1, 1.5, fill=False, edgecolor='none', visible=False)

        # labels_gpu = [l[7::] for l in labels_gpu] + ["GPU Perfect\hspace{13.15pt}"]
        # handles_gpu = handles_ck + (r,)

        # labels_tv = [l[7::] for l in labels_tv] + ["GPU TV\hspace{13.3pt}"]

        # handles_tv = handles_tv + (r,)

        # labels_hy = [l[11::] for l in labels_hy] + ["GPU Hybrid"]
        # handles_hy = handles_hy + (r,)

        # legend = ax.legend(handles_ck[::-1], labels_ck[::-1], loc='lower left', bbox_to_anchor=(-.27,-0.104),
        #                    fancybox=True, shadow=False, ncol=3, frameon=False)
        # legend1 = ax.legend(handles_tv[::-1], labels_tv[::-1], loc='lower left', bbox_to_anchor=(-.27,-0.129),
        #                    fancybox=True, shadow=False, ncol=4, frameon=False)
        # legend2 = ax.legend(handles_hy[::-1], labels_hy[::-1], loc='lower left', bbox_to_anchor=(-.27, -.154),
        #                    fancybox=True, shadow=False, ncol=5, frameon=False)
        # plt.gca().add_artist(legend)
        # plt.gca().add_artist(legend1)

    # elif experiment["size"] == sizes_in_inches["allGraphs"]:
    #     fig.tight_layout(rect=[0, 0.05, 1, 1])
    #     handles, labels = ax.get_legend_handles_labels()
    #     legend = ax.legend(handles[::-1], labels[::-1], loc='center', bbox_to_anchor=(0.4, -0.05),
    #                        fancybox=True, shadow=True, ncol=3)
    # elif experiment["size"] == sizes_in_inches["wideDetailed"]:
    #     fig.tight_layout(rect=[0, 0.13, 1, 1])

    #     rect = patches.Rectangle((0,5.75),750,1,linewidth=1,edgecolor='black',facecolor='white', clip_on=False)
    #     shadow = patches.Rectangle((5,5.8),750,1,linewidth=1,edgecolor='grey',facecolor='grey', clip_on=False)
    #     ax.add_patch(shadow)
    #     ax.add_patch(rect)

    #     handles_and_labels = zip(ax.get_legend_handles_labels()[0], ax.get_legend_handles_labels()[1])

    #     # hl_ck = [hl for hl in handles_and_labels if "CK" in hl[1]]
    #     # handles_ck, labels_ck = zip(*hl_ck)

    #     # hl_tv = [hl for hl in handles_and_labels if "TV" in hl[1]]
    #     # handles_tv, labels_tv = zip(*hl_tv)

    #     # r = mpl.patches.Rectangle((0,0), 1, 1.5, fill=False, edgecolor='none', visible=False)

    #     # labels_ck = [l[7::] for l in labels_ck] + ["GPU CK"]
    #     # handles_ck = handles_ck + (r,)

    #     # labels_tv = [l[7::] for l in labels_tv] + ["GPU TV"]
    #     # handles_tv = handles_tv + (r,)

    #     # legend = ax.legend(handles_ck[::-1], labels_ck[::-1], loc='lower left', bbox_to_anchor=(-.05,-0.35),
    #     #                    fancybox=True, shadow=False, ncol=3, frameon=False)
    #     # legend1 = ax.legend(handles_tv[::-1], labels_tv[::-1], loc='lower left', bbox_to_anchor=(-.05,-0.43),
    #     #                    fancybox=True, shadow=False, ncol=4, frameon=False)
    #     # plt.gca().add_artist(legend)
    # else:
    #     fig.tight_layout(rect=[0, 0.2, 1, 1])
    #     handles, labels = ax.get_legend_handles_labels()
    #     legend = ax.legend(handles[::-1], labels[::-1], loc='center', bbox_to_anchor=(0.4, -0.45),
    #                        fancybox=True, shadow=True, ncol=2)
    else:  # lines
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        if "xlog" in experiment and experiment["xlog"]:
            ax.set_xscale("log")
        if "ylog" in experiment and experiment["ylog"]:
            ax.set_yscale("log")
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc="best")
        fig.tight_layout()

    if "yformatter" in experiment:
        ax.yaxis.set_major_formatter(experiment["yformatter"])
    if "ylim" in experiment:
        ax.set_ylim(experiment["ylim"])

    if experiment["type"] != "detailed":
        ax.yaxis.grid(b=True, which="major", zorder=0)
    else:
        ax.xaxis.grid(b=True, which="major", zorder=0)
    # ax.grid(True)

    # Saving path
    if len(sys.argv) <= 2:
        dir_to_save_to = os.path.join("plots")
    else:
        dir_to_save_to = sys.argv[2]

    if not os.path.exists(dir_to_save_to):
        os.makedirs(dir_to_save_to)

    # Saving filename
    filename_to_save = ""
    if "out_prefix" in experiment:
        filename_to_save += experiment["out_prefix"] + "."
    filename_to_save += ntpath.basename(sys.argv[1]) + "."
    # for res_name, res_val in experiment["restrictions"]:
    #     filename_to_save += "_" + res_name + "-" + str(res_val)

    # filename_to_save += "x=" + experiment["x_param"]
    # if "y_param" in experiment:
    #     filename_to_save += "y=" + experiment["y_param"]
    # if "y_divide" in experiment:
    #     filename_to_save += "div" + experiment["y_divide"]

    filename_to_save += experiment["type"]
    filename_to_save = filename_to_save.replace("# ", "num")

    # Save
    plt.gcf().set_size_inches(w=xSize, h=ySize)

    if usePgf:
        if experiment["type"] == "detailed":
            fig.savefig(
                os.path.join(dir_to_save_to, filename_to_save + ".pgf"),
                bbox_extra_artists=(legend),
            )
        else:
            fig.savefig(os.path.join(dir_to_save_to, filename_to_save + ".pgf"))
    else:
        # fig.set_size_inches(10,5, forward=True)
        if experiment["type"] == "detailed":
            fig.savefig(
                os.path.join(dir_to_save_to, filename_to_save + ".png"),
                bbox_extra_artists=(legend),
            )
        else:
            fig.savefig(os.path.join(dir_to_save_to, filename_to_save + ".png"))
