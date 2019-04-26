#!/usr/bin/env python3
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from scipy.stats import describe


def main(fitness_dict=""):
    with open(fitness_dict, "r") as f:
        data_dict = json.load(f)
    fitness = data_dict.get("Fitness")
    df = pd.DataFrame(data=data_dict)

    # pixel char...
    df = df.replace("PixelcharGen", "PixelChar")
    # keys: Model Temperature Fitness Perplexity Length Parseability Executability
    #describe(df[df.Model == "TokenGen"]["Fitness"])

    models = df.Model.unique()

    # ax = sns.violinplot(x="Model", y="Fitness", data=df, split=True)
    SHOW_OVERALL_FITNESS = True
    if SHOW_OVERALL_FITNESS:
        ax = sns.boxplot(
            x="Model", y="Fitness", data=df, order=["GitHub", "CharGen", "TokenGen", "PixelChar"])
        ax.set_yscale("log")
        plt.title("Distribution of Source Code Fitness")
        plt.xlabel("Source Code Origin")
        plt.savefig("fitness_distribution.pdf")
        plt.clf()

    SHOW_CHARGEN_TEMPERATURE = True
    if SHOW_CHARGEN_TEMPERATURE:
        ax = sns.boxplot(x="Temperature", y="Fitness",
                         data=df[df.Model == "CharGen"])
        # ax = sns.violinplot(x="Temperature", y="Fitness", data=df[df.Model == "CharGen"])
        ax.set_yscale("log")
        plt.title("Recurrent CharGen Fitness by Temperature")
        plt.ylim(-0.05, None)
        plt.xlabel("Temperature")
        #plt.show()
        plt.savefig("chargen_temp_fitness.pdf")
        plt.clf()

    SHOW_CHARGEN_PARSE_EXEC = True
    if SHOW_CHARGEN_PARSE_EXEC:
        ax = sns.countplot(x="Temperature", hue="Executability", data=df[df.Model == "CharGen"][df["Parseability"] == 1], palette=sns.color_palette("colorblind"))
        ax.legend_.texts[0]._text = "Parsed"
        ax.legend_.texts[1]._text = "Parsed & Executed"
        plt.title("Parseable and Executable CharGen Created Code Files")
        plt.xlabel("Temperature")
        plt.ylabel("Count")
        #plt.show()
        plt.savefig("chargen_temp_parse_exec.pdf")
        plt.clf()

    SHOW_TOKENGEN_TEMPERATURE = True
    if SHOW_TOKENGEN_TEMPERATURE:
        ax = sns.boxplot(x="Temperature", y="Fitness",
                         data=df[df.Model == "TokenGen"])
        # ax = sns.violinplot(x="Temperature", y="Fitness", data=df[df.Model == "TokenGen"])
        ax.set_yscale("log")
        plt.title("Recurrent TokenGen Fitness by Temperature")
        plt.ylim(-0.05, None)
        plt.xlabel("Temperature")
        # plt.show()
        plt.savefig("tokengen_temp_fitness.pdf")
        plt.clf()

    SHOW_TOKENGEN_PARSE_EXEC = True
    if SHOW_TOKENGEN_PARSE_EXEC:
        ax = sns.countplot(x="Temperature", hue="Executability", data=df[df.Model == "TokenGen"][df["Parseability"] == 1], palette=sns.color_palette("colorblind"))
        ax.legend_.texts[0]._text = "Parsed"
        ax.legend_.texts[1]._text = "Parsed & Executed"
        plt.title("Parseable and Executable TokenGen Created Code Files")
        plt.xlabel("Temperature")
        plt.ylabel("Count")
        #plt.show()
        plt.savefig("tokengen_temp_parse_exec.pdf")
        plt.clf()

    print()

    # looks awful
    # pal = sns.cubehelix_palette(10, rot=-25, light=.7)
    # g = sns.FacetGrid(df[df.Model == "CharGen"], row="Temperature", hue="Temperature", aspect=15, height=.5, palette=pal)
    # g.map(sns.kdeplot, "Fitness", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
    # g.map(sns.kdeplot, "Fitness", clip_on=False, color="w", lw=2, bw=.2)
    # g.map(plt.axhline, y=0, lw=2, clip_on=False)
    # def label(x, color, label):
    #     ax = plt.gca()
    #     ax.text(0, .2, label, fontweight="bold", color=color, ha="left", va="center", transform=ax.transAxes)
    # g.map(label, "Fitness")
    # g.fig.subplots_adjust(hspace=-.25)
    # g.despine(bottom=True, left=True)
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.show()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--fitness-dict", help="name of fitness data dict", type=str, default="fitness.json")

    args = parser.parse_args()
    main(fitness_dict=args.fitness_dict)
