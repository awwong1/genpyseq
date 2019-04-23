#!/usr/bin/env python3
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from argparse import ArgumentParser


def main(fitness_dict=""):
    with open(fitness_dict, "r") as f:
        data_dict = json.load(f)
    df = pd.DataFrame(data=data_dict)

    print(stats.describe(df["Fitness"]))

    ax = sns.distplot(df["Fitness"], bins=40, norm_hist=True, label="GitHub")

    plt.legend()
    plt.title("Distribution of Source Code Fitness")
    plt.ylabel("Density")
    plt.xlim(0, None)
    plt.show()
    plt.savefig("fitness_distribution.pdf")



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--fitness-dict", help="name of fitness data dict", type=str, default="fitness.json")

    args = parser.parse_args()
    main(fitness_dict=args.fitness_dict)
