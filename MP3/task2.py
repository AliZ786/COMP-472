import random

import Task1
import matplotlib.pyplot as plt
import pandas as pd


# First model: glove-wiki-gigaword-300 -- C1-E1
# Second model: fasttext-wiki-news-subwords-300 -- C2-E2
# Third model: glove-twitter-200 -- C3-E3
# Fourth model: glove-twitter-25 -- C4 -E4

def modelComparison():
    # Obtains the success rate from the gold standard, useful for later calculations
    success_rate_array = pd.read_csv("Crowdsourced Gold-Standard for MP3.csv", usecols=["Sucess Rate"])

    goldstdlist = success_rate_array["Sucess Rate"].tolist()
    goldstdlistclean = [x for x in goldstdlist if x == x]
    proc_list = [x[:4] for x in goldstdlistclean]
    float_goldstd = [float(i) for i in proc_list]
    # creates a random baseline
    random_baseline = float_goldstd.copy()

    for i in range(len(random_baseline)):
        random_baseline[i] = round(random.uniform(0, 100), 1)

    # Evaluate each of the four models
    glovewiki300perf = Task1.modelEvaluation(model_name="glove-wiki-gigaword-300", filemode="a")
    fastwiki300perf = Task1.modelEvaluation(model_name="fasttext-wiki-news-subwords-300", filemode="a")
    glovetwitter200perf = Task1.modelEvaluation(model_name="glove-twitter-200", filemode="a")
    glovetwitter25perf = Task1.modelEvaluation(model_name="glove-twitter-25", filemode="a")
    goldstandardperf = round((sum(float_goldstd) / len(float_goldstd)),
                             2)  # Calculates the average from the Crowd Sourced Gold Standard
    randomperf = round((sum(random_baseline) / len(random_baseline)), 2)
    x = ["glove-wiki-gigaword-300", "fasttext-wiki-news-subwords-300", "glove-twitter-200", "glove-twitter-25",
         "Crowd Sourced Gold Standard", "Random Baseline"]
    y = [glovewiki300perf, fastwiki300perf, glovetwitter200perf, glovetwitter25perf, goldstandardperf, randomperf]

    # Create a bar chart showing the differences between the models
    plt.bar(x, y, color=['yellow', 'red', 'green', 'blue', 'black', 'purple'], width=0.8)
    plt.xlabel("Model Name")
    plt.ylabel("Performance")
    plt.title("Model Performance")

    plt.tick_params(axis='x', which='major', labelsize=3.5)

    for index, data in enumerate(y):
        plt.text(x=index - 0.3, y=data + 1, s=f"{data}",
                 fontdict=dict(fontsize=12, color='maroon'))

    # Saves the figure to a PDF file
    plt.savefig("model-performance.pdf")


if __name__ == '__main__':
    modelComparison()
