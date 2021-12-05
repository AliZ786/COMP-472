import task1
import matplotlib.pyplot as plt

#First model: glove-wiki-gigaword-300 -- C1-E1
#Second model: fasttext-wiki-news-subwords-300 -- C2-E2
#Third model: glove-twitter-200 -- C3-E3 
#Fourth model: glove-twitter-25 -- C4 -E4

def modelComparison():
    #Evaluate each of the four models
    glovewiki300perf = task1.modelEvaluation(model_name = "glove-wiki-gigaword-300", filemode = "a")
    fastwiki300perf = task1.modelEvaluation(model_name = "fasttext-wiki-news-subwords-300", filemode = "a")
    glovetwitter200perf = task1.modelEvaluation(model_name = "glove-twitter-200", filemode = "a")
    glovetwitter25perf = task1.modelEvaluation(model_name = "glove-twitter-25", filemode = "a")

    #Missing -- random baseline and crowdsourcing (waiting on that)

    #fig = plt.figure()

    x = ["glove-wiki-gigaword-300", "fasttext-wiki-news-subwords-300", "glove-twitter-200", "glove-twitter-25"]
    y = [glovewiki300perf, fastwiki300perf, glovetwitter200perf, glovetwitter25perf]

    #Create a bar chart showing the differences between the models
    plt.bar(x, y, color = ['yellow', 'red', 'green', 'blue'], width = 0.8)
    plt.xlabel("Model Name")
    plt.ylabel("Performance")
    plt.title("Model Performance")

    plt.tick_params(axis='x', which='major', labelsize=5)

    for index, data in enumerate(y):
        plt.text(x=index, y=data+1, s=f"{data}",
             fontdict=dict(fontsize=12, color='maroon'))

    #Saves the figure to a PDF file
    plt.savefig("model-performance.pdf")

if __name__ == '__main__':
    modelComparison()
