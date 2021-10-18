import matplotlib.pyplot as plt
import os
from sklearn import metrics
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from tabulate import tabulate
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score


# Task 1.2

business = 'MP1/BBC/business'
entertainment = 'MP1/BBC/entertainment'
politics = 'MP1/BBC/politics'
sport = 'MP1/BBC/sport'
tech = 'MP1/BBC/tech'

totalBusinessFiles = 0
totalEntertainmentFiles = 0
totalPoliticsFiles = 0
totalSportFiles = 0
totalTechFiles = 0

for base, dirs, files in os.walk(business):
    for Files in files:
        totalBusinessFiles += 1

for base, dirs, files in os.walk(entertainment):
    for Files in files:
        totalEntertainmentFiles += 1

for base, dirs, files in os.walk(politics):
    for Files in files:
        totalPoliticsFiles += 1

for base, dirs, files in os.walk(sport):
    for Files in files:
        totalSportFiles += 1

for base, dirs, files in os.walk(tech):
    for Files in files:
        totalTechFiles += 1

x = range(5)
x_labels = ["Business", "Entertainment", "Politics", "Sports", "Tech"]
y = [totalBusinessFiles, totalEntertainmentFiles,
     totalPoliticsFiles, totalSportFiles, totalTechFiles]


plt.bar(x, y, color=['black', 'red', 'green',
        'blue', 'yellow'], align='center')
plt.title('BBC Distribution')
plt.ylabel("Total number of instances per class", fontsize=10)
plt.xticks(x, x_labels)


for index, data in enumerate(y):
    plt.text(x=index, y=data+1, s=f"{data}",
             fontdict=dict(fontsize=12, color='maroon'))

plt.savefig('BBC-distribution.pdf')

 # Task 1.3

bbc_data = load_files('MP1/BBC', load_content=True, encoding='latin1')


# Task 1.4

vectorizer = CountVectorizer()
vectorizer.fit(bbc_data.data)

vectorizer.get_feature_names_out()

# X_train
bbc_data_transformed = vectorizer.transform(bbc_data.data)

bbc_data_transformed.toarray()

vocabulary = pd.DataFrame(bbc_data_transformed.toarray(),
                          columns=vectorizer.get_feature_names_out())

# print(pd.DataFrame(bbc_data_transformed.toarray(), columns=vectorizer.get_feature_names_out()))

# print("\nSparse matrix\n" , bbc_data_transformed)

# print ("\nDense matrix\n", bbc_data_transformed.toarray())


# Task 1.5

# y_train
BBC_Y = bbc_data.target

X_train, X_test, y_train, y_test = train_test_split(
    bbc_data_transformed, BBC_Y,  test_size=0.20, train_size=0.8, random_state=None)

f = open("bbc-performance.txt", "w")

# Task 1.6 and Task 1.8 (a repetition of Task 6 and 7)
for x in range(2):
   
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
   
    #Task 1.6 
    classifier_MNB = MultinomialNB()
    classifier_MNB.fit(X_train, y_train)
    y_pred = classifier_MNB.predict(X_test)

    def get_vector(target, data, data_range):
        arr = sum(data[target == data_range])
        return arr
   
    #Task 1.7
    # Task 1.7 a
    f.write("\n\n(a) ****** MultinomialNB default values, try " +
            str(x+1) + " ******\n\n")
    

    # Task 1.7 b
    f.write("\n(b) Confusion Matrix \n")
    cm = confusion_matrix(y_test, y_pred)
    confusion_matrix = pd.DataFrame(cm, index=x_labels)
    f.write(tabulate(confusion_matrix, x_labels,
            tablefmt="grid", stralign="center") + "\n")

    # Task 1.7 c
    class_report = classification_report(
        y_test, y_pred, target_names=bbc_data.target_names)
    f.write("\n(c) Precision, recall, and F1-measure for each class \n")
    # Index = [''], because we has to LHS values to put
    class_repo = pd.DataFrame({class_report}, index=[''])
    f.write(tabulate(class_repo, tablefmt="grid",
            stralign="right", numalign="right"))

    # Task 1.7 (d)
    f.write("\n\n(d) Accuracy, Macro-average F1 and Weighted-average F1 of the model \n")
    headers = ["Accuracy_score", "Macro-average F1", "Weighted-average F1"]
    accuracy = accuracy_score(y_test, y_pred)
    f1_macroavg = f1_score(y_test, y_pred, average='macro')
    f1_weightedavg = f1_score(y_test, y_pred, average='weighted')
    f1_scores = pd.DataFrame(
        {accuracy, f1_macroavg, f1_weightedavg}, index=headers)
    f.write(tabulate(f1_scores, tablefmt="grid"))

    classes = ["business", "entertainment", "politics", "sport", "tech"]
    prob_table = []
    f.write("\n\n(e) The prior probabilities of each class \n")
    class_report_prob = classification_report(
        y_test, y_pred, target_names=bbc_data.target_names, output_dict=True)
    # Want to compute the average for each class seperately by using the total value of which was 445
    for i in range(len(classes)):
        prob_total = class_report_prob['macro avg']['support']
        prob1 = (class_report_prob[str(classes[i])]['support']/prob_total)

        if (classes[i] == "sport"):
            classes[i] = str(classes[i]) + "s"
        else:
            classes[i] = str(classes[i])

        column1 = classes[i].capitalize()
        column2 = prob1
        columns = column1, column2
        prob_table.append(columns)
   
    f.write(tabulate(prob_table, headers=[
        "Class", "Probability"], tablefmt="grid"))

# Task 1.7 f
    words = " ".join(vocabulary).split()
    f.write("\n\n(f) The size in the vocabulary is: " + str(len(words)))

    # Task 1.7 g
    f.write("\n\n(g) The number of word tokens in each class is:\n")
    word_tokens = []
    table_headers = ["Class", "Word-Tokens"]
    for i in range(len(x_labels)):
        words_class = np.sum(get_vector(BBC_Y, bbc_data_transformed, i))
        words = (x_labels[i]) + ": " + (str(words_class)) + "\n"
        data1 = x_labels[i]
        data2 = (words_class)
        column = data1, data2
        word_tokens.append(column)

    f.write(tabulate(word_tokens, tablefmt="grid",
            headers=table_headers, stralign="right", numalign="right"))

    # Task 1.7 h
    f.write("\n\n(h) The number of word tokens in the whole corpus is: ")
    words_corpus = str(sum(map(np.sum, bbc_data_transformed)))
    f.write(words_corpus)

    # Task 1.7 i
    f.write("\n\n(i) Number and percentage of words with a frequency of zero in each class :\n ")

    freq_table = []
    table_headers = ["Class", "Words", "Percentage"]
    for i in range(len(x_labels)):
        total_words = get_vector(BBC_Y, bbc_data_transformed, i)
        nonzerofreq_words = np.count_nonzero(total_words.toarray())
        zerofreq_words = total_words.toarray().size - nonzerofreq_words
        percentage = "{:.2f}".format(
            (zerofreq_words/total_words.toarray().size) * 100)
        formatted_percentage = str(percentage) + " %"
        data_1 = x_labels[i]
        data2 = zerofreq_words
        data3 = formatted_percentage
        column = data_1, data2, data3
        freq_table.append(column)

    f.write(tabulate(freq_table, tablefmt="grid", headers=table_headers))

    # Task 1.7 j
    f.write("\n\n(j) Number and percentage of words with a frequency of one in entire corpus :\n ")
    words_corpus = sum(bbc_data_transformed).toarray()
    onefreq_words = np.count_nonzero(words_corpus ==1)
    percentage = "{:.2f}".format((onefreq_words/words_corpus.size) * 100)
    formatted_percentage = str(percentage) + " %"
    f.write("\n Number of words: " + str(onefreq_words) + "\n")
    f.write(" Percentage of words: " + formatted_percentage + "\n")

    # Task 1.7 k
    f.write("\n\n(k) 2 favourite words and their log prob:\n\n ")

    word1 = "potato"
    word2 = "zombies"

    vocabularylist = vectorizer.get_feature_names_out()
    fav1 = np.where(vocabularylist == word1)[0][0]
    fav2 = np.where(vocabularylist == word2)[0][0]

    logprob_table1 =[]
    logprob_table2 = []
    # First fav word
    for i in range(len(x_labels)):
        logprob = classifier_MNB.feature_log_prob_[i][fav1]
        data1 = x_labels[i]
        data2 = logprob
        columns = data1, data2
        logprob_table1.append(columns)



    f.write("Favourite word 1: " + word1 + "\n")
    table_headers = ["Class", "Log prob"]
    
    f.write(tabulate(logprob_table1, headers=table_headers, tablefmt="grid"))

    # Second fav word
    for i in range(len(x_labels)):
        logprob = classifier_MNB.feature_log_prob_[i][fav2]
        data1 = x_labels[i]
        data2 = logprob
        columns = data1, data2
        logprob_table2.append(columns)


    f.write("\n\n" + "Favourite word 2: " + word2 + "\n")
    table_headers = ["Class", "Log prob"]
    f.write(tabulate(logprob_table2, headers=table_headers, tablefmt="grid"))
    f.write(
        "\n\n***********************************************************************")

# Task 1.9
f.write("\n\n(a) ****** MultinomialNB smoothing = [0.0001]  " "******\n\n")

# Repeat of task 6
classifier_MNBs1 = MultinomialNB(alpha=0.0001)
classifier_MNBs1.fit(X_train, y_train)
ys1_pred = classifier_MNBs1.predict(X_test)

# Repeat of task 7
f.write("\n(b) Confusion Matrix \n")
cm = metrics.confusion_matrix(y_test, ys1_pred)
confusion_mat = pd.DataFrame(cm, index=x_labels)
f.write(tabulate(confusion_mat, x_labels,
        tablefmt="grid", stralign="center") + "\n")

# Task 1.7 c
class_report = classification_report(
    y_test, ys1_pred, target_names=bbc_data.target_names)
f.write("\n(c) Precision, recall, and F1-measure for each class \n")
# Index = [''], because we has to LHS values to put
class_repo = pd.DataFrame({class_report}, index=[''])
f.write(tabulate(class_repo, tablefmt="grid",
        stralign="right", numalign="right"))

# Task 1.7 d
f.write("\n\n(d) Accuracy, Macro-average F1 and Weighted-average F1 of the model \n")
headers = ["Accuracy_score", "Macro-average F1", "Weighted-average F1"]
acc_score = metrics.accuracy_score(y_test, ys1_pred)
f1_macroavg = f1_score(y_test, ys1_pred, average='macro')
f1_weightedavg = f1_score(y_test, ys1_pred, average='weighted')
f1_scores = pd.DataFrame(
    {acc_score, f1_macroavg, f1_weightedavg}, index=headers)
f.write(tabulate(f1_scores, tablefmt="grid"))

# Task 1.7 e

classes = ["business", "entertainment", "politics", "sport", "tech"]
prob_table = []
f.write("\n\n(e) The prior probabilities of each class \n")
class_report_prob = classification_report(
    y_test, ys1_pred, target_names=bbc_data.target_names, output_dict=True)
# Want to compute the average for each class seperately by using the total value of which was 445
for i in range(len(classes)):
    prob_total = class_report_prob['macro avg']['support']
    prob1 = (class_report_prob[str(classes[i])]['support']/prob_total)

    if (classes[i] == "sport"):
        classes[i] = str(classes[i]) + "s"
    else:
        classes[i] = str(classes[i])

    column1 = classes[i].capitalize()
    column2 = prob1
    columns = column1, column2
    prob_table.append(columns)



f.write(tabulate(prob_table, headers=[
        "Class", "Probability"], tablefmt="grid"))

# Task 1.7 f
words = " ".join(vocabulary).split()
f.write("\n\n(f) The size in the vocabulary is: " + str(len(words)))

# Task 1.7 g
f.write("\n\n(g) The number of word tokens in each class is:\n")
word_tokens = []
table_headers = ["Class", "Word-Tokens"]
for i in range(len(x_labels)):
    words_class = np.sum(get_vector(BBC_Y, bbc_data_transformed, i))
    words = (x_labels[i]) + ": " + (str(words_class)) + "\n"
    data1 = x_labels[i]
    data2 = (words_class)
    column = data1, data2
    word_tokens.append(column)

f.write(tabulate(word_tokens, tablefmt="grid",
        headers=table_headers, stralign="right", numalign="right"))

# Task 1.7 h
f.write("\n\n(h) The number of word tokens in the whole corpus is: ")
words_corpus = str(sum(map(np.sum, bbc_data_transformed)))
f.write(words_corpus)

# Task 1.7 i
f.write("\n\n(i) Number and percentage of words with a frequency of zero in each class :\n ")

freq_table = []
table_headers = ["Class", "Words", "Percentage"]
for i in range(len(x_labels)):
    total_words = get_vector(BBC_Y, bbc_data_transformed, i)
    nonzerofreq_words = np.count_nonzero(total_words.toarray())
    zerofreq_words = total_words.toarray().size - nonzerofreq_words
    percentage = "{:.2f}".format(
        (zerofreq_words/total_words.toarray().size) * 100)
    formatted_percentage = str(percentage) + " %"
    data_1 = x_labels[i]
    data2 = zerofreq_words
    data3 = formatted_percentage
    column = data_1, data2, data3
    freq_table.append(column)

f.write(tabulate(freq_table, tablefmt="grid", headers=table_headers))

# Task 1.7 j

f.write("\n\n(j) Number and percentage of words with a frequency of one in entire corpus :\n ")
words_corpus = sum(bbc_data_transformed).toarray()
onefreq_words = np.count_nonzero(words_corpus ==1)
percentage = "{:.2f}".format((onefreq_words/words_corpus.size) * 100)
formatted_percentage = str(percentage) + " %"
f.write("\n Number of words: " + str(onefreq_words) + "\n")
f.write(" Percentage of words: " + formatted_percentage + "\n")


# Task 1.7 k
f.write("\n\n(k) 2 favourite words and their log prob:\n\n ")

word1 = "potato"
word2 = "zombies"

vocabularylist = vectorizer.get_feature_names_out()
fav1 = np.where(vocabularylist == word1)[0][0]
fav2 = np.where(vocabularylist == word2)[0][0]

table1 = []
table2 = []

# First fav word
for i in range(len(x_labels)):
    logprob = classifier_MNBs1.feature_log_prob_[i][fav1]
    data1 = x_labels[i]
    data2 = logprob
    columns1 = data1, data2
    table1.append(columns1)


f.write("Favourite word 1: " + word1 + "\n")
table_headers = ["Class", "Log prob"]
f.write(tabulate(table1, headers=table_headers, tablefmt="grid"))

# Second fav word

for i in range(len(x_labels)):
    logprob = classifier_MNBs1.feature_log_prob_[i][fav2]
    data1 = x_labels[i]
    data2 = logprob
    columns2 = data1, data2
    table2.append(columns2)

f.write("\n\n" + "Favourite word 2: " + word2 + "\n")
table_headers = ["Class", "Log prob"]
f.write(tabulate(table2, headers=table_headers, tablefmt="grid"))

f.write("\n\n***********************************************************************") 

# Task 1.10
f.write("\n\n(a) ****** MultinomialNB smoothing = [0.9]  " "******\n\n")

# Repeat of task 6
classifier_MNBs2 = MultinomialNB(alpha=0.9)
classifier_MNBs2.fit(X_train, y_train)
ys2_pred = classifier_MNBs2.predict(X_test)

# Repeat of task 7
f.write("\n(b) Confusion Matrix \n")
cm = metrics.confusion_matrix(y_test, ys2_pred)
confusion_mat = pd.DataFrame(cm, index=x_labels)
f.write(tabulate(confusion_mat, x_labels,
        tablefmt="grid", stralign="center") + "\n")

# Task 1.7 c
class_report = classification_report(
    y_test, ys2_pred, target_names=bbc_data.target_names)
f.write("\n(c) Precision, recall, and F1-measure for each class \n")
# Index = [''], because we has to LHS values to put
class_repo = pd.DataFrame({class_report}, index=[''])
f.write(tabulate(class_repo, tablefmt="grid",
        stralign="right", numalign="right"))

# Task 1.7 d
f.write("\n\n(d) Accuracy, Macro-average F1 and Weighted-average F1 of the model \n")
headers = ["Accuracy_score", "Macro-average F1", "Weighted-average F1"]
acc_score = metrics.accuracy_score(y_test, ys2_pred)
f1_macroavg = f1_score(y_test, ys2_pred, average='macro')
f1_weightedavg = f1_score(y_test, ys2_pred, average='weighted')
f1_scores = pd.DataFrame(
    {acc_score, f1_macroavg, f1_weightedavg}, index=headers)
f.write(tabulate(f1_scores, tablefmt="grid"))

# Task 1.7 e
classes = ["business", "entertainment", "politics", "sport", "tech"]
prob_table = []
f.write("\n\n(e) The prior probabilities of each class \n")
class_report_prob = classification_report(
    y_test, ys2_pred, target_names=bbc_data.target_names, output_dict=True)
# Want to compute the average for each class seperately by using the total value of which was 445
for i in range(len(classes)):
    prob_total = class_report_prob['macro avg']['support']
    prob1 = (class_report_prob[str(classes[i])]['support']/prob_total)

    if (classes[i] == "sport"):
        classes[i] = str(classes[i]) + "s"
    else:
        classes[i] = str(classes[i])

    column1 = classes[i].capitalize()
    column2 = prob1
    columns = column1, column2
    prob_table.append(columns)



f.write(tabulate(prob_table, headers=[
        "Class", "Probability"], tablefmt="grid"))

# Task 1.7 f
words = " ".join(vocabulary).split()
f.write("\n\n(f) The size in the vocabulary is: " + str(len(words)))

# Task 1.7 g
f.write("\n\n(g) The number of word tokens in each class is:\n")
word_tokens = []
table_headers = ["Class", "Word-Tokens"]
for i in range(len(x_labels)):
    words_class = np.sum(get_vector(BBC_Y, bbc_data_transformed, i))
    words = (x_labels[i]) + ": " + (str(words_class)) + "\n"
    data1 = x_labels[i]
    data2 = (words_class)
    column = data1, data2
    word_tokens.append(column)

f.write(tabulate(word_tokens, tablefmt="grid",
        headers=table_headers, stralign="right", numalign="right"))

# Task 1.7 h
f.write("\n\n(h) The number of word tokens in the whole corpus is: ")
words_corpus = str(sum(map(np.sum, bbc_data_transformed)))
f.write(words_corpus)

# Task 1.7 i
f.write("\n\n(i) Number and percentage of words with a frequency of zero in each class :\n ")

freq_table = []
table_headers = ["Class", "Words", "Percentage"]
for i in range(len(x_labels)):
    total_words = get_vector(BBC_Y, bbc_data_transformed, i)
    nonzerofreq_words = np.count_nonzero(total_words.toarray())
    zerofreq_words = total_words.toarray().size - nonzerofreq_words
    percentage = "{:.2f}".format(
        (zerofreq_words/total_words.toarray().size) * 100)
    formatted_percentage = str(percentage) + " %"
    data_1 = x_labels[i]
    data2 = zerofreq_words
    data3 = formatted_percentage
    column = data_1, data2, data3
    freq_table.append(column)

f.write(tabulate(freq_table, tablefmt="grid", headers=table_headers))

# Task 1.7 j
f.write("\n\n(j) Number and percentage of words with a frequency of one in entire corpus :\n ")

f.write("\n\n(j) Number and percentage of words with a frequency of one in entire corpus :\n ")
words_corpus = sum(bbc_data_transformed).toarray()
onefreq_words = np.count_nonzero(words_corpus ==1)
percentage = "{:.2f}".format((onefreq_words/words_corpus.size) * 100)
formatted_percentage = str(percentage) + " %"
f.write("\n Number of words: " + str(onefreq_words) + "\n")
f.write(" Percentage of words: " + formatted_percentage + "\n")


# Task 1.7 k
f.write("\n\n(k) 2 favourite words and their log prob:\n\n ")

word1 = "potato"
word2 = "zombies"

vocabularylist = vectorizer.get_feature_names_out()
fav1 = np.where(vocabularylist == word1)[0][0]
fav2 = np.where(vocabularylist == word2)[0][0]

table1_s2  =[]
table2_s2 = []

# First fav word
for i in range(len(x_labels)):
    logprob = classifier_MNBs2.feature_log_prob_[i][fav1]
    data1 = x_labels[i]
    data2 = logprob
    columns_1 = data1, data2
    table1_s2.append(columns_1)


f.write("Favourite word 1: " + word1 + "\n")
table_headers = ["Class", "Log prob"]
f.write(tabulate(table1_s2, headers=table_headers, tablefmt="grid"))

# Second fav word
for i in range(len(x_labels)):
    logprob = classifier_MNBs2.feature_log_prob_[i][fav2]
    data1 = x_labels[i]
    data2 = logprob
    columns_2 = data1, data2
    table2_s2.append(columns_2)
f.write("\n\n" + "Favourite word 2: " + word2 + "\n")
table_headers = ["Class", "Log prob"]
f.write(tabulate(table2_s2, headers=table_headers, tablefmt="grid"))

f.close()
