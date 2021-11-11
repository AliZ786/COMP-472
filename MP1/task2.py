import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
import warnings
from tabulate import tabulate


warnings.filterwarnings('ignore')
# Task 2.2
drug_file = pd.read_csv('MP1/drug200.csv')

# Task 2.3
drug_array = np.array(drug_file["Drug"])

(unique_drug, frequency) = np.unique(drug_array, return_counts=True)

#Creates a bar chart showing the frequency of the drugs
plt.bar(unique_drug, frequency,color=['black', 'red', 'green',
        'blue', 'yellow'], width = 0.8)

plt.xlabel("Drug")
plt.ylabel("Frequency")
plt.title("Drug distribution")


for index, data in enumerate(frequency):
    plt.text(x=index, y=data+1, s=f"{data}",
             fontdict=dict(fontsize=12, color='maroon'))


#Saves the figure to a PDF file
# plt.savefig("drug-distribution.pdf")

# Task 2.4
drug_file["BP"] = pd.Categorical(drug_file["BP"], ['LOW', 'NORMAL', 'HIGH'], ordered=True)
drug_file["Cholesterol"] = pd.Categorical(drug_file["Cholesterol"], ['LOW','NORMAL', 'HIGH'], ordered=True)

numerical_data = pd.get_dummies(drug_file, columns=['Sex', 'BP', 'Cholesterol'])

#Task 2.5
X_data = numerical_data[['Age', 'Na_to_K', 'Sex_F', 'Sex_M', 'BP_LOW', 'BP_NORMAL', 'BP_HIGH','Cholesterol_LOW', 'Cholesterol_NORMAL', 'Cholesterol_HIGH']]

X_train, X_test, y_train, y_test = train_test_split(X_data, drug_array)

# Task 2.6
#a) Gaussian Naive Bayes Classifier with default parameters
gnb_classifier = GaussianNB()
gnb_classifier.fit(X_train, y_train)

#b) Decision Tree with default parameters
base_dt_classifier = DecisionTreeClassifier()
base_dt_classifier.fit(X_train, y_train)

#c) Better performing Decision Tree
hyper_params = {'criterion':['gini', 'entropy'], 'max_depth': [2,10,20], 'min_samples_split': [2, 20, 50]}

top_dt_classifier = GridSearchCV(DecisionTreeClassifier(), hyper_params)
top_dt_classifier.fit(X_train, y_train)

#d) Perceptron with default parameters
perceptron_classifier = Perceptron()
perceptron_classifier.fit(X_train, y_train)

#e) Multi-Layered Perceptron with the provided parameters -- RETURNS A WARNING 
base_ml_perceptron_classifier = MLPClassifier(hidden_layer_sizes=(100,1), activation='logistic', solver='sgd', max_iter=5000)
base_ml_perceptron_classifier.fit(X_train, y_train)

#f) Better performing Multi-Layered Perceptron -- RETURNS A WARNING
params = {'activation':['identity', 'logistic', 'tanh', 'relu'], 'solver': ['sgd', 'adam'], 'hidden_layer_sizes': [[10,10,10], [50,30]]} 

top_ml_perceptron_classifier = GridSearchCV(MLPClassifier(), params)
top_ml_perceptron_classifier.fit(X_train, y_train)

# Task 2.7
separator = '\n\n*****************************************\n\n'
class_arr = ['DrugA', 'DrugB', 'DrugC', 'DrugX', 'DrugY']

f = open('drugs-performance.txt', 'w')

def dostepseven(classifier_obj):
	cf_matrix = confusion_matrix(y_test, classifier_obj.predict(X_test))
	cf_mat = pd.DataFrame(cf_matrix, index=class_arr)
	f.write("\nConfusion Matrix\n")
	f.write(tabulate(cf_mat, headers=class_arr, tablefmt="grid"))
	# np.savetxt(f, cf_matrix, fmt='%.0f', delimiter='|')

	precision = precision_score(y_test, classifier_obj.predict(X_test), average=None, zero_division=0)
	f.write("\n\nPrecision Measures by Class\n")
	precision_arr =[]

	for i in range(len(class_arr)):
		c1 = class_arr[i]
		c2 = "{:.2f}".format(precision[i])
		columns = c1, c2
		precision_arr.append(columns)

		# f.write(class_arr[i] + ':' + format(precision[i], '.2f') + '\t\n')
	f.write(tabulate(precision_arr, headers=["Class", "Precision"], tablefmt="grid", numalign="right"))

	recall = recall_score(y_test, classifier_obj.predict(X_test), average=None, zero_division=0)
	f.write("\n\nRecall Measures by Class\n")
	recall_arr =[]

	for i in range(len(class_arr)):
		c1 = class_arr[i]
		c2 = "{:.2f}".format(recall[i])
		columns = c1, c2
		recall_arr.append(columns)
		# f.write(class_arr[i] + ':' + format(recall[i], '.2f') + '\t\n')

	f.write(tabulate(recall_arr, headers=["Class", "Recall"], tablefmt="grid", numalign="right"))

	f1 = f1_score(y_test, classifier_obj.predict(X_test), average=None, zero_division=0)
	f.write("\n\nF1 Measures by Class\n")
	f1_array = [] 

	for i in range(len(class_arr)):
		c1 = class_arr[i]
		c2 = "{:.2f}".format(f1[i])
		columns = c1, c2
		f1_array.append(columns)
		# f.write(class_arr[i] + ':' + format(f1[i], '.2f') + '\t\n')
	
	f.write(tabulate(f1_array, headers=["Class", "F1-Score"], tablefmt="grid", numalign="right"))
	
	indexes = ["Accuracy", "Macro-Average F1", "Weighted-Average F1"]
	f.write("\n\nAccuracy, Macro-Average F1, and the Weighted-Averaged F1 of the Model\n")
	f1_scores = []
	f1_values = []
	
	for i in range(len(indexes)):
	 accuracy = accuracy_score(y_test, classifier_obj.predict(X_test))
	 f1_macroavg = f1_score(y_test, classifier_obj.predict(X_test), average='macro')
	 f1_weightedavg = f1_score(y_test, classifier_obj.predict(X_test), average='weighted')
	 f1_values = [accuracy, f1_macroavg, f1_weightedavg]
	 c1 = indexes[i]
	 c2 = f1_values[i]
	 columns = c1, c2
	 f1_scores.append(columns)

	f.write(tabulate(f1_scores, tablefmt="grid"))
	
# NB
f.write("\na) Gaussian Naive Bayes Classifier\n")
dostepseven(gnb_classifier)
f.write(separator)

# Base-DT
f.write("\nb) Base Decision Tree Classifier\n")
dostepseven(base_dt_classifier)
f.write(separator)

# Top-DT
f.write("\nc) Top Decision Tree Classifier\n")
f.write("Hyper-parameters: " +str(hyper_params) +"\n")
f.write("It's best estimator are: " +str(top_dt_classifier.best_estimator_)+ "\n")
dostepseven(top_dt_classifier)
f.write(separator)

# PER
f.write("\nd) Perceptron Classifier\n")
dostepseven(perceptron_classifier)
f.write(separator)

# Base-MLP
f.write("\ne) Base Multi-Layered Perceptron Classifier\n")
f.write("The parameters used for this are: hidden_layer_sizes=(100,1), activation='logistic', solver='sgd', max_iter=5000\n")
dostepseven(base_ml_perceptron_classifier)
f.write(separator)

# Top-MLP
f.write("\nf) Top Multi-Layered Perceptron Classifier\n")
f.write("Hyper-parameters: " +str(params) +"\n")
f.write("It's best estimator are: " +str(top_ml_perceptron_classifier.best_estimator_) +"\n")
dostepseven(top_ml_perceptron_classifier)
f.write(separator)

features = ["Average accuracy", "Accuracy standard deviation","Average macro-average F1",
"Macro-average F1 standard deviation", "Average weighted-average F1", "Weighted-average F1 standard deviation" ]
values = []

# task 2.8
def dostepeight(classifier_obj):
	acc=[]
	mac_avg=[]
	wei_avg=[]

	for x in range(10):
		# print("Step 8, loop " +str(x+1))
		classifier_obj.fit(X_train, y_train)
		acc.append(accuracy_score(y_test, classifier_obj.predict(X_test)))
		mac_avg.append(f1_score(y_test, classifier_obj.predict(X_test), average='macro'))
		wei_avg.append(f1_score(y_test, classifier_obj.predict(X_test), average='weighted'))
		


	acc_avg = sum(acc) / len(acc)
	mac_avg_avg = sum(mac_avg) / len(mac_avg)
	wei_avg_avg = sum(wei_avg) / len(wei_avg)
	acc_stdev = statistics.stdev(acc)
	mac_avg_stdev = statistics.stdev(mac_avg)
	wei_avg_stdev = statistics.stdev(wei_avg)

	values = [acc_avg, acc_stdev, mac_avg_avg, mac_avg_stdev, wei_avg_avg, wei_avg_stdev] 
	final_values = []
	for i in range(len(features)):
		c1 = features[i]
		c2 = values[i]
		columns = c1,c2
		final_values.append(columns)

	f.write(tabulate(final_values, tablefmt="grid"))

f.write("\n\na) Gaussian Naive Bayes Averages & Standard Deviations\n")
dostepeight(gnb_classifier)

f.write("\n\nb) Base Decision Tree Averages & Standard Deviations\n")
dostepeight(base_dt_classifier)

f.write("\n\nc) Top Decision Tree Averages & Standard Deviations\n")
dostepeight(top_dt_classifier)

f.write("\n\nd) Perceptron Averages & Standard Deviations\n")
dostepeight(perceptron_classifier)

f.write("\n\ne) Base Multi-Layered Perceptron Averages & Standard Deviations\n")
dostepeight(base_ml_perceptron_classifier)

f.write("\n\nf) Top Multi-Layered Perceptron Averages & Standard Deviations\n")
dostepeight(top_ml_perceptron_classifier)
