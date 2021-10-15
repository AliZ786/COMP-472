import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Task 2.2
drugfile = pd.read_csv('drug200.csv')

# Task 2.3
drug_array = np.array(drugfile["Drug"])

(unique_drug, frequency) = np.unique(drug_array, return_counts=True)

#Creates a bar chart showing the frequencing of the drugs
plt.bar(unique_drug, frequency, width = 0.8)

plt.xlabel("Drug")
plt.ylabel("Frequency")

plt.title("Drug distribution")

#Saves the figure to a PDF file
plt.savefig("drug-distribution.pdf")

# Task 2.4
