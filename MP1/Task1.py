import matplotlib.pyplot as plt
import os
from sklearn.datasets import  load_files

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
x_labels = ['Business', 'Entertainment', 'Politics', 'Sport', 'Tech']
y = [totalBusinessFiles, totalEntertainmentFiles, totalPoliticsFiles, totalSportFiles, totalTechFiles]


plt.bar(x, y, color=['black', 'red', 'green', 'blue', 'yellow'], align='center')
plt.title('BBC Distribution')
plt.ylabel("Total number of instances per class", fontsize = 10)
plt.xticks(x, x_labels)
plt.grid(True)

for index,data in enumerate(y):
    plt.text(x=index , y =data+1 , s=f"{data}" , fontdict=dict(fontsize=12, color = 'maroon'))

plt.savefig('BBC-distribution.pdf')

 # Task 1.3

bbc_data = load_files('MP1/BBC',load_content=True, encoding = 'latin1')
