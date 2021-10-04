import matplotlib.pyplot as plt
import os
from sklearn.datasets import load_files

# Task 1.2

business = "/Users/Saleha1/Downloads/BBC/business"
entertainment = "/Users/Saleha1/Downloads/BBC/entertainment"
politics = "/Users/Saleha1/Downloads/BBC/politics"
sport = "/Users/Saleha1/Downloads/BBC/sport"
tech = "/Users/Saleha1/Downloads/BBC/tech"

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

plt.bar(x, y, color='maroon', align='center')
plt.title('BBC Distribution')
plt.xticks(x, x_labels)
plt.savefig('BBC-distribution.pdf')
plt.show()

# Task 1.3

bbc_data = load_files("/Users/Saleha1/Downloads/BBC", encoding = 'latin1')
