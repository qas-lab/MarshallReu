import pandas as pd
from collections import Counter
import re

def countDevs(doc, devs, counter):
    for dev in devs:
        match = re.findall(rf'\b{dev}\b', doc, re.IGNORECASE)
        counter[dev] += len(match)


data = pd.read_csv('dataset.csv')
#print(data)

devs = data['Assignee'].unique().tolist()


count = Counter()


for description in data['Assignee']:
    countDevs(description, devs, count)

print('Printing Devs')

sortedDev = count.most_common()

for dev, count in sortedDev:
    print(f'{dev} : {count}')
