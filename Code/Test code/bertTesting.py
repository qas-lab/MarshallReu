# # file for testing bert

# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

# from bertopic import BERTopic
# import pandas as pd
# from sklearn.metrics import classification_report


# def findTopKDevs(dev_models, bugReport, k=5):
#     dev_probabilities = {}
#     for dev_name, model in dev_models.items():
#         topics, probabilities = model.transform([bugReport])
#         if not isinstance(probabilities[0], (list, tuple)):
#             max_prob = probabilities[0]
#         else:
#             max_prob = max(probabilities[0])
#         dev_probabilities[dev_name] = max_prob
    
#     sorted_devs = sorted(dev_probabilities.items(), key=lambda item: item[1], reverse=True)
#     top_k_devs = [dev for dev, _ in sorted_devs[:k]] 
#     return top_k_devs

# # Loading data
# data = pd.read_csv('test_documents.csv')
# data['Text'] = data['Product'].fillna('') + ' ' + data['Component'].fillna('') + ' ' + data['Summary'].fillna('')

# # Developer names and models
# developer_names = [
#     'Darin Wright', 'Dani Megert', 'Kevin Barnes', 'Curtis Windatt',
#     'Martin Aeschlimann', 'Boris Bokowski', 'Tod Creasey', 'Darin Swanson',
#     'Ayushman Jain', 'Jared Burns', 'Benno Baumgartner', 'Douglas Pollock',
#     'Joe Szurszewski', 'Tom Hofmann', 'Brian de Alwis', 'Adam Kiezun', 
#     'Samantha Chan', 'Andre Weinand'
# ]

# model_names = [
#     '0_Darin Wright', '1_Dani Megert', '2_Kevin Barnes', '3_Curtis Windatt',
#     '4_Martin Aeschlimann', '5_Boris Bokowski', '6_Tod Creasey', '7_Darin Swanson',
#     '8_Ayushman Jain', '9_Jared Burns', '10_Benno Baumgartner', '11_Douglas Pollock',
#     '12_Joe Szurszewski', '13_Tom Hofmann', '14_Brian de Alwis', '15_Adam Kiezun',
#     '16_Samantha Chan', '17_Andre Weinand'
# ]

# dev_model_mapping = dict(zip(developer_names, model_names))

# # Loading models
# docDictionary = {}
# for name, model_file in dev_model_mapping.items():
#     model = BERTopic.load(model_file)
#     docDictionary[name] = model

# dev_product_component_mapping = {}
# for name in developer_names:
#     dev_docs = data[data['Developer'] == name]
#     products = dev_docs['Product'].unique()
#     components = dev_docs['Component'].unique()
#     dev_product_component_mapping[name] = (products, components)

# ordered_docDictionary = docDictionary


# # Changed the following code below num = 10 and added from :num 

# data = data.sample(frac=1).reset_index(drop=True) # Shuffling data

# num = 10

# testingDoc = data['Text'][:num]
# true_devs = data['Developer'][:num]

# totalBugs = true_devs.value_counts()
# totalBugs.to_dict()

# dev_index_mapping = {name: index for index, name in enumerate(developer_names)}

# #num = len(testingDoc)
# num = 10

# if len(testingDoc) > 0:
#     true_labels = [dev_index_mapping.get(dev, -1) for dev in true_devs]  # Map developer names to indices
#     predicted_labels = []
#     top_5_correct = 0

#     for i in range(num):
#         product = data.iloc[i]['Product']
#         component = data.iloc[i]['Component']

#         relevant_devs = [dev for dev, (products, components) in dev_product_component_mapping.items() 
#                          if product in products and component in components]

#         filtered_docDictionary = {dev: ordered_docDictionary[dev] for dev in relevant_devs if dev in ordered_docDictionary}

#         if filtered_docDictionary:
#             top_5_devs = findTopKDevs(filtered_docDictionary, testingDoc.iloc[i], k=5)
#             best_dev = top_5_devs[0] if top_5_devs else None
            
#             print(best_dev)
#             print(top_5_devs)
            
#             if best_dev:
#                 matched_index = dev_index_mapping.get(best_dev, -1)
#                 if matched_index != -1:
#                     predicted_labels.append(matched_index)
#                 else:
#                     predicted_labels.append(-1)  
                
#                 if true_labels[i] in [dev_index_mapping.get(dev, -1) for dev in top_5_devs]:
#                     top_5_correct += 1
                
#                 print(f"Document {i}: Best Developer - {best_dev}")
#                 print(true_devs.iloc[i])
#                 print()

#             else:
#                 predicted_labels.append(-1) 
#         else:
#             predicted_labels.append(-1)  

#     # Classification Report
#     report = classification_report(true_labels, predicted_labels, zero_division=0)
#     print(report)
    
#     # Top 1 and Top 5 accuracy
#     top_1_accuracy = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred) / num
#     top_5_accuracy = top_5_correct / num
#     print(f"Top-1 Accuracy: {top_1_accuracy:.2f}")
#     print(f"Top-5 Accuracy: {top_5_accuracy:.2f}")

# else:
#     print("Not enough documents for evaluation.")

import os
from bertopic import BERTopic
import pandas as pd
from sklearn.metrics import classification_report

# Set environment variable to disable oneDNN optimizations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

def findTopKDevs(dev_models, bugReport, k=5, used_devs=None):
    dev_probabilities = {}
    for dev_name, model in dev_models.items():
        if used_devs and dev_name in used_devs:
            continue
        topics, probabilities = model.transform([bugReport])
        if not isinstance(probabilities[0], (list, tuple)):
            max_prob = probabilities[0]
        else:
            max_prob = max(probabilities[0])
        dev_probabilities[dev_name] = max_prob
    
    sorted_devs = sorted(dev_probabilities.items(), key=lambda item: item[1], reverse=True)
    top_k_devs = [dev for dev, _ in sorted_devs[:k]] 
    return top_k_devs

# Loading data
data = pd.read_csv('test_documents.csv')
data['Text'] = data['Product'].fillna('') + ' ' + data['Component'].fillna('') + ' ' + data['Summary'].fillna('')

# Developer names and models
developer_names = [
    'Darin Wright', 'Dani Megert', 'Kevin Barnes', 'Curtis Windatt',
    'Martin Aeschlimann', 'Boris Bokowski', 'Tod Creasey', 'Darin Swanson',
    'Ayushman Jain', 'Jared Burns', 'Benno Baumgartner', 'Douglas Pollock',
    'Joe Szurszewski', 'Tom Hofmann', 'Brian de Alwis', 'Adam Kiezun', 
    'Samantha Chan', 'Andre Weinand'
]

model_names = [
    '0_Darin Wright', '1_Dani Megert', '2_Kevin Barnes', '3_Curtis Windatt',
    '4_Martin Aeschlimann', '5_Boris Bokowski', '6_Tod Creasey', '7_Darin Swanson',
    '8_Ayushman Jain', '9_Jared Burns', '10_Benno Baumgartner', '11_Douglas Pollock',
    '12_Joe Szurszewski', '13_Tom Hofmann', '14_Brian de Alwis', '15_Adam Kiezun',
    '16_Samantha Chan', '17_Andre Weinand'
]

dev_model_mapping = dict(zip(developer_names, model_names))

# Loading models
docDictionary = {}
for name, model_file in dev_model_mapping.items():
    model = BERTopic.load(model_file)
    docDictionary[name] = model

dev_product_component_mapping = {}
for name in developer_names:
    dev_docs = data[data['Developer'] == name]
    products = dev_docs['Product'].unique()
    components = dev_docs['Component'].unique()
    dev_product_component_mapping[name] = (products, components)

ordered_docDictionary = docDictionary

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True) # Shuffling data

num = 10
testingDoc = data['Text'][:num]
true_devs = data['Developer'][:num]

# Calculate the total number of bugs for each developer
totalBugs = true_devs.value_counts().to_dict()

dev_index_mapping = {name: index for index, name in enumerate(developer_names)}

# Initialize counters
dev_usage = {dev: 0 for dev in developer_names}

predicted_labels = []
top_5_correct = 0

if len(testingDoc) > 0:
    true_labels = [dev_index_mapping.get(dev, -1) for dev in true_devs]

    for i in range(num):
        product = data.iloc[i]['Product']
        component = data.iloc[i]['Component']

        # Filter developers based on availability
        available_devs = [dev for dev in developer_names if dev_usage[dev] < totalBugs.get(dev, float('inf'))]
        relevant_devs = [dev for dev, (products, components) in dev_product_component_mapping.items() 
                         if product in products and component in components and dev in available_devs]

        filtered_docDictionary = {dev: ordered_docDictionary[dev] for dev in relevant_devs if dev in ordered_docDictionary}

        if filtered_docDictionary:
            top_5_devs = findTopKDevs(filtered_docDictionary, testingDoc.iloc[i], k=5)
            best_dev = top_5_devs[0] if top_5_devs else None
            
            print(f"Document {i}: True Developer - {true_devs.iloc[i]}")
            print(f"Document {i}: Top 5 Developers - {top_5_devs}")
            print(f"Document {i}: Best Developer - {best_dev}")
            
            if best_dev:
                matched_index = dev_index_mapping.get(best_dev, -1)
                if matched_index != -1:
                    predicted_labels.append(matched_index)
                    dev_usage[best_dev] += 1  # Increment usage for this developer
                else:
                    predicted_labels.append(-1)  
                
                if true_labels[i] in [dev_index_mapping.get(dev, -1) for dev in top_5_devs]:
                    top_5_correct += 1
                
                print()

            else:
                predicted_labels.append(-1) 
        else:
            predicted_labels.append(-1)  

    # Classification Report
    report = classification_report(true_labels, predicted_labels, zero_division=0)
    print(report)
    
    # Top 1 and Top 5 accuracy
    top_1_accuracy = sum(1 for true, pred in zip(true_labels, predicted_labels) if true == pred) / num
    top_5_accuracy = top_5_correct / num
    print(f"Top-1 Accuracy: {top_1_accuracy:.2f}")
    print(f"Top-5 Accuracy: {top_5_accuracy:.2f}")

else:
    print("Not enough documents for evaluation.")
