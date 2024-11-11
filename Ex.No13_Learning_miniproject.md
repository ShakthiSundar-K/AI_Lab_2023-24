# Ex.No: 10 Learning – Use Supervised Learning  
### DATE: 11-11-2024                                                                         
### REGISTER NUMBER : 212222040152
### AIM: 
To write a program to train the classifier for Medicine Recommendation System.
###  Algorithm:


1. **Data Loading and Preprocessing**:
   - Load training data and other relevant datasets (symptoms, precautions, medications, workouts, diets).
   - Separate features (`X`) from the target label (`y`), which is the medical condition or prognosis.
   - Encode the target label (`y`) to convert it to a numerical format for machine learning processing.

2. **Train-Test Split**:
   - Split the dataset into training and testing sets, with 80% of the data used for training and 20% for testing.

3. **Model Selection and Training**:
   - Define a dictionary of machine learning models: Support Vector Classifier, Random Forest, K-Nearest Neighbors, Gradient Boosting, and Multinomial Naive Bayes.
   - For each model:
     - Train the model on the training set.
     - Make predictions on the test set.
     - Calculate the model’s accuracy and generate a confusion matrix to evaluate performance.

4. **Model Saving**:
   - Save the trained SVC model using `pickle` for future use without retraining.

5. **Model Testing**:
   - Reload the saved model.
   - Select test cases to predict conditions and compare predicted labels with actual labels to check model correctness.

6. **Prediction Helper Function**:
   - Define a helper function that retrieves the symptoms, precautions, medications, diet plan, and workout recommendations based on a predicted disease.

7. **Symptom-to-Condition Prediction**:
   - Accept user-input symptoms.
   - Transform symptoms into a numerical vector matching the input format of the trained model.
   - Use the model to predict the likely disease based on user symptoms.

8. **Result Display**:
   - Display the predicted condition.
   - Use the helper function to fetch and display related information, including disease description, precautions, medications, workouts, and diet plans, to guide the user.

### Program:
```
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
import pandas as pd
import numpy as np
import warnings
import pickle
import os

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

df = pd.read_csv('/content/Training.csv')
sym_des = pd.read_csv('/content/Symptom-severity(1).csv')
precautions = pd.read_csv('/content/precautions_df(1).csv')
workout = pd.read_csv('/content/workout_df(1).csv')
description = pd.read_csv('/content/description(1).csv')
medications = pd.read_csv('/content/medications(1).csv')
diets = pd.read_csv('/content/diets(1).csv')

df.head()
df.tail()
df.shape
df['prognosis'].unique()
len(df['prognosis'].unique())

X = df.drop('prognosis', axis=1)
y = df['prognosis']

le = LabelEncoder()
le.fit(y)
Y = le.transform(y)

# split the data
X_test, X_train, y_test, y_train = train_test_split(X, Y, test_size=0.2, random_state=42)
X_test.shape, X_train.shape, y_test.shape, y_train.shape

# Here we can train the top best models which can give best accuracy So I can create a dictionary of models
models = {
    'SVC': SVC(kernel='linear'),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'KNeighbors': KNeighborsClassifier(n_neighbors=5),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100),
    'MultinomialNB': MultinomialNB()
}

for model_name, model in models.items():
    # Train model
    model.fit(X_train, y_train)
    # test model
    predictions = model.predict(X_test)
    # calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    # calculate confusion matrix
    cm = confusion_matrix(y_test, predictions)
    # print results
    print(f"{model_name} accuracy : {accuracy}")
    print(f"{model_name} confusion matrix :")
    print(np.array2string(cm, separator=', '))

svc = SVC(kernel='linear')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
acc = accuracy_score(y_pred, y_test)
acc

# Path to save the model
path = '/kaggle/working//svc.pkl'
# Ensure the directory exists
os.makedirs(os.path.dirname(path), exist_ok=True)
# Save the model
with open(path, 'wb') as file:
    pickle.dump(svc, file)

# load the dataset
svc = pickle.load(open(path, 'rb'))

# 2D array conversion
X_test.iloc[0].values.reshape(1, -1)

# Now take a prediction on this 2D array to check that our model pred correctly or not

# Test 1
print('Model Predictions :', svc.predict(X_test.iloc[0].values.reshape(1, -1)))
print('Actual Labels :', y_test[0])

# Test 2
print('Model Predictions :', svc.predict(X_test.iloc[40].values.reshape(1, -1)))
print('Actual Labels :', y_test[40])

def helper(dis): # This function can give us the symptoms Description, Precautions, Medication, Diet plan, workout
    desc = description[description['Disease'] == dis]['Description']
    desc = " ".join([w for w in desc])
    pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3']].values.flatten()
    med = medications[medications['Disease'] == dis]['Medication'].values.flatten()
    die = diets[diets['Disease'] == dis]['Diet'].values.flatten()
    wrkout = workout[workout['disease'] == dis]['workout'].values.flatten()
    return desc, pre, med, die, wrkout

# Create a dictionary of symptoms and diseases
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis'}

# Model Prediction function
def given_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))
    for item in patient_symptoms:
        input_vector[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([input_vector])[0]]

symptoms = input("Enter your symptoms.......")
user_symptoms = [s.strip() for s in symptoms.split(',')]
predicted_disease = given_predicted_value(user_symptoms)
desc, pre, med, die, wrkout = helper(predicted_disease)

print("=================predicted disease============")
print(predicted_disease)
print("=================description==================")
print(desc)
print("=================precautions==================")
i = 1
for p_i in pre:
    print(i, ": ", p_i)
    i += 1
print("=================medications==================")
for m_i in med:
    print(i, ": ", m_i)
    i += 1
print("=================workout==================")
for w_i in wrkout:
    print(i, ": ", w_i)
    i += 1
print("=================diets==================")
for d_i in die:
    print(i, ": ", d_i)
    i += 1

```

### Output:
![image](https://github.com/user-attachments/assets/1ee0892c-7f61-42e3-883f-efde32902e73)


### Result:
Thus the system was trained successfully and the prediction was carried out.
