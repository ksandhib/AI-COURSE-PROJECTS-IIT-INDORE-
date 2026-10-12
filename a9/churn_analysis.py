import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

# Load the dataset (assuming standard Telco Customer Churn dataset, as the provided Excel is data dictionary)
url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
customer = pd.read_csv(url)

# Clean data: Convert TotalCharges to numeric, handle missing values
customer['TotalCharges'] = pd.to_numeric(customer['TotalCharges'], errors='coerce')
customer = customer.dropna()

# A) Data Manipulation
print("A) Data Manipulation:")

# a. Total number of male customers
male_count = sum(customer['gender'] == "Male")
print(f"a. Total number of male customers: {male_count}")

# b. Total number of customers whose Internet Service is ‘DSL’
dsl_count = sum(customer['InternetService'] == "DSL")
print(f"b. Total number of customers whose Internet Service is ‘DSL’: {dsl_count}")

# c. Extract all the Female senior citizens whose Payment Method is Mailed check & store in ‘new_customer’
new_customer = customer[(customer['gender'] == 'Female') & (customer['SeniorCitizen'] == 1) & (customer['PaymentMethod'] == 'Mailed check')]
print(f"c. Number of Female senior citizens with Payment Method 'Mailed check': {len(new_customer)}")
print("First 5 rows of new_customer:")
print(new_customer.head())

# d. Extract all those customers whose tenure is less than 10 months or Total charges is less than 500$
new_customer = customer[(customer['tenure'] < 10) | (customer['TotalCharges'] < 500)]
print(f"d. Number of customers with tenure < 10 or TotalCharges < 500: {len(new_customer)}")
print("First 5 rows of new_customer:")
print(new_customer.head())

# B) Data Visualization
print("\nB) Data Visualization:")

# a. Pie-chart for distribution of customers churning out
names = customer["Churn"].value_counts().keys().tolist()
sizes = customer["Churn"].value_counts().tolist()
plt.figure(figsize=(6,6))
plt.pie(sizes, labels=names, autopct="%0.1f%%")
plt.title('Distribution of Customers Churning Out')
plt.savefig('churn_pie_chart.png')
plt.show()
print("a. Pie-chart saved as 'churn_pie_chart.png'")

# b. Bar-plot for distribution of ‘Internet Service’
plt.figure(figsize=(8,6))
plt.bar(customer['InternetService'].value_counts().keys().tolist(), customer['InternetService'].value_counts().tolist(), color='orange')
plt.xlabel('Categories of Internet Service')
plt.ylabel('Count of categories')
plt.title('Distribution of Internet Service')
plt.savefig('internet_service_bar_plot.png')
plt.show()
print("b. Bar-plot saved as 'internet_service_bar_plot.png'")

# C) Model Building
print("\nC) Model Building:")

# Prepare Churn as binary
customer['Churn'] = customer['Churn'].map({'Yes': 1, 'No': 0})

# a. Model 1: Using ‘tenure’ as feature
x = customer[['tenure']]
y = customer[['Churn']]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

model = Sequential()
model.add(Dense(12, input_dim=1, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=150, validation_data=(x_test, y_test), verbose=0)

print("a. Model 1 Results:")
print(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
print(f"Mean validation accuracy: {np.mean(history.history['val_accuracy']):.4f}")

y_pred = (model.predict(x_test) > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure()
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model 1: Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('model1_accuracy.png')
plt.show()
print("Accuracy vs Epochs plot saved as 'model1_accuracy.png'")

# b. Model 2: With dropout
model2 = Sequential()
model2.add(Dense(12, input_dim=1, activation='relu'))
model2.add(Dropout(0.3))
model2.add(Dense(8, activation='relu'))
model2.add(Dropout(0.2))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history2 = model2.fit(x_train, y_train, epochs=150, validation_data=(x_test, y_test), verbose=0)

print("\nb. Model 2 Results:")
print(f"Final validation accuracy: {history2.history['val_accuracy'][-1]:.4f}")
print(f"Mean validation accuracy: {np.mean(history2.history['val_accuracy']):.4f}")

y_pred2 = (model2.predict(x_test) > 0.5).astype(int)
cm2 = confusion_matrix(y_test, y_pred2)
print("Confusion Matrix:")
print(cm2)

plt.figure()
plt.plot(history2.history['accuracy'], label='Train Accuracy')
plt.plot(history2.history['val_accuracy'], label='Val Accuracy')
plt.title('Model 2: Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('model2_accuracy.png')
plt.show()
print("Accuracy vs Epochs plot saved as 'model2_accuracy.png'")

# c. Model 3: Using ‘Tenure’, ’Monthly Charges’ & ‘Total Charges’ as features
x3 = customer[['tenure', 'MonthlyCharges', 'TotalCharges']]
y3 = customer[['Churn']]
x_train3, x_test3, y_train3, y_test3 = train_test_split(x3, y3, test_size=0.30, random_state=42)

model3 = Sequential()
model3.add(Dense(12, input_dim=3, activation='relu'))
model3.add(Dense(8, activation='relu'))
model3.add(Dense(1, activation='sigmoid'))
model3.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history3 = model3.fit(x_train3, y_train3, epochs=150, validation_data=(x_test3, y_test3), verbose=0)

print("\nc. Model 3 Results:")
print(f"Final validation accuracy: {history3.history['val_accuracy'][-1]:.4f}")
print(f"Mean validation accuracy: {np.mean(history3.history['val_accuracy']):.4f}")

y_pred3 = (model3.predict(x_test3) > 0.5).astype(int)
cm3 = confusion_matrix(y_test3, y_pred3)
print("Confusion Matrix:")
print(cm3)

plt.figure()
plt.plot(history3.history['accuracy'], label='Train Accuracy')
plt.plot(history3.history['val_accuracy'], label='Val Accuracy')
plt.title('Model 3: Accuracy vs Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('model3_accuracy.png')
plt.show()
print("Accuracy vs Epochs plot saved as 'model3_accuracy.png'")

print("\nConclusion:")
print("Model 1 mean val acc: {:.4f}".format(np.mean(history.history['val_accuracy'])))
print("Model 2 mean val acc: {:.4f}".format(np.mean(history2.history['val_accuracy'])))
print("Model 3 mean val acc: {:.4f}".format(np.mean(history3.history['val_accuracy'])))
