from sklearn.model_selection import train_test_split


from ucimlrepo import fetch_ucirepo 
import numpy as np 

isolet = fetch_ucirepo(id=54) 
# Extract features and targets
X = isolet.data.features
y = isolet.data.targets['class']

# Split the dataset into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Alternatively, save as CSV for compatibility with C++ programs
np.savetxt('isolet_X_train.csv', X_train, delimiter=',')  # Save X_train to a .csv file
np.savetxt('isolet_y_train.csv', y_train, delimiter=',', fmt='%d')  # Save y_train to a .csv file
np.savetxt('isolet_X_test.csv', X_test, delimiter=',')    # Save X_test to a .csv file
np.savetxt('isolet_y_test.csv', y_test, delimiter=',', fmt='%d')    # Save y_test to a .csv file
