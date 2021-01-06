#Arif Bipu challenge based on  'Learn Python for Data Science #1' by @Sirajology on YouTube.
from sklearn import tree
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier

#[height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

# Calling decision tree classifier and fitting
clf1 = tree.DecisionTreeClassifier()
clfDT = clf1.fit(X,Y)

# Calling support vector machine and fitting
clf2 = svm.SVC(probability=True)
clfSVM = clf2.fit(X,Y)

# Calling Kneighbors classifier and fitting
clf3 = KNeighborsClassifier(n_neighbors=3)
clfKN = clf3.fit(X,Y)

# Calling Gaussian Process classifier and fitting
clf4 = GaussianProcessClassifier()
clfGPC = clf4.fit(X,Y)

# Calling MLPClassifier and fitting
clf5 = MLPClassifier(learning_rate='constant', learning_rate_init=0.001)
clfMLP = clf5.fit(X,Y)


testing = [[175, 74, 41]]

# Results stored

predictionDT = clfDT.predict(testing)
predictionSVM = clfSVM.predict(testing)
predictionKN = clfKN.predict(testing)
predictionGPC = clfGPC.predict(testing)
predictionMLP = clfMLP.predict(testing)

# Probabilities stored
probaDT = clfDT.predict_proba(testing)
probaSVM = clfSVM.predict_proba(testing)
probaKN = clfKN.predict_proba(testing)
probaGPC = clfGPC.predict_proba(testing)
probaMLP = clfMLP.predict_proba(testing)

#printing statements with results

print("DT classifier test data {} is predicted as {} with probability of {}".format(testing, predictionDT, probaDT))

print("SVM classifier test data {} is predicted as {} with probability of {}".format(testing, predictionSVM, probaSVM))

print("KN classifier test data {} is predicted as {} with probability of {}".format(testing, predictionKN, probaKN))

print("GPC classifier test data {} is predicted as {} with probability of {}".format(testing, predictionGPC, probaGPC))

print("MLP classifier test data {} is predicted as {} with probability of {}".format(testing, predictionMLP, probaMLP))
