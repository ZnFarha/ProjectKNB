from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


def dtree(X_train, X_test, y_train, y_test):
    dtree_model = DecisionTreeClassifier(criterion='entropy', min_samples_leaf=5).fit(X_train, y_train)
    dtree_predictions = dtree_model.predict(X_test)
    acc = accuracy_score(y_test, dtree_predictions)
    print(acc)


def kNN(X_train, X_test, y_train, y_test):
    knn = KNeighborsClassifier(n_neighbors=4).fit(X_train, y_train)
    knn_predictions = knn.predict(X_test)
    acc = accuracy_score(y_test, knn_predictions)
    print(acc)


def naiveB(X_train, X_test, y_train, y_test):
    gnb = GaussianNB().fit(X_train, y_train)
    gnb_predictions = gnb.predict(X_test)
    acc = accuracy_score(y_test, gnb_predictions)

    print("Accuracy: ", acc)

    print('Predicted labels for the first 15 instances: ')
    for i in range(1, 16):
        print("%s) Predicted=%s" % (i, gnb_predictions[i]))


def rforest(X_train, X_test, y_train, y_test):
    randfor = RandomForestClassifier(max_depth=2, random_state=0).fit(X_train, y_train)
    randfor_predictions = randfor.predict(X_test)
    acc = accuracy_score(y_test, randfor_predictions)
    print(acc)
