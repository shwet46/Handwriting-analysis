import os
import zipfile
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import extract
import categorize

# Feature arrays
X_baseline_angle = []
X_top_margin = []
X_letter_size = []
X_line_spacing = []
X_word_spacing = []
X_pen_pressure = []
X_slant_angle = []
y_t1, y_t2, y_t3, y_t4, y_t5 = [], [], [], [], []
page_ids = []

if os.path.isfile("label_list"):
    print("Info: label_list found.")

    with open("label_list", "r") as labels:
        for line in labels:
            content = line.split()

            X_baseline_angle.append(float(content[0]))
            X_top_margin.append(float(content[1]))
            X_letter_size.append(float(content[2]))
            X_line_spacing.append(float(content[3]))
            X_word_spacing.append(float(content[4]))
            X_pen_pressure.append(float(content[5]))
            X_slant_angle.append(float(content[6]))

            y_t1.append(float(content[7]))
            y_t2.append(float(content[8]))
            y_t3.append(float(content[9]))
            y_t4.append(float(content[10]))
            y_t5.append(float(content[11]))

            page_ids.append(content[12])

    # Combine features for each trait
    X_t1 = list(zip(X_baseline_angle, X_slant_angle))
    X_t2 = list(zip(X_letter_size, X_top_margin))
    X_t3 = list(zip(X_line_spacing, X_word_spacing))
    X_t4 = list(zip(X_slant_angle, X_top_margin))
    X_t5 = list(zip(X_line_spacing, X_word_spacing))

    # Helper function for training and testing
    def train_and_evaluate(clf, X, y, name):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=5)
        clf.fit(X_train, y_train)
        acc = accuracy_score(clf.predict(X_test), y_test)
        print(f"{name} accuracy: {acc:.4f}")
        return clf

    print("\n=== Accuracy of Classifiers using SVM ===")
    clf1 = train_and_evaluate(SVC(kernel='rbf'), X_t1, y_t1, "Classifier 1 (Neuroticism)")
    clf2 = train_and_evaluate(SVC(kernel='rbf'), X_t2, y_t2, "Classifier 2 (Agreeableness)")
    clf3 = train_and_evaluate(SVC(kernel='rbf'), X_t3, y_t3, "Classifier 3 (Openness)")
    clf4 = train_and_evaluate(SVC(kernel='rbf'), X_t4, y_t4, "Classifier 4 (Conscientiousness)")
    clf5 = train_and_evaluate(SVC(kernel='rbf'), X_t5, y_t5, "Classifier 5 (Extraversion)")

    print("\n=== Accuracy of Classifiers using KNN ===")
    clf6 = train_and_evaluate(KNeighborsClassifier(n_neighbors=5), X_t1, y_t1, "Classifier 1 (KNN)")
    clf7 = train_and_evaluate(KNeighborsClassifier(n_neighbors=5), X_t2, y_t2, "Classifier 2 (KNN)")
    clf8 = train_and_evaluate(KNeighborsClassifier(n_neighbors=5), X_t3, y_t3, "Classifier 3 (KNN)")
    clf9 = train_and_evaluate(KNeighborsClassifier(n_neighbors=5), X_t4, y_t4, "Classifier 4 (KNN)")
    clf10 = train_and_evaluate(KNeighborsClassifier(n_neighbors=5), X_t5, y_t5, "Classifier 5 (KNN)")

    print("\n=== Accuracy of Classifiers using Random Forest ===")
    clf11 = train_and_evaluate(RandomForestClassifier(criterion="gini", n_estimators=10), X_t1, y_t1, "Classifier 1 (RF)")
    clf12 = train_and_evaluate(RandomForestClassifier(criterion="gini", n_estimators=10), X_t2, y_t2, "Classifier 2 (RF)")
    clf13 = train_and_evaluate(RandomForestClassifier(criterion="gini", n_estimators=10), X_t3, y_t3, "Classifier 3 (RF)")
    clf14 = train_and_evaluate(RandomForestClassifier(criterion="gini", n_estimators=10), X_t4, y_t4, "Classifier 4 (RF)")
    clf15 = train_and_evaluate(RandomForestClassifier(criterion="gini", n_estimators=10), X_t5, y_t5, "Classifier 5 (RF)")

    # Prediction loop
    print("\n=== Personality Trait Prediction ===")
    while True:
        file_name = input("Enter file name to predict or 'z' to exit: ")
        if file_name.lower() == 'z':
            break

        raw_features = extract.start(file_name)

        raw_baseline_angle = raw_features[0]
        baseline_angle, comment = categorize.determine_baseline_angle(raw_baseline_angle)
        print("Baseline Angle:", comment)

        raw_top_margin = raw_features[1]
        top_margin, comment = categorize.determine_top_margin(raw_top_margin)
        print("Top Margin:", comment)

        raw_letter_size = raw_features[2]
        letter_size, comment = categorize.determine_letter_size(raw_letter_size)
        print("Letter Size:", comment)

        raw_line_spacing = raw_features[3]
        line_spacing, comment = categorize.determine_line_spacing(raw_line_spacing)
        print("Line Spacing:", comment)

        raw_word_spacing = raw_features[4]
        word_spacing, comment = categorize.determine_word_spacing(raw_word_spacing)
        print("Word Spacing:", comment)

        raw_pen_pressure = raw_features[5]
        pen_pressure, comment = categorize.determine_pen_pressure(raw_pen_pressure)
        print("Pen Pressure:", comment)

        raw_slant_angle = raw_features[6]
        slant_angle, comment = categorize.determine_slant_angle(raw_slant_angle)
        print("Slant:", comment)

        print()
        print("Neuroticism:", clf1.predict([[baseline_angle, slant_angle]])[0])
        print("Agreeableness:", clf2.predict([[letter_size, top_margin]])[0])
        print("Openness:", clf3.predict([[line_spacing, word_spacing]])[0])
        print("Conscientiousness:", clf4.predict([[slant_angle, top_margin]])[0])
        print("Extraversion:", clf5.predict([[line_spacing, word_spacing]])[0])
        print("---------------------------------------------------\n")

else:
    print("Error: 'label_list' file not found.")