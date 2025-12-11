import sys
sys.path.insert(0, '/path/to/project')

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from TwoPatternsDataset import TwoPatternsDataset

def main():
    train_data = TwoPatternsDataset("/path/to/project/TwoPatterns/TwoPatterns_TRAIN.tsv")
    test_data = TwoPatternsDataset("/path/to/project/TwoPatterns/TwoPatterns_TEST.tsv")
    
    train_X = np.array([x.squeeze().numpy() for x, _ in train_data])
    train_y = np.array([y.item() if hasattr(y, 'item') else y for _, y in train_data])
    
    test_X = np.array([x.squeeze().numpy() for x, _ in test_data])
    test_y = np.array([y.item() if hasattr(y, 'item') else y for _, y in test_data])
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_X, train_y)
    
    train_pred = clf.predict(train_X)
    test_pred = clf.predict(test_X)
    
    print(f"Train Accuracy: {accuracy_score(train_y, train_pred):.4f}")
    print(f"Test Accuracy: {accuracy_score(test_y, test_pred):.4f}")
    print("\nClassification Report (Test):")
    print(classification_report(test_y, test_pred))

if __name__ == "__main__":
    main()
