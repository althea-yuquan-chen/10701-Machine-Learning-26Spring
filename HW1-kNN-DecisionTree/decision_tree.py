import numpy as np
from collections import Counter
import pandas as pd
import sys
import matplotlib.pyplot as plt



class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None, counts=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.counts = counts
    
    def is_leaf_node(self):
        return self.left is None and self.right is None
    

class DecisionTree:

    def __init__(self, min_sample_split = 2, max_depth = None, n_features = None, min_info_gain = None):
        self.min_sample_split = min_sample_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.min_info_gain = min_info_gain
        self.root = None

        
    # Calculate entropy: H(S) = - \sum p(i)log_2p(i)
    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        entropy = -np.sum([p * np.log2(p) for p in ps if p > 0])

        return entropy
    

    # Calculate information gain
    def _information_gain(self, y, X, threshold):
        parent_entropy = self._entropy(y)

        # split
        left_index = np.argwhere(X <= threshold).flatten()
        right_index = np.argwhere(X > threshold).flatten()

        # if pure node, return 0 (no information gain)
        if len(left_index) == 0 or len(right_index) == 0:
            return 0

        # calculate child entropy
        n = len(y)
        n_l, n_r = len(left_index), len(right_index)
        e_l, e_r = self._entropy(y[left_index]), self._entropy(y[right_index])

        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        return parent_entropy - child_entropy


    # find the best split rule for each node, return feature and its threshold
    def _best_split(self, X, y, feature_index):
        best_gain = self.min_info_gain if self.min_info_gain is not None else 0
        split_index, split_threshold = None, None

        for feature in feature_index:
            X_column = X[:, feature]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_index = feature
                    split_threshold = threshold
        
        return split_index, split_threshold
    

    
    def _most_common_label(self, y):
        counter = Counter(y)

        most_common = counter.most_common(1)[0][0]

        return most_common


    def _grow_tree(self, X, y, depth = 0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        counts_arr = np.bincount(y.astype(int), minlength=2)
        node_counts = {0: counts_arr[0], 1: counts_arr[1]}
        majority_value = self._most_common_label(y)

        # Stopping criteria
        if ((self.max_depth is not None and depth >= self.max_depth) or n_labels == 1 or n_samples < self.min_sample_split):
            return Node(value=majority_value, counts=node_counts)

        # find the best split rule
        feature_index = range(n_features)
        best_feature, best_threshold = self._best_split(X, y, feature_index) # use all the features
        if best_feature is None:
            return Node(value=majority_value, counts=node_counts)
        
        # split the node
        left_index = np.argwhere(X[:, best_feature] <= best_threshold).flatten()
        right_index = np.argwhere(X[:, best_feature] > best_threshold).flatten()

        # recursively split
        left = self._grow_tree(X[left_index, :], y[left_index], depth + 1)
        right = self._grow_tree(X[right_index, :], y[right_index], depth + 1)

        return Node(best_feature, best_threshold, left, right, value=majority_value, counts=node_counts)
    

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        
        return self._traverse_tree(x, node.right)


    # Reduced Error Pruning
    def prune(self, X_val, y_val):
        X_val = np.array(X_val)
        y_val = np.array(y_val).flatten().astype(int)
        
        self._prune_node(self.root, X_val, y_val)

    def _prune_node(self, node, X_val, y_val):
        if node.is_leaf_node():
            return

        if len(y_val) == 0: # prune
            node.left = None
            node.right = None
            return

        left_mask = X_val[:, node.feature] <= node.threshold
        right_mask = ~left_mask

        X_left, y_left = X_val[left_mask], y_val[left_mask]
        X_right, y_right = X_val[right_mask], y_val[right_mask]

        # recursively prune, bottom-up
        self._prune_node(node.left, X_left, y_left)
        self._prune_node(node.right, X_right, y_right)

        # compare the error between pruned and subtree
        # Error if Pruned
        leaf_pred = node.value
        leaf_errors = np.sum(y_val != leaf_pred)

        # Error of Subtree
        subtree_preds = np.array([self._traverse_tree(x, node) for x in X_val])
        subtree_errors = np.sum(y_val != subtree_preds)

        if leaf_errors <= subtree_errors:
            node.left = None
            node.right = None


    def calculate_error(self, X_test, y_test):
        predictions = self.predict(X_test)
        accuracy = np.sum(y_test == predictions) / len(y_test)

        return accuracy
    

    def fit(self, X, y):
        y = np.array(y).flatten().astype(int)
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)


def print_tree(node, features, file = None):
    def _format_counts(counts):
        if counts is None: return ""
        c0 = counts.get(0,0)
        c1 = counts.get(1,0)

        return f"[{c0} 0/{c1} 1]"
    
    def _recursive_print(curr_node, depth, split_feature_name = None, split_val =None):
        if curr_node is None:
            return 
        
        indent = "| " * depth
        info_str = ""

        if split_feature_name is not None:
            info_str += f"{indent}{split_feature_name} = {split_val}: "
        else: # root node
            info_str += indent

        info_str += _format_counts(curr_node.counts)

        print(info_str, file=file)

        if not curr_node.is_leaf_node():
            feature = features[curr_node.feature]

            _recursive_print(curr_node.left, depth + 1, feature, 0)
            _recursive_print(curr_node.right, depth + 1, feature, 1)
    
    _recursive_print(node, 0)


def plot_depth_metrics(X_train, y_train, X_val, y_val):
    depths = range(9)
    train_accs = []
    val_accs = []

    print(f"{'Depth':<10} {'Train Acc':<15} {'Val Acc':<15}")
    print("-" * 40)

    for d in depths:
        clf = DecisionTree(max_depth=d) 
        clf.fit(X_train, y_train)

        train_acc = clf.calculate_error(X_train, y_train)
        val_acc = clf.calculate_error(X_val, y_val)

        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        print(f"{d:<10} {train_acc:.6f}        {val_acc:.6f}")

    max_val_acc = max(val_accs)
    best_depths = [d for d, acc in zip(depths, val_accs) if acc == max_val_acc]
    
    print("-" * 40)
    print(f"Max Validation Accuracy: {max_val_acc:.6f}")
    print(f"Achieved at depth(s): {best_depths}")

    plt.figure(figsize=(10, 6))
    plt.plot(depths, train_accs, label='Training Accuracy', marker='o')
    plt.plot(depths, val_accs, label='Validation Accuracy', marker='o')
    
    plt.title('Training & Validation Accuracy vs. Max Depth')
    plt.xlabel('Maximum Depth')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.xticks(depths)
    
    plt.savefig('depth_metrics.png')
    print("\nPlot saved to 'depth_metrics.png'.")
    plt.show()


def find_optimal_split_val(X_train, y_train, X_val, y_val):
    n_train = len(y_train)

    best_acc = -1.0
    best_c = -1

    print(f"Searching optimal min_samples_split from 0 to {n_train}...")
    
    for c in range(n_train + 1):
        clf = DecisionTree(min_sample_split=c, max_depth=None)
        
        clf.fit(X_train, y_train)

        val_acc = clf.calculate_error(X_val, y_val)

        if val_acc > best_acc:
            best_acc = val_acc
            best_c = c
            # print(f"New Best: C={c}, Acc={val_acc:.6f}")

    print("-" * 40)
    print(f"Optimal minimum splitting size (C): {best_c}")
    print(f"Validation accuracy: {best_acc:.6f}")
    print("-" * 40)


def find_optimal_info_gain(X_train, y_train, X_val, y_val):
    thresholds = [i / 100 for i in range(101)]
    
    best_acc = -1.0
    best_tau = -1.0
    
    print(f"Searching optimal min_info_gain from 0.00 to 1.00...")
    
    for tau in thresholds:
        clf = DecisionTree(min_info_gain=tau, max_depth=None)
        
        clf.fit(X_train, y_train)
        
        val_acc = clf.calculate_error(X_val, y_val)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_tau = tau
            # print(f"New Best: tau={tau}, Acc={val_acc:.6f}")

    print("-" * 40)
    print(f"Optimal information gain threshold: {best_tau}")
    print(f"Validation accuracy: {best_acc:.6f}")
    print("-" * 40)



if __name__ == "__main__":

    if len(sys.argv) < 4:
        print("Usage: python decision_tree.py <train_file> <val_file> <max_depth>")
        sys.exit(1)
    
    train_file_path = 'data/' + sys.argv[1]
    val_file_path = 'data/' + sys.argv[2]
    max_depth = int(sys.argv[3])

    df_train = pd.read_csv(train_file_path, sep='\t')
    df_val = pd.read_csv(val_file_path, sep='\t')

    # print(df_train.head())

    train_data = df_train.to_numpy()
    val_data = df_val.to_numpy()

    X_train = train_data[:, :-1]
    y_train = train_data[:, -1]

    X_val = val_data[:, :-1]
    y_val = val_data[:, -1]

    tree = DecisionTree(max_depth=max_depth, min_info_gain=0)

    tree.fit(X_train, y_train)

    feature_names = list(df_train.columns[:-1])


    if len(sys.argv) >= 5:
        mode = sys.argv[4]
        output_filename = ""
        
        if mode == "train":
            output_filename = "trained_tree.txt"
        elif mode == "prune":
            output_filename = "pruned_tree.txt"
            tree.prune(X_val, y_val)
        elif mode == "metrics":
            plot_depth_metrics(X_train, y_train, X_val, y_val)
        elif mode == "optimal_split":
            find_optimal_split_val(X_train, y_train, X_val, y_val)
        elif mode == "optimal_gain":
            find_optimal_info_gain(X_train, y_train, X_val, y_val)
        else:
            print(f"[Warning] Unknown mode '{mode}', defaulting output to console.")
            output_filename = None

        if output_filename:
            with open(output_filename, "w") as f:
                print_tree(tree.root, feature_names, file=f)
            print(f"Tree structure saved to {output_filename}")
        else:
             print_tree(tree.root, feature_names)

    else:
        print("\n--- Tree Structure (Console Output) ---")
        print_tree(tree.root, feature_names)

    # print(tree.calculate_error(X_train, y_train))
    # print(tree.calculate_error(X_val, y_val))
    