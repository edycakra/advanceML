import pandas as pd

# Get the dataset filename from user input
user_input = input("Enter the dataset filename (dummy or iris): ")

# Add condition
if user_input == "dummy" or user_input == "iris":
    """"""
else:
    print("Invalid dataset name. Please enter 'dummy' or 'iris'.")
    exit()

filename = "./data/{}.csv".format(user_input)
data = pd.read_csv(filename)

# Get the column names
header = data.columns.tolist()

# Get the column indices based on the header
column_indices = {header[i]: i for i in range(len(header))}


def unique_vals(rows, col):
    """Find the unique values for a column in a dataset."""
    return set(rows.iloc[:, col])


def class_counts(rows):
    """Counts the number of each type of example in a dataset."""
    counts = {}
    for row in rows.iloc[:, -1]:
        if row not in counts:
            counts[row] = 0
        counts[row] += 1
    return counts


def is_numeric(value):
    """Simple function to check if a value is numeric."""
    return isinstance(value, (int, float))


class Question:
    """A Question is used to partition a dataset. To measure the value within the features."""

    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        # This is where we compare the feature value of sample and the feature value of "question"
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        # Helper method to make it readable
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return "Is %s %s %s?" % (header[self.column], condition, str(self.value))


def partition(rows, question):
    """Partitions a dataset based on a given question. If true, add to true_rows"""
    true_rows = rows[rows.apply(lambda row: question.match(row), axis=1)]
    false_rows = rows[~rows.apply(lambda row: question.match(row), axis=1)]
    return true_rows, false_rows


def gini_calc(rows):
    """Calculate the Gini Impurity for a list of rows."""
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl] / float(len(rows))
        impurity -= prob_of_lbl**2
    return impurity


def info_gain(left, right, current_uncertainty):
    """Information Gain. How much information do we gain after partitioning.

    The uncertainty of the starting node, minus the weighted impurity of
    two child nodes.
    """
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini_calc(left) - (1 - p) * gini_calc(right)


def find_best_split(rows):
    """Find the best question to ask by iterating over every feature/value pair and calculating the information gain."""
    best_gain = 0
    best_question = None
    current_uncertainty = gini_calc(rows)
    n_features = len(rows.columns) - 1

    for col in range(n_features):
        values = unique_vals(rows, col)

        for val in values:
            question = Question(col, val)
            true_rows, false_rows = partition(rows, question)

            if len(true_rows) == 0 or len(false_rows) == 0:
                continue

            gain = info_gain(true_rows, false_rows, current_uncertainty)

            if gain >= best_gain:
                best_gain, best_question = gain, question

    return best_gain, best_question


class Leaf:
    """A Leaf node classifies data."""

    def __init__(self, rows):
        self.predictions = class_counts(rows)


class Decision_Node:
    """A Decision Node asks a question."""

    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    """Builds the decision tree recursively."""

    # Find the best question
    gain, question = find_best_split(rows)

    # Base case: when there's no more information to gain
    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)

    # Build true branch recursively
    true_branch = build_tree(true_rows)

    # Build false branch recursively
    false_branch = build_tree(false_rows)

    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing=""):
    """Prints the decision tree. The purpose is to visually understand the decision making of the tree."""
    if isinstance(node, Leaf):
        print("Prediction:", node.predictions)
        return

    print(spacing + str(node.question))
    print(spacing + "--> True:")
    print_tree(node.true_branch, spacing + "  ")
    print(spacing + "--> False:")
    print_tree(node.false_branch, spacing + "  ")


def classify(row, node):
    """Classifies a row using the decision tree."""

    if isinstance(node, Leaf):
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    """Prints the predictions at a leaf."""
    total = sum(counts.values())
    probs = {}

    for lbl in counts.keys():
        probs[lbl] = str(round(counts[lbl] / total * 100, 2)) + "%"
    return probs


# Build the decision tree
my_tree = build_tree(data)


print("==== DECISION TREE RULES:")
print_tree(my_tree)
print("=========================")
# Custom input by user
if user_input == "dummy":
    # Ask the user for input
    color = input("Color of the fruit: ")
    diameter = float(input("Diameter of the fruit: "))
    print(print_leaf(classify([color.lower(), diameter], my_tree)))
elif user_input == "iris":
    # Ask the user for input
    sepalLgth = input("Length of sepal: ")
    sepalWdth = float(input("Width of sepal: "))
    petalLgth = input("Length of petal: ")
    petalWdth = float(input("Width of petal: "))
    print(print_leaf(classify([sepalLgth, sepalWdth, petalLgth, petalWdth], my_tree)))
