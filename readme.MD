# Learn Decision Tree

This project provides an implementation of a decision tree algorithm from scratch, without relying on any external libraries such as scikit-learn.

The purpose is to learn the inner workings of a decision tree algorithm and understand the underlying concepts.

<p align="center">
    <img src="https://wcs.smartdraw.com/decision-tree/img/structure-of-a-decision-tree.png"/>
    Structure of a Decision Tree
    </a>
</p>

## Project Description

There are 2 datasets: The first is dummy dataset containing variables of color, diameter, and fruit's label. The second is iris dataset from https://www.kaggle.com/datasets/uciml/iris

> Before play around with the `main.py` script, it is highly recommended to read the accompanying Jupyter Notebook (`notebooks/main.ipynb`) first. The notebook provides a detailed explanation of the decision tree algorithm, implementation, and example usage.

### **How it Works (A Brief Explanation)**

The steps of doing a Decision Tree are as described below:

1. Get the list of rows in the dataset, check the unique values and dominant labels:

- This step involves examining the dataset and extracting unique values for each feature (column) and identifying the dominant label (class) for the given set of rows

2. Calculate the gini impurity

- Gini impurity is a measure of impurity or randomness in a set of rows. (0 if pure, 0.5 if 50:50)

3. Generate Question that needs to be asked at a node

- The question refers to a feature-value pair that will be used to split the data at a particular node in the decision tree. The goal is to find the question that minimizes impurity or maximizes information gain.

4. Partition the rows into **true_rows** and **false_rows**

- Based on the question generated in the previous step, the rows are divided into two subsets: true_rows, which satisfy the question, and false_rows, which do not satisfy the question

5. Check the information gain

- Information gain is a measure of the reduction in entropy (or increase in purity) achieved by splitting the data based on a particular question

6. Update using highest information gain

- Select the question with the highest information gain as the best split at the current node

7. Update best question based on the highest information gain

- The best question is updated based on the highest information gain

8. Do it all recursively until reaching stop condition (no more impurity, or reach maximum depth)
   ![](https://github.com/edycakra/advanceML/blob/main/docs/workflow.png)

## Installation

To run the project, you need to clone it first and you need to have Python and Jupyter Notebook installed on your machine

## Usage

Read the jupyter notebook, or just play around using this command

> $ python -B main.py

Then you will need to input the name of the dataset: 'dummy' or 'iris'

## Result

The best question or ruleset generated by the Decision Tree, followed by the prediction result of user's input
![](https://github.com/edycakra/advanceML/blob/main/docs/result.png)

## References

- https://en.wikipedia.org/wiki/Decision_tree_learning#Gini_impurity
- https://towardsdatascience.com/under-the-hood-decision-tree-454f8581684e
