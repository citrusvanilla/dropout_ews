# An "Early Warning System" Supervised Learning Trigger for Student Dropout Intervention 

>*NOTE: Sample data is included in this repository as `student_data.csv`.

This repository contains a module of proprietary functions as `dropout_ews.py` and a Jupyter notebook as `dropout_ews_walkthrough.ipynb` that walks the user through the implementation of an "Early Warning System" (EWS) trigger for intervention in potential student dropouts.  The optimized trigger uses a 'feature weighted nearest neighbor' supervised learning model.  A sample dataset is included as `student_data.csv` in the same directory.

The notebook contains code for fitting a variety of supervised learning models to the problem and evaluating their effectivness:

![alt text](http://i.imgur.com/LReRrYe.jpg)

The notebook then demonstrates a method for obtaining a ["feature weighted nearest neighbor" model](http://www.isle.org/~langley/papers/diet.ecml97.pdf) using an optimized Decision Tree and a KNN model.  The final model is fit to the dataset and performance metrics accompany:

![alt text](http://i.imgur.com/3XbSeGQ.jpg)

## Goals

This notebook compares a range of supervised learning models on the dataset, and shows that a "feature weighted nearest neighbor" model has best-in-class accuracy and speed. 

## Software and Library Requirements

* Python 2.7.11
* Jupyter Notebook 4.2.2
* Numpy 1.11.2
* scikit-image 0.12.3
* matplotlib 1.5.2

## Data
-----------
The sample data used for this notebook comes from the [UCI Irvine Machine Learning repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance) and has been modified to fit a classification problem.

> "This data approach student achievement in secondary education of two Portuguese schools. The data attributes include student grades, demographic, social and school related features) and it was collected by using school reports and questionnaires. Two datasets are provided regarding the performance in two distinct subjects: Mathematics (mat) and Portuguese language (por). In [Cortez and Silva, 2008], the two datasets were modeled under binary/five-level classification and regression tasks. Important note: the target attribute G3 has a strong correlation with attributes G2 and G1. This occurs because G3 is the final year grade (issued at the 3rd period), while G1 and G2 correspond to the 1st and 2nd period grades. It is more difficult to predict G3 without G2 and G1, but such prediction is much more useful (see paper source for more details)."

The sample dataset used in this project is included as `student_data.csv`. The last column 'passed' is the target/label, all other are feature columns.  The CSV contains a header with the following 30 attributes:

- __school__ : student's school (binary: "GP" or "MS")
- __sex__ : student's sex (binary: "F" - female or "M" - male)
- __age__ : student's age (numeric: from 15 to 22)
- __address__ : student's home address type (binary: "U" - urban or "R" - rural)
- __famsize__ : family size (binary: "LE3" - less or equal to 3 or "GT3" - greater than 3)
- __Pstatus__ : parent's cohabitation status (binary: "T" - living together or "A" - apart)
- __Medu__ : mother's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to - __9th grade, 3 - secondary education or 4 - higher education)
- __Fedu__ : father's education (numeric: 0 - none, 1 - primary education (4th grade), 2 - 5th to - __9th grade, 3 - secondary education or 4 - higher education)
- __Mjob__ : mother's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
- __Fjob__ : father's job (nominal: "teacher", "health" care related, civil "services" (e.g. administrative or police), "at_home" or "other")
- __reason__ : reason to choose this school (nominal: close to "home", school "reputation", "course" preference or "other")
- __guardian__ : student's guardian (nominal: "mother", "father" or "other")
- __traveltime__ : home to school travel time (numeric: 1 - <15 min., 2 - 15 to 30 min., 3 - 30 min. to 1 hour, or 4 - >1 hour)
- __studytime__ : weekly study time (numeric: 1 - <2 hours, 2 - 2 to 5 hours, 3 - 5 to 10 hours, or 4 - >10 hours)
- __failures__ : number of past class failures (numeric: n if 1<=n<3, else 4)
- __schoolsup__ : extra educational support (binary: yes or no)
- __famsup__ : family educational support (binary: yes or no)
- __paid__ : extra paid classes within the course subject (Math or Portuguese) (binary: yes or no)
- __activities__ : extra-curricular activities (binary: yes or no)
- __nursery__ : attended nursery school (binary: yes or no)
- __higher__ : wants to take higher education (binary: yes or no)
- __internet__ : Internet access at home (binary: yes or no)
- __romantic__ : with a romantic relationship (binary: yes or no)
- __famrel__ : quality of family relationships (numeric: from 1 - very bad to 5 - excellent)
- __freetime__ : free time after school (numeric: from 1 - very low to 5 - very high)
- __goout__ : going out with friends (numeric: from 1 - very low to 5 - very high)
- __Dalc__ : workday alcohol consumption (numeric: from 1 - very low to 5 - very high)
- __Walc__ : weekend alcohol consumption (numeric: from 1 - very low to 5 - very high)
- __health__ : current health status (numeric: from 1 - very bad to 5 - very good)
- __absences__ : number of school absences (numeric: from 0 to 93)

Each student has a target that takes two discrete labels:

- __passed__ : did the student pass the final exam (binary: yes or no)


## Getting up and running

While in the `dropout_ews` directory, use the following command in your command line interface:

> `ipython notebook dropout_ews.ipynb`
