# PROPRIETARY FUNCTIONS
import pandas as pd
import time
from sklearn.metrics import f1_score


def preprocess_features(X):
    '''Preprocesses feature columns for non-numeric data'''
    outX = pd.DataFrame(index=X.index)  # output dataframe, initially empty

    # Check each column
    for col, col_data in X.iteritems():
        # If data type is non-numeric, try to replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])
        # Note: This should change the data type for yes/no columns to int

        # If still non-numeric, convert to one or more dummy variables
        if col_data.dtype == object:
            col_data = pd.get_dummies(col_data, prefix=col)  # e.g. 'school' => 'school_GP', 'school_MS'

        outX = outX.join(col_data)  # collect column(s) in output dataframe

    return outX


def train_predict(name, X_train, y_train, X_test, y_test, set_size_1, set_size_2, set_size_3, clf_default, clf_tuned = None):
    '''Train supervised models, and predict using different training set sizes.
    Requires SKLEARN.'''
    #print table headers
    table_headers = ["tr_size", "tr_time","tr_ptime","tr_f1","tst_ptime","tst_f1"]
    row_format ="{:>12}" * (len(table_headers) + 1)
    print "\n"
    print row_format.format("model", *table_headers)
    
    sizes = (set_size_1, set_size_2, set_size_3)
    
    #calculate and print the default classifier metrics
    for size in sizes:
        #train classifier and calculate time
        start_tr_time = time.time()
        clf_default.fit(X_train[:size], y_train[:size])
        end_tr_time = time.time()
        tr_time = end_tr_time - start_tr_time
    
        #predict with classifier on the training set and calculate time
        start_tr_ptime = time.time()
        y_pred = clf_default.predict(X_train[:size])
        end_tr_ptime = time.time()
        tr_ptime = end_tr_ptime - start_tr_ptime
    
        #calculate f1 score for the training set
        tr_f1 = f1_score(y_train[:size].values, y_pred, pos_label='yes')
    
        #predict with classifier on the test set and calculate time
        start_tst_ptime = time.time()
        y_pred = clf_default.predict(X_test)
        end_tst_ptime = time.time()
        tst_ptime = end_tst_ptime - start_tst_ptime
    
        #calculate f1 score for the test set
        tst_f1 = f1_score(y_test.values, y_pred, pos_label='yes')
        
        #print the metrics to the table
        print row_format.format(name + "_default", 
                                size, 
                                "{:.4f}".format(tr_time), 
                                "{:.4f}".format(tr_ptime), 
                                "{:.4f}".format(tr_f1), 
                                "{:.4f}".format(tst_ptime), 
                                "{:.4f}".format(tst_f1))

    #calculate and print the tuned classifier metrics
    if clf_tuned != None:
        for size in sizes:
            #train classifier and calculate time
            start_tr_time = time.time()
            clf_tuned.fit(X_train[:size], y_train[:size])
            end_tr_time = time.time()
            tr_time = end_tr_time - start_tr_time

            #predict with classifier on the training set and calculate time
            start_tr_ptime = time.time()
            y_pred = clf_tuned.predict(X_train[:size])
            end_tr_ptime = time.time()
            tr_ptime = end_tr_ptime - start_tr_ptime

            #calculate f1 score for the training set
            tr_f1 = f1_score(y_train[:size].values, y_pred, pos_label='yes')

            #predict with classifier on the test set and calculate time
            start_tst_ptime = time.time()
            y_pred = clf_tuned.predict(X_test)
            end_tst_ptime = time.time()
            tst_ptime = end_tst_ptime - start_tst_ptime

            #calculate f1 score for the test set
            tst_f1 = f1_score(y_test.values, y_pred, pos_label='yes')

            #print the metrics to the table
            print row_format.format(name + "_tuned", 
                                    size, 
                                    "{:.4f}".format(tr_time), 
                                    "{:.4f}".format(tr_ptime), 
                                    "{:.4f}".format(tr_f1), 
                                    "{:.4f}".format(tst_ptime), 
                                    "{:.4f}".format(tst_f1))