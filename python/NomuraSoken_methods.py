#-*- coding: utf-8 -*-

# import sqlite3
import csv
import os.path
import codecs
import sklearn
from sklearn import svm
import scipy
import numpy
from datetime import datetime,time
import xgboost
from sklearn.linear_model import LogisticRegression
import random


##################
# ###How to import:
# import sys
# import os.path
# PythonPath = os.path.join(os.sep, 'usr', 'local', 'data', 'python')
# sys.path.append(os.path.abspath(PythonPath))
# ## if you want to use the method only: method()
# from NomuraSoken_methods import *
### if you want to use the library name first: NomuraSoken_methods.method()
# import NomuraSoken_methods
##################

#ProgressBar printing methods
 #Go up a line in the terminal to print over something
def up():
    # My terminal breaks if we don't flush after the escape-code
    sys.stdout.write('\x1b[1A')
    sys.stdout.flush()

#Go down a line in the terminal (like print "")
def down():
    # I could use '\x1b[1B' here, but newline is faster and easier
    sys.stdout.write('\n')
    sys.stdout.flush()

#Print to terminal and to a file at the same time
def printSTDlog(strlog, log_file):
    with codecs.open(log_file, 'a', 'utf-8') as logf:
            print(strlog)
            strlog+= "\n"
            logf.write(strlog)

#Print to a file
def printLog(strlog, log_file):
    with codecs.open(log_file, 'a', 'utf-8') as logf:
            strlog+= "\n"
            logf.write(strlog)

#Make a list like [1,[2,3],[4,[5]]] into [1,2,3,4,5]
def flatten(container):
    for i in container:
        if isinstance(i, (list,tuple)):
            for j in flatten(i):
                yield j
        else:
            yield i

#Read a CSV file
def readCSV(filename, titlesplit=True):
    f = open(filename, 'rt', encoding='utf-8')
    reader = csv.reader(f, delimiter = ',')
    table = []
    for row in reader:
        table.append(tuple(row))
    f.close()
    if titlesplit:
        titles = table[0]
        table = table[1:]
        return titles, table
    else:
        return table

######################
### get File Paths ###
######################

def getHalfPath():
    halfpath = os.path.join(os.path.dirname(os.getcwd()), 'data', "送付データ")
    return halfpath

def getSurveyDataCSVPath():
    halfpath = getHalfPath()
    surveydata_csv = os.path.join(halfpath, "アンケートデータ", "CSV")
    return surveydata_csv

def getSurveyDataSQLPath():
    halfpath = getHalfPath()
    surveydata_sql = os.path.join(halfpath, "アンケートデータ", "SQL")
    return surveydata_sql

def getPrintDataCSVPath():
    halfpath = getHalfPath()
    printdata_csv = os.path.join(halfpath, "出稿データ")
    return printdata_csv

def getProcessedDataFolderPath():
    processed_data_folder_path = os.path.join(os.path.dirname(os.getcwd()), 'processed_data')
    return processed_data_folder_path



##
# Don't use these
#
# tvviewdata_sql_path = os.path.join(surveydata_sql, "tvviewdata.sqlite")
# magazinedata_sql_path = os.path.join(surveydata_sql, "magazines.sqlite")
# commercialdata_sql_path = os.path.join(printdata_csv, "SQL", "commercials.sqlite")
##

def getMainDataCSVPat):
    surveydata_csv = getSurveyDataCSVPath()
    maindata_csv_path = os.path.join(surveydata_csv, "メインデータ_2017.csv")
    return maindata_csv_path

def getMainDataSQLPat):
    getSurveyDataSQLPath()
    maindata_sql_path = os.path.join(surveydata_sql, "maindata.sqlite")
    return maindata_sql_path

def getTVdataCSVPat):
    surveydata_csv = getSurveyDataCSVPath()
    tvviewdata_csv_path = os.path.join(surveydata_csv, "テレビ番組別視聴状況_2017.csv")
    return tvviewdata_csv_path

def getMagazinesCSVPat):
    surveydata_csv = getSurveyDataCSVPath()
    magazinedata_csv_path = os.path.join(surveydata_csv, "雑誌閲読状況_2017.csv")
    return magazinedata_csv_path

def getCommercialCSVPat):
    printdata_csv = getPrintDataCSVPath()
    commercialdata_csv_path = os.path.join(printdata_csv, "テレビCM出稿データ_2017.csv")
    return commercialdata_csv_path

def getProductLabelCSVPat):
    processed_data_folder_path = getProcessedDataFolderPath()
    label_csv_path = os.path.join(processed_data_folder_path, "20171018_商品購入意欲データ.csv")
    return label_csv_path

def getProductID_dic):
    processed_data_folder_path = getProcessedDataFolderPath()
    product_csv_path = os.path.join(processed_data_folder_path, "20171018_商品サービス名.csv")
    _,product_csv = readCSV(product_csv_path)
    product_csv = [(int(row[0]), row[1]) for row in product_csv]
    products = dict(product_csv)
    return products

def getUserLabelCSVPat):
    processed_data_folder_path = getProcessedDataFolderPath()
    label_csv_path = os.path.join(processed_data_folder_path, "User_ClassLabel_by_Product.csv")
    return label_csv_path

def getUserLabelCSVPath_iyok):
    processed_data_folder_path = getProcessedDataFolderPath()
    label_csv_path = os.path.join(processed_data_folder_path, "User_ClassLabel_by_Product_iyoku.csv")
    return label_csv_path

def getMainVectorize):
    processed_data_folder_path = getProcessedDataFolderPath()
    vec_csv_path = os.path.join(processed_data_folder_path, "Main_Vectorized.csv")
    titles,data = readCSV(vec_csv_path)
    return titles,data

def getMatchingCMdataCSVPat):
    processed_data_folder_path = getProcessedDataFolderPath()
    mcd_csv_path = os.path.join(processed_data_folder_path, "matching_cm_data.csv")
    return mcd_csv_path

def getProgPrimeCSVPat):
    processed_data_folder_path = getProcessedDataFolderPath()
    ptgp_csv_path = os.path.join(processed_data_folder_path, "prog_time_genre_prime.csv")
    return ptgp_csv_path

def getCM_time_count_CSVPat):
    processed_data_folder_path = getProcessedDataFolderPath()
    cmtime_csv_path = os.path.join(processed_data_folder_path, "cm_time_count.csv")
    return cmtime_csv_path

def getUser_demographics_CSVPat):
    processed_data_folder_path = getProcessedDataFolderPath()
    user_demographics_csv_path = os.path.join(processed_data_folder_path, "User_demographics.csv")
    return user_demographics_csv_path

def getProductList():
    product_list = [1,5,6,7,10,26,29,30,32,33,36,39,42,43,50,54,55,58,66,80,83,97,113,114,118,147,154,155,157,158,164,181,185,186,187,197]
    return product_list

def MakeLogFile(filename):
    '''
    Make Log file path string
    '''
    halfpath = os.path.join(os.path.dirname(os.getcwd()), 'logs')
    log_file = os.path.join(halfpath, filename)
    file_parent = log_file.rsplit("/",1)[0]+"/"
    pathlib.Path(file_parent).mkdir(parents = True, exist_ok=True)
    return log_file

def MakeModelPath(filename):
    '''
    Make Model file path string
    '''
    halfpath = os.path.join(os.path.dirname(os.getcwd()), 'models')
    model_path = os.path.join(halfpath, filename)
    file_parent = model_path.rsplit("/",1)[0]+"/"
    pathlib.Path(file_parent).mkdir(parents = True, exist_ok=True)
    return model_path

#######################
### Processing Data ###
#######################

def getTimePeriod(product_id):
    _,label_csv = readCSV(getProductLabelCSVPath())
    pindex = ((product_id*4)+1, (product_id*4)+3)
    start_date = '2017/'+label_csv[pindex[0]][1].rsplit('（',1)[1].replace('）','')
    end_date = '2017/'+label_csv[pindex[1]][1].rsplit('（',1)[1].replace('）','')
    start_datetime = datetime.strptime(start_date, '%Y/%m/%d')
    end_datetime = datetime.strptime(end_date, '%Y/%m/%d')
    return start_datetime, end_datetime

def getTVdate(tvwatch_id):
    _,ptgp_table = readCSV(getProgPrimeCSVPath())
    tv_date = [row[3] for row in ptgp_table if row[0]==tvwatch_id][0]
    tv_datetime = datetime.strptime(tv_date, '%Y/%m/%d')
    return tv_datetime

def getTVtimes(tvwatch_id):
    _,ptgp_table = readCSV(getProgPrimeCSVPath())
    tv_times = [[row[5],row[6],row[7],row[8]] for row in ptgp_table if row[0]==tvwatch_id][0]
    tv_start_time = time(int(tv_times[0])%24,int(tv_times[1])%24)
    tv_end_time = time(int(tv_times[2])%24,int(tv_times[3])%24)
    return tv_start_time,tv_end_time

def getCMseconds(tvwatch_id, product_id, cmtime_csv):
    cmseconds_list = [row[2] for row in cmtime_csv if row[0]==tvwatch_id and int(float(row[1]))==product_id]
    if len(cmseconds_list)>0:
        cmseconds = int(cmseconds_list[0])
    else:
        cmseconds = 0
    return cmseconds

def getCMinfo(product_id):
    cm_titles,cm_table = readCSV(getTVdataCSVPath())
    mcd_titles,mcd_table = readCSV(getMatchingCMdataCSVPath())
    _,main_data = readCSV(getMainDataCSVPath())
    users = [user[0] for user in main_data]
    ### List of TV programs where a CM with the product was aired.
    product_tvban_set = set([row[0] for row in mcd_table if row[4]==str(product_id)])
    ### REMOVE TV programs outside time period
    _,ptgp_table = readCSV(getProgPrimeCSVPath())
    start_date,end_date = getTimePeriod(product_id)
    time_tvban_set = set([ptgprow[0] for ptgprow in ptgp_table if start_date<=datetime.strptime(ptgprow[3], '%Y/%m/%d')<end_date])
    tvban_set = time_tvban_set & product_tvban_set
    ### Filter the main CM info table from tv programs relating to the product in the related time.
    cm_table_T = list(map(list,zip(*([cm_titles]+cm_table))))
    cminfo_T = [cm_table_T[0]]+[col for col in cm_table_T if col[0] in tvban_set]
    cminfo = list(map(list,zip(*cminfo_T)))
    cminfo_titles = cminfo[0]
    cminfo_table = cminfo[1:]
    cminfo_table = [cmrow for cmrow in cminfo_table if cmrow[0] in users]
    return cminfo_titles, cminfo_table

########################
### MACHINE LEARNING ###
########################

# ### product_id = 0 ~ 199
# ### categories:
# ### 0: ⭕ -> ❌
# ### 1: ❌ -> ❌
# ### 2: ❌ -> ⭕
# ### 3: ⭕ -> ⭕
# ### 4: 2 and 3
# ### 5: 0 and 1

def getYlabel_product_purchase(product_id, category=2):
    _,label_data=readCSV(getUserLabelCSVPath())
    labels = [int(row[product_id+1]) for row in label_data]
    if category<4:
        y_labels = [1 if cat==category else 0 for cat in labels]
    elif category == 4:
        y_labels = [1 if cat==2 or cat==3 else 0 for cat in labels]
    elif category == 5:
        y_labels = [1 if cat==0 or cat==1 else 0 for cat in labels]
    return y_labels

def getYlabel_product_intent(product_id, category=2):
    _,label_data=readCSV(getUserLabelCSVPath_iyoku())
    labels = [int(row[product_id+1]) for row in label_data]
    if category<4:
        y_labels = [1 if cat==category else 0 for cat in labels]
    elif category == 4:
        y_labels = [1 if cat==2 or cat==3 else 0 for cat in labels]
    elif category == 5:
        y_labels = [1 if cat==0 or cat==1 else 0 for cat in labels]
    return y_labels

def getYlabel_user_purchase(user_id, category=2):
    _,label_data=readCSV(getUserLabelCSVPath())
    labels = [[int(val) for val in row][1:] for row in label_data if row[0]==user_id][0]
    if category<4:
        y_labels = [1 if cat==category else 0 for cat in labels]
    elif category == 4:
        y_labels = [1 if cat==2 or cat==3 else 0 for cat in labels]
    elif category == 5:
        y_labels = [1 if cat==0 or cat==1 else 0 for cat in labels]
    products = getProductList()
    y = [val for num,val in enumerate(y_labels) if num in products]
    return y

def getYlabel_user_intent(user_id, category=2):
    _,label_data=readCSV(getUserLabelCSVPath_iyoku())
    labels = [[int(val) for val in row][1:] for row in label_data if row[0]==user_id][0]
    if category<4:
        y_labels = [1 if cat==category else 0 for cat in labels]
    elif category == 4:
        y_labels = [1 if cat==2 or cat==3 else 0 for cat in labels]
    elif category == 5:
        y_labels = [1 if cat==0 or cat==1 else 0 for cat in labels]
    products = getProductList()
    y = [val for num,val in enumerate(y_labels) if num in products]
    return y

def getXvector_product(product_id, target='purchase',mltype='cm_only', prime_inclusion=False):
    processed_data_folder_path = getProcessedDataFolderPath()
    if mltype=='cm_only':
        x_titles,pre_x = readCSV(os.path.join(processed_data_folder_path, "Xvector_cmtime_by_product.csv"))
        x = [[int(val) for val in row][2:16] for row in pre_x if int(row[1])==product_id]
        if not prime_inclusion:
            x = [[row[i*2]+row[(i*2)+1] for i in range(int(len(row)/2))] for row in x]
        return x
    elif mltype=='demographics':
        demo_titles, demo_table = readCSV(getUser_demographics_CSVPath())
        if target=='intent':
            x = [[int(val) for val in row] for row in demo_table]
            x = [row[1:] for row in x]
            return x
        elif target=='purchase':
            _,label_data=readCSV(getUserLabelCSVPath_iyoku())
            labels = [int(row[product_id+1]) for row in label_data]
            intentions = []
            for label in labels:
                if label==0:
                    intentions.append([1,0])
                elif label==1:
                    intentions.append([0,0])
                elif label==2:
                    intentions.append([0,1])
                elif label==3:
                    intentions.append([1,1])
            x = [[int(val) for val in row] for row in demo_table]
            x = [row[1:]+intentions[i] for i,row in enumerate(x)]
            return x
    elif mltype=='cm_demo':
        x_titles,pre_x = readCSV(os.path.join(processed_data_folder_path, "Xvector_cmtime_by_product.csv"))
        demo_titles, demo_table = readCSV(getUser_demographics_CSVPath())
        if target=='intent':
            cm_x = [[int(val) for val in row][2:16] for row in pre_x if int(row[1])==product_id]
            if not prime_inclusion:
                cm_x = [[row[i*2]+row[(i*2)+1] for i in range(int(len(row)/2))] for row in cm_x]
            demo_x = [[int(val) for val in row] for row in demo_table]
            x = [cm_x[i]+row[1:] for i,row in enumerate(demo_x)]
            return x
        elif target=='purchase':
            _,label_data=readCSV(getUserLabelCSVPath_iyoku())
            cm_x = [[int(val) for val in row][2:16] for row in pre_x if int(row[1])==product_id]
            if not prime_inclusion:
                cm_x = [[row[i*2]+row[(i*2)+1] for i in range(int(len(row)/2))] for row in cm_x]
            demo_x = [[int(val) for val in row] for row in demo_table]
            labels = [int(row[product_id+1]) for row in label_data]
            intentions = []
            for label in labels:
                if label==0:
                    intentions.append([1,0])
                elif label==1:
                    intentions.append([0,0])
                elif label==2:
                    intentions.append([0,1])
                elif label==3:
                    intentions.append([1,1])
            x = [cm_x[i]+row[1:]+intentions[i] for i,row in enumerate(demo_x)]
            return x

def getXvector_user(user_id, target='purchase',mltype='cm_only', prime_inclusion=False):
    products = getProductList()
    processed_data_folder_path = getProcessedDataFolderPath()
    if mltype=='cm_only':
        x_titles,pre_x = readCSV(os.path.join(processed_data_folder_path, "Xvector_cmtime_by_product.csv"))
        x = [[int(val) for val in row][2:16] for row in pre_x if row[0]==user_id and int(row[1]) in products]
        if not prime_inclusion:
            x = [[row[i*2]+row[(i*2)+1] for i in range(int(len(row)/2))] for row in x]
        return x
    elif mltype=='demographics':
        demo_titles, demo_table = readCSV(getUser_demographics_CSVPath())
        if target=='intent':
            x = [[[int(val) for val in row] for row in demo_table if row[0]==user_id][0][1:] for product in products]
            return x
        elif target=='purchase':
            _,label_data=readCSV(getUserLabelCSVPath_iyoku())
            labels = [[int(val) for val in row][1:] for row in label_data if row[0]==user_id][0]
            labels = [val for num,val in enumerate(labels) if num in products]
            intentions = []
            for label in labels:
                if label==0:
                    intentions.append([1,0])
                elif label==1:
                    intentions.append([0,0])
                elif label==2:
                    intentions.append([0,1])
                elif label==3:
                    intentions.append([1,1])
            x = [[[int(val) for val in row] for row in demo_table if row[0]==user_id][0][1:] for product in products]
            x = [row+intentions[i] for i,row in enumerate(x)]
            return x
    elif mltype=='cm_demo':
        x_titles,pre_x = readCSV(os.path.join(processed_data_folder_path, "Xvector_cmtime_by_product.csv"))
        demo_titles, demo_table = readCSV(getUser_demographics_CSVPath())
        if target=='intent':
            cm_x = [[int(val) for val in row][2:16] for row in pre_x if row[0]==user_id and int(row[1]) in products]
            if not prime_inclusion:
                cm_x = [[row[i*2]+row[(i*2)+1] for i in range(int(len(row)/2))] for row in cm_x]
            demo_x = [[[int(val) for val in row] for row in demo_table if row[0]==user_id][0][1:] for product in products]
            x = [cm_x[i]+row for i,row in enumerate(demo_x)]
            return x
        elif target=='purchase':
            _,label_data=readCSV(getUserLabelCSVPath_iyoku())
            cm_x = [[int(val) for val in row][2:16] for row in pre_x if row[0]==user_id and int(row[1]) in products]
            if not prime_inclusion:
                cm_x = [[row[i*2]+row[(i*2)+1] for i in range(int(len(row)/2))] for row in cm_x]
            labels = [[int(val) for val in row][1:] for row in label_data if row[0]==user_id][0]
            labels = [val for num,val in enumerate(labels) if num in products]
            intentions = []
            for label in labels:
                if label==0:
                    intentions.append([1,0])
                elif label==1:
                    intentions.append([0,0])
                elif label==2:
                    intentions.append([0,1])
                elif label==3:
                    intentions.append([1,1])
            demo_x = [[[int(val) for val in row] for row in demo_table if row[0]==user_id][0][1:] for product in products]
            x = [cm_x[i]+row+intentions[i] for i,row in enumerate(demo_x)]
            return x

####################
### SVM Learning ###
####################

def SVMkfolds(x, y, k, kernel = 'linear', C = 1.0, gamma = 0.001):
    precisions = []
    recalls = []
    accuracies = []
    f1s = []
    testsize = len(y)//k
    # Correct Prediction, True Positive, True Negative, Incorrect Prediction, False Positive, False Negative
    counts = [{"CP":0, "TP": 0, "TN":0, "IP":0, "FP":0, "FN":0} for t in range(k)]
    xysets = [row+[y[num]] for num,row in enumerate(x)]
    for t in range(k):
        numpy.random.shuffle(xysets)
        y_list = [xyset[-1] for xyset in xysets]
        X_list = [xyset[:-1] for xyset in xysets]
        X = numpy.array(X_list)
        y = numpy.array(y_list)
        #Define classifier
        clf = svm.SVC(kernel = kernel, C = C, gamma = gamma)
        clf.fit(X[:-testsize],y[:-testsize])
        #Test data
        for i in range(1, testsize+1):
            predicted = clf.predict(X[-i].reshape(1,-1))[0]
            true_value = y[-i]
            if (predicted == true_value): # test data
                counts[t]["CP"] += 1
                if predicted == 1:
                    counts[t]["TP"] += 1
                else:
                    counts[t]["TN"] += 1
            else:
                counts[t]["IP"] += 1
                if predicted == 1:
                    counts[t]["FP"] += 1
                else:
                    counts[t]["FN"] += 1
        # precision = true_positives / (true_positives + false_positives)
        # recall = true_positives / (true_positives + false_negatives)
        # accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        if counts[t]["TP"]+counts[t]["FP"]>0:
            precision = counts[t]["TP"] / (counts[t]["TP"] + counts[t]["FP"])
        else:
            precision = 0
        if counts[t]["TP"]+counts[t]["FN"]>0:
            recall = counts[t]["TP"] / (counts[t]["TP"] + counts[t]["FN"])
        else:
            recall = 0
        accuracy = counts[t]["CP"]/testsize
        if precision>0 or recall>0:
            F1 = 2* ((precision*recall)/(precision+recall))
        else:
            F1 = 0
        #
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(F1)
    avpr = sum(precisions)/len(precisions)
    stpr = scipy.std(precisions)
    avre = sum(recalls)/len(recalls)
    stre = scipy.std(recalls)
    avac = sum(accuracies)/len(accuracies)
    stac = scipy.std(accuracies)
    avf1 = sum(f1s)/len(f1s)
    stf1 = scipy.std(f1s)
    results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
    return results

def SVM_weights(x, y, feature_names, kernel = 'linear', C = 1.0, gamma = 0.001):
    if type(x) == type([]):
        x = numpy.array(x)
    if type(y) == type([]):
        y = numpy.array(y)
    clf = svm.SVC(kernel = kernel, C = C, gamma = gamma)
    clf.fit(x,y)
    weights = clf.coef_.tolist()[0]
    influences = zip(feature_names, weights)
    return influences

###############
### XGBoost ###
###############

def XGBoost_Kfolds(x, y, k, probability_cutoff=0.5, max_depth=3, learning_rate=0.1, n_estimators=100, eta=1, silent=1, objective='binary:logistic', min_child_weight=1, num_round=2):
    precisions = []
    recalls = []
    accuracies = []
    f1s = []
    testsize = len(y)//k
    counts = [{"CP":0, "TP": 0, "TN":0, "IP":0, "FP":0, "FN":0} for t in range(k)]
    if type(x) == type(numpy.array([])):
        x = x.tolist()
    if type(y) == type(numpy.array([])):
        y = y.tolist()
    xysets = [row+[y[num]] for num,row in enumerate(x)]
    for t in range(k):
        numpy.random.shuffle(xysets)
        y_list = [xyset[-1] for xyset in xysets]
        X_list = [xyset[:-1] for xyset in xysets]
        X = numpy.array(X_list)
        y = numpy.array(y_list)
        #Define classifier
        # specify parameters via map
        param = {'max_depth':max_depth, 'learning_rate':learning_rate, 'eta':eta, 'silent':silent, 'objective':objective, 'n_estimators':n_estimators}
        dtrain = xgboost.DMatrix(X[:-testsize], label=y[:-testsize])
        clf = xgboost.train(param, dtrain, num_round)
        #Test data
        dtest = xgboost.DMatrix(X[-testsize:])
        predicted_probs = clf.predict(dtest)
        ypreds = []
        for ypred in predicted_probs:
            if ypred>probability_cutoff:
                ypreds.append(1)
            elif ypred==probability_cutoff:
                ypreds.append(random.randint(0,1))
            else:
                ypreds.append(0)
        for i in range(1, testsize+1):
            predicted = ypreds[i-1]
            true_value = y[-i]
            if (predicted == true_value): # test data
                counts[t]["CP"] += 1
                if predicted == 1:
                    counts[t]["TP"] += 1
                else:
                    counts[t]["TN"] += 1
            else:
                counts[t]["IP"] += 1
                if predicted == 1:
                    counts[t]["FP"] += 1
                else:
                    counts[t]["FN"] += 1
        # precision = true_positives / (true_positives + false_positives)
        # recall = true_positives / (true_positives + false_negatives)
        # accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        if counts[t]["TP"]+counts[t]["FP"]>0:
            precision = counts[t]["TP"] / (counts[t]["TP"] + counts[t]["FP"])
        else:
            precision = 0
        if counts[t]["TP"]+counts[t]["FN"]>0:
            recall = counts[t]["TP"] / (counts[t]["TP"] + counts[t]["FN"])
        else:
            recall = 0
        accuracy = counts[t]["CP"]/testsize
        if precision>0 or recall>0:
            F1 = 2* ((precision*recall)/(precision+recall))
        else:
            F1 = 0
        #
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(F1)
    avpr = sum(precisions)/len(precisions)
    stpr = scipy.std(precisions)
    avre = sum(recalls)/len(recalls)
    stre = scipy.std(recalls)
    avac = sum(accuracies)/len(accuracies)
    stac = scipy.std(accuracies)
    avf1 = sum(f1s)/len(f1s)
    stf1 = scipy.std(f1s)
    results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
    return results

def LogisticRegression_Kfolds(x,y,k):
    precisions = []
    recalls = []
    accuracies = []
    f1s = []
    testsize = len(y)//k
    # Correct Prediction, True Positive, True Negative, Incorrect Prediction, False Positive, False Negative
    counts = [{"CP":0, "TP": 0, "TN":0, "IP":0, "FP":0, "FN":0} for t in range(k)]
    if type(x) == type(numpy.array([])):
        x = x.tolist()
    if type(y) == type(numpy.array([])):
        y = y.tolist()
    xysets = [row+[y[num]] for num,row in enumerate(x)]
    for t in range(k):
        numpy.random.shuffle(xysets)
        y_list = [xyset[-1] for xyset in xysets]
        X_list = [xyset[:-1] for xyset in xysets]
        X = numpy.array(X_list)
        y = numpy.array(y_list)
        #Define classifier
        clf = LogisticRegression()
        clf.fit(X[:-testsize],y[:-testsize])
        #Test data
        for i in range(1, testsize+1):
            predicted = clf.predict(X[-i].reshape(1,-1))[0]
            true_value = y[-i]
            if (predicted == true_value): # test data
                counts[t]["CP"] += 1
                if predicted == 1:
                    counts[t]["TP"] += 1
                else:
                    counts[t]["TN"] += 1
            else:
                counts[t]["IP"] += 1
                if predicted == 1:
                    counts[t]["FP"] += 1
                else:
                    counts[t]["FN"] += 1
        # precision = true_positives / (true_positives + false_positives)
        # recall = true_positives / (true_positives + false_negatives)
        # accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives)
        if counts[t]["TP"]+counts[t]["FP"]>0:
            precision = counts[t]["TP"] / (counts[t]["TP"] + counts[t]["FP"])
        else:
            precision = 0
        if counts[t]["TP"]+counts[t]["FN"]>0:
            recall = counts[t]["TP"] / (counts[t]["TP"] + counts[t]["FN"])
        else:
            recall = 0
        accuracy = counts[t]["CP"]/testsize
        if precision>0 or recall>0:
            F1 = 2* ((precision*recall)/(precision+recall))
        else:
            F1 = 0
        #
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)
        f1s.append(F1)
    avpr = sum(precisions)/len(precisions)
    stpr = scipy.std(precisions)
    avre = sum(recalls)/len(recalls)
    stre = scipy.std(recalls)
    avac = sum(accuracies)/len(accuracies)
    stac = scipy.std(accuracies)
    avf1 = sum(f1s)/len(f1s)
    stf1 = scipy.std(f1s)
    results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
    return results

if __name__ == '__main__':
    pass