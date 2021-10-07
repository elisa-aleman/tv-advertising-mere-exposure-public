#-*- coding: utf-8 -*-

from NomuraSoken_methods import *

def getResultLog(model='svm', base='user', server=True):
    model_str = ""
    if model == 'svm': model_str = "SVM_tests";
    elif model == 'xgboost': model_str = "XGBoost_tests";
    elif model == 'logit': model_str = "LogisticRegression_tests";
    filename = "{}_{}.csv".format(model_str, base)
    csv_path = MakeLogFile(filename, server=server)
    titles, table = readCSV(csv_path)
    return titles, table

def getPredictableUsers(server=True):
    log_file = MakeLogFile('Predictable_Users.csv')
    titles = ['Model', 'Model Type','Target', 'Category', 'Predictable User Percentage (f>0.5)']
    strlog = ','.join(titles)
    printLog(strlog,log_file)
    for model in ['svm', 'xgboost', 'logit']:
        _,table = getResultLog(model,'user',server)
        for mltype in ['cm_only_prime', 'cm_only_nonprime', 'demographics', 'cm_demo_prime', 'cm_demo_nonprime']:
            for target in ['purchase','intent']:
                for category in [0,1,2,3,4,5]:
                    f1s = [float(trow[10]) for trow in table if trow[1]==target and trow[2]==mltype and int(trow[3])==category]
                    total = len(f1s)
                    predictables = len([f1 for f1 in f1s if f1>0.5])
                    per = predictables/total
                    strlog = '{},{},{},{},{}'.format(model,mltype,target,category,per)
                    printLog(strlog,log_file)

def main():
    getPredictableUsers(server=True)


if __name__ == '__main__':
    main()

