
from NomuraSoken_methods import *
import scipy

def getResultLog(model='svm', base='product', server=True):
    model_str = ""
    if model == 'svm': model_str = "SVM_tests";
    elif model == 'xgboost': model_str = "XGBoost_tests";
    elif model == 'logit': model_str = "LogisticRegression_tests";
    filename = "{}_{}.csv".format(model_str, base)
    csv_path = MakeLogFile(filename, server=server)
    titles, table = readCSV(csv_path)
    return titles, table

def test_H1(server=True):
    # cm_only vs demographics
    log_file = MakeLogFile("Test_H1_cm_only-demographics.csv")
    titles = ['Model', 'Base', 'Configuration', 'purchase_0', 'purchase_1','purchase_2', 'purchase_3','purchase_4','purchase_5', 'intent_0', 'intent_1','intent_2', 'intent_3','intent_4','intent_5']
    strlog = ','.join(titles)
    if os.path.exists(log_file)==False:
        printLog(strlog,log_file)
    for model in ['svm', 'xgboost', 'logit']:
        for base in ['product','user']:
            _,table = getResultLog(model,base,server)
            for prime_inclusion in [True,False]:
                prime_inclusion_str = "Weekday and Time Slot" if prime_inclusion else "Weekday Only"
                cm_mltype_str = 'cm_only_prime' if prime_inclusion else 'cm_only_nonprime'
                row = [model,base,prime_inclusion_str]
                for target in ['purchase', 'intent']:
                    for category in [0,1,2,3,4,5]:
                        cm_only_f1s = [float(trow[10]) for trow in table if trow[1]==target and trow[2]==cm_mltype_str and int(trow[3])==category]
                        demographics_f1s = [float(trow[10]) for trow in table if trow[1]==target and trow[2]=='demographics' and int(trow[3])==category]
                        p = scipy.stats.ttest_ind(cm_only_f1s, demographics_f1s, equal_var=False)[1]
                        row.append(str(p))
                strlog = ','.join(row)
                printLog(strlog,log_file)

def test_H2(server=True):
    # cm_demo vs demographics
    log_file = MakeLogFile("Test_H2_demographics-cm_demo.csv")
    titles = ['Model', 'Base', 'Configuration', 'purchase_0', 'purchase_1','purchase_2', 'purchase_3','purchase_4','purchase_5', 'intent_0', 'intent_1','intent_2', 'intent_3','intent_4','intent_5']
    strlog = ','.join(titles)
    if os.path.exists(log_file)==False:
        printLog(strlog,log_file)
    for model in ['svm', 'xgboost', 'logit']:
        for base in ['product','user']:
            _,table = getResultLog(model,base,server)
            for prime_inclusion in [True,False]:
                prime_inclusion_str = "Weekday and Time Slot" if prime_inclusion else "Weekday Only"
                cm_mltype_str = 'cm_demo_prime' if prime_inclusion else 'cm_demo_nonprime'
                row = [model,base,prime_inclusion_str]
                for target in ['purchase', 'intent']:
                    for category in [0,1,2,3,4,5]:
                        cm_demo_f1s = [float(trow[10]) for trow in table if trow[1]==target and trow[2]==cm_mltype_str and int(trow[3])==category]
                        demographics_f1s = [float(trow[10]) for trow in table if trow[1]==target and trow[2]=='demographics' and int(trow[3])==category]
                        p = scipy.stats.ttest_ind(demographics_f1s, cm_demo_f1s, equal_var=False)[1]
                        row.append(str(p))
                strlog = ','.join(row)
                printLog(strlog,log_file)

def test_H3(server=True):
    # cm_only vs cm_demo
    log_file = MakeLogFile("Test_H3_cm_only-cm_demo.csv")
    titles = ['Model', 'Base', 'Configuration', 'purchase_0', 'purchase_1','purchase_2', 'purchase_3','purchase_4','purchase_5', 'intent_0', 'intent_1','intent_2', 'intent_3','intent_4','intent_5']
    strlog = ','.join(titles)
    if os.path.exists(log_file)==False:
        printLog(strlog,log_file)
    for model in ['svm', 'xgboost', 'logit']:
        for base in ['product','user']:
            _,table = getResultLog(model,base,server)
            for prime_inclusion in [True,False]:
                prime_inclusion_str = "Weekday and Time Slot" if prime_inclusion else "Weekday Only"
                cm_only_mltype_str = 'cm_only_prime' if prime_inclusion else 'cm_only_nonprime'
                cm_demo_mltype_str = 'cm_demo_prime' if prime_inclusion else 'cm_demo_nonprime'
                row = [model,base,prime_inclusion_str]
                for target in ['purchase', 'intent']:
                    for category in [0,1,2,3,4,5]:
                        cm_demo_f1s = [float(trow[10]) for trow in table if trow[1]==target and trow[2]==cm_demo_mltype_str and int(trow[3])==category]
                        cm_only_f1s = [float(trow[10]) for trow in table if trow[1]==target and trow[2]==cm_only_mltype_str and int(trow[3])==category]
                        p = scipy.stats.ttest_ind(cm_only_f1s, cm_demo_f1s, equal_var=False)[1]
                        row.append(str(p))
                strlog = ','.join(row)
                printLog(strlog,log_file)

def test_H4(server=True):
    # purchase vs intent
    for mltype in ['cm_only_prime', 'cm_only_nonprime', 'demographics', 'cm_demo_prime', 'cm_demo_nonprime']:
        log_file = MakeLogFile("Test_H4_{}_purchase-intent.csv".format(mltype))
        titles = [
            'Model', 
            'Base', 
            'Configuration', 
            '{}_0'.format(mltype),
            '{}_1'.format(mltype),
            '{}_2'.format(mltype),
            '{}_3'.format(mltype),
            '{}_4'.format(mltype),
            '{}_5'.format(mltype)]
        strlog = ','.join(titles)
        if os.path.exists(log_file)==False:
            printLog(strlog,log_file)
        for model in ['svm', 'xgboost', 'logit']:
            for base in ['product','user']:
                _,table = getResultLog(model,base,server)
                for prime_inclusion in [True,False]:
                    prime_inclusion_str = "Weekday and Time Slot" if prime_inclusion else "Weekday Only"
                    row = [model,base,prime_inclusion_str]
                    for category in [0,1,2,3,4,5]:
                        purchase_f1s = [float(trow[10]) for trow in table if trow[1]=='purchase' and trow[2]==mltype and int(trow[3])==category]
                        intent_f1s = [float(trow[10]) for trow in table if trow[1]=='intent' and trow[2]==mltype and int(trow[3])==category]
                        p = scipy.stats.ttest_ind(purchase_f1s, intent_f1s, equal_var=False)[1]
                        row.append(str(p))
                    strlog = ','.join(row)
                    printLog(strlog,log_file)

def test_H5(server=True):
    # prime vs nonprime
    log_file = MakeLogFile("Test_H5_prime-nonprime.csv")
    titles = ['Model', 'Base', 'Model Type', 'purchase_0', 'purchase_1','purchase_2', 'purchase_3','purchase_4','purchase_5', 'intent_0', 'intent_1','intent_2', 'intent_3','intent_4','intent_5']
    strlog = ','.join(titles)
    if os.path.exists(log_file)==False:
        printLog(strlog,log_file)
    for model in ['svm', 'xgboost', 'logit']:
        for base in ['product','user']:
            _,table = getResultLog(model,base,server)
            for mltype in ['cm_only', 'cm_demo']:
                if mltype == 'cm_only':
                    mltype_str = 'Advert Viewing Time'
                if mltype == 'cm_demo':
                    mltype_str = 'Advert Viewint Time and Demographics'
                row = [model,base,mltype_str]
                for target in ['purchase', 'intent']:
                    for category in [0,1,2,3,4,5]:
                        prime_f1s = [float(trow[10]) for trow in table if trow[1]==target and trow[2]==mltype+'_prime' and int(trow[3])==category]
                        nonprime_f1s = [float(trow[10]) for trow in table if trow[1]==target and trow[2]==mltype+'_nonprime' and int(trow[3])==category]
                        p = scipy.stats.ttest_ind(prime_f1s, nonprime_f1s, equal_var=False)[1]
                        row.append(str(p))
                strlog = ','.join(row)
                printLog(strlog,log_file)

def main():
    test_H1()
    test_H2()
    test_H3()
    test_H4()
    test_H5()


if __name__ == '__main__':
    main()
