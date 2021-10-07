#-*- coding: utf-8 -*-

from NomuraSoken_methods import *

def testSVMs_purchase_product():
    titles = ['Product_id','Machine_type','Purchase_Category','Precision_Av','Precision_StDv','Recall_Av','Recall_StDv','Accuracy_Av','Accuracy_StDv','F1_Av','F1_StDv']
    log_file = "/home/ealeman/SVM_tests.csv"
    strlog = ','.join(titles)
    printSTDlog(strlog,log_file)
    products = getProductList()
    types = ['total_only','day_prime','combined']
    categories = [2,4,5,3,0,1]
    k = 5
    for category in categories:
        for product_id in products:
            for mltype in types:
                x = getXvector_product(product_id, mltype=mltype)
                y = getYlabel_product_purchase(product_id, category=category)
                r = SVMkfolds(x, y, k)
                # results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
                log_row = [str(product_id),mltype,str(category),str(r[0][0]),str(r[0][1]),str(r[1][0]),str(r[1][1]),str(r[2][0]),str(r[2][1]),str(r[3][0]),str(r[3][1])]
                strlog = ",".join(log_row)
                printSTDlog(strlog,log_file)

def testSVMs_purchase_user():
    titles = ['User_id','Machine_type','Purchase_Category','Precision_Av','Precision_StDv','Recall_Av','Recall_StDv','Accuracy_Av','Accuracy_StDv','F1_Av','F1_StDv']
    log_file = "/home/ealeman/SVM_tests_by_user.csv"
    strlog = ','.join(titles)
    printSTDlog(strlog,log_file)
    _,main_data = readCSV(getMainDataCSVPath())
    users = [user[0] for user in main_data]
    types = ['total_only','day_prime','combined']
    categories = [2,4,5,3,0,1]
    k = 4
    for category in categories:
        for user_id in users:
            for mltype in types:
                x = getXvector_user(user_id, mltype=mltype)
                y = getYlabel_user_purchase(user_id, category=category)
                for tries in xrange(30):
                    try:
                        r = SVMkfolds(x, y, k)
                        break
                    except ValueError:
                        print "ValueError"
                # results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
                log_row = [user_id,mltype,str(category),str(r[0][0]),str(r[0][1]),str(r[1][0]),str(r[1][1]),str(r[2][0]),str(r[2][1]),str(r[3][0]),str(r[3][1])]
                strlog = ",".join(log_row)
                printSTDlog(strlog,log_file)

def testSVMs_intent_product():
    titles = ['Product_id','Machine_type','Purchase_Category','Precision_Av','Precision_StDv','Recall_Av','Recall_StDv','Accuracy_Av','Accuracy_StDv','F1_Av','F1_StDv']
    log_file = "/home/ealeman/SVM_tests_iyoku.csv"
    strlog = ','.join(titles)
    printSTDlog(strlog,log_file)
    products = getProductList()
    types = ['total_only','day_prime','combined']
    categories = [2,4,5,3,0,1]
    k = 5
    for category in categories:
        for product_id in products:
            for mltype in types:
                x = getXvector_product(product_id, mltype=mltype)
                y = getYlabel_product_intent(product_id, category=category)
                r = SVMkfolds(x, y, k)
                # results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
                log_row = [str(product_id),mltype,str(category),str(r[0][0]),str(r[0][1]),str(r[1][0]),str(r[1][1]),str(r[2][0]),str(r[2][1]),str(r[3][0]),str(r[3][1])]
                strlog = ",".join(log_row)
                printSTDlog(strlog,log_file)

def testSVMs_intent_user():
    titles = ['User_id','Machine_type','Purchase_Category','Precision_Av','Precision_StDv','Recall_Av','Recall_StDv','Accuracy_Av','Accuracy_StDv','F1_Av','F1_StDv']
    log_file = "/home/ealeman/SVM_tests_iyoku_by_user.csv"
    strlog = ','.join(titles)
    printSTDlog(strlog,log_file)
    _,main_data = readCSV(getMainDataCSVPath())
    users = [user[0] for user in main_data]
    types = ['total_only','day_prime','combined']
    categories = [2,4,5,3,0,1]
    k = 5
    for category in categories:
        for user_id in users:
            for mltype in types:
                x = getXvector_user(user_id, mltype=mltype)
                y = getYlabel_user_intent(user_id, category=category)
                r = None
                for tries in xrange(30):
                    try:
                        r = SVMkfolds(x, y, k)
                        break
                    except ValueError:
                        print "ValueError"
                if r:
                    # results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
                    log_row = [user_id,mltype,str(category),str(r[0][0]),str(r[0][1]),str(r[1][0]),str(r[1][1]),str(r[2][0]),str(r[2][1]),str(r[3][0]),str(r[3][1])]
                    strlog = ",".join(log_row)
                    printSTDlog(strlog,log_file)
                else:
                    print "Skipped because of ValueError limit"

def main():
    testSVMs_purchase_product()
    testSVMs_purchase_user()
    testSVMs_intent_product()
    testSVMs_intent_user()

if __name__ == '__main__':
    main()