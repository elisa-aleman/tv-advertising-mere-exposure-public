#-*- coding: utf-8 -*-

from NomuraSoken_methods import *


def testSVMs_product(prime_inclusion=False, server=True):
    titles = ['Product_id','Target','Model_type','Category','Precision_Av','Precision_StDv','Recall_Av','Recall_StDv','Accuracy_Av','Accuracy_StDv','F1_Av','F1_StDv']
    if not prime_inclusion:
        log_file = MakeLogFile("SVM_tests_product_PrimeNotIncluded.csv", server=server)
    else:
        log_file = MakeLogFile("SVM_tests_product_PrimeIncluded.csv", server=server)
    strlog = ','.join(titles)
    printLog(strlog,log_file)
    products = getProductList()
    types = ['cm_only','demographics','cm_demo']
    targets = ['purchase','intent']
    categories = [2,4,5,3,0,1]
    k = 5
    for target in targets:
        for category in categories:
            print(target,category)
            for product_id in products:
                for mltype in types:
                    x = getXvector_product(product_id, target=target,mltype=mltype, prime_inclusion=prime_inclusion, server=server)
                    if target=='purchase':
                        y = getYlabel_product_purchase(product_id, category=category, server=server)
                    elif target=='intent':
                        y = getYlabel_product_intent(product_id, category=category, server=server)
                    for tries in range(30):
                        try:
                            r = SVMkfolds(x, y, k)
                            break
                        except ValueError:
                            pass
                            # print("ValueError")
                    if r:
                        # results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
                        log_row = [str(product_id),target,mltype,str(category),str(r[0][0]),str(r[0][1]),str(r[1][0]),str(r[1][1]),str(r[2][0]),str(r[2][1]),str(r[3][0]),str(r[3][1])]
                        strlog = ",".join(log_row)
                        printLog(strlog,log_file)
                    else:
                        print("Skipped because of ValueError limit")


def testSVMs_user(prime_inclusion=False, server=True):
    titles = ['User_id','Target','Model_type','Category','Precision_Av','Precision_StDv','Recall_Av','Recall_StDv','Accuracy_Av','Accuracy_StDv','F1_Av','F1_StDv']
    if not prime_inclusion:
        log_file = MakeLogFile("SVM_tests_user_PrimeNotIncluded.csv", server=server)
    else:
        log_file = MakeLogFile("SVM_tests_user_PrimeIncluded.csv", server=server)
    strlog = ','.join(titles)
    printLog(strlog,log_file)
    _,main_data = readCSV(getMainDataCSVPath(server=server))
    users = [user[0] for user in main_data]
    types = ['cm_only','demographics','cm_demo']
    targets = ['purchase','intent']
    categories = [2,4,5,3,0,1]
    k = 5
    for target in targets:
        for category in categories:
            print(target,category)
            for user_id in users:
                for mltype in types:
                    x = getXvector_user(user_id, target=target, mltype=mltype, prime_inclusion=prime_inclusion, server=server)
                    if target=='purchase':
                        y = getYlabel_user_purchase(user_id, category=category, server=server)
                    elif target=='intent':
                        y = getYlabel_user_intent(user_id, category=category, server=server)
                    for tries in range(30):
                        try:
                            r = SVMkfolds(x, y, k)
                            break
                        except ValueError:
                            pass
                            # print("ValueError")
                    if r:
                        # results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
                        log_row = [user_id,target,mltype,str(category),str(r[0][0]),str(r[0][1]),str(r[1][0]),str(r[1][1]),str(r[2][0]),str(r[2][1]),str(r[3][0]),str(r[3][1])]
                        strlog = ",".join(log_row)
                        printLog(strlog,log_file)
                    else:
                        print("Skipped because of ValueError limit")
            print(target,category)


def testXGBoost_product(prime_inclusion=False, server=True):
    titles = ['Product_id','Target','Model_type','Category','Precision_Av','Precision_StDv','Recall_Av','Recall_StDv','Accuracy_Av','Accuracy_StDv','F1_Av','F1_StDv']
    if not prime_inclusion:
        log_file = MakeLogFile("XGBoost_tests_product_PrimeNotIncluded.csv", server=server)
    else:
        log_file = MakeLogFile("XGBoost_tests_product_PrimeIncluded.csv", server=server)
    strlog = ','.join(titles)
    printLog(strlog,log_file)
    products = getProductList()
    types = ['cm_only','demographics','cm_demo']
    targets = ['purchase','intent']
    categories = [2,4,5,3,0,1]
    k = 5
    for target in targets:
        for category in categories:
            print(target,category)
            for product_id in products:
                for mltype in types:
                    x = getXvector_product(product_id, target=target,mltype=mltype, prime_inclusion=prime_inclusion, server=server)
                    if target=='purchase':
                        y = getYlabel_product_purchase(product_id, category=category, server=server)
                    elif target=='intent':
                        y = getYlabel_product_intent(product_id, category=category, server=server)
                    for tries in range(30):
                        try:
                            r = XGBoost_Kfolds(x, y, k)
                            break
                        except ValueError:
                            pass
                            # print("ValueError")
                    if r:
                        # results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
                        log_row = [str(product_id),target,mltype,str(category),str(r[0][0]),str(r[0][1]),str(r[1][0]),str(r[1][1]),str(r[2][0]),str(r[2][1]),str(r[3][0]),str(r[3][1])]
                        strlog = ",".join(log_row)
                        printLog(strlog,log_file)
                    else:
                        print('Skipped because of ValueError limit')


def testXGBoost_user(prime_inclusion=False, server=True):
    titles = ['User_id','Target','Model_type','Category','Precision_Av','Precision_StDv','Recall_Av','Recall_StDv','Accuracy_Av','Accuracy_StDv','F1_Av','F1_StDv']
    if not prime_inclusion:
        log_file = MakeLogFile("XGBoost_tests_user_PrimeNotIncluded.csv", server=server)
    else:
        log_file = MakeLogFile("XGBoost_tests_user_PrimeIncluded.csv", server=server)
    strlog = ','.join(titles)
    printLog(strlog,log_file)
    _,main_data = readCSV(getMainDataCSVPath(server=server))
    users = [user[0] for user in main_data]
    types = ['cm_only','demographics','cm_demo']
    targets = ['purchase','intent']
    categories = [2,4,5,3,0,1]
    k = 5
    for target in targets:
        for category in categories:
            print(target,category)
            for user_id in users:
                for mltype in types:
                    x = getXvector_user(user_id, target=target, mltype=mltype, prime_inclusion=prime_inclusion, server=server)
                    if target=='purchase':
                        y = getYlabel_user_purchase(user_id, category=category, server=server)
                    elif target=='intent':
                        y = getYlabel_user_intent(user_id, category=category, server=server)
                    for tries in range(30):
                        try:
                            r = XGBoost_Kfolds(x, y, k)
                            break
                        except ValueError:
                            pass
                            # print("ValueError")
                    if r:
                        # results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
                        log_row = [user_id,target,mltype,str(category),str(r[0][0]),str(r[0][1]),str(r[1][0]),str(r[1][1]),str(r[2][0]),str(r[2][1]),str(r[3][0]),str(r[3][1])]
                        strlog = ",".join(log_row)
                        printLog(strlog,log_file)
                    else:
                        print('Skipped because of ValueError limit')


def testLogisticRegression_product(prime_inclusion=False, server=True):
    titles = ['Product_id','Target','Model_type','Category','Precision_Av','Precision_StDv','Recall_Av','Recall_StDv','Accuracy_Av','Accuracy_StDv','F1_Av','F1_StDv']
    if not prime_inclusion:
        log_file = MakeLogFile("LogisticRegression_tests_product_PrimeNotIncluded.csv", server=server)
    else:
        log_file = MakeLogFile("LogisticRegression_tests_product_PrimeIncluded.csv", server=server)
    strlog = ','.join(titles)
    printLog(strlog,log_file)
    products = getProductList()
    types = ['cm_only','demographics','cm_demo']
    targets = ['purchase','intent']
    categories = [2,4,5,3,0,1]
    k = 5
    for target in targets:
        for category in categories:
            print(target,category)
            for product_id in products:
                for mltype in types:
                    x = getXvector_product(product_id, target=target,mltype=mltype, prime_inclusion=prime_inclusion, server=server)
                    if target=='purchase':
                        y = getYlabel_product_purchase(product_id, category=category, server=server)
                    elif target=='intent':
                        y = getYlabel_product_intent(product_id, category=category, server=server)
                    for tries in range(30):
                        try:
                            r = LogisticRegression_Kfolds(x, y, k)
                            break
                        except ValueError:
                            pass
                            # print("ValueError")
                    if r:
                        # results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
                        log_row = [str(product_id),target,mltype,str(category),str(r[0][0]),str(r[0][1]),str(r[1][0]),str(r[1][1]),str(r[2][0]),str(r[2][1]),str(r[3][0]),str(r[3][1])]
                        strlog = ",".join(log_row)
                        printLog(strlog,log_file)
                    else:
                        print('Skipped because of ValueError limit')


def testLogisticRegression_user(prime_inclusion=False, server=True):
    titles = ['User_id','Target','Model_type','Category','Precision_Av','Precision_StDv','Recall_Av','Recall_StDv','Accuracy_Av','Accuracy_StDv','F1_Av','F1_StDv']
    if not prime_inclusion:
        log_file = MakeLogFile("LogisticRegression_tests_user_PrimeNotIncluded.csv", server=server)
    else:
        log_file = MakeLogFile("LogisticRegression_tests_user_PrimeIncluded.csv", server=server)
    strlog = ','.join(titles)
    printLog(strlog,log_file)
    _,main_data = readCSV(getMainDataCSVPath(server=server))
    users = [user[0] for user in main_data]
    types = ['cm_only','demographics','cm_demo']
    targets = ['purchase','intent']
    categories = [2,4,5,3,0,1]
    k = 5
    for target in targets:
        for category in categories:
            print(target,category)
            for user_id in users:
                for mltype in types:
                    x = getXvector_user(user_id, target=target, mltype=mltype, prime_inclusion=prime_inclusion, server=server)
                    if target=='purchase':
                        y = getYlabel_user_purchase(user_id, category=category, server=server)
                    elif target=='intent':
                        y = getYlabel_user_intent(user_id, category=category, server=server)
                    for tries in range(30):
                        try:
                            r = LogisticRegression_Kfolds(x, y, k)
                            break
                        except ValueError:
                            pass
                            # print("ValueError")
                    if r:
                        # results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
                        log_row = [user_id,target,mltype,str(category),str(r[0][0]),str(r[0][1]),str(r[1][0]),str(r[1][1]),str(r[2][0]),str(r[2][1]),str(r[3][0]),str(r[3][1])]
                        strlog = ",".join(log_row)
                        printLog(strlog,log_file)
                    else:
                        print('Skipped because of ValueError limit')


def main():
    # testSVMs_product(prime_inclusion=False)
    # testSVMs_user(prime_inclusion=False)
    # testXGBoost_product(prime_inclusion=False)
    # testXGBoost_user(prime_inclusion=False)
    # testLogisticRegression_product(prime_inclusion=False)
    # testLogisticRegression_user(prime_inclusion=False)
    testSVMs_product(prime_inclusion=True)
    testSVMs_user(prime_inclusion=True)
    testXGBoost_product(prime_inclusion=True)
    testXGBoost_user(prime_inclusion=True)
    testLogisticRegression_product(prime_inclusion=True)
    testLogisticRegression_user(prime_inclusion=True)

if __name__ == '__main__':
    main()