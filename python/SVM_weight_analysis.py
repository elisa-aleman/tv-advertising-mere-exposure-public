
from NomuraSoken_methods import *

##########################################################
######## IYOKU Category 4 product weight analysis ########
##########################################################

def weights():
    csvtitles = ['Product_id','Machine_type','Purchase_Category','Feature','Weight']
    log_file = "/home/ealeman/SVM_iyoku_goodProducts_weights.csv"
    strlog = ','.join(csvtitles)
    printSTDlog(strlog,log_file)
    products = [29,154,26,147,30,163,155,157]
    category = 4
    titles = getXtitles()
    features = titles[0:22] + [titles[35]]
    mltype = 'combined'
    for product_id in products:
        x = getXvector(product_id, mltype=mltype)
        y = getYlabel_iyoku(product_id, category=category)
        inf = SVM_weights(x, y, features)
        for feature,weight in inf:
            strlog = '{},{},{},{},{}'.format(product_id,mltype,category,feature,weight)
            printSTDlog(strlog,log_file)

def times_confirm():
    titles = ['Product_id','Machine_type','Purchase_Category','Precision_Av','Precision_StDv','Recall_Av','Recall_StDv','Accuracy_Av','Accuracy_StDv','F1_Av','F1_StDv']
    log_file = "/home/ealeman/SVM_tests_iyoku_times_small.csv"
    strlog = ','.join(titles)
    printSTDlog(strlog,log_file)
    products = [29,154,26,147,30,163,155,157]
    category = 4
    k = 5
    mltype = 'combined'
    for product_id in products:
        x = getXvector_times(product_id, mltype=mltype)
        y = getYlabel_iyoku(product_id, category=category)
        r = SVMkfolds(x, y, k, kernel = 'linear', C = 1.0, gamma = 0.001, times = 1)
        # results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
        log_row = [str(product_id),mltype,str(category),str(r[0][0]),str(r[0][1]),str(r[1][0]),str(r[1][1]),str(r[2][0]),str(r[2][1]),str(r[3][0]),str(r[3][1])]
        strlog = ",".join(log_row)
        printSTDlog(strlog,log_file)

def testSVMs_iyoku_times():
    titles = ['Product_id','Machine_type','Purchase_Category','Precision_Av','Precision_StDv','Recall_Av','Recall_StDv','Accuracy_Av','Accuracy_StDv','F1_Av','F1_StDv']
    log_file = "/home/ealeman/SVM_tests_iyoku_times.csv"
    strlog = ','.join(titles)
    printSTDlog(strlog,log_file)
    products = getProductList()
    types = ['total_only','day_prime','combined']
    categories = [2,4,5,3,0,1]
    # categories = [1]
    k = 5
    for category in categories:
        for product_id in products:
            for mltype in types:
                x = getXvector_times(product_id, mltype=mltype)
                y = getYlabel_iyoku(product_id, category=category)
                r = SVMkfolds(x, y, k)
                # results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
                log_row = [str(product_id),mltype,str(category),str(r[0][0]),str(r[0][1]),str(r[1][0]),str(r[1][1]),str(r[2][0]),str(r[2][1]),str(r[3][0]),str(r[3][1])]
                strlog = ",".join(log_row)
                printSTDlog(strlog,log_file)

def main():
    weights()
    times_confirm()
    testSVMs_iyoku_times()

if __name__ == '__main__':
    main()


