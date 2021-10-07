#-*- coding: utf-8 -*-

from NomuraSoken_methods import *

####################################
############ Y_Purchase ############
####################################

### ALWAYS USE FOR original CSV data, not vectorized
### returns list of users with their category for that product
def purchase_label_users(original_data, product_id): 
    label_csv_path = getProductLabelCSVPath()
    _,label_csv = readCSV(label_csv_path)
    product_index_to_check = ((product_id*4)+841, (product_id*4)+843)
    label_checkers = (label_csv[(product_id*4)+1][3], label_csv[(product_id*4)+3][3])
    labeled_users = []
    for row in original_data:
        user_id = row[0]
        purchase_answers = (row[product_index_to_check[0]],row[product_index_to_check[1]])
        if purchase_answers[0]<label_checkers[0] and purchase_answers[1]>=label_checkers[1]:
            label = 0
        elif purchase_answers[0]>=label_checkers[0] and purchase_answers[1]>=label_checkers[1]:
            label = 1
        elif purchase_answers[0]>=label_checkers[0] and purchase_answers[1]<label_checkers[1]:
            label = 2
        elif purchase_answers[0]<label_checkers[0] and purchase_answers[1]<label_checkers[1]:
            label = 3
        labeled_users.append((user_id, label))
    return labeled_users

def Make_Label_Table_purchase():
    _,data = readCSV(getMainDataCSVPath())
    products = getProductID_dict()
    label_table = []
    users = [row[0] for row in data]
    label_table.append(users)
    for product_id in xrange(200):
        labels_raw = purchase_label_users(data,product_id)
        labels = [lable_row[1] for lable_row in labels_raw]
        label_table.append(labels)
    label_table = map(list, zip(*label_table))
    label_titles = ["{}_category".format(products[i]) for i in xrange(200)]
    label_titles = ["SampleID"]+label_titles
    label_table_csv = [label_titles]+label_table
    log_file = MakeLogFile('User_ClassLabel_by_Product.csv')
    strlog = ','.join(label_titles)
    printLog(strlog,log_file)
    for in_row in label_table:
        strlog = ",".join([unicode(str(x),'utf-8') for x in in_row])
        printLog(strlog,log_file)

##################################
############ Y_Intent ############
##################################

def intent_label_users(original_data, product_id): 
    label_csv_path = getProductLabelCSVPath()
    _,label_csv = readCSV(label_csv_path)
    product_index_to_check = ((product_id*4)+840, (product_id*4)+842)
    label_checkers = (label_csv[(product_id*4)][3], label_csv[(product_id*4)+2][3])
    labeled_users = []
    for row in original_data:
        user_id = row[0]
        purchase_answers = (row[product_index_to_check[0]],row[product_index_to_check[1]])
        if purchase_answers[0]<label_checkers[0] and purchase_answers[1]>=label_checkers[1]:
            label = 0
        elif purchase_answers[0]>=label_checkers[0] and purchase_answers[1]>=label_checkers[1]:
            label = 1
        elif purchase_answers[0]>=label_checkers[0] and purchase_answers[1]<label_checkers[1]:
            label = 2
        elif purchase_answers[0]<label_checkers[0] and purchase_answers[1]<label_checkers[1]:
            label = 3
        labeled_users.append((user_id, label))
    return labeled_users

def Make_Label_Table_intent():
    _,data = readCSV(getMainDataCSVPath())
    products = getProductID_dict()
    label_table = []
    users = [row[0] for row in data]
    label_table.append(users)
    for product_id in xrange(200):
        labels_raw = intent_label_users(data,product_id)
        labels = [lable_row[1] for lable_row in labels_raw]
        label_table.append(labels)
    label_table = map(list, zip(*label_table))
    label_titles = ["{}_category".format(products[i]) for i in xrange(200)]
    label_titles = ["SampleID"]+label_titles
    label_table_csv = [label_titles]+label_table
    log_file = MakeLogFile('User_ClassLabel_by_Product_iyoku.csv')
    strlog = ','.join(label_titles)
    printLog(strlog,log_file)
    for in_row in label_table:
        strlog = ",".join([unicode(str(x),'utf-8') for x in in_row])
        printLog(strlog,log_file)

def main():
    Make_Label_Table_purchase()
    Make_Label_Table_intent()

if __name__ == '__main__':
    main()