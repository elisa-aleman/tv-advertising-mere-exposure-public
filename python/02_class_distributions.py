#-*- coding: utf-8 -*-

from NomuraSoken_methods import *


def Make_Class_purchase_Pivot_csv_200():
    products = range(200)
    product_dict = getProductID_dict()
    _,label_data=readCSV(getUserLabelCSVPath())
    log_file = MakeLogFile("Class_purchase_Pivot_200.csv")
    titles = ['Product_ID','Product_Name','Category_0_counts','Category_1_counts','Category_2_counts','Category_3_counts','Category_4_counts','Category_5_counts']
    strlog = ','.join(titles)
    printSTDlog(strlog,log_file)
    for product_id in products:
        Y = [int(row[product_id+1]) for row in label_data]
        cat0_counts = Y.count(0)
        cat1_counts = Y.count(1)
        cat2_counts = Y.count(2)
        cat3_counts = Y.count(3)
        cat4_counts = cat2_counts+cat3_counts
        cat5_counts = cat0_counts+cat1_counts
        product_name = product_dict[product_id]
        strlog = "{},{},{},{},{},{},{},{}".format(product_id,product_name,cat0_counts,cat1_counts,cat2_counts,cat3_counts,cat4_counts,cat5_counts)     
        printSTDlog(strlog,log_file)


def Make_Class_purchase_Pivot_csv_38():
    products = getProductList()
    product_dict = getProductID_dict()
    _,label_data=readCSV(getUserLabelCSVPath())
    log_file = MakeLogFile("Class_purchase_Pivot_38.csv")
    titles = ['Product_ID','Product_Name','Category_0_counts','Category_1_counts','Category_2_counts','Category_3_counts','Category_4_counts','Category_5_counts']
    strlog = ','.join(titles)
    printSTDlog(strlog,log_file)
    for product_id in products:
        Y = [int(row[product_id+1]) for row in label_data]
        cat0_counts = Y.count(0)
        cat1_counts = Y.count(1)
        cat2_counts = Y.count(2)
        cat3_counts = Y.count(3)
        cat4_counts = cat2_counts+cat3_counts
        cat5_counts = cat0_counts+cat1_counts
        product_name = product_dict[product_id]
        strlog = "{},{},{},{},{},{},{},{}".format(product_id,product_name,cat0_counts,cat1_counts,cat2_counts,cat3_counts,cat4_counts,cat5_counts)     
        printSTDlog(strlog,log_file)


def Make_Class_intent_Pivot_csv_200():
    products = range(200)
    product_dict = getProductID_dict()
    _,label_data=readCSV(getUserLabelCSVPath_iyoku())
    log_file = MakeLogFile("Class_intent_Pivot_200.csv")
    titles = ['Product_ID','Product_Name','Category_0_counts','Category_1_counts','Category_2_counts','Category_3_counts','Category_4_counts','Category_5_counts']
    strlog = ','.join(titles)
    printSTDlog(strlog,log_file)
    for product_id in products:
        Y = [int(row[product_id+1]) for row in label_data]
        cat0_counts = Y.count(0)
        cat1_counts = Y.count(1)
        cat2_counts = Y.count(2)
        cat3_counts = Y.count(3)
        cat4_counts = cat2_counts+cat3_counts
        cat5_counts = cat0_counts+cat1_counts
        product_name = product_dict[product_id]
        strlog = "{},{},{},{},{},{},{},{}".format(product_id,product_name,cat0_counts,cat1_counts,cat2_counts,cat3_counts,cat4_counts,cat5_counts)     
        printSTDlog(strlog,log_file)


def Make_Class_intent_Pivot_csv_38():
    products = getProductList()
    product_dict = getProductID_dict()
    _,label_data=readCSV(getUserLabelCSVPath_iyoku())
    log_file = MakeLogFile("Class_intent_Pivot_38.csv")
    titles = ['Product_ID','Product_Name','Category_0_counts','Category_1_counts','Category_2_counts','Category_3_counts','Category_4_counts','Category_5_counts']
    strlog = ','.join(titles)
    printSTDlog(strlog,log_file)
    for product_id in products:
        Y = [int(row[product_id+1]) for row in label_data]
        cat0_counts = Y.count(0)
        cat1_counts = Y.count(1)
        cat2_counts = Y.count(2)
        cat3_counts = Y.count(3)
        cat4_counts = cat2_counts+cat3_counts
        cat5_counts = cat0_counts+cat1_counts
        product_name = product_dict[product_id]
        strlog = "{},{},{},{},{},{},{},{}".format(product_id,product_name,cat0_counts,cat1_counts,cat2_counts,cat3_counts,cat4_counts,cat5_counts)     
        printSTDlog(strlog,log_file)

def Make_Class_purchase_Pivot_csv_200_percent():
    products = range(200)
    product_dict = getProductID_dict()
    _,label_data=readCSV(getUserLabelCSVPath())
    log_file = MakeLogFile("Class_purchase_percent_Pivot_200.csv")
    titles = ['Product_ID','Product_Name','Category_0_percent','Category_1_percent','Category_2_percent','Category_3_percent','Category_4_percent','Category_5_percent']
    strlog = ','.join(titles)
    printSTDlog(strlog,log_file)
    for product_id in products:
        Y = [int(row[product_id+1]) for row in label_data]
        cat0_counts = (Y.count(0))/3000
        cat1_counts = (Y.count(1))/3000
        cat2_counts = (Y.count(2))/3000
        cat3_counts = (Y.count(3))/3000
        cat4_counts = (Y.count(2)+Y.count(3))/3000
        cat5_counts = (Y.count(0)+Y.count(1))/3000
        product_name = product_dict[product_id]
        strlog = "{},{},{},{},{},{},{},{}".format(product_id,product_name,cat0_counts,cat1_counts,cat2_counts,cat3_counts,cat4_counts,cat5_counts)     
        printSTDlog(strlog,log_file)


def Make_Class_purchase_Pivot_csv_38_percent():
    products = getProductList()
    product_dict = getProductID_dict()
    _,label_data=readCSV(getUserLabelCSVPath())
    log_file = MakeLogFile("Class_purchase_percent_Pivot_38.csv")
    titles = ['Product_ID','Product_Name','Category_0_percent','Category_1_percent','Category_2_percent','Category_3_percent','Category_4_percent','Category_5_percent']
    strlog = ','.join(titles)
    printSTDlog(strlog,log_file)
    for product_id in products:
        Y = [int(row[product_id+1]) for row in label_data]
        cat0_counts = (Y.count(0))/3000
        cat1_counts = (Y.count(1))/3000
        cat2_counts = (Y.count(2))/3000
        cat3_counts = (Y.count(3))/3000
        cat4_counts = (Y.count(2)+Y.count(3))/3000
        cat5_counts = (Y.count(0)+Y.count(1))/3000
        product_name = product_dict[product_id]
        strlog = "{},{},{},{},{},{},{},{}".format(product_id,product_name,cat0_counts,cat1_counts,cat2_counts,cat3_counts,cat4_counts,cat5_counts)     
        printSTDlog(strlog,log_file)


def Make_Class_intent_Pivot_csv_200_percent():
    products = range(200)
    product_dict = getProductID_dict()
    _,label_data=readCSV(getUserLabelCSVPath_iyoku())
    log_file = MakeLogFile("Class_intent_percent_Pivot_200.csv")
    titles = ['Product_ID','Product_Name','Category_0_percent','Category_1_percent','Category_2_percent','Category_3_percent','Category_4_percent','Category_5_percent']
    strlog = ','.join(titles)
    printSTDlog(strlog,log_file)
    for product_id in products:
        Y = [int(row[product_id+1]) for row in label_data]
        cat0_counts = (Y.count(0))/3000
        cat1_counts = (Y.count(1))/3000
        cat2_counts = (Y.count(2))/3000
        cat3_counts = (Y.count(3))/3000
        cat4_counts = (Y.count(2)+Y.count(3))/3000
        cat5_counts = (Y.count(0)+Y.count(1))/3000
        product_name = product_dict[product_id]
        strlog = "{},{},{},{},{},{},{},{}".format(product_id,product_name,cat0_counts,cat1_counts,cat2_counts,cat3_counts,cat4_counts,cat5_counts)     
        printSTDlog(strlog,log_file)


def Make_Class_intent_Pivot_csv_38_percent():
    products = getProductList()
    product_dict = getProductID_dict()
    _,label_data=readCSV(getUserLabelCSVPath_iyoku())
    log_file = MakeLogFile("Class_intent_percent_Pivot_38.csv")
    titles = ['Product_ID','Product_Name','Category_0_percent','Category_1_percent','Category_2_percent','Category_3_percent','Category_4_percent','Category_5_percent']
    strlog = ','.join(titles)
    printSTDlog(strlog,log_file)
    for product_id in products:
        Y = [int(row[product_id+1]) for row in label_data]
        cat0_counts = (Y.count(0))/3000
        cat1_counts = (Y.count(1))/3000
        cat2_counts = (Y.count(2))/3000
        cat3_counts = (Y.count(3))/3000
        cat4_counts = (Y.count(2)+Y.count(3))/3000
        cat5_counts = (Y.count(0)+Y.count(1))/3000
        product_name = product_dict[product_id]
        strlog = "{},{},{},{},{},{},{},{}".format(product_id,product_name,cat0_counts,cat1_counts,cat2_counts,cat3_counts,cat4_counts,cat5_counts)     
        printSTDlog(strlog,log_file)

def main():
    # Make_Class_purchase_Pivot_csv_200()
    # Make_Class_purchase_Pivot_csv_38()
    # Make_Class_intent_Pivot_csv_200()
    # Make_Class_intent_Pivot_csv_38()
    Make_Class_purchase_Pivot_csv_200_percent()
    Make_Class_purchase_Pivot_csv_38_percent()
    Make_Class_intent_Pivot_csv_200_percent()
    Make_Class_intent_Pivot_csv_38_percent()


if __name__ == '__main__':
    main()

