#-*- coding: utf-8 -*-

from NomuraSoken_methods import *
from datetime import datetime, time

def is_time_between(begin_time, end_time, check_time=None):
    # If check time is not given, default to current UTC time
    check_time = check_time or datetime.utcnow().time()
    if begin_time < end_time:
        return check_time >= begin_time and check_time <= end_time
    else: # crosses midnight
        return check_time >= begin_time or check_time <= end_time

def Make_Broadcast_distributions_Product_38():
    products = getProductList()
    product_dict = getProductID_dict()
    log_file = MakeLogFile("Broadcast_distributions_product_38.csv")
    titles = ['Product_ID','Product_Name','Broadcast_3_to_7','Broadcast_7_to_11','Broadcast_11_to_15','Broadcast_15_to_19','Broadcast_19_to_23','Broadcast_23_to_3']
    strlog = ','.join(titles)
    printSTDlog(strlog,log_file)
    brakets = [
        (3,  7),
        (7,  11),
        (11, 15),
        (15, 19),
        (19, 23),
        (23, 3)
    ]
    for product_id in products:
        braket_counts = {(3,7):0,(7,11):0,(11,15):0,(15,19):0,(19,23):0,(23,3):0}
        tvwatch_ids,_ = getCMinfo(product_id)
        tvwatch_ids = tvwatch_ids[1:]
        for tvwatch_id in tvwatch_ids:
            tv_start_time,_ = getTVtimes(tvwatch_id)
            for braket in brakets:
                if is_time_between(time(braket[0]),time(braket[1]),tv_start_time):
                    braket_counts[(braket[0],braket[1])] += 1
        product_name = product_dict[product_id]
        strlog = "{},{},{},{},{},{},{},{}".format(product_id,product_name,braket_counts[(3,7)],braket_counts[(7,11)],braket_counts[(11,15)],braket_counts[(15,19)],braket_counts[(19,23)],braket_counts[(23,3)])
        printSTDlog(strlog,log_file)

def main():
    Make_Broadcast_distributions_Product_38()

if __name__ == '__main__':
    main()
