#-*- coding: utf-8 -*-

from NomuraSoken_methods import *

#####################
### Product Based ###
#####################

def getXvector_product_times(product_id, mltype='day_prime'):
    cminfo_titles,cminfo_table = getCMinfo(product_id)
    ### Total
    total_tvban_set = set(cminfo_titles[1:])
    ### Days
    monday_tvban_set = set([tvban for tvban in total_tvban_set if getTVdate(tvban).weekday()==0])
    tuesday_tvban_set = set([tvban for tvban in total_tvban_set if getTVdate(tvban).weekday()==1])
    wednesday_tvban_set = set([tvban for tvban in total_tvban_set if getTVdate(tvban).weekday()==2])
    thursday_tvban_set = set([tvban for tvban in total_tvban_set if getTVdate(tvban).weekday()==3])
    friday_tvban_set = set([tvban for tvban in total_tvban_set if getTVdate(tvban).weekday()==4])
    saturday_tvban_set = set([tvban for tvban in total_tvban_set if getTVdate(tvban).weekday()==5])
    sunday_tvban_set = set([tvban for tvban in total_tvban_set if getTVdate(tvban).weekday()==6])
    heijitsu_tvban_set = monday_tvban_set|tuesday_tvban_set|wednesday_tvban_set|thursday_tvban_set|friday_tvban_set
    weekend_tvban_set = saturday_tvban_set|sunday_tvban_set
    # Holidays
    #http://www.officeholidays.com/countries/japan/index.php
    holidays_str = ['2017/01/09','2017/03/20']
    holidays = [datetime.strptime(holiday_date, '%Y/%m/%d') for holiday_date in holidays_str]
    holiday_tvban_set = set([tvban for tvban in total_tvban_set if getTVdate(tvban) in holidays])
    rest_day_tvban_set = weekend_tvban_set & holiday_tvban_set
    ###
    days_tvban_sets = [monday_tvban_set,tuesday_tvban_set,wednesday_tvban_set,thursday_tvban_set,friday_tvban_set,saturday_tvban_set,sunday_tvban_set,heijitsu_tvban_set,weekend_tvban_set,holiday_tvban_set,rest_day_tvban_set]
    ###
    _,ptgp_table = readCSV(getProgPrimeCSVPath())
    ### Prime Status
    prime_tvban_set = total_tvban_set & set([ptgprow[0] for ptgprow in ptgp_table if ptgprow[4]=='1'])
    nonprime_tvban_set = total_tvban_set - prime_tvban_set
    ###
    prime_status_tvban_sets = [prime_tvban_set,nonprime_tvban_set]
    ###
    data_tvban_sets = [a&b for a in days_tvban_sets for b in prime_status_tvban_sets]+days_tvban_sets+prime_status_tvban_sets+[total_tvban_set]
    ###
    _,cmtime_csv = readCSV(getCM_time_count_CSVPath())
    pre_x = []
    for user in cminfo_table:
        user_tvban_set = set([title for num,title in enumerate(cminfo_titles) if user[num]=='1'])
        sum_tvban_sets = [ds&user_tvban_set for ds in data_tvban_sets]
        cmtimes_by_set = [len(tvban_set) for tvban_set in sum_tvban_sets]
        pre_x.append(cmtimes_by_set)
    if mltype=='total_only':
        Xvector = [[row[35]] for row in pre_x]
    elif mltype=='day_prime':
        Xvector = [row[0:22] for row in pre_x]
    elif mltype=='combined':
        Xvector = [row[0:22]+[row[35]] for row in pre_x]
    return Xvector


#### X
def create_x_vector_csv_purchase_product(products):
    x_titles = getXtitles()
    titles = ['SampleID','Product_id']+x_titles
    strlog = ','.join(titles)
    processed_data_folder_path = getProcessedDataFolderPath(server=server)
    log_file = os.path.join(processed_data_folder_path, "Xvector_cmtime_by_product.csv")
    printSTDlog(strlog,log_file)
    _,main_data = readCSV(getMainDataCSVPath())
    users = [user[0] for user in main_data]
    for product_id in products:
        x = getXvector_product_times(product_id)
        in_x = [[users[num],product_id]+xrow for num,xrow in enumerate(x)]
        for in_row in in_x:
            strlog = ','.join([str(val) for val in in_row])
            printSTDlog(strlog,log_file)

#### Y
def create_y_vector_csv_purchase_product(products):
    y_titles = ['SampleID','Product_id','Y_Category0','Y_Category1','Y_Category2','Y_Category3','Y_Category4','Y_Category5']
    strlog = ','.join(y_titles)
    processed_data_folder_path = getProcessedDataFolderPath(server=server)
    log_file = os.path.join(processed_data_folder_path, "Yvector_purchase_status_by_product.csv")
    printSTDlog(strlog,log_file)
    _,main_data = readCSV(getMainDataCSVPath())
    users = [user[0] for user in main_data]
    for product_id in products:
        ys = [getYlabel(product_id, category=cat) for cat in xrange(6)]
        y = [list(i) for i in zip(*ys)]
        in_y = [[users[num],product_id]+yrow for num,yrow in enumerate(y)]
        for in_row in in_y:
            strlog = ','.join([str(val) for val in in_row])
            printSTDlog(strlog,log_file)


def main():
    products = getProductList()
    create_x_vector_csv_purchase_product(products)
    create_y_vector_csv_purchase_product(products)

if __name__ == '__main__':
    main()

