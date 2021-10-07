#-*- coding: utf-8 -*-

from NomuraSoken_methods import *

def make_Xvector_demographics():
    main_titles,main_data = readCSV(getMainDataCSVPath())
    # 0 SampleID
    # 2 AGE (number)
    # 1 SEX_CD (1= male, 2= female)
    # 3 MARRIAGE (1 = single, 2 = married, 3 = divorced or widowed)
    # 4 CHILD_CD (1 = parent, 2 = not parent)
    # 17 INCOM_SA (
    #     1 = No Income,
    #     2 = Under 1,000,000 yen,
    #     3 = From 1,000,000 yen to 2,000,000 yen,
    #     4 = From 2,000,000 yen to 3,000,000 yen,
    #     5 = From 3,000,000 yen to 4,000,000 yen,
    #     6 = From 4,000,000 yen to 5,000,000 yen,
    #     7 = From 5,000,000 yen to 6,000,000 yen,
    #     8 = From 6,000,000 yen to 7,000,000 yen,
    #     9 = From 7,000,000 yen to 10,000,000 yen,
    #     10 = From 10,000,000 yen to 15,000,000 yen,
    #     11 = From 15,000,000 yen to 20,000,000 yen,
    #     12 = Over 20,000,000 yen
    # )
    x = []
    for user in main_data:
        row = [0 for i in range(26)]
        #  0: SampleID
        #  1: age_18 to 25 years old
        #  2: age_26 to 35 years old
        #  3: age_36 to 45 years old
        #  4: age_46 to 55 years old
        #  5: age_56 or older
        #  6: Male
        #  7: Female
        #  8: Single
        #  9: Married
        # 10: Divorced or Widowed
        # 11: Parent
        # 12: Not a Parent
        # 13: Not disclosed
        # 14: No Income
        # 15: income_Under 1,000,000 yen
        # 16: income_From 1,000,000 yen to 2,000,000 yen
        # 17: income_From 2,000,000 yen to 3,000,000 yen
        # 18: income_From 3,000,000 yen to 4,000,000 yen
        # 19: income_From 4,000,000 yen to 5,000,000 yen
        # 20: income_From 5,000,000 yen to 6,000,000 yen
        # 21: income_From 6,000,000 yen to 7,000,000 yen
        # 22: income_From 7,000,000 yen to 10,000,000 yen
        # 23: income_From 10,000,000 yen to 15,000,000 yen
        # 24: income_From 15,000,000 yen to 20,000,000 yen
        # 25: income_Over 20,000,000 yen
        age = int(user[2])
        sex = int(user[1])
        marriage = int(user[3])
        parent = int(user[4])
        income =  user[17]
        if income!=' ':
            income = int(income)
        else:
            income = 0
        row[0] = int(user[0])
        if age>=18 and age<=25: row[1]=1
        if age>=26 and age<=35: row[2]=1
        if age>=36 and age<=45: row[3]=1
        if age>=46 and age<=55: row[4]=1
        if age>=56: row[5]=1
        if sex==1: row[6]=1
        if sex==2: row[7]=1
        if marriage==1: row[8]=1
        if marriage==2: row[9]=1
        if marriage==3: row[10]=1
        if parent==1: row[11]=1
        if parent==2: row[12]=1
        if income==0: row[13]
        if income==1: row[14]=1
        if income==2: row[15]=1
        if income==3: row[16]=1
        if income==4: row[17]=1
        if income==5: row[18]=1
        if income==6: row[19]=1
        if income==7: row[20]=1
        if income==8: row[21]=1
        if income==9: row[22]=1
        if income==10: row[23]=1
        if income==11: row[24]=1
        if income==12: row[25]=1
        x.append(row)
    log_file = MakeLogFile("User_demographics.csv")
    titles = [
        "SampleID",
        "age_18_to_25_years_old",
        "age_26_to_35_years_old",
        "age_36_to_45_years_old",
        "age_46_to_55_years_old",
        "age_56_or_older",
        "Male",
        "Female",
        "Single",
        "Married",
        "Divorced_or_Widowed",
        "Parent",
        "Not_a_Parent",
        "No_Income",
        "income_Under_1000000_yen",
        "income_From_1000000_yen_to_2000000_yen",
        "income_From_2000000_yen_to_3000000_yen",
        "income_From_3000000_yen_to_4000000_yen",
        "income_From_4000000_yen_to_5000000_yen",
        "income_From_5000000_yen_to_6000000_yen",
        "income_From_6000000_yen_to_7000000_yen",
        "income_From_7000000_yen_to_10000000_yen",
        "income_From_10000000_yen_to_15000000_yen",
        "income_From_15000000_yen_to_20000000_yen",
        "income_Over_20000000_yen"
    ]
    strlog = ",".join(titles)
    printSTDlog(strlog, log_file)
    for x_row in x:
        strlog = ",".join([str(val) for val in x_row])
        printSTDlog(strlog,log_file)

def main():
    make_Xvector_demographics()


if __name__ == '__main__':
    main()


