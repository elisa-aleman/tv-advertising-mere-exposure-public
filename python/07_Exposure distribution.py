#-*- coding: utf-8 -*-

from NomuraSoken_methods import *

def MakeExposureDist(server=True):
	titles = ['Product_ID', 'Monday_Primetime', 'Monday_NonPrimetime', 'Tuesday_Primetime', 'Tuesday_NonPrimetime', 'Wednesday_Primetime', 'Wednesday_NonPrimetime', 'Thursday_Primetime', 'Thursday_NonPrimetime', 'Friday_Primetime', 'Friday_NonPrimetime','Saturday_Primetime', 'Saturday_NonPrimetime','Sunday_Primetime', 'Sunday_NonPrimetime',]
	strlog = ','.join(titles)
	log_file = MakeLogFile('Exposure_distribution.csv')
	printLog(strlog,log_file)
	products = getProductList()
	for product_id in products:
		x = getXvector_product(product_id, mltype='cm_only', prime_inclusion=True, server=server)
		row_x = [str(product_id)]+[str(sum(i)) for i in zip(*x)]
		strlog = ','.join(row_x)
		printLog(strlog,log_file)


def main():
    MakeExposureDist()

if __name__ == '__main__':
    main()