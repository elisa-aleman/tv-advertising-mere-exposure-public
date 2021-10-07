#-*- coding: utf-8 -*-

from NomuraSoken_methods import *

def Vectorize_Demographics_OLD(data,titles):
    data = [orig_row[1:840] for orig_row in data]
    titles = titles[1:840]
    data_T = map(list,zip(*data))
    ### Dividing discrete data into categories
    ### Split continuous data into ranges
    # Split age ones by 12,19,29,39,49,59,69,70
    ages = [1,5,6,7,8,9]
    data_T_sets = []
    for num,col in enumerate(data_T):
        if num in ages:
            data_T_sets.append([12,19,29,39,49,59,60])
        else:
            data_T_sets.append(sorted(list(set(col))))
    ####
    newtitles = [["{}__{}".format(title,val) if val!=-1 else "{}".format(title) for val in vset] for title,vset in zip(titles,data_T_sets)]
    newdata = []
    for row in data:
        newrow = []
        for column_num, value in enumerate(row):
            newtitle_range = newtitles[column_num]
            if column_num in ages:
                if value<=12:
                    newrow+= [1,0,0,0,0,0,0]
                elif value<=19:
                    newrow+= [0,1,0,0,0,0,0]
                elif value<=29:
                    newrow+= [0,0,1,0,0,0,0]
                elif value<=39:
                    newrow+= [0,0,0,1,0,0,0]
                elif value<=49:
                    newrow+= [0,0,0,0,1,0,0]
                elif value<=59:
                    newrow+= [0,0,0,0,0,1,0]
                else:
                    newrow+= [0,0,0,0,0,0,1]
            else:
                add_section = []
                for newtitle in newtitle_range:
                    if newtitle == "{}__{}".format(titles[column_num], value):
                        add_section.append(1)
                    else:
                        add_section.append(0)
                newrow+=add_section
        newdata.append(newrow)
    newtitles_flat = list(flatten(newtitles))
    ###
    return newtitles_flat, newdata

if __name__ == '__main__':
	pass
	