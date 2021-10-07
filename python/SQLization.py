#-*- coding: utf-8 -*-

from NomuraSoken_methods import *


def SQLization():
    # maindata
    # print "maindata"
    # titles, table = readCSV(maindata_csv_path)
    # CSVtoSQL(titles, table, maindata_sql_path, tablename='users')
    # tvviewdata
    print "tv view data"
    titles, table = readCSV(tvviewdata_csv_path)
    CSVcreateSQL(titles, tvviewdata_sql_path, tablename="tv_views")
    CSVtoSQL(titles, table, tvviewdata_sql_path, tablename="tv_views")
    # magazines
    print "magazines"
    titles, table = readCSV(magazinedata_csv_path)
    CSVcreateSQL(titles, magazinedata_sql_path, tablename="magazine")
    CSVtoSQL(titles, table, magazinedata_sql_path, tablename="magazine")
    # commercials
    # print "commercials"
    # titles, table = readCSV(commercialdata_csv_path)
    # CSVcreateSQL(titles, commercialdata_sql_path, tablename="commercials")
    # CSVtoSQL(titles, table, commercialdata_sql_path, tablename="commercials")

######################################
#### Assuming we don't need these ####
######################################

def CSVcreateSQL(titles, dbname, tablename):
    conn, c = Connect(dbname)
    size = len(titles)
    # titles_quotes = ["'{}'".format(i) for i in titles]
    # titles_str = ", ".join(titles_quotes)
    c.execute("CREATE TABLE {tn} ('{nf}' INTEGER PRIMARY KEY)".format(tn=tablename, nf=titles[0]))
    for title in titles[1:]:
        c.execute("ALTER TABLE {tn} ADD COLUMN '{cn}' INTEGER".format(tn=tablename, cn=title))
        conn.commit()
    conn.commit()

def CSVtoSQL(titles, table, dbname, tablename='users'):
    times = len(titles)/100
    conn, c = Connect(dbname)
    columns = titles[0:100]
    columns_quote = ["'{}'".format(i) for i in columns]
    columns_str = ", ".join(columns_quote)
    sentence = "?{}".format(",?"*(len(columns)-1))
    sql ="INSERT INTO {} ({}) VALUES({})".format(tablename, columns_str, sentence)
    ins = [tuple(row[0:100]) for row in table]
    c.executemany(sql,ins)
    conn.commit()
    print "Finished first insert 0:100"
    for time in xrange(1, times+1):
        if time*100+100<len(titles):
            finish = start+99
            columns = titles[time*100:time*100+100]
            upd = [tuple(list(row[time*100:time*100+100])+[row[0]]) for row in table]
        else:
            finish = len(titles)
            columns = titles[time*100:]
            upd = [tuple(list(row[time*100:])+[row[0]]) for row in table]
        columns_quote = ["'{}'=?".format(i) for i in columns]
        columns_str = ", ".join(columns_quote)
        sql ="UPDATE {} SET {} WHERE SampleID =?".format(tablename, columns_str)
        c.executemany(sql,upd)
        conn.commit()
        start = time*100
        print "Finished updating from {} to {}".format(start, finish)
    conn.close()