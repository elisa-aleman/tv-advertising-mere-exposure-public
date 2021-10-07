
from NomuraSoken_methods import *
from minepy import MINE
from sklearn.svm import SVR


def getYoutubeX():
    youtube_path = os.path.join(os.sep, "usr", "local", "data", "加工データ","youtube_x_label.csv")
    _,youtube_csv = readCSV(youtube_path)
    products = [int(ytx[0]) for ytx in youtube_csv]
    X = [[int(val) for val in row][1:] for row in youtube_csv]
    return products, X

def getYperuser_ALT(user_id, products, category=2):
    _,label_data=readCSV(getUserLabelCSVPath())
    labels = [[int(val) for val in row][1:] for row in label_data if row[0]==user_id][0]
    if category<4:
        y_labels = [1 if cat==category else 0 for cat in labels]
    elif category == 4:
        y_labels = [1 if cat==2 or cat==3 else 0 for cat in labels]
    elif category == 5:
        y_labels = [1 if cat==0 or cat==1 else 0 for cat in labels]
    y = [val for num,val in enumerate(y_labels) if num in products]
    return y

def testSVMs_youtube_user():
    titles = ['User_id','Purchase_Category','Precision_Av','Precision_StDv','Recall_Av','Recall_StDv','Accuracy_Av','Accuracy_StDv','F1_Av','F1_StDv']
    log_file = "/home/ealeman/SVM_tests_youtube_by_user.csv"
    strlog = ','.join(titles)
    printSTDlog(strlog,log_file)
    _,main_data = readCSV(getMainDataCSVPath())
    users = [user[0] for user in main_data]
    categories = [2,4,5,3,0,1]
    k = 4
    products, x = getYoutubeX()
    for category in categories:
        for user_id in users:
            y = getYperuser_ALT(user_id,  products=products, category=category)
            r = None
            for tries in xrange(30):
                try:
                    r = SVMkfolds(x, y, k)
                    break
                except ValueError:
                    print "ValueError"
            if r:
                # results = [[avpr, stpr, precisions], [avre, stre, recalls], [avac, stac, accuracies], [avf1, stf1, f1s], counts]
                log_row = [user_id,str(r[0][0]),str(r[0][1]),str(r[1][0]),str(r[1][1]),str(r[2][0]),str(r[2][1]),str(r[3][0]),str(r[3][1])]
                strlog = ",".join(log_row)
                printSTDlog(strlog,log_file)
            else:
                print "Skipped because of ValueError limit"

def Youtube_Regression_y():
    products, x = getYoutubeX()
    _,main_data = readCSV(getMainDataCSVPath())
    users = [user[0] for user in main_data]
    categories = [2,4,5,3,0,1]
    for category in categories:
        user_ys = [getYperuser_ALT(user_id,  products=products, category=category) for user_id in users]
        y = [sum(x) for x in zip(*user_ys)]
        print y

def getYoutubeY(category=2):
    youtube_path = os.path.join(os.sep, "usr", "local", "data", "加工データ","youtube_y.csv")
    _,youtube_csv = readCSV(youtube_path)
    y = [int(row[category+1]) for row in youtube_csv]
    return y

def RegressionYoutube():
    titles = ['Purchase_Category','SVR_RBF_Score','SVR_Linear_Score','SVR_Poly_score']
    # titles = ['Purchase_Category','SVR_RBF_Score','SVR_Poly_score']
    log_file = "/home/ealeman/svr_youtube.csv"
    strlog = ','.join(titles)
    printSTDlog(strlog,log_file)
    products, X = getYoutubeX()
    categories = [2,4,5,3,0,1]
    for category in categories:
        y = getYoutubeY(category)
        ####FIT DATA
        svr_rbf = SVR(kernel='rbf', C=0.0003, gamma=0.1)
        svr_lin = SVR(kernel='linear', C=1)
        svr_poly = SVR(kernel='poly', C=0.0003, degree=2)
        svr_rbf.fit(X, y)
        svr_lin.fit(X, y)
        svr_poly.fit(X, y)
        ####Scores
        score_rbf = svr_rbf.score(X,y)
        score_lin = svr_lin.score(X,y)
        score_poly = svr_poly.score(X,y)
        ####
        strlog = '{},{},{},{}'.format(category,score_rbf,score_lin,score_poly)
        # strlog = '{},{},{},{}'.format(category,score_rbf,score_poly)
        printSTDlog(strlog,log_file)

def RegressionYoutube():
    titles = ['Purchase_Category','SVR_Poly_Score']
    log_file = "/home/ealeman/svr_youtube_poly.csv"
    strlog = ','.join(titles)
    printSTDlog(strlog,log_file)
    products, X = getYoutubeX()
    categories = [2,4,5,3,0,1]
    for category in categories:
        y = getYoutubeY(category)
        print X
        print y
        ####FIT DATA
        # svr_rbf = SVR(kernel='rbf', C=0.0003, gamma=0.1)
        # svr_lin = SVR(kernel='linear', C=1)
        svr_poly = SVR(kernel='poly', C=0.0003, degree=2)
        # svr_rbf.fit(X, y)
        # svr_lin.fit(X, y)
        svr_poly.fit(X, y)
        ####Scores
        # score_rbf = svr_rbf.score(X,y)
        # score_lin = svr_lin.score(X,y)
        score_poly = svr_poly.score(X,y)
        ####
        strlog = '{},{}'.format(category,score_poly)
        printSTDlog(strlog,log_file)

def getMIC(x, y):
    mine = MINE(alpha=0.6, c=15)
    mine.compute_score(x, y)
    mic = mine.mic()
    return mic

def MICyoutube():
    titles = ['Purchase_Category','Likes-Users_MIC','Dislikes-Users_MIC','YoutubeViews-Users_MIC','TVBroadcastTimes-Users_MIC']
    log_file = "/home/ealeman/MIC_youtube.csv"
    strlog = ','.join(titles)
    printSTDlog(strlog,log_file)
    #
    products, x = getYoutubeX()
    categories = [2,4,5,3,0,1]
    for category in categories:
        y = getYoutubeY(category)
        likes = [row[0] for row in x]
        likes_mic = getMIC(likes,y)
        dislikes = [row[1] for row in x]
        dislikes_mic = getMIC(dislikes,y)
        ytviews = [row[2] for row in x]
        ytviews_mic = getMIC(ytviews,y)
        tvbroads = [row[3] for row in x]
        tvbroads_mic = getMIC(tvbroads,y)
        strlog = '{},{},{},{},{}'.format(category,likes_mic,dislikes_mic,ytviews_mic,tvbroads_mic)
        printSTDlog(strlog,log_file)


def main():
    MICyoutube()
    RegressionYoutube()
    testSVMs_youtube_user()

if __name__ == '__main__':
    main()