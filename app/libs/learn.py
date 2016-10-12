# -*- coding: utf-8 -*-
import os.path
import sys

import cv2
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import RandomizedPCA
from sklearn.externals import joblib
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import (classification_report, precision_score,
                             recall_score, confusion_matrix)
from sklearn.svm import SVC


IMAGE_PATH = "data/new_cutout_results/{}/{}.jpg"
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
TARGET_NUM = 4
DATA_NUM = 40  # 一人あたりの画像データ数
PARAM_GRID = [
    {
        'C': [1e3, 5e3, 1e4, 5e4, 1e5],
        'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
        'kernel': ['linear']
    }, {
        'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']
    }
]
PICKLE_ROOT = '/var/www/7faces/app/libs/data/'
# NAME_LIST = ('渡辺麻友', '指原莉乃', '柏木由紀', '松井珠理奈', '松井玲奈',
#             '山本彩', '島崎遥香')

# NAME_LIST = ('渡辺麻友', '柏木由紀', '山本彩', '松井玲奈')
NAME_LIST = ('MAYU WATANABE', 'YUKI KASHIWAGI', 'SAYAKA YAMAMOTO', 'REINA MATSUI')

# ラベルの対応は以下
# 0,渡辺麻友
# 1,指原莉乃
# 2,柏木由紀
# 3,松井珠理奈
# 4,松井玲奈
# 5,山本彩
# 6,島崎遥香


def show_img(image):
    cv2.imshow('result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def load_data(test_size=0.25):
    X = np.zeros((TARGET_NUM*DATA_NUM, IMAGE_WIDTH*IMAGE_HEIGHT))
    for target in range(TARGET_NUM):
        for i in range(DATA_NUM):
            path = IMAGE_PATH.format(target, i)
            if not os.path.isfile(path):
                sys.exit("ERROR: Data not found at {}".format(path))

            image = cv2.imread(path)
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 1次元配列でグレースケールの値を格納
            X[(target)*DATA_NUM+i] = image_gray.flatten()

    # ラベル値を設定
    y = np.zeros(TARGET_NUM*DATA_NUM)
    for i in range(DATA_NUM):
        y[i*DATA_NUM:(i+1)*DATA_NUM] = i

    # テスト:トレーニング比をtest-sizeで決める
    result = train_test_split(X, y, test_size=test_size)

    return result


def svm_calc(X_train, X_test, y_train, y_test):
    print("モノクロ画像を入力としてSVMで学習")

    param_grid = PARAM_GRID

    # 第三引数でどちらの点数を重視するか
    clf = GridSearchCV(SVC(), param_grid)
    clf.fit(X_train, y_train)  # このfitでcross validationが入ってる
    y_pred = clf.predict(X_test)

    # print("Best parameters set found on development set:")
    # print("  {}\n".format(clf.best_estimator_))

    # print("Grid scores on development set:")
    # for params, mean_score, scores in clf.grid_scores_:
    #     print("  %0.3f (+/-%0.03f) for %r" % (mean_score, scores.std()/2
    #                                          , params))
    # print("Best score = {}".format(clf.best_score_))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


def learn_pca_svm(X_train, X_test, y_train, y_test):
    print("主成分分析とSVMで学習")

    # 特徴ベクトルの次元
    n_components = 30

    pca = RandomizedPCA(n_components=n_components, whiten=True)
    pca.fit(X_train)
    joblib.dump(pca, PICKLE_ROOT + 'pca.pkl')

    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    param_grid = PARAM_GRID
    clf = GridSearchCV(SVC(class_weight='auto'), param_grid)
    clf = clf.fit(X_train_pca, y_train)

    joblib.dump(clf, PICKLE_ROOT + 'clf.pkl')

    y_pred = clf.predict(X_test_pca)

    # print("Best estimator found by grid search:")
    # print("  {}\n".format(clf.best_estimator_))

    # Quantitative evaluation of the model quality on the test set

    # print("Grid scores on development set:")
    # for params, mean_score, scores in clf.grid_scores_:
    #     print("  %0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2,
    #                                           params))
    # print("Best score = {}".format(clf.best_score_))

    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))


def classify_pca_svm(path):
    image = cv2.imread(path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = cv2.resize(image_gray, (300, 300))

    # 1次元配列でグレースケールの値を格納
    X = image_gray.flatten()

    # 特徴ベクトルの次元
    n_components = 30

    pca = joblib.load(PICKLE_ROOT + 'pca.pkl')
    # pca = joblib.load('/var/www/7faces/app/libs/data/pca.pkl')

    X_pca = pca.transform(X)

    clf = joblib.load(PICKLE_ROOT + 'clf.pkl')
	# clf = joblib.load('/var/www/7faces/app/libs/data/clf.pkl')
    y = clf.predict(X_pca)
    return NAME_LIST[int(y[0])]


if __name__ == '__main__':
    data = load_data()
    learn_pca_svm(*data)
