# -*- coding: utf-8 -*- 
from __future__ import print_function
from os import path
import sys

import cv2
import numpy

COLOR = (255, 255, 255)
CASCADE_PATH = "/var/www/7faces/app/libs/data/haarcascade_frontalface_alt2.xml"
# CASCADE_PATH = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt2.xml"

def get_cropped_face(image_path):
    if not path.isfile(image_path):
        sys.exit("ERROR: Data not found at {}".format(image_path))

    image = cv2.imread(image_path)
    # cropped_image = crop_face(image)
    cropped_image = crop_face(image_path)
    cropped_image_path = image_path + '_cropped.png'
    cv2.imwrite(cropped_image_path, cropped_image)
    return cropped_image_path


def crop_face(image):
    # グレースケール変換
    image = cv2.imread(image)    
    #image = cv2.imread("/var/www/7faecs/app/uploads/test.png")    
    cv2.imwrite("/var/www/7faces/app/uploads/test.png", image)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # cv2.imwrite("/var/www/7faces/app/uploads/test_gray.png", image_gray)

    # カスケード分類器の特徴量を取得する
    cascade = cv2.CascadeClassifier(CASCADE_PATH)

    # 物体認識（顔認識）の実行
    face_rects = cascade.detectMultiScale(
        image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))
    f = open("/var/www/7faces/app/uploads/test.txt", "w")
    f.write("heelo")
    # 検出した顔を囲む矩形の作成(カラー)
    #for rect in face_rects:
    #    f.write(str(rect))
    #    f.write("hello\n")
    #    cv2.rectangle(

    #        image,
    #        tuple(rect[0:2]),
    #        tuple(rect[0:2]+rect[2:4]),
    #        COLOR,
    #        thickness=2
    #    )
    f.write(str(len(face_rects)))
    for rect in face_rects:
        f.write(str(rect))
        f.write("\n")
    f.close()
    #cv2.imwrite("/var/www/7faces/app/uploads/test_rects.png", image)
    # raise Exception(len(face_rects))
    if len(face_rects) == 0:
	    return False
    # raise Exception(print(face_rects))
    #cv2.imwrite("/var/www/7faces/app/uploads/test_gray_cut.png", image)
    # print(face_rects, file=sys.stderr)
    x, y, w, h = face_rects[-1]

    # 線で囲われた所だけの画像のデータ
    cut_image = image[y:y+h, x:x+w]

    # 線で囲われた所だけの画像のデータ
    cut_image_gray = image_gray[y:y+h, x:x+w]

    # リサイズする(カラー)
    # re_size_cut_image = cv2.resize(cut_image,(300,300))

    # リサイズする(グレースケール)
    re_size_cut_image_gray = cv2.resize(cut_image_gray, (300, 300))
	# re_size_cut_image_gray = cv2.resize(image_gray, (300, 300))

    # 表示して確認
    # cv2.imshow("IMAGE", re_size_cut_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return re_size_cut_image_gray
