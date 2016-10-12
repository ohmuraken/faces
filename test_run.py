# from app import app
from app.libs.crop_face import get_cropped_face
from app.libs.learn import classify_pca_svm


is_cropped = get_cropped_face("7faces/a.png")
print is_cropped # a.png_cropped.png

if not is_cropped:
	print "error"

print classify_pca_svm(is_cropped)
