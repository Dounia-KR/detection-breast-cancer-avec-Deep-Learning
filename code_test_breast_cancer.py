import tensorflow as tf
import cv2
font = cv2.FONT_HERSHEY_SIMPLEX
bottmleftCorneroftext = (10,28)
fontScale = 0.80
fontColor = (0,0,255)
lineType = 2
my_text = ''
img = cv2.imread('image_de_test/positive1.png')
image_after1 = cv2.resize(img,(224,224))
image_after = image_after1.reshape((1,224,224,3))

print("breast cancer")
model = tf.keras.models.load_model('breast_cancer_model.h5')
resulta = model.predict_classes(image_after)
if resulta == 0:
    my_text = 'negative'
else :
    my_text = 'positive'

print(" le résultat est écrit sur l’image de test ")
cv2.putText(image_after1,my_text,bottmleftCorneroftext,font,fontScale,fontColor,lineType)
cv2.imshow('image',image_after1)
cv2.waitKey(0)
cv2.destroyAllWindows()

