import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade  = cv2.CascadeClassifier('haarcascade_eye.xml')

image = cv2.imread('gaffney-group.jpg')

gray_image = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray_image , scaleFactor = 1.06, minNeighbors = 5)

for x,y,w,h in faces:
    image = cv2.rectangle(image , (x,y) , (x+w , y+h) , (0, 255, 255) , 4)
    eye_gray = gray_image[y:y+h, x:x+w]
    eye_color =  image[y:y+h , x:x+w]
    eyes = eye_cascade.detectMultiScale(eye_gray, scaleFactor=1.06, minNeighbors=5)
    for (ex , ey , ew, eh) in eyes:
        cv2.rectangle(eye_color , (ex ,ey) , (ex+ew , ey+eh) , (0, 255, 0) , 2)


resized_filtered_image = cv2.resize(image , (int(image.shape[1]/2) , int(image.shape[0]/2)))

cv2.imshow("DETECTOR " , resized_filtered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()