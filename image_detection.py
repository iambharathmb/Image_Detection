import cv2 as cv

#img = cv.imread("image.png")
#img = cv.imread("peoples_1.jpg")
img = cv.imread("peoples_2.jpg")
#cv.imshow("Person", img)

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#cv.imshow("Gray", gray)

haar_cascade = cv.CascadeClassifier("hfd.xml")

faces_rect = haar_cascade.detectMultiScale(gray,1.1,1)
#haar_cascade.detectMultiScale(gray, scalsFactor = 1.1, minNeighbors = 3)

print(f'number of faces found = {len(faces_rect)}')

for(x,y,w,h) in faces_rect:
    cv.rectangle(img, (x,y), (x+w, y+h), (0,255,0), thickness=2)

cv.imshow("Dectected Faces", img)


cv.waitKey(0)
cv.destroyAllWindows()