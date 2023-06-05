import cv2
import numpy as np

#video dosyasını açar.
vid = cv2.VideoCapture("traffic.mp4")

#Arka planı çıkarmak için video çerçeveleri üzerinde kullanılacak olan bir arka plan çıkarma algoritmasıdır
# Arka plan çıkarma, bir çerçevedeki hareketli nesneleri belirlemek için mevcut çerçeveyi arka plana karşı kıyaslar.
backsub = cv2.createBackgroundSubtractorMOG2()

c = 0
while True:
    ret, frame = vid.read()
    if ret == 1:
        fgmask = backsub.apply(frame)

        cv2.line(frame, (20, 0), (20, 300), (0, 255, 0), 2)
        cv2.line(frame, (40, 0), (40, 300), (0, 255, 0), 2)

        contours, hiers = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        try:
            hiers = hiers[0]
        except:
            hiers = []

        for contour, hier in zip(contours, hiers):
            (x, y, w, h) = cv2.boundingRect(contour)
            if w > 40 and h > 40:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)
                if x > 50 and x < 70:
                    c = c + 1

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "" + str(c), (90, 100), font, 2, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("vid", fgmask)
        cv2.imshow("vid", frame)
        if cv2.waitKey(40) & 0xFF == ord("q"):
            break


vid.release()
cv2.destroyAllWindows()
