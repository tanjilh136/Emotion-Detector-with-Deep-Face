# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 21:23:07 2020

@author: Tanjil
"""

from tkinter import *
import tkinter.font as tkFont
import cv2
from deepface import DeepFace

def run():
    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap= cv2.VideoCapture(1)

    if not cap.isOpened():
      cap= cv2.VideoCapture(0)
    if not cap.isOpened():
      raise IOError('Web Cam Failure')

    while True:
      ret,frame=cap.read()
      result=DeepFace.analyze(frame, actions = ['emotion'])

      gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

      faces = faceCascade.detectMultiScale(gray,1.1,4)

      for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+w),(0,255,0),2)

        text2=result['dominant_emotion'].upper()

        font = cv2.FONT_HERSHEY_SIMPLEX

        cv2.putText(frame,
                    text2,
                    (30,180),
                    font,1,
                    (255,255,255),
                    2,
                    cv2.LINE_4);
        cv2.imshow('Orginal video',frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
          break
    cap.release()
    cv2.destroyAllWindows()

screen = Tk()
screen.title("Face Emotion Detection with WebCam ")
screen.geometry("800x550")

fontStyle = tkFont.Font(family="Lucida Grande", size=30)
fontStyle_result = tkFont.Font(family="Lucida Grande", size=30)
welcome_text = Label(text = "Emotion Detector", fg = "white", bg = "black", font=fontStyle)
welcome_text.pack()

fontStyle1=tkFont.Font(family="Lucida Grande", size=16)
test_h_text= Label(text = "After The Click Wait for a while to Initiate the camera",font=fontStyle1)
test_h_text.place(x=120,y=80)

msg="Note that before click You must keep your camera open\n if you use an external camera,if it's built in camera. Don't worry that "
test_h_text1= Label(text = msg,font=fontStyle1)
test_h_text1.place(x=120,y=150)

click = Button(text = "Click Here", fg = "blue", bg = "grey",font=fontStyle1, command = run,)
click.place(x=350, y=360)

mainloop()