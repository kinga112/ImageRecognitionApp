from kivy.app import App
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.camera import Camera
from kivy.uix.button import Button
#from kivy.core.window import Window
from kivy.uix.image import Image
import tensorflow as tf
import cv2
from app import imageRecog
import time
from kivy.config import Config
from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
import cv2
import tensorflow as tf
import os
from app import hockeyPuck
from app import baseball
from app import basketball



Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '480')
Config.set('graphics', 'height', '640')

done = False

class Widgets(Widget):
    pass

class kivyFile(App):
    def build(self):
        return Widgets()

    def imageRec(self):
        imageRecog()
        done = True
        time.sleep(2)
        if(done == True):
            if hockeyPuck == 1:
                self.label.text = "Hockey Puck"
            if baseball == 1:
                self.label.text = "Baseball"
            if basketball == 1:
                self.label.text = "Basketball"

if __name__ == '__main__':
    kivyFile().run()

