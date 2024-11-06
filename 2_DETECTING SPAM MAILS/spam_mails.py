#pip install -r requirements.txt
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm

spam = pd.read_csv('data.csv')
z = spam['EmailText']
y = spam["Label"]
z_train, z_test,y_train, y_test = train_test_split(z,y,test_size = 0.2)

cv = CountVectorizer()
features = cv.fit_transform(z_train)
model = svm.SVC()
model.fit(features,y_train)
features_test = cv.transform(z_test)
print(model.score(features_test,y_test))
features_test = cv.transform(z_test)
print("Accuracy: {}".format(model.score(features_test,y_test)))

import pickle
from tkinter import *

def check_spam():
    text = spam_text_Entry.get()
    with open('data.csv') as file:
        contents = file.read()
        if text in contents:
            print(text,"text is spam")
            my_string_var.set("Result: text is spam")
        else:


            print(text,"text not found in database")
            my_string_var.set("Result: text not found in database")

win = Tk()
win.geometry("400x600")
win.configure(background="cyan")
win.title("Email Spam Detector")

title = Label(win, text="Email Spam Detector",
             bg="gray",width="300",height="2",fg="white",font=("Calibri", 20, "bold", "italic", "underline"))
title.pack()

spam_text = Label(win, text="Enter your Text: ",bg="cyan", font=("Verdana", 12))
spam_text.place(x=12,y=100)

spam_text_Entry = Entry(win, width=33)
spam_text_Entry.place(x=155, y=105)

my_string_var = StringVar()
my_string_var.set("Result: ")

print_spam = Label(win,textvariable=my_string_var,bg="cyan",font=("Verdana", 12))
print_spam.place(x=12,y=200)

submit_button = Button(win, text="Submit",width="12",height="1",
                      activebackground="red",bg="Pink",command=check_spam,
                      font=("Verdana", 12))
submit_button.place(x=12,y=150)

win.mainloop()