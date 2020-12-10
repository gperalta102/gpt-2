
import fire
import json
import os
import numpy as np
import tensorflow as tf
from tkinter import *
import model, sample, encoder
import threading
from tkinter import *


model_name='anime00',
seed=None,
nsamples=1,
batch_size=1,
length=25,
temperature=0.8,
top_k=40,
top_p=1,
models_dir='models'

models_dir = os.path.expanduser(os.path.expandvars(models_dir))
if batch_size is None:
    batch_size = 1
assert nsamples % batch_size == 0

if length is None:
    length = hparams.n_ctx // 2
elif length > hparams.n_ctx:
    raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)
with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k, top_p=top_p
        )
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join(models_dir, model_name))
        saver.restore(sess, ckpt)

        


enc = encoder.get_encoder(model_name)
hparams = model.default_hparams()
with open(os.path.join('models', model_name, 'hparams.json')) as f:
    hparams.override_from_dict(json.load(f))
root = Tk()
root.title("Chat Bot")
root.geometry("400x500")
root.resizable(width = False, height = FALSE)

main_menu = Menu(root)

def retrieve_input():
    inputValue = messageWindow.get("1.0", 'end-1c')
    print(inputValue)
    recieveMessage(inputValue)
    

def recieveMessage(message):
    chatWindow.config(state=NORMAL)
    chatWindow.insert(END, message)
    chatWindow.config(state = DISABLED) 
    chatWindow.see(END) 

            

#Create the submenu


chatWindow= Text(root, bd=1, bg="black", width="50", height="8", font=("Arial", 23), foreground="#00ffff")
chatWindow.place(x=6,y=6,height=385, width=370)

messageWindow= Text(root, bd=0, bg="black", width="30", height="4", font=("Arial", 23), foreground="#00ffff")
messageWindow.pack()
messageWindow.place(x=128, y=400, height=88, width=260)



scrollbar = Scrollbar(root, command=chatWindow.yview, cursor="star")
scrollbar.place(x=375, y=5, height=385)

button = Button(root, text="send", width="12", height="5", bd=0, bg="#0080ff", activebackground="#00bfff", foreground="#ffffff", font=("Arial", 12),command=lambda: retrieve_input())
button.pack()
button.place(x=6, y=400, height=88)


        

root.mainloop()
