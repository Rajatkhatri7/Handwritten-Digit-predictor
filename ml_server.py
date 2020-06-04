
import json
import numpy as np
import random
from flask import Flask,request
import tensorflow as tf


app=Flask(__name__)
model=tf.keras.models.load_model("model.h5")
#using keras api to show output of all layers
feature_model=tf.keras.models.Model(model.inputs,
                    [layer.output for layer in model.layers])
_,(x_test,_)=tf.keras.datasets.mnist.load_data()
x_test=x_test/255

def get_prediction():
    index=np.random.choice(x_test.shape[0])
    image=x_test[index,:,:]
    image_arr=np.reshape(image,(1,784))
    return feature_model.predict(image_arr),image

@app.route("/",methods=["GET","POST"])
def index():
    if request.method=="POST":
        preds,image=get_prediction()
        final_prediction=[p.tolist() for p in preds]# returning json obj
        return json.dumps({
        
        "prediction": final_prediction,
        "image": image.tolist()
    })
    return "welcome to server"
if __name__=="__main__":
    app.run()
