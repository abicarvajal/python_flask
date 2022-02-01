from urllib import response
from flask import Flask
from itsdangerous import json
import requests
import tensorflow as tf
import urllib

app = Flask(__name__)

@app.route('/ping')
def ping():
    return 'Pong'

@app.route('/predecir/<path:url>')
def predecir(url):
    filename = 'imagen.jpg'
    #The URL of tensorflow serving
    # endpoint = "http://localhost:8601/v1/models/modelo_flores:predict"
    endpoint = "http://20.127.94.155:8601/v1/models/modelo_flores:predict"

    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)

    image = tf.io.decode_image(open(filename,'rb').read(),channels=3)
    image = tf.image.resize(image, [224,224])
    image = image/255.

    image_tensor = tf.expand_dims(image,0)
    image_tensor = image_tensor.numpy().tolist()

    # Prepare the data that is going to be sent in the POST request
    json_data = {
        "instances": image_tensor
    }

    # Send the request to the Prediction API
    response = requests.post(endpoint, json=json_data)

    #Retrieve the highest probablity index of the Tensor (actual prediction)
    prediction = tf.argmax(response.json()['predictions'][0])

    return str(prediction.numpy())


if __name__ == '__main__':
    app.run(debug=True,port=4000)
