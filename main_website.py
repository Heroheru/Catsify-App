import tensorflow_hub as hub
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import imghdr
from PIL import Image 
import os 
import pillow_heif
pillow_heif.register_heif_opener()

app = Flask(__name__)
IMAGE_SHAPE = (224,224)
pets_labels_dict = {
   0    :   'Abyssinian'                 ,  
   1    :   'Bengal'                     ,
   2    :   'Birman'                     ,
   3    :   'Bombay'                     ,
   4    :   'British_Shorthair'          ,
   5    :   'Egyptian_Mau'               ,
   6    :   'Maine_Coon'                 ,
   7    :   'Persian'                    ,
   8    :   'Ragdoll'                    ,
   9    :   'Russian_Blue'               ,
   10    :   'Siamese'                    ,
   11    :   'Sphynx'                     ,
}

model = load_model('Model_Penelitian_FInal.h5',custom_objects={'KerasLayer': hub.KerasLayer})
model.make_predict_function()

def predict_label(img_path):
	gambar = image.load_img(img_path).resize(IMAGE_SHAPE)
	gambar = image.img_to_array(gambar)/255.0
	p = model.predict(gambar[np.newaxis, ...])
	print(p)
	predicted_label = np.argmax(p)
	return pets_labels_dict[predicted_label]

@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about", methods=['GET', 'POST'])
def about_page():
	return render_template("about.html")


@app.route("/predict", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']
		img_path = "static/" + "Foto_kucing"
		img.save(img_path)
		type_image = imghdr.what(img_path)
		if type_image != 'jpg'	:
			im = Image.open(img_path)
			rgb_im = im.convert("RGB")
			img_path = img_path+".jpg"
			rgb_im.save(img_path)
		p = predict_label(img_path)


	return render_template("hasilpredik.html", prediction = p, img_path = img_path)


@app.errorhandler(404)
def not_found_error(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'),500

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = False,host='0.0.0.0')