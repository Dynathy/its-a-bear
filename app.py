from flask import Flask, render_template, request, redirect
from fastai.vision.widgets import *
from fastai.vision.all import *
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
path = os.getcwd()  # Gets the current working directory
learn_inf = load_learner(os.path.join(path, 'export.pkl'))  # Join paths in a platform-independent way

# Ensure there's a folder to save uploaded images
os.makedirs('static/uploads', exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    if request.method == 'POST':
        image_file = request.files['image']
        if image_file:
            image_location = os.path.join(
                'static/uploads',
                secure_filename(image_file.filename)
            )
            image_file.save(image_location)
            img = PILImage.create(image_location)
            pred, pred_idx, probs = learn_inf.predict(img)
            result = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
            return render_template('index.html', prediction=result, image_loc=image_file.filename)
    return render_template('index.html', prediction=0, image_loc=None)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
