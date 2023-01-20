import gradio as gr
from fastai.vision.all import *
import skimage

learn = load_learner('export.pkl')

labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred, pred_idx, probs = learn.predict(cd img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}

title = 'Greek God Classifier'
description = 'A greek god classifier trained on data from the internet. It classifies images (often inaccurately) of the six main gods'
gr.Interface(fn=predict, inputs=gr.inputs.Image(shape=(512, 512)), outputs=gr.outputs.Label(num_top_classes=3), title = title, description = description).launch()