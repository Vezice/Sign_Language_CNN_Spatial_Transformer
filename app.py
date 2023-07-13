from imutils.video import VideoStream
from flask import Response, request
from flask import Flask
from flask import render_template
import threading
import argparse
import time
from flask import jsonify
#import autocomplete
import torchvision
from PIL import Image
import PIL
import glob
from torchvision import transforms
from numpy import asarray

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from collections import OrderedDict
#from model import Network

from model_Putih import Network
#from model_Hitam import Network
#from model_Karpet import Network
#from model_Pagi import Network
#from model_4data import Network

#model = torch.load('model_trained.pt', map_location='cpu') # Original
#model = torch.load('NewData_notransform.pt', map_location='cpu')
#model = torch.load('NewData_transform.pt', map_location='cpu')
#model = torch.load('NewData_experiment.pt', map_location='cpu')
#model = torch.load('NewData_experiment1.pt', map_location='cpu')
#model = torch.load('NewData_grayscale2.pt', map_location='cpu')
#model = torch.load('BigData_grayscale.pt', map_location='cpu')
#model = torch.load('BigData_transform.pt', map_location='cpu')
#model = torch.load('NewData_28x28.pt', map_location='cpu')
#model = torch.load('NewData_28x28grayscale.pt', map_location='cpu')
#model = torch.load('Model_28x28grayscale.pt', map_location='cpu')
#model = torch.load('olddata.pt', map_location='cpu')
#model = torch.load('olddata_inverse.pt', map_location='cpu')
#model = torch.load('olddata_single.pt', map_location='cpu')
#model = torch.load('BigData_single.pt', map_location='cpu') # GOOD, but A M N S missing

#model = torch.load('combi_normal.pt', map_location='cpu')
#model = torch.load('combi_small.pt', map_location='cpu')
#model = torch.load('combi_small_gray.pt', map_location='cpu')
#model = torch.load('combi_custom1.pt', map_location='cpu')
#model = torch.load('combi_single.pt', map_location='cpu') # GOOD, but M N missing (70x70 btw)
#model = torch.load('combi_single_224.pt', map_location='cpu')
#model = torch.load('combi_single_100.pt', map_location='cpu') # GOOD, but A M N missing
#model = torch.load('combi_single_100_notransform.pt', map_location='cpu') # GOOD but some are barely recognizable
#model = torch.load('combi_single_STN2.pt', map_location='cpu')
#model = torch.load('combi_STN_ResNet2_IMPORTANT.pt', map_location='cpu') #not so good...
#model = torch.load('4data_resnet.pt', map_location='cpu') # Notbad
#model = torch.load('4data_resnet100.pt', map_location='cpu') # Notbad

#Real Pengujian
model = Network()

model = torch.load('Putih.pt', map_location='cpu') # 64 Hidden Layer
#model = torch.load('Hitam.pt', map_location='cpu') # 32 Hidden Layer
#model = torch.load('Sajadah.pt', map_location='cpu') # 64 Hidden Layer
#model = torch.load('Pagi2.pt', map_location='cpu') # 64 Hidden Layer
#model = torch.load('4_data.pt', map_location='cpu') # 256 Hidden Layer

#python app.py -i 0.0.0.0 -o 8080

model.eval()

#signs = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I',
#        '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
#        '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y'}

signs = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I', '9': 'J',
        '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
        '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y', '25': 'Z' }

#autocomplete.load()

outputFrame = None
lock = threading.Lock()
trigger_flag = False

app = Flask(__name__)

vc = VideoStream(src=0).start()
time.sleep(2.0)

# Remove Background ---------------------
""" def load_model():
    model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
    model.eval()
    return model

def make_transparent_foreground(pic, mask):
    # split the image into channels
    b, g, r = cv2.split(np.array(pic).astype('uint8'))
    # add an alpha channel with and fill all with transparent pixels (max 255)
    a = np.ones(mask.shape, dtype='uint8') * 255
    # merge the alpha channel back
    alpha_im = cv2.merge([b, g, r, a], 4)
    # create a transparent background
    bg = np.zeros(alpha_im.shape)
    # setup the new mask
    new_mask = np.stack([mask, mask, mask, mask], axis=2)
    # copy only the foreground color pixels from the original image where mask is set
    foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)

    return foreground

def remove_background(model, input_file):
    input_image = input_file
    #grayscale = transforms.Grayscale(num_output_channels=3)
    preprocess = transforms.Compose([
        #grayscale,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0)

    # create a binary (black and white) mask of the profile foreground
    mask = output_predictions.byte().cpu().numpy()
    background = np.zeros(mask.shape)
    bin_mask = np.where(mask, 255, background).astype(np.uint8)

    foreground = make_transparent_foreground(input_image ,bin_mask)

    return foreground, bin_mask

def custom_background(background_file, foreground):
    final_foreground = Image.fromarray(foreground)
    background = Image.open(background_file)
    x = (background.size[0]-final_foreground.size[0])/2 + 0.5
    y = (background.size[1]-final_foreground.size[1])/2 + 0.5
    box = (x, y, final_foreground.size[0] + x, final_foreground.size[1] + y)
    crop = background.crop(box)
    final_image = crop.copy()
    # put the foreground in the centre of the background
    paste_box = (0, final_image.size[1] - final_foreground.size[1], final_image.size[0], final_image.size[1])
    final_image.paste(final_foreground, paste_box, mask=final_foreground)
    return final_image

background_file = 'white.jpg' """

# ---------------------


            
def detect_gesture(frameCount):

    global vc, outputFrame, lock, trigger_flag

    while True:
        frame = vc.read()

        width = 720
        height = 480
            
        frame = cv2.resize(frame, (width,height))

        img = frame[20:250, 20:250]

        
        
        #foreground_file = 'foreground.png'
        #image = img
        #deeplab_model = load_model()
        #foreground, bin_mask = remove_background(deeplab_model, image)
        #image_fg = Image.fromarray(foreground)
        #if foreground_file.endswith(('jpg', 'jpeg')):
        #    image_fg = image_fg.convert('RGB')
        #image_fg.save(foreground_file)
        
        #final_image = custom_background(background_file, foreground)
        #numpydata = asarray(final_image)

        #img = numpydata

        #res = cv2.resize(img, dsize=(100, 100), interpolation = cv2.INTER_LINEAR)
        res = cv2.resize(img, dsize=(224, 224), interpolation = cv2.INTER_LINEAR) # Not that effective
        #res = cv2.resize(img, dsize=(28, 28), interpolation = cv2.INTER_LINEAR)
        #res = cv2.cvtColor(res, cv2.COLOR_RGBA2RGB)
        #res = cv2.cvtColor(res, cv2.COLOR_BGRA2GRAY) # Not that effective
        res = cv2.cvtColor(res, cv2.COLOR_RGBA2GRAY)
        #res = np.stack((res,)*3, axis=-1)

        #res1 = np.reshape(res, (1, 3, 70, 70)) / 255
        #res1 = np.reshape(res, (1, 1, 70, 70)) / 255
        #res1 = np.reshape(res, (1, 1, 28, 28)) / 255
        #res1 = np.reshape(res, (1, 1, 100, 100)) / 255
        res1 = np.reshape(res, (1, 1, 224, 224)) / 255

        res1 = torch.from_numpy(res1)
        res1 = res1.type(torch.FloatTensor)

        # Only for STN
        #res1_transformed = model.stn(res1).cpu()
        #for e in range(1, 4):
        #  res1_transformed = model.stn(res1_transformed)#.cpu()
        #res1 = torchvision.transforms.functional.rgb_to_grayscale(res1, num_output_channels = 3) 
        #torch.nn.Module.dump_patches = True

        #out = model(res1_transformed)

        out = model(res1)
        probs, label = torch.topk(out, 26)
        probs = torch.nn.functional.softmax(probs, 1)

        pred = out.max(1, keepdim=True)[1]

        if float(probs[0,0]) < 0.4:
            detected = ' '
            #detected = signs[str(int(pred))] + ': ' + '{:.2f}'.format(float(probs[0,0]))
        else:
            detected = signs[str(int(pred))] + ': ' + '{:.2f}'.format(float(probs[0,0]))

        """temp_probs = torch.nn.functional.softmax(out, 1)

        if float(probs[0,0]) < 0.4:
            #detected = 'None'
            detected = 'A' + ': ' + '{:.2f}'.format(float(temp_probs[0,0]))
        else:
            detected = 'A' + ': ' + '{:.2f}'.format(float(temp_probs[0,0]))"""
            

        """ if trigger_flag:
            full_sentence+=signs[str(int(pred))].lower()
            trigger_flag=False
        
        if(text_suggestion!=''):
            if(text_suggestion==' '):
                full_sentence+=' '
                text_suggestion=''
            else:
                full_sentence_list = full_sentence.strip().split()
                if(len(full_sentence_list)!=0):
                    full_sentence_list.pop()
                full_sentence_list.append(text_suggestion)
                full_sentence = ' '.join(full_sentence_list)
                full_sentence+=' '
                text_suggestion='' """

        font = cv2.FONT_ITALIC
        frame = cv2.putText(frame, detected, (60,285), font, 1, (0,255,255), 2, cv2.LINE_AA)

        frame = cv2.rectangle(frame, (20, 20), (250, 250), (0, 255, 255), 3)

        with lock:
            outputFrame = frame.copy()

		
def generate():
    global outputFrame, lock
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            if not flag:
                continue

        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


""" def get_suggestion(prev_word='my', next_semi_word='na'):
    global full_sentence
    separated = full_sentence.strip().split(' ')

    print(separated)

    if(len(separated)==0):
        return ['i', 'me', 'the', 'my', 'there']
    elif(len(separated)==1):
        suggestions = autocomplete.predict(full_sentence, '')[:5]
    elif(len(separated)>=2):
        first = ''
        second = ''

        first = separated[-2]
        second = separated[-1]
        
        suggestions = autocomplete.predict(first, second)[:5]
        
    return [word[0] for word in suggestions] """

@app.route("/")
def index():
    #alphabet = request.form.get('alphabet')
    #print(alphabet)
    return render_template("index.html")


""" @app.route('/char') 
def char():
    global text_suggestion
    recommended = get_suggestion()
    option = request.args.get('character')
    if(option=='space'):
        text_suggestion=" "
    else:
        text_suggestion=recommended[int(option)-1]
    print(text_suggestion)
    return Response("done")"""

@app.route("/video_feed")
def video_feed():
	return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")

""" @app.route('/suggestions')
def suggestion():
    suggestions = get_suggestion()
    return jsonify(suggestions) """

""" @app.route('/sentence')
def sentence():
    global full_sentence
    return jsonify(full_sentence) """

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, required=True, help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, required=True, help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=32, help="# of frames used to construct the background model")
    args = vars(ap.parse_args())

    alphabet = ''

    t = threading.Thread(target=detect_gesture, args=(args["frame_count"],))
    t.daemon = True
    t.start()

    app.run(host=args["ip"], port=args["port"], debug=True, threaded=True, use_reloader=False)

vc.stop()