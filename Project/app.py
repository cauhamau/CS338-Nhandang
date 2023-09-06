from flask import Flask, render_template, request
from PIL import Image
import cv2
import glob
import json
import os
import shutil
import numpy as np

from PIL import Image



#det

import argparse
import glob
import multiprocessing as mp
import os
import time

import cv2
import tqdm
from adet.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from demo.predictor import VisualizationDemo


#rec 

from vietocr.vietocr.tool.predictor import Predictor
from vietocr.vietocr.tool.config import Cfg
import cv2
import os
from PIL import Image
import argparse
from tqdm import tqdm


app = Flask(__name__)


#setup det sss
mp.set_start_method("spawn", force=True)
cfg = get_cfg()
config_file = './configs/BAText/VinText/attn_R_50.yaml'
opts = []
confidence_threshold = 0.3
cfg.merge_from_file(config_file)
cfg.merge_from_list(opts)

cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
cfg.MODEL.FCOS.INFERENCE_TH_TEST = confidence_threshold
cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
cfg.freeze()
demo = VisualizationDemo(cfg)


#set rec

cfg_name = 'vgg_transformer'
config = Cfg.load_config_from_name(cfg_name)
#    cfg_name = 'vgg_transformer'
weights = './vietocr/weights/vietocr.pth'
# path_img = path_image
device = 'cuda:0'
# path_output = './result'

config['vocab'] = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~° ' + '̉'+ '̀' + '̃'+ '́'+ '̣'
config['device'] = device
config['weights'] = weights

detector = Predictor(config)



def detect(path_test, output):
   
    filename = path_test.split('/')[-1].split('.')[0]
    # output = './result'
    path_out = os.path.join(output, f'res_{filename}.txt')
    img = read_image(path_test, format="BGR")
    _, predictions = demo.run_on_image(img, path_test, path_out, confidence_threshold)
    
    return predictions


def reg(path_image, path_output):
 
	predictions = detect(path_image, path_output)

	filename = path_image.split('/')[-1].split('.')[0]
    #filename = 'reg_'+filename

	img = cv2.imread(path_image, 0)
	out_txt = open(os.path.join(path_output, f'{filename}.txt'), 'w', encoding="utf-8")
	if len(predictions) == 0:
		out_txt.close()
		return None
	else :
		for line in predictions:
			prob_detect = line[1]
			bbox = line[0].strip()
			bbox = bbox.split(',')
			bbox = [int(ele) for ele in bbox]
			top = min(bbox[1:8:2])
			bottom = max(bbox[1:8:2])
			left = min(bbox[0:7:2])
			right = max(bbox[0:7:2])

			bxxout = [left, top, right, bottom]

			if top == 0: top = 1
			if left == 0: left = 1
			x = int((left + right)/2)
			y = int((top+ bottom)/2)
			height = bottom - y
			width = right - x
			crop_img = img[y - height: y + height, x - width: x + width]
			try:
				crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
			except:
				continue

			im_pil = Image.fromarray(crop_img)
			pred, prob = detector.predict(im_pil , return_prob = True)
			
			content = ' '.join([str(p) for p in bxxout]) + ' ' + pred + '\n'
			out_txt.write(content)
			
		out_txt.close()   



@app.route('/', methods=['Get','POST'])
def index():
	if os.path.exists("static"):
		shutil.rmtree("static")
	return render_template('index.html')

@app.route("/label", methods=['POST'])
def home():
	img_files = request.files.getlist("img_files")
	for file in img_files:
		filename = file.filename.replace("(","").replace(")","").replace(" ","")
		if not os.path.exists("static/images"):
				os.makedirs("static/images")
		file.save(os.path.join("static/images", filename))

	if not os.path.exists('static/labels'):
		os.makedirs('static/labels')

	global width, height, image_path,ratio
	image_path = sorted(glob.glob("static/images/*.jpg") + glob.glob("static/images/*.jpeg") + glob.glob("static/images/*.png"))
	# 
	
	for path in image_path:
		
		reg(path, 'static/labels')

	width = 720
	height = 960

	all_polygons = []
	for i,image in enumerate(image_path):
		img = cv2.imread(image)
		#img = img.resize()
		h, w,_ = img.shape
		ratio_w = w/width
		ratio_h = h/height
		polygon = []
		txt_path = str(f"static/labels/{image.split('/')[-1].split('.')[0]}.txt")
		with open(txt_path, 'r') as f:
			lines = f.readlines()

	
		if len(lines) != 0:
			for line in lines:
				x1 = int(int(line.split(' ')[0]) / ratio_w)
				y1 = int(int(line.split(' ')[1]) / ratio_h)
				x2 = int(int(line.split(' ')[2]) / ratio_w)
				y2 = int(int(line.split(' ')[3]) / ratio_h)
				text = line.split(' ')[4].strip()
				
				
				rectangles = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,'text':text}
				#print(list(map(lambda x: x * ratio, rectangles.values())))
				polygon.append(rectangles)
			all_polygons.append(polygon)
		else :
			all_polygons.append([])
	return render_template("label.html",image0=image_path[0], image_path=image_path, width_canvas=width,height_canvas=height,all_polygons=all_polygons)

if __name__ == "__main__":
	app.run(debug=True, host="0.0.0.0", port=8080)
