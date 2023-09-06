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



#viet ocr

from vietocr.vietocr.tool.predictor import Predictor
from vietocr.vietocr.tool.config import Cfg
import cv2
import os
from PIL import Image
import argparse
from tqdm import tqdm





def detect():
    mp.set_start_method("spawn", force=True)
    cfg = get_cfg()
    config_file = './configs/BAText/VinText/attn_R_50.yaml'
    opts = []
    confidence_threshold = 0.5
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(opts)

    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.FCOS.INFERENCE_TH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    cfg.freeze()
    demo = VisualizationDemo(cfg)

    path_test = './images/test.jpg'
    filename = path_test.split('/')[-1].split('.')[0]
    output = './result'
    path_out = os.path.join(output, f'res_{filename}.txt')
    img = read_image(path_test, format="BGR")
    _, predictions = demo.run_on_image(img, path_test, path_out, confidence_threshold)
    
    return predictions
    
def reg(path_image, predictions):
    cfg_name = 'vgg_transformer'
    config = Cfg.load_config_from_name(cfg_name)
#    cfg_name = 'vgg_transformer'
    weights = './vietocr/weights/vietocr.pth'
    path_img = path_image
    device = 'cuda:0'
    path_output = './result'

    config['vocab'] = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~° ' + '̉'+ '̀' + '̃'+ '́'+ '̣'
    config['device'] = device
    config['weights'] = weights

    detector = Predictor(config)


    filename = path_image.split('/')[-1].split('.')[0]
    #filename = 'reg_'+filename

    img = cv2.imread(path_img, 0)
    out_txt = open(os.path.join(path_output, f'reg_{filename}.txt'), 'w', encoding="utf-8")
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
            
            # if submission==True:
            content = ','.join([str(p) for p in bbox]) + ',' + pred + '\n'
            # else:
            #     content = str(prob_detect[:-1]) + ' '
            out_txt.write(content)
            #print(content)
            
        out_txt.close()    
        
       
    


if __name__ == "__main__":
    path_image = './images/test.jpg'
    predictions = detect()
    print(predictions)
    reg(path_image, predictions)

    # for path in tqdm.tqdm(args.input):
    #     img = read_image(path, format="BGR")
    #     start_time = time.time()
    #     file_name = path.split('/')[-1]
    #     file_name = file_name.split('.')[0]

    #     path_out = os.path.join(args.output, f'res_{file_name}.txt')
    #     predictions = demo.run_on_image(img, path, path_out, args.confidence_threshold)




