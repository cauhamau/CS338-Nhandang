from vietocr.tool.predictor import Predictor
from vietocr.tool.config import Cfg
import cv2
import os
from PIL import Image
import argparse
from tqdm import tqdm
import sys




def parse_args():
    parser = argparse.ArgumentParser(description='Inference')
    parser.add_argument('--cfg_name', type=str, default = 'vgg_transformer',  help='Config file name')
    parser.add_argument('--weights', default = './weights/vietocr.pth' ,type=str, help='Checkpoint file path')
    parser.add_argument('--path_detect', type=str, default="../../Results/dict-guided", help='Path of results detector: txt file ')
    parser.add_argument('--path_img', type=str, 
                                        default = "images",
                                        help='input image path of folder')
    parser.add_argument('--device', default="cuda:0")                           
    parser.add_argument('--path_output', type=str, default="../../Results/vietocr" , help='paht out of results')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    config = Cfg.load_config_from_name(args.cfg_name)
    config['vocab'] = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~° ' + '̉'+ '̀' + '̃'+ '́'+ '̣'
    config['device'] = args.device
    config['weights'] = args.weights

    print(f'LOAD CONFIG: {args.cfg_name}')

    detector = Predictor(config)

    if not os.path.exists(args.path_output):
        os.makedirs(args.path_output)

    for img_name in tqdm(sorted(os.listdir(args.path_img))):
        filename = img_name.split('.')[0]
        
        img = cv2.imread(os.path.join(args.path_img, img_name), 0)
        out_txt = open(os.path.join(args.path_output, f'{filename}.txt'), 'w', encoding="utf-8")
        try:
            file_txt = open(os.path.join(args.path_detect, f'{filename}.txt'), 'r', encoding='utf-8')
        except:
            out_txt.close()
            file_txt.close()
            continue
            
        lines = file_txt.readlines()
        for line in lines:
            prob_detect = line.split('\t')[-1][:-1]
            bbox = line.split('\t')[0]
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
            
            if args.submission==True:
                content = ','.join([str(p) for p in bbox]) + ',' + pred + '\n'
            else:
                content = str(prob_detect[:-1]) + ' '
            out_txt.write(content)
            
        out_txt.close()
        file_txt.close()
if __name__ == '__main__':
    main()
