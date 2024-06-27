import argparse
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from torchvision import transforms
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import os

def resize_keep_aspectratio(image_src,dst_size):
    src_h,src_w = image_src.shape[:2]
    #print(src_h,src_w)
    dst_h,dst_w = dst_size 
    h = dst_w * (float(src_h)/src_w)
    w = dst_h * (float(src_w)/src_h)
    h = int(h)
    w = int(w)
    if h <= dst_h:
        image_dst = cv2.resize(image_src,(dst_w,int(h)))
    else:
        image_dst = cv2.resize(image_src,(int(w),dst_h))
    h_,w_ = image_dst.shape[:2]
    top = int((dst_h - h_) / 2);
    down = int((dst_h - h_+1) / 2);
    left = int((dst_w - w_) / 2);
    right = int((dst_w - w_+1) / 2);
    
    value = [0,0,0]
    borderType = cv2.BORDER_CONSTANT
    image_dst = cv2.copyMakeBorder(image_dst, top, down, left, right, borderType, None, value)
  
    return image_dst

def detect(frame1):
    old_img_w = old_img_h = imgsz
    old_img_b = 1
    frame1 = cv2.resize(frame1,(640,640))
    img = frame1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Warmup
    if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
        old_img_b = img.shape[0]
        old_img_h = img.shape[2]
        old_img_w = img.shape[3]
    #img = image_tensor
    pred = model(img, augment=opt.augment)[0]
    # Apply NMS
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    pred = pred[0].cpu().detach().numpy()

    return pred

def parser_get():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=custom_weights, help='model.pt path(s)')
    #parser.add_argument('--source', type=str, default=custom_source, help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()

    return opt


def bounding_box(frame, label):
    color_array = [(0, 0, 255), 
    (128, 42, 42), 
    (8, 46, 84), 
    (34, 139, 34), 
    (138, 43, 226), 
    (244, 164, 96)]

    row = label.shape[0]
    for i in range (row):
        x = int(float(label[i][0]))
        y = int(float(label[i][1]))
        w = int(float(label[i][2]))
        h = int(float(label[i][3]))
        #frame = cv2.rectangle(frame, (int(x-w/2), int(y+h/2)), (int(x+w/2), int(y-h/2)), color_array[i], 2, cv2.LINE_AA)
        try:
            frame = cv2.rectangle(frame, (x, y), (w,h), color_array[0], 2, cv2.LINE_AA)
        except:
            print('too much cow')
    return frame


def count_seat(cx,cy, box_w):
    
    for m in range(int(640/box_w)):
        if m*box_w <= cx < (m+1)*box_w:
            i1 = m
        
    for n in range(int(640/box_w)):
        if n*box_w <= cy < (n+1)*box_w:
            j1 = n
    
    return int(i1), int(j1)



    
def act_calculate(path):  
    box_w = 20
    w = int(640/box_w)
    h = int(640/box_w)
    density_before2 = np.zeros((w, h))
    
    vc = cv2.VideoCapture(path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    frame_count = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    print(frame_count)
    video = []
    count = 0
    count_loss = 0
    for idx in range(frame_count):
        vc.set(10, idx)
        _, img = vc.read()
        t12=time.time()
        dtime55 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        txt_path = str(dtime55[0:10])+'.txt'
        if count_loss == 30:
            count_loss = 0
            print("\rprocess: {}/{}".format(idx+1 , frame_count), end = '')
            old_img_w = old_img_h = imgsz
            old_img_b = 1
            img = resize_keep_aspectratio(img, (640,640))
            im2 = img
            pred2 = detect(img)
            t22 = time.time()
            density_now2 = np.zeros((w, h))

            ######
            ##########
            for j2 in range (len(pred2)):
                obj2 = pred2[j2]
                x12 = int(float(obj2[0]))
                y12 = int(float(obj2[1]))
                x22 = int(float(obj2[2]))
                y22 = int(float(obj2[3]))
                cx2 = (x12+x22)/2
                cy2 = (y12+y22)/2
                m2, n2 = count_seat(cx2,cy2,box_w)
                density_now2[n2][m2] = density_now2[n2][m2] + 1
            act2 = density_now2 - density_before2
            act_num2 = 0
            for acty2 in range(w):
                for actx2 in range(h):
                    act_num2 = act_num2 + abs(act2[acty2][actx2])
            act_num2 = act_num2
            density_before2 = density_now2
            print(act_num2, 'yolov7:' , t22-t12, 'act:', time.time()-t22)
            dtime2 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) 
            print(dtime2)
            cv2.imshow('1', im2)

            
            with open('./result/'+txt_path, 'a') as f:
                f.write(str(dtime2)+' '+str(act_num2)+'\n')
        #except:
            #print('cam2 lose')
            if cv2.waitKey(100) == ord('q'):
                break
        else:
            count_loss = count_loss + 1
    cv2.destroyAllWindows()

if __name__ == '__main__':
    
    custom_weights = 'best_chi.pt' 
    opt = parser_get()

    ######################################################
    weights, imgsz= opt.weights, opt.img_size
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    model = TracedModel(model, device, opt.img_size)
    model.half()  # to FP16
    ##########################################################

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

    #path = 'C:/Users/user/Desktop/chicken_activity_dataset/Generic_DAHUA-001-20230713-150240-1689231760247-7.mp4'
    file = './sourse/'
    l = os.listdir(file)
    for k in range(len(l)):
        act_calculate(file + l[k])


