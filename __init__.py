import os,sys
now_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(now_dir)

import cv2
import shutil
import numpy as np
import torch
import torchvision
from hivision.error import FaceError
from hivision.utils import hex_to_rgb, resize_image_to_kb, add_background,add_watermark,save_image_dpi_to_bytes
from hivision import IDCreator
from hivision.creator.layout_calculator import (
    generate_layout_photo,
    generate_layout_image,
)
from hivision.demo.utils import csv_to_size_list,csv_to_color_list
from hivision.creator.choose_handler import choose_handler, HUMAN_MATTING_MODELS


size_list_dict_CN = csv_to_size_list(os.path.join(now_dir, "hivision/demo/assets/size_list_CN.csv"))
size_list_CN = list(size_list_dict_CN.keys())
size_list_dict_EN = csv_to_size_list(os.path.join(now_dir, "hivision/demo/assets/size_list_EN.csv"))
size_list_EN = list(size_list_dict_EN.keys())
color_list_dict_CN = csv_to_color_list(os.path.join(now_dir, "hivision/demo/assets/color_list_CN.csv"))
color_list_CN = list(color_list_dict_CN.keys())

color_list_dict_EN = csv_to_color_list(os.path.join(now_dir, "hivision/demo/assets/color_list_EN.csv"))
color_list_EN = list(color_list_dict_EN.keys())

HUMAN_MATTING_MODELS_EXIST = [
    os.path.splitext(file)[0]
    for file in os.listdir(os.path.join(now_dir, "hivision/creator/weights"))
    if file.endswith(".onnx") or file.endswith(".mnn")
]
# 在HUMAN_MATTING_MODELS中的模型才会被加载到Gradio中显示
HUMAN_MATTING_MODELS = [
    model for model in HUMAN_MATTING_MODELS if model in HUMAN_MATTING_MODELS_EXIST
]


FACE_DETECT_MODELS = ["mtcnn"]
FACE_DETECT_MODELS_EXPAND = (
    ["retinaface-resnet50"]
    if os.path.exists(
        os.path.join(
            now_dir, "hivision/creator/retinaface/weights/retinaface-resnet50.onnx"
        )
    )
    else []
)
FACE_DETECT_MODELS += FACE_DETECT_MODELS_EXPAND

class ENHivisionParamsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "size":(size_list_EN,),
                "bgcolor":(color_list_EN,),
                "render":(["pure_color", "updown_gradient", "center_gradient"],),
                "kb":("INT",{
                    "default": 300,
                }),
                "dpi":("INT",{
                    "default": 300,
                }),
            }
        }

    RETURN_TYPES = ("PARAMS",)
    RETURN_NAMES = ("normal_params",)

    FUNCTION = "get_params"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_HivisionIDPhotos"

    def get_params(self,size,bgcolor,render,kb,dpi):
        parmas = {
            "size":size_list_dict_EN[size],
            "bgcolor": color_list_dict_EN[bgcolor],
            "render":render,
            "kb":kb,
            "dpi":dpi
        }
        return (parmas,)

class ZHHivisionParamsNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "size":(size_list_CN,),
                "bgcolor":(color_list_CN,),
                "render":(["纯色", "上下渐变", "中心渐变"],),
                "kb":("INT",{
                    "default": 300,
                }),
                "dpi":("INT",{
                    "default": 300,
                })
            }
        }

    RETURN_TYPES = ("PARAMS",)
    RETURN_NAMES = ("normal_params",)

    FUNCTION = "get_params"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_HivisionIDPhotos"

    def get_params(self,size,bgcolor,render,kb,dpi):
        if render == "纯色":
            render = "pure_color"
        elif render == "上下渐变":
            render = "updown_gradient"
        else:
            render = "center_gradient"
        parmas = {
            "size":size_list_dict_CN[size],
            "bgcolor": color_list_dict_CN[bgcolor],
            "render":render,
            "kb":kb,
            "dpi":dpi
        }
        return (parmas,)

class AddWaterMarkNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "input_img":("IMAGE",),
                "text":("STRING",{
                    "default": "AIFSH",
                    "multiline": True,
                }),
                "text_color":("STRING",{
                    "default": "#FFFFFF"
                }),
                "text_size":("INT",{
                    "default": 20,
                    "min": 10,
                    "max": 100,
                    "step": 1,
                    "display":"slider"
                }),
                "text_opacity":("FLOAT",{
                    "min":0,
                    "max":1,
                    "default":0.15,
                    "step":0.01,
                    "round":0.001,
                    "display":"slider"
                }),
                "text_angle":("INT",{
                    "default": 30,
                    "min": 0,
                    "max": 360,
                    "step": 1,
                    "display":"slider"
                }),
                "text_space":("INT",{
                    "default": 25,
                    "min": 10,
                    "max": 200,
                    "step": 1,
                    "display":"slider"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "gen_img"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_HivisionIDPhotos"

    def gen_img(self,input_img,text,text_color,text_size,
                text_opacity,text_angle,text_space):
        
        img_np = input_img.numpy()[0] * 255
        img_np = img_np.astype(np.uint8)
        input_image_standard = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
        result_image = add_watermark(image=input_image_standard,
                                     text=text,size=text_size,opacity=text_opacity,
                                     angle=text_angle,space=text_space,color=text_color)

        standard_cv2 = cv2.cvtColor(result_image,cv2.COLOR_BGR2RGB)
        # hd_cv2 = cv2.cvtColor(result_image,cv2.COLOR_BGR2RGBA)
        standard_img = torchvision.transforms.ToTensor()(standard_cv2)
        standard_img = standard_img.permute(1,2,0).unsqueeze(0)
        # hd_img = torchvision.transforms.ToTensor()(hd_cv2).permute(1,2,0).unsqueeze(0)
        return (standard_img,)

class AddBackgroundNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "input_img":("IMAGE",),
                "normal_params":("PARAMS",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("3ch_standard_img","4ch_hd_img",)

    FUNCTION = "gen_img"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_HivisionIDPhotos"

    def gen_img(self,input_img,normal_params):
        img_np = input_img.numpy()[0] * 255
        img_np = img_np.astype(np.uint8)
        input_image = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGBA)
        
        color = normal_params["bgcolor"]
        render = normal_params["render"]

        # print(color)
        color = hex_to_rgb(color)
        # print(color)
        # 将元祖的 0 和 2 号数字交换
        color = (color[2], color[1], color[0])

        result_image = add_background(
            input_image, bgr=color, mode=render
        )
        result_image = result_image.astype(np.uint8)

        standard_cv2 = cv2.cvtColor(result_image,cv2.COLOR_BGR2RGB)
        standard_img = torchvision.transforms.ToTensor()(standard_cv2).permute(1,2,0).unsqueeze(0)
        # hd_cv2 = cv2.cvtColor(result_image,cv2.COLOR_BGR2RGBA)
        # hd_img = torchvision.transforms.ToTensor()(hd_cv2).permute(1,2,0).unsqueeze(0)
        #print(result_image.shape)
        return (standard_img,)

class HivisionLayOutNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "input_img":("IMAGE",),
                "normal_params":("PARAMS",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("3ch_standard_img","4ch_hd_img",)

    FUNCTION = "gen_img"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_HivisionIDPhotos"

    def gen_img(self,input_img,normal_params):
        img_np = input_img.numpy()[0] * 255
        img_np = img_np.astype(np.uint8)
        input_image = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)

        size = normal_params["size"]

        typography_arr, typography_rotate = generate_layout_photo(
            input_height=size[0], input_width=size[1]
        )

        result_layout_image = generate_layout_image(
            input_image,
            typography_arr,
            typography_rotate,
            height=size[0],
            width=size[1],
        )
        # result_layout_image = cv2.cvtColor(result_layout_image, cv2.COLOR_RGB2BGR)
        standard_cv2 = cv2.cvtColor(result_layout_image,cv2.COLOR_BGR2RGB)
        standard_img = torchvision.transforms.ToTensor()(standard_cv2).permute(1,2,0).unsqueeze(0)
        # hd_cv2 = cv2.cvtColor(result_layout_image,cv2.COLOR_BGR2RGBA)
        # hd_img = torchvision.transforms.ToTensor()(hd_cv2).permute(1,2,0).unsqueeze(0)
        return (standard_img,)

class LaterProcessNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "input_img":("IMAGE",),
                "normal_params":("PARAMS",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    # RETURN_NAMES = ("standard_img","hd_img",)

    FUNCTION = "gen_img"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_HivisionIDPhotos"

    def gen_img(self,input_img,normal_params):
        img_np = input_img.numpy()[0] * 255
        img_np = img_np.astype(np.uint8)
        input_image = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
        tmp_img_path = "tmp.png"
        resize_image_to_kb(
            input_image, tmp_img_path, normal_params["kb"]
        )

        result_layout_image = cv2.imread(tmp_img_path)
        save_image_dpi_to_bytes(result_layout_image,tmp_img_path,normal_params['dpi'])
        result_layout_image = cv2.imread(tmp_img_path)
        os.remove(tmp_img_path)

        # result_layout_image = cv2.cvtColor(result_layout_image, cv2.COLOR_RGB2BGR)
        standard_cv2 = cv2.cvtColor(result_layout_image,cv2.COLOR_BGR2RGB)
        standard_img = torchvision.transforms.ToTensor()(standard_cv2).permute(1,2,0).unsqueeze(0)
        # hd_cv2 = cv2.cvtColor(result_layout_image,cv2.COLOR_BGR2RGBA)
        # hd_img = torchvision.transforms.ToTensor()(hd_cv2).permute(1,2,0).unsqueeze(0)
        return (standard_img,)



class HivisionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "input_img":("IMAGE",),
                "normal_params":("PARAMS",),
                "face_alignment":("BOOLEAN",{
                    "default": True
                }),
                "change_bg_only":("BOOLEAN",{
                    "default": False
                }),
                "crop_only":("BOOLEAN",{
                    "default": False
                }),
                "matting_model":(HUMAN_MATTING_MODELS,),
                "face_detect_model":(FACE_DETECT_MODELS,),
                "head_measure_ratio":("FLOAT",{
                    "default": 0.2,
                    "min":0.1,
                    "max":0.5,
                    "step":0.01,
                    "round": 0.001,
                    "display":"slider"
                }),
                "top_distance":("FLOAT",{
                    "default": 0.12,
                    "min":0.02,
                    "max":0.5,
                    "step":0.01,
                    "round": 0.001,
                    "display":"slider"
                }),
                "whitening_strength":("INT",{
                    "default": 2,
                    "min":0,
                    "max":15,
                    "step":1,
                    "display":"slider"
                }),
                 "brightness_strength":("INT",{
                    "default": 0,
                    "min":-5,
                    "max":25,
                    "step":1,
                    "display":"slider"
                }),
                "contrast_strength":("INT",{
                    "default": 0,
                    "min":-10,
                    "max":50,
                    "step":1,
                    "display":"slider"
                }),
                "saturation_strength":("INT",{
                    "default": 0,
                    "min":-10,
                    "max":50,
                    "step":1,
                    "display":"slider"
                }),
                "sharpen_strength":("INT",{
                    "default": 0,
                    "min":0,
                    "max":5,
                    "step":1,
                    "display":"slider"
                }),
            }
        }
    
    RETURN_TYPES = ("IMAGE","IMAGE",)
    RETURN_NAMES = ("standard_img","hd_img",)

    FUNCTION = "gen_img"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_HivisionIDPhotos"
        
    def gen_img(self,input_img,normal_params,face_alignment,change_bg_only,crop_only,matting_model,
                face_detect_model,head_measure_ratio,top_distance,whitening_strength,
                brightness_strength,contrast_strength,saturation_strength,sharpen_strength):
        creator = IDCreator()

        # ------------------- 人像抠图模型选择 -------------------
        choose_handler(creator,matting_model_option=matting_model,
                       face_detect_option=face_detect_model)

        img_np = input_img.numpy()[0] * 255
        img_np = img_np.astype(np.uint8)
        input_image = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
        
        size = normal_params["size"]
        try:
            result = creator(input_image, size=size,
                                head_measure_ratio=head_measure_ratio,
                                head_top_range=(top_distance, top_distance-0.02),
                                change_bg_only=change_bg_only,
                                crop_only=crop_only,
                                face_alignment=face_alignment,
                                whitening_strength=whitening_strength,
                                brightness_strength=brightness_strength,
                                contrast_strength = contrast_strength,
                                sharpen_strength=sharpen_strength,
                                saturation_strength=saturation_strength,)
        except FaceError:
            print("人脸数量不等于 1，请上传单张人脸的图像。")
        else:
            standard_cv2 = cv2.cvtColor(result.standard,cv2.COLOR_BGRA2RGBA)
            hd_cv2 = cv2.cvtColor(result.hd,cv2.COLOR_BGRA2RGBA)
            standard_img = torchvision.transforms.ToTensor()(standard_cv2)
            standard_img = standard_img.permute(1,2,0).unsqueeze(0)
            hd_img = torchvision.transforms.ToTensor()(hd_cv2).permute(1,2,0).unsqueeze(0)
            # out_put = torch.cat((standard_img,hd_img))
            # print(hd_img.shape)
            return (standard_img, hd_img)

        

NODE_CLASS_MAPPINGS = {
    "LaterProcessNode":LaterProcessNode,
    "HivisionNode": HivisionNode,
    "ZHHivisionParamsNode":ZHHivisionParamsNode,
    "ENHivisionParamsNode":ENHivisionParamsNode,
    "AddWaterMarkNode":AddWaterMarkNode,
    "AddBackgroundNode":AddBackgroundNode,
    "HivisionLayOutNode":HivisionLayOutNode,
}

            
                
