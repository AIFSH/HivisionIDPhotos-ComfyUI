import os,sys
now_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.append(now_dir)

import cv2
import numpy as np
import torch
import torchvision
from hivision.error import FaceError
from hivision.utils import hex_to_rgb, resize_image_to_kb, add_background
from hivision import IDCreator
from hivision.creator.layout_calculator import (
    generate_layout_photo,
    generate_layout_image,
)
from hivision.demo.utils import csv_to_size_list
from hivision.creator.choose_handler import choose_handler

size_list_dict_CN = csv_to_size_list(os.path.join(now_dir, "hivision/demo/size_list_CN.csv"))
size_list_CN = list(size_list_dict_CN.keys())
color_list_dict_CN = {
    "蓝色": (86, 140, 212),
    "白色": (255, 255, 255),
    "红色": (233, 51, 35),
}
color_list = list(color_list_dict_CN.keys())

class HivisionNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "input_img":("IMAGE",),
                "type":([
                    "idphoto",
                    "human_matting",
                    "add_background",
                    "generate_layout_photos",
                ],),
                "kb":("INT",{
                    "default": 300
                }),
                "size":(size_list_CN,),
                "bgcolor":(color_list,),
                "matting_model":(["hivision_modnet", "modnet_photographic_portrait_matting", "mnn_hivision_modnet","rmbg-1.4"],),
                "render":(["pure_color", "updown_gradient", "center_gradient"],)
            }
        }
    
    RETURN_TYPES = ("IMAGE","IMAGE",)
    RETURN_NAMES = ("3ch_standard_img","4ch_hd_img",)

    FUNCTION = "gen_img"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_HivisionIDPhotos"

    def gen_img(self,input_img,type,kb,size,bgcolor,
                matting_model,render):
        
        print(type)
        print(input_img.shape)
        creator = IDCreator()

        # ------------------- 人像抠图模型选择 -------------------
        choose_handler(creator,matting_model_option=matting_model,
                       face_detect_option="mtcnn")

        img_np = input_img.numpy()[0] * 255
        img_np = img_np.astype(np.uint8)
        input_image = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)
        height, width = size_list_dict_CN[size]

        # 如果模式是生成证件照
        if type == "idphoto":
            # 将字符串转为元组
            size = (height, width)
            try:
                result = creator(input_image, size=size)
            except FaceError:
                print("人脸数量不等于 1，请上传单张人脸的图像。")
            else:
                standard_cv2 = cv2.cvtColor(result.standard,cv2.COLOR_BGR2RGB)
                hd_cv2 = cv2.cvtColor(result.hd,cv2.COLOR_BGR2RGBA)
                standard_img = torchvision.transforms.ToTensor()(standard_cv2)
                standard_img = standard_img.permute(1,2,0).unsqueeze(0)
                hd_img = torchvision.transforms.ToTensor()(hd_cv2).permute(1,2,0).unsqueeze(0)
                # out_put = torch.cat((standard_img,hd_img))
                # print(hd_img.shape)
                return (standard_img, hd_img)

        # 如果模式是人像抠图
        elif type == "human_matting":
            result = creator(input_image, change_bg_only=True)
            standard_cv2 = cv2.cvtColor(result.standard,cv2.COLOR_BGR2RGB)
            standard_img = torchvision.transforms.ToTensor()(standard_cv2).permute(1,2,0).unsqueeze(0)
            hd_cv2 = cv2.cvtColor(result.hd,cv2.COLOR_BGR2RGBA)
            hd_img = torchvision.transforms.ToTensor()(hd_cv2).permute(1,2,0).unsqueeze(0)
            # print(hd_img.shape)
            return (standard_img, hd_img,)


        # 如果模式是添加背景
        elif type == "add_background":
            # 将字符串转为元组
            input_image = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGBA)
            color = color_list_dict_CN[bgcolor]
            
            # 将元祖的 0 和 2 号数字交换
            color = (color[2], color[1], color[0])

            result_image = add_background(
                input_image, bgr=color, mode=render
            )
            result_image = result_image.astype(np.uint8)

            result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            tmp_img_path = "tmp.png"
            resize_image_to_kb(
                result_image, tmp_img_path, kb
            )

            result_image = cv2.imread(tmp_img_path)

            standard_cv2 = cv2.cvtColor(result_image,cv2.COLOR_BGR2RGB)
            standard_img = torchvision.transforms.ToTensor()(standard_cv2).permute(1,2,0).unsqueeze(0)
            hd_cv2 = cv2.cvtColor(result_image,cv2.COLOR_BGR2RGBA)
            hd_img = torchvision.transforms.ToTensor()(hd_cv2).permute(1,2,0).unsqueeze(0)
            #print(result_image.shape)
            return (standard_img, hd_img,)
            

        # 如果模式是生成排版照
        elif type == "generate_layout_photos":

            size = (height,width)

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
            result_layout_image = cv2.cvtColor(result_layout_image, cv2.COLOR_RGB2BGR)

            tmp_img_path = "tmp.png"
            resize_image_to_kb(
                result_layout_image, tmp_img_path, kb
            )

            result_layout_image = cv2.imread(tmp_img_path)

            standard_cv2 = cv2.cvtColor(result_layout_image,cv2.COLOR_BGR2RGB)
            standard_img = torchvision.transforms.ToTensor()(standard_cv2).permute(1,2,0).unsqueeze(0)
            hd_cv2 = cv2.cvtColor(result_layout_image,cv2.COLOR_BGR2RGBA)
            hd_img = torchvision.transforms.ToTensor()(hd_cv2).permute(1,2,0).unsqueeze(0)
            return (standard_img, hd_img,)

NODE_CLASS_MAPPINGS = {
    "HivisionNode": HivisionNode
}

            
                