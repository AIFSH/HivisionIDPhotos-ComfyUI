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
from hivision.creator.human_matting import (
    extract_human_modnet_photographic_portrait_matting,
    extract_human,
    extract_human_mnn_modnet,
)

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
                "height":("INT",{
                    "default": 413
                }),
                "width":("INT",{
                    "default": 295
                }),
                "bgcolor":("STRING",{
                    "default": "638cce"
                }),
                "matting_model":(["hivision_modnet", "modnet_photographic_portrait_matting", "mnn_hivision_modnet"],),
                "render":(["pure_color", "updown_gradient", "center_gradient"],)
            }
        }
    
    RETURN_TYPES = ("IMAGE","IMAGE",)
    RETURN_NAMES = ("3ch_img","4ch_img",)

    FUNCTION = "gen_img"

    #OUTPUT_NODE = False

    CATEGORY = "AIFSH_HivisionIDPhotos"

    def gen_img(self,input_img,type,height,width,bgcolor,
                matting_model,render):
        
        print(type)
        print(input_img.shape)
        creator = IDCreator()

        # ------------------- 人像抠图模型选择 -------------------
        if matting_model == "hivision_modnet":
            creator.matting_handler = extract_human
        elif matting_model == "modnet_photographic_portrait_matting":
            creator.matting_handler = extract_human_modnet_photographic_portrait_matting
        elif matting_model == "mnn_hivision_modnet":
            creator.matting_handler = extract_human_mnn_modnet

        img_np = input_img.numpy()[0] * 255
        img_np = img_np.astype(np.uint8)
        input_image = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGB)

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
            hd_cv2 = cv2.cvtColor(result.hd,cv2.COLOR_BGR2RGBA)
            hd_img = torchvision.transforms.ToTensor()(hd_cv2).permute(1,2,0).unsqueeze(0)
            # print(hd_img.shape)
            return (hd_img, hd_img,)


        # 如果模式是添加背景
        elif type == "add_background":
            # 将字符串转为元组
            input_image = cv2.cvtColor(img_np,cv2.COLOR_BGR2RGBA)
            color = hex_to_rgb(bgcolor)
            # 将元祖的 0 和 2 号数字交换
            color = (color[2], color[1], color[0])

            result_image = add_background(
                input_image, bgr=color, mode=render
            )
            result_image = result_image.astype(np.uint8)
            hd_cv2 = cv2.cvtColor(result_image,cv2.COLOR_BGR2RGBA)
            result_image = torchvision.transforms.ToTensor()(hd_cv2).permute(1,2,0).unsqueeze(0)
            #print(result_image.shape)
            return (result_image, result_image,)
            

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
            result_layout_image = cv2.cvtColor(result_layout_image,cv2.COLOR_BGR2RGB)
            result_image = torchvision.transforms.ToTensor()(result_layout_image).permute(1,2,0).unsqueeze(0)
            return (result_image, result_image,)

NODE_CLASS_MAPPINGS = {
    "HivisionNode": HivisionNode
}

            
                