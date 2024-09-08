# HivisionIDPhotos-ComfyUI
a custom node for [HivisionIDPhotos](https://github.com/Zeyi-Lin/HivisionIDPhotos),you can find [workflow](./doc/证件照_workflow.json)

## Example
| 输入 | 抠图 | 换背景 | 排版 |
| -- | -- | -- | -- |
| ![](./doc/demo.jpg) [source](https://www.liblib.art/imageinfo/b7cb6b18b2af4c37be8607b648b52979) | ![](./doc/ComfyUI_temp_movvp_00002_.png) ![](./doc/ComfyUI_temp_igzmq_00002_.png) | ![](./doc/ComfyUI_temp_byppo_00004_.png) | ![](./doc/ComfyUI_temp_jeppc_00005_.png) |

## 教程
- [Demo](https://www.bilibili.com/video/BV1iFpvegEY3/)
- [一键包](https://b23.tv/QFgmoXM)
  
## weights
存到项目的`ComfyUI/custom_nodes/HivisionIDPhotos-ComfyUI/hivision/creator/weights`目录下：
- `modnet_photographic_portrait_matting.onnx` (24.7MB): [MODNet](https://github.com/ZHKKKe/MODNet)官方权重，[下载](https://github.com/Zeyi-Lin/HivisionIDPhotos/releases/download/pretrained-model/modnet_photographic_portrait_matting.onnx)
- `hivision_modnet.onnx` (24.7MB): 对纯色换底适配性更好的抠图模型，[下载](https://github.com/Zeyi-Lin/HivisionIDPhotos/releases/download/pretrained-model/hivision_modnet.onnx)
- `mnn_hivision_modnet.mnn` (24.7MB): mnn转换后的抠图模型 by [zjkhahah](https://github.com/zjkhahah)，[下载](https://github.com/Zeyi-Lin/HivisionIDPhotos/releases/download/pretrained-model/mnn_hivision_modnet.mnn)
- `rmbg-1.4.onnx` (176.2MB): [BRIA AI](https://huggingface.co/briaai/RMBG-1.4) 开源的抠图模型，[下载](https://huggingface.co/briaai/RMBG-1.4/resolve/main/model.pth?download=true)后重命名为`rmbg-1.4.onnx`

