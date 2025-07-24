#ComfyUI_MaskImagePlacement
适用于ComfyUI的图像通过遮罩（图片）匹配的节点插件，实现替换背景图像内容的一个基础功能

创建这个插件的原因主要在于我找不到合适的方法去解决图像通过遮罩快速匹配到对应位置的这个头疼问题

目前插件没有额外的节点内容，只有“MaskImagePlacement遮罩图像替换”

#ComfyUI_MaskImagePlacement 
A node plug-in for ComfyUI that matches images through masks (pictures), realizing a basic function of replacing background image content

The main reason for creating this plug-in is that I can't find a suitable way to solve the headache of quickly matching images to corresponding positions through masks

Currently, the plug-in has no additional node content, only "MaskImagePlacement遮罩图像替换"

## 功能
- 自动裁剪前景图像的透明边缘，避免无效区域影响缩放
- 根据mask自动定位前景位置，支持多种适配策略（fit/fill/stretch）
- 对mask区域的背景进行模糊和羽化处理，提升融合效果
- 可调节padding、alpha阈值等参数，灵活适配不同场景

- ## Function
- Automatically crop the transparent edges of the foreground image to avoid invalid areas affecting scaling
- Automatically locate the foreground position according to the mask, supporting multiple adaptation strategies (fit/fill/stretch)
- Blur and feather the background of the mask area to improve the fusion effect
- Adjustable parameters such as padding and alpha threshold to flexibly adapt to different scenes

## 安装方法
1. 确保已安装ComfyUI
2. 进入ComfyUI的`custom_nodes`目录
3. 克隆本仓库：
   ```bash
   git clone https://github.com/2559923437/ComfyUI_MaskImagePlacement.git

## Installation method
1. Make sure ComfyUI is installed
2. Enter the `custom_nodes` directory of ComfyUI
3. Clone this repository:
   ```bash
   git clone https://github.com/2559923437/ComfyUI_MaskImagePlacement.git
