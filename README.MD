#ComfyUI_MaskImagePlacement
适用于ComfyUI的图像通过遮罩（图片）匹配的节点插件，实现替换背景图像内容的一个基础功能

创建这个插件的原因主要在于我找不到合适的方法去解决图像通过遮罩快速匹配到对应位置的这个头疼问题

目前插件没有额外的节点内容，只有“MaskImagePlacement遮罩图像替换”

## 功能
- 自动裁剪前景图像的透明边缘，避免无效区域影响缩放
- 根据mask自动定位前景位置，支持多种适配策略（fit/fill/stretch）
- 对mask区域的背景进行模糊和羽化处理，提升融合效果
- 可调节padding、alpha阈值等参数，灵活适配不同场景

## 安装方法
1. 确保已安装ComfyUI
2. 进入ComfyUI的`custom_nodes`目录
3. 克隆本仓库：
   ```bash
   git clone https://github.com/2559923437/ComfyUI_MaskImagePlacement.git
