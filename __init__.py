# ComfyUI_MaskImagePlacement
# 版本: 1.0.0
# 作者: WuMIn_059
# 描述: 基于mask自动定位、缩放前景图像，并支持背景模糊和羽化的ComfyUI节点



import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from comfy.utils import common_upscale

class MaskBasedImagePlacement:
    """MaskImagePlacement遮罩图像替换"""
    
    CATEGORY = "图像编辑/合成"
    DESCRIPTION = "裁剪前景透明边缘，对mask区域背景进行模糊和羽化处理后自动合成"
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "background_image": ("IMAGE",),  # 背景图
                "foreground_image": ("IMAGE",),  # 前景图（可含透明通道）
                "mask_image": ("IMAGE",),        # 位置mask（黑白图）
                "padding": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),  # 边界留白
                "fit_strategy": (["fit", "fill", "stretch"], {"default": "fit"}),      # 适配策略
                "alpha_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),  # alpha裁剪阈值
                "feather_amount": ("INT", {"default": 5, "min": 0, "max": 50, "step": 1}),  # 羽化程度（像素）
                "blur_strength": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1}),  # 模糊强度
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("合成图像",)
    FUNCTION = "place_image"
    
    def find_mask_bounds(self, mask):
        """找到mask非零区域的边界框"""
        if len(mask.shape) > 2:
            mask = mask.mean(axis=2)  # 转单通道
        non_zero = np.where(mask > 0.5)  # mask阈值0.5
        if not non_zero[0].size:
            return (0, 0, mask.shape[1], mask.shape[0])  # 全零则返回整图
        min_y, max_y = np.min(non_zero[0]), np.max(non_zero[0])
        min_x, max_x = np.min(non_zero[1]), np.max(non_zero[1])
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def crop_transparent_edges(self, image, alpha_threshold=0.5):
        """裁剪前景图像的透明边缘（保留有效内容）"""
        if image.shape[-1] == 3:
            return image  # 无alpha通道，直接返回
        alpha = image[..., 3] / 255.0  # 归一化到0-1
        non_zero = np.where(alpha > alpha_threshold)  # 超过阈值的像素视为有效
        if not non_zero[0].size:
            return image  # 全透明，返回原图
        min_y, max_y = np.min(non_zero[0]), np.max(non_zero[0])
        min_x, max_x = np.min(non_zero[1]), np.max(non_zero[1])
        return image[min_y:max_y+1, min_x:max_x+1, :]  # 裁剪有效区域
    
    def resize_foreground(self, foreground, target_w, target_h, fit_strategy):
        """调整前景尺寸（基于裁剪后的有效区域）"""
        h, w = foreground.shape[:2]
        if fit_strategy == "stretch":
            scale_w, scale_h = target_w / w, target_h / h
        else:
            scale = min(target_w / w, target_h / h) if fit_strategy == "fit" else max(target_w / w, target_h / h)
            scale_w, scale_h = scale, scale
        new_w, new_h = int(w * scale_w), int(h * scale_h)
        
        # 转换为ComfyUI支持的张量格式
        if len(foreground.shape) == 3:
            tensor = torch.from_numpy(foreground).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        else:
            tensor = torch.from_numpy(foreground).unsqueeze(0).unsqueeze(0).float() / 255.0
        
        # 缩放
        resized = common_upscale(tensor, new_w, new_h, "bilinear", False)
        resized_np = resized.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (resized_np * 255).astype(np.uint8)
    
    def feather_and_blur_background(self, background, mask, x, y, w, h, feather_amount, blur_strength, padding=0):
        """对mask区域的背景进行羽化和模糊处理"""
        # 扩展边界以处理羽化和padding
        expanded_x = max(0, x - padding - feather_amount)
        expanded_y = max(0, y - padding - feather_amount)
        expanded_w = min(background.shape[1] - expanded_x, w + 2 * (padding + feather_amount))
        expanded_h = min(background.shape[0] - expanded_y, h + 2 * (padding + feather_amount))
        
        # 提取扩展区域
        background_region = background[expanded_y:expanded_y+expanded_h, expanded_x:expanded_x+expanded_w].copy()
        
        # 创建区域内的mask
        mask_roi = mask[expanded_y:expanded_y+expanded_h, expanded_x:expanded_x+expanded_w]
        if len(mask_roi.shape) > 2:
            mask_roi = mask_roi.mean(axis=2)  # 转为单通道
        mask_roi = mask_roi / 255.0  # 归一化到0-1
        
        # 应用羽化（通过高斯模糊实现边缘过渡）
        if feather_amount > 0:
            mask_roi = gaussian_filter(mask_roi, sigma=feather_amount)
        
        # 对背景区域应用模糊
        blurred_region = np.zeros_like(background_region, dtype=np.float32)
        if blur_strength > 0:
            # 分别对每个通道进行模糊
            for c in range(3):
                blurred_region[..., c] = gaussian_filter(background_region[..., c].astype(np.float32), sigma=blur_strength)
        else:
            blurred_region = background_region.astype(np.float32)
        
        # 混合原始背景和模糊背景（基于mask实现渐变过渡）
        result_region = np.zeros_like(background_region, dtype=np.uint8)
        for c in range(3):
            original = background_region[..., c].astype(np.float32)
            blurred = blurred_region[..., c]
            # 核心混合逻辑：mask中心区域完全模糊，边缘渐变过渡到原始背景
            mixed = original * (1 - mask_roi) + blurred * mask_roi
            result_region[..., c] = mixed.astype(np.uint8)
        
        # 将处理后的区域放回原图
        background_copy = background.copy()
        background_copy[expanded_y:expanded_y+expanded_h, expanded_x:expanded_x+expanded_w] = result_region
        
        return background_copy
    
    def place_image(self, background_image, foreground_image, mask_image, padding=0, fit_strategy="fit", 
                   alpha_threshold=0.5, feather_amount=5, blur_strength=2.0):
        """主逻辑：裁剪→处理背景→缩放→合成"""
        # 转换为numpy数组（0-255）
        background = (background_image[0].cpu().numpy() * 255).astype(np.uint8)
        foreground = (foreground_image[0].cpu().numpy() * 255).astype(np.uint8)
        mask = (mask_image[0].cpu().numpy() * 255).astype(np.uint8)
        
        # 1. 定位mask区域
        x, y, w, h = self.find_mask_bounds(mask)
        
        # 2. 对mask区域的背景进行羽化和模糊处理
        processed_background = self.feather_and_blur_background(
            background, mask, x, y, w, h, feather_amount, blur_strength, padding
        )
        
        # 3. 裁剪前景透明边缘
        cropped_foreground = self.crop_transparent_edges(foreground, alpha_threshold)
        if cropped_foreground.size == 0:
            return (torch.from_numpy(processed_background).unsqueeze(0).float() / 255.0,)  # 空图像，返回处理后的背景
        
        # 4. 调整目标区域大小（考虑padding）
        x_padded = max(0, x - padding)
        y_padded = max(0, y - padding)
        target_w = min(processed_background.shape[1] - x_padded, w + 2 * padding)
        target_h = min(processed_background.shape[0] - y_padded, h + 2 * padding)
        
        # 5. 调整前景尺寸以适配mask区域
        resized_foreground = self.resize_foreground(cropped_foreground, target_w, target_h, fit_strategy)
        
        # 6. 处理alpha通道（确保存在）
        if resized_foreground.shape[-1] == 3:
            alpha = np.ones((resized_foreground.shape[0], resized_foreground.shape[1]), dtype=np.uint8) * 255
            resized_foreground = np.concatenate([resized_foreground, alpha[..., np.newaxis]], axis=2)
        alpha = resized_foreground[..., 3] / 255.0  # 归一化到0-1
        
        # 7. 计算放置位置（居中）
        h_fg, w_fg = resized_foreground.shape[:2]
        place_x = x_padded + (target_w - w_fg) // 2
        place_y = y_padded + (target_h - h_fg) // 2
        place_x = max(0, min(place_x, processed_background.shape[1] - w_fg))
        place_y = max(0, min(place_y, processed_background.shape[0] - h_fg))
        
        # 8. 图像合成（alpha混合）
        result = processed_background.copy()
        fg_rgb = resized_foreground[..., :3]
        for c in range(3):
            bg_region = result[place_y:place_y+h_fg, place_x:place_x+w_fg, c].astype(np.float32)
            fg_region = fg_rgb[..., c].astype(np.float32)
            blended = bg_region * (1 - alpha) + fg_region * alpha
            result[place_y:place_y+h_fg, place_x:place_x+w_fg, c] = blended.astype(np.uint8)
        
        # 转换回ComfyUI格式
        result_tensor = torch.from_numpy(result).unsqueeze(0).float() / 255.0
        return (result_tensor,)

# 节点映射
NODE_CLASS_MAPPINGS = {
    "MaskBasedImagePlacement": MaskBasedImagePlacement
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MaskBasedImagePlacement": "MaskImagePlacement遮罩图像替换"
}
    
