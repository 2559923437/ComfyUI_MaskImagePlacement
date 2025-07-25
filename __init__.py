# ComfyUI_MaskImagePlacement
# 版本: 1.0.2
# 作者: WuMIn_059
# 描述: 基于mask自动定位、缩放前景图像，并支持背景模糊和羽化的ComfyUI节点



import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from comfy.utils import common_upscale

class MaskBasedImagePlacement:
    """MaskImagePlacement遮罩图像替换"""
    
    CATEGORY = "图像编辑/合成"
    DESCRIPTION = "裁剪前景透明边缘，支持缩放比例、位置偏移，对mask区域背景进行模糊和羽化"
    
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
                # 新增参数：百分比缩放
                "scale_percent": ("INT", {"default": 100, "min": 10, "max": 500, "step": 10}),  # 缩放百分比（10%-500%）
                # 新增参数：XY偏移
                "x_offset": ("INT", {"default": 0, "min": -200, "max": 200, "step": 1}),  # X方向偏移（像素）
                "y_offset": ("INT", {"default": 0, "min": -200, "max": 200, "step": 1}),  # Y方向偏移（像素）
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("合成图像",)
    FUNCTION = "place_image"
    
    # 以下方法保持不变：find_mask_bounds、crop_transparent_edges、feather_and_blur_background
    def find_mask_bounds(self, mask):
        if len(mask.shape) > 2:
            mask = mask.mean(axis=2)
        non_zero = np.where(mask > 0.5)
        if not non_zero[0].size:
            return (0, 0, mask.shape[1], mask.shape[0])
        min_y, max_y = np.min(non_zero[0]), np.max(non_zero[0])
        min_x, max_x = np.min(non_zero[1]), np.max(non_zero[1])
        return (min_x, min_y, max_x - min_x, max_y - min_y)
    
    def crop_transparent_edges(self, image, alpha_threshold=0.5):
        if image.shape[-1] == 3:
            return image
        alpha = image[..., 3] / 255.0
        non_zero = np.where(alpha > alpha_threshold)
        if not non_zero[0].size:
            return image
        min_y, max_y = np.min(non_zero[0]), np.max(non_zero[0])
        min_x, max_x = np.min(non_zero[1]), np.max(non_zero[1])
        return image[min_y:max_y+1, min_x:max_x+1, :]
    
    def feather_and_blur_background(self, background, mask, x, y, w, h, feather_amount, blur_strength, padding=0):
        expanded_x = max(0, x - padding - feather_amount)
        expanded_y = max(0, y - padding - feather_amount)
        expanded_w = min(background.shape[1] - expanded_x, w + 2 * (padding + feather_amount))
        expanded_h = min(background.shape[0] - expanded_y, h + 2 * (padding + feather_amount))
        background_region = background[expanded_y:expanded_y+expanded_h, expanded_x:expanded_x+expanded_w].copy()
        
        mask_roi = mask[expanded_y:expanded_y+expanded_h, expanded_x:expanded_x+expanded_w]
        if len(mask_roi.shape) > 2:
            mask_roi = mask_roi.mean(axis=2)
        mask_roi = mask_roi / 255.0
        
        if feather_amount > 0:
            mask_roi = gaussian_filter(mask_roi, sigma=feather_amount)
        
        blurred_region = np.zeros_like(background_region, dtype=np.float32)
        if blur_strength > 0:
            for c in range(3):
                blurred_region[..., c] = gaussian_filter(background_region[..., c].astype(np.float32), sigma=blur_strength)
        else:
            blurred_region = background_region.astype(np.float32)
        
        result_region = np.zeros_like(background_region, dtype=np.uint8)
        for c in range(3):
            original = background_region[..., c].astype(np.float32)
            blurred = blurred_region[..., c]
            mixed = original * (1 - mask_roi) + blurred * mask_roi
            result_region[..., c] = mixed.astype(np.uint8)
        
        background_copy = background.copy()
        background_copy[expanded_y:expanded_y+expanded_h, expanded_x:expanded_x+expanded_w] = result_region
        return background_copy
    
    # 调整resize逻辑，兼容百分比缩放
    def resize_foreground(self, foreground, target_w, target_h, fit_strategy, scale_percent):
        h, w = foreground.shape[:2]
        # 先根据适配策略计算基础缩放比例
        if fit_strategy == "stretch":
            scale_w, scale_h = target_w / w, target_h / h
        else:
            scale = min(target_w / w, target_h / h) if fit_strategy == "fit" else max(target_w / w, target_h / h)
            scale_w, scale_h = scale, scale
        
        # 应用百分比缩放（转换为0-1比例）
        scale_percent = max(0.1, scale_percent / 100.0)  # 限制最小10%
        scale_w *= scale_percent
        scale_h *= scale_percent
        
        new_w, new_h = int(w * scale_w), int(h * scale_h)
        # 确保尺寸不为0
        new_w = max(1, new_w)
        new_h = max(1, new_h)
        
        # 转换为张量并缩放
        if len(foreground.shape) == 3:
            tensor = torch.from_numpy(foreground).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        else:
            tensor = torch.from_numpy(foreground).unsqueeze(0).unsqueeze(0).float() / 255.0
        
        resized = common_upscale(tensor, new_w, new_h, "bilinear", False)
        resized_np = resized.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return (resized_np * 255).astype(np.uint8)
    
    def place_image(self, background_image, foreground_image, mask_image, padding=0, fit_strategy="fit", 
                   alpha_threshold=0.5, feather_amount=5, blur_strength=2.0, scale_percent=100, x_offset=0, y_offset=0):
        # 转换为numpy数组
        background = (background_image[0].cpu().numpy() * 255).astype(np.uint8)
        foreground = (foreground_image[0].cpu().numpy() * 255).astype(np.uint8)
        mask = (mask_image[0].cpu().numpy() * 255).astype(np.uint8)
        
        # 1. 定位mask区域
        x, y, w, h = self.find_mask_bounds(mask)
        
        # 2. 处理背景模糊和羽化
        processed_background = self.feather_and_blur_background(
            background, mask, x, y, w, h, feather_amount, blur_strength, padding
        )
        
        # 3. 裁剪前景透明边缘
        cropped_foreground = self.crop_transparent_edges(foreground, alpha_threshold)
        if cropped_foreground.size == 0:
            return (torch.from_numpy(processed_background).unsqueeze(0).float() / 255.0,)
        
        # 4. 计算目标区域尺寸（含padding）
        x_padded = max(0, x - padding)
        y_padded = max(0, y - padding)
        target_w = min(processed_background.shape[1] - x_padded, w + 2 * padding)
        target_h = min(processed_background.shape[0] - y_padded, h + 2 * padding)
        
        # 5. 调整前景尺寸（应用百分比缩放）
        resized_foreground = self.resize_foreground(cropped_foreground, target_w, target_h, fit_strategy, scale_percent)
        
        # 6. 处理alpha通道
        if resized_foreground.shape[-1] == 3:
            alpha = np.ones((resized_foreground.shape[0], resized_foreground.shape[1]), dtype=np.uint8) * 255
            resized_foreground = np.concatenate([resized_foreground, alpha[..., np.newaxis]], axis=2)
        alpha = resized_foreground[..., 3] / 255.0
        
        # 7. 计算放置位置（居中 + XY偏移）
        h_fg, w_fg = resized_foreground.shape[:2]
        # 基础居中位置
        base_x = x_padded + (target_w - w_fg) // 2
        base_y = y_padded + (target_h - h_fg) // 2
        # 应用偏移
        place_x = base_x + x_offset
        place_y = base_y + y_offset
        # 边界限制（防止超出背景）
        place_x = max(0, min(place_x, processed_background.shape[1] - w_fg))
        place_y = max(0, min(place_y, processed_background.shape[0] - h_fg))
        
        # 8. 图像合成
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
