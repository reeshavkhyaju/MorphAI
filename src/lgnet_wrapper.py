# import torch
# import numpy as np
# from PIL import Image
# import torchvision.transforms as T
# import torch.nn.functional as F

# # Import your landmark generator and config
# from src.landmarks import generate_landmark_heatmap
# from config import MODEL_WEIGHTS_PATH, DEVICE, IMAGE_SIZE

# class LGNetInference:
#     def __init__(self, weights_path: str = MODEL_WEIGHTS_PATH, device: str = DEVICE):
#         self.device = torch.device(device)
#         self.image_size = IMAGE_SIZE  # e.g., (256, 256)
        
#         # 1. Load LGNet Model Architecture
#         # Replace LGNetModel() with your actual PyTorch model class
#         self.model = LGNetModel().to(self.device)
        
#         # 2. Load Weights
#         checkpoint = torch.load(weights_path, map_location=self.device)
#         state_dict = checkpoint.get("state_dict", checkpoint)
#         self.model.load_state_dict(state_dict)
#         self.model.eval()

#     @torch.inference_mode()
#     def predict(self, image_pil: Image.Image, mask_pil: Image.Image) -> Image.Image:
#         """
#         Main prediction method for the backend API.
        
#         Args:
#             image_pil: Original RGB image (PIL)
#             mask_pil: Grayscale mask image (PIL, white pixels = area to inpaint)
            
#         Returns:
#             Inpainted RGB image (PIL)
#         """
#         # Preprocessing & Resizing
#         image_rgb = image_pil.convert("RGB").resize(self.image_size)
#         mask_gray = mask_pil.convert("L").resize(self.image_size)

#         img_np = np.array(image_rgb)
#         mask_np = (np.array(mask_gray) > 127).astype(np.float32)  # Binary mask [H, W]

#         # 1. Generate 1-ch Landmark Heatmap on the fly
#         heatmap_np = generate_landmark_heatmap(img_np, target_size=self.image_size)

#         # 2. Mask the RGB Image
#         masked_img_np = img_np * (1.0 - mask_np[..., None])

#         # 3. Build Tensors & Normalize
#         # RGB normalized to [-1, 1]
#         masked_tensor = torch.from_numpy(masked_img_np).permute(2, 0, 1).float() / 127.5 - 1.0
#         # Mask and Heatmap normalized to [0, 1]
#         mask_tensor = torch.from_numpy(mask_np).unsqueeze(0).float()
#         heatmap_tensor = torch.from_numpy(heatmap_np).unsqueeze(0).float()

#         # 4. Concatenate into 5-channel Tensor [1, 5, H, W]
#         input_5ch = torch.cat([masked_tensor, mask_tensor, heatmap_tensor], dim=0).unsqueeze(0).to(self.device)

#         # 5. Model Inference
#         output_tensor = self.model(input_5ch)

#         # 6. Post-processing (Denormalize [-1, 1] -> [0, 255])
#         output_np = ((output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

#         # 7. Alpha Blend: Paste restored pixels only inside the mask area
#         final_np = output_np * mask_np[..., None].astype(np.uint8) + img_np * (1 - mask_np[..., None].astype(np.uint8))

#         return Image.fromarray(final_np)