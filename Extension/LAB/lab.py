import torch, numpy as np
from skimage import color


class LAB():
    """
    LAB color transfer
    """
    def transform(self, images: torch.Tensor, target_images: torch.Tensor) -> torch.Tensor:
        """
        Applies LAB color transfer from target_images to images for a batch.

        Args:
            images (torch.Tensor): Batch of images in RGB, shape (B, C, H, W), range [0, 1] or [0, 255].
            target_images (torch.Tensor): Batch of target images in RGB, shape (B, C, H, W), same range as images.

        Returns:
            torch.Tensor: Batch of LAB color transferred images, same shape as input.
        """
        assert images.shape == target_images.shape, "Source and target batch must have same shape"
        images_np = images.permute(0, 2, 3, 1).cpu().numpy()  # (B, H, W, C)
        targets_np = target_images.permute(0, 2, 3, 1).cpu().numpy()
        
        if images_np.max() > 1.1:
            images_np, targets_np = images_np / 255.0, targets_np / 255.0

        out_images = []
        for i in range(images_np.shape[0]): 
            src_lab = color.rgb2lab(images_np[i])
            trg_lab = color.rgb2lab(targets_np[i])

            lab_trans = ((src_lab - np.mean(src_lab, axis=(0, 1))) / np.std(src_lab, axis=(0, 1))) * np.std(trg_lab, axis=(0, 1)) + np.mean(trg_lab, axis=(0, 1))
            out_images.append((np.clip(color.lab2rgb(lab_trans), 0, 1) * 255).astype(np.uint8))

        return torch.from_numpy(np.stack(out_images, axis=0)).permute(0, 3, 1, 2).float() / 255.0