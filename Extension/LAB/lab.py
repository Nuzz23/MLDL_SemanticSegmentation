import torch
from skimage import color


class LAB():
    """
    LAB color transfer.
    It does so by converting images from RGB to LAB color space,
    normalizing the LAB channels of the source image, and then applying
    the mean and standard deviation of the target image's LAB channels.
    The output is then converted back to RGB color space.
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
        images, target_images = images.clone().permute(0, 2, 3, 1), target_images.clone().permute(0, 2, 3, 1)  # (B, H, W, C)

        if images.max() > 1.1:
            images, target_images = images / 255.0, target_images / 255.0

        out_images = []
        for i in range(images.shape[0]):
            src_lab = color.rgb2lab(images[i])
            trg_lab = color.rgb2lab(target_images[i])

            lab_trans = ((src_lab - torch.mean(src_lab, axis=(0, 1))) / (torch.std(src_lab, axis=(0, 1)) + 1e-8)) * torch.std(trg_lab, axis=(0, 1)) + torch.mean(trg_lab, axis=(0, 1))
            out_images.append((torch.clip(color.lab2rgb(lab_trans), 0, 1) * 255).byte())

        return torch.stack(out_images, axis=0).permute(0, 3, 1, 2).float() / 255.0