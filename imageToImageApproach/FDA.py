import torch
import numpy as np


def extractAmplitudePhase(image:torch.Tensor)->tuple[torch.Tensor, torch.Tensor]:
    """
    Extract the amplitude and phase components of the Fourier Transform.
    
    Args:
        image (torch.Tensor): input image tensor after the fft2.
        
    Returns:
        Amplitude and Phase (tuple): amplitude and phase components of the Fourier Transform.
    """
    return torch.abs(image), torch.angle(image)


def FDASourceToTarget(source: torch.Tensor, target: torch.Tensor, beta:float=0.05)->torch.Tensor:
    """
    Apply Fourier Domain Adaptation (FDA) to the source image using the target image.
    
    Args:
        source (torch.Tensor): Source image tensor.
        target (torch.Tensor): Target image tensor.
        beta (float): Parameter to control the frequency swapping.
    
    Returns:
        FDAImage (torch.Tensor): Adapted source image in the target domain. 
    """
    source, target = source.clone(), target.clone()
    
    fftSource, fftTarget = torch.fft.fft2(source, dim=(-2, -1)),torch.fft.fft2(target, dim=(-2, -1))  
    
    ampSource, phaSource = extractAmplitudePhase(fftSource)
    ampTarget, _ = extractAmplitudePhase(fftTarget)

    amp_src = swapLowFrequencies(ampSource, ampTarget, beta)

    return torch.fft.ifft2(amp_src*torch.exp(1j*phaSource))


def swapLowFrequencies(ampSource:torch.Tensor, ampTarget:torch.Tensor, beta:float=0.1)->torch.Tensor:
    """
    Swap the low frequencies of the source and target images.
    
    Args:
        ampSource (torch.Tensor): Amplitude of the source image.
        ampTarget (torch.Tensor): Amplitude of the target image.
        beta (float): Parameter to control the frequency swapping.
    
    Returns:
        swappedAmpSource (torch.Tensor): Amplitude of the source image with swapped low frequencies.
    """

    b = (np.floor(np.amin(ampSource.shape[-2:])*beta)).astype(int)  

    ampSource[:,:,0:b,0:b] = ampTarget[:,:,0:b,0:b]
    return ampSource 