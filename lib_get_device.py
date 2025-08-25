#
### Import Modules. ###
#
from typing import cast
#
import torch


#
def get_best_device() -> torch.device | str:

    """
    Automatically selects the best available device for PyTorch.

    The selection priority is:
    1. NVIDIA GPU (CUDA) and AMD GPU (ROCm) [uses CUDA interface]
    2. Apple Silicon (MPS)
    3. Intel GPU (XPU via Intel Extension for PyTorch)
    4. Huawei NPU (NPU via torch_npu)
    5. Google TPU (XLA via torch_xla)
    6. CPU (fallback)

    Returns:
        torch.device: The selected device object.
    """

    #
    device: torch.device = torch.device('cpu')

    #
    ### NVIDIA GPU (CUDA) and AMD GPU (ROCm) [uses CUDA interface]. ###
    #
    if torch.cuda.is_available():
        #
        device = torch.device('cuda')
        #
        print(f"Using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
        #
        return device

    #
    ### Apple Silicon (MPS). ###
    #
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
        print("Using Apple Silicon GPU (MPS)")
        return device

    #
    ### Intel's XPU (requires Intel Extension for PyTorch). ###
    #
    try:
        #
        import intel_extension_for_pytorch as ipex  # type: ignore
        #
        if torch.xpu.is_available():
            #
            device = torch.device('xpu')
            #
            print("Using Intel GPU (XPU)")
            #
            return device
    #
    except ImportError:
        #
        pass

    #
    ### Huawei's NPU (requires torch_npu). ###
    #
    try:
        #
        import torch_npu  # type: ignore
        #
        if torch_npu.npu.is_available():  # type: ignore
            #
            device = torch.device('npu')
            #
            print("Using Huawei NPU")
            #
            return device
    #
    except ImportError:
        #
        pass

    #
    ### Google's TPU (primarily for Google Cloud). ###
    #
    try:
        #
        import torch_xla.core.xla_model as xm  # type: ignore
        #
        ### This check is for environments where torch_xla is already configured ###
        ### to find a TPU, like on Google Cloud. ###
        #
        if xm.get_xla_supported_devices('TPU'):  # type: ignore
            #
            device: torch.device = cast(torch.device, xm.xla_device())  # type: ignore
            #
            print("Using Google TPU")
            #
            return device
    #
    except ImportError:
        #
        pass

    #
    ### Fallback to CPU if nothing else found. ###
    #
    device = torch.device('cpu')
    #
    print("No accelerator found, defaulting to CPU")
    #
    return device
