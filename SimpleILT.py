
from openFuILT import FuILT
from openFuILT.semantic import SGD
from openFuILT import opc
from pylitho import Abbe
import torch
import torch.nn.functional as func

sigmoidStepness = 4.0
stepness = 50.0
targetIntensity = 0.225

def initializer(mask : torch.Tensor) -> torch.Tensor:
    return  mask * 2.0 - 1.0

def binarizer(mask : torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(sigmoidStepness * mask)

@opc(initializer=initializer, binarizer=binarizer)
def simpleILT(mask : torch.Tensor, target : torch.Tensor) -> torch.Tensor:
    canvas = target.shape[0]
    simulator = Abbe(
        canvas=canvas,
        pixel=14,
        defocus=[0, 30, 60],
        )
    mask = binarizer(mask)
    image = simulator(mask) # NCHW
    image = torch.sigmoid(stepness * (image - targetIntensity))
    return func.mse_loss(image[0][0], target, reduction='sum') + func.mse_loss(image[0][1], image[0][2], reduction='sum')


if __name__ == "__main__":
    # initial FuILT engine
    engine = FuILT(num_worker=2, device_ids=[0, 2]).set_verbose()

    # set layout and pixel size
    engine.set_layout("benchmarks/gcd.gds", layer=11).set_pixel(14)

    # set macro and micro parameters
    engine.set_macro_params(macro_size=[2, 2], 
                            macro_olrate=0.1).set_micro_params(level=2, 
                                                               micro_olrate=0.1)

    # set optimizer, learning rate, and epochs
    engine.set_optimizer(SGD()).set_lr(0.5).set_epoch(10)

    # run FuILT
    engine.solve(simpleILT)
    
    