"""Growing Neural Cellular Automata for the CRITICALLY THOMAS wordmark.

Following Mordvintsev, Randazzo, Niklasson & Levin, "Growing Neural Cellular
Automata", Distill 2020. https://distill.pub/2020/growing-ca/

Target: target.npy, a 25 x 109 RGBA image. Alpha is binary and doubles as the
alive mask; RGB is premultiplied, so dead cells are exactly zero.

The cell state is 16 channels:
    0..2    RGB      visible colour
    3       alpha    visible, and the "this cell is alive" flag
    4..15   hidden   scratch space, meaning learned during training
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import logging
logger = logging.getLogger(__name__)

HERE = Path(__file__).parent
TARGET = HERE / "target.npy"
NN_PATH = HERE / "nca.pt"

CHANNELS = 16
ALPHA = 3          # channel index of alpha
ALIVE_THRESHOLD = 0.1
FIRING_PROBABILITY = 0.5

BATCH_SIZE = 8
POOL_SIZE = 1024


def pick_device() -> torch.device:
    """Step 1."""
    # Set the pytorch device
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"

    logger.debug(f"Torch device: {device_type}")
    return torch.device(device_type)



def load_target(device: torch.device, target_path: Path = TARGET) -> torch.Tensor:
    """Step 1. -> (1, 4, H, W) float32 on `device`, values in [0, 1]."""
    target = np.load(target_path).astype("float32")
    tensor = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0).to(device)
    logger.debug(f"Loaded tensor with shape {tensor.shape} and type {tensor.dtype}")
    return tensor



def make_seed(height: int, width: int, batch: int, device: torch.device) -> torch.Tensor:
    """Step 2. -> (batch, 16, H, W), a single living cell at the centre."""
    seed = torch.zeros(size=(batch, CHANNELS, height, width), device=device)
    seed[:, ALPHA:, height//2, width//2] = 1.0
    return seed


def to_rgb(state: torch.Tensor) -> torch.Tensor:
    """Step 3. -> (batch, 3, H, W) visible image, composited onto white."""
    mask = state[:, ALPHA:ALPHA+1, :, :].clamp(0.0, 1.0)
    source = state[:, :ALPHA, :, :] 
    return source + (1.0 - mask)  # white background

def perception_kernel(device: torch.device) -> torch.Tensor:
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=device)/8.0
    sobel_y = sobel_x.T
    identity = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32, device=device)
    return torch.stack([identity, sobel_x, sobel_y], dim=0).unsqueeze(1).repeat(CHANNELS, 1, 1, 1)

def apply_kernel(state: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    return F.conv2d(state, kernel, padding=1, groups=CHANNELS) # Precompute on CPU for efficiency

class NCA(nn.Module):

    def __init__(self, perception_kernel: torch.Tensor, channels: int = CHANNELS):
        super().__init__()
        self.channels = channels
        self.register_buffer("perception_kernel", perception_kernel)
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=channels * 3, out_channels=128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=channels, kernel_size=1)
        )
        nn.init.zeros_(self.model[-1].weight)
        nn.init.zeros_(self.model[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pre_life = F.max_pool2d(x[:, ALPHA:ALPHA+1, :, :], kernel_size=3, stride=1, padding=1) > ALIVE_THRESHOLD

        dx = apply_kernel(x, self.perception_kernel)
        dx = self.model(dx)

        update_mask = torch.rand(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device) > FIRING_PROBABILITY

        x = x + dx * update_mask.float()

        post_life = F.max_pool2d(x[:, ALPHA:ALPHA+1, :, :], kernel_size=3, stride=1, padding=1) > ALIVE_THRESHOLD

        x = x * (pre_life & post_life).float()

        return x

def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return ((pred - target) ** 2).mean(dim=[1,2,3])

def train(nca: NCA, target, batch, optimizer) -> float:
    optimizer.zero_grad()

    for _ in range(np.random.randint(64, 97)):
        batch = nca(batch)

    loss = loss_fn(batch[:, :4], target).mean()
    loss.backward()
    for p in nca.parameters():
        p.grad /= (p.grad.norm() + 1e-8)
    optimizer.step()
    return batch, loss


def run_training_loop(nca: NCA, target: torch.Tensor, steps: int = 1000, lr: float = 1e-3):
    optimizer = torch.optim.Adam(nca.parameters(), lr=lr)

    main_pool = make_seed(target.shape[2], target.shape[3], batch=POOL_SIZE, device=target.device)

    for step in range(steps):
        with torch.no_grad():

            batch_indices = np.random.choice(POOL_SIZE, size=BATCH_SIZE, replace=False)
            batch = main_pool[batch_indices]

            # reset the worst-performing seed in the pool to a new random seed
            batch_loss = loss_fn(batch[:, :4], target)
            index_max = batch_loss.argmax()
            batch[index_max] = make_seed(target.shape[2], target.shape[3], batch=1, device=target.device)

        batch, loss = train(nca, target, batch, optimizer)
        if step % 100 == 0:
            logger.info(f"Step {step}: Loss {loss.item()}")

        batch = batch.detach()

        main_pool[batch_indices] = batch

def generate_result(nca: NCA, target: torch.Tensor, steps: int = 10000) -> torch.Tensor:
    """Generate a result from a trained NCA."""
    images = []
    with torch.no_grad():
        batch = make_seed(target.shape[2], target.shape[3], batch=1, device=target.device)
        for _ in range(steps):
            batch = nca(batch)
            images.append(to_rgb(batch).cpu().numpy())
    return np.concatenate(images, axis=0)



if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    device = pick_device()
    target = load_target(device)

    # see if nn exists already
    if NN_PATH.exists():
        logger.info(f"Loading NCA from {NN_PATH}")
        nca = torch.load(NN_PATH, map_location=device, weights_only=False)
    else:
        logger.info("Training NCA from scratch")
        nca = NCA(perception_kernel(device)).to(device)

    # train the NCA

    run_training_loop(nca, target, steps=10000, lr=1e-3)

    # save the NCA
    torch.save(nca, NN_PATH)

    # display a GIF in PIL
    from PIL import Image
    images = generate_result(nca, target, steps=1000)
    images = (images * 255).astype(np.uint8)
    pil_images = [Image.fromarray(img.transpose(1, 2, 0)) for img in images]
    pil_images[0].save("result.gif", save_all=True, append_images=pil_images[1:], duration=50, loop=0)



