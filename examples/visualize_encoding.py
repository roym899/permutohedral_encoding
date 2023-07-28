import time

import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import torch
from PIL import Image

import permutohedral_encoding as permuto_enc

rr.init("permuto", spawn=True)

# create encoding
pos_dim = 2
capacity = pow(2, 10)
nr_levels = 24
nr_feat_per_level = 2
nr_points = 1000
coarsest_scale = 0.1
finest_scale = 0.001

for i, dtype in enumerate([torch.float16, torch.float16, torch.float32, torch.float64]):
    scale_list = np.geomspace(coarsest_scale, finest_scale, num=nr_levels)
    encoding = permuto_enc.PermutoEncoding(
        pos_dim, capacity, nr_levels, nr_feat_per_level, scale_list, dtype=dtype
    )
    network = torch.nn.Sequential(
        encoding,
        torch.nn.Linear(encoding.output_dims(), 32, dtype=dtype),
        torch.nn.GELU(),
        torch.nn.Linear(32, 32, dtype=dtype),
        torch.nn.GELU(),
        torch.nn.Linear(32, 32, dtype=dtype),
        torch.nn.GELU(),
        torch.nn.Linear(32, 3, dtype=dtype),
        # torch.nn.Sigmoid(),
    ).cuda()
    image = np.asarray(Image.open("./test_images/test8.png")) / 255.0
    image = torch.from_numpy(image).to("cuda", dtype=dtype)
    resize_factor = torch.tensor(image.shape[:-1], device="cuda")
    rr.log_image(f"image{dtype}", image, timeless=True)
    optimizer = torch.optim.Adam(
        network.parameters(),
        lr=1e-3 if dtype is torch.float16 else 1e-2,
        eps=1e-5 if dtype is torch.float16 else 1e-8,
    )
    all_points = torch.cartesian_prod(torch.arange(0, 100), torch.arange(0, 100)) / 100
    all_points = all_points.to("cuda", dtype) + (1 / 200.0)

    start_time = time.time()
    for iteration in range(3000):
        points = torch.rand(nr_points, 2, device="cuda", dtype=dtype)
        target_indices = (points * resize_factor).long()
        targets = image[target_indices[:, 0], target_indices[:, 1]]
        predictions = network(points)
        loss = ((targets - predictions).abs()).mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        predictions = network(all_points)
        image_prediction = torch.clamp(predictions.reshape(100, 100, 3), 0, 1)
        if i != 0:
            rr.set_time_sequence("iteration", iteration)
            rr.set_time_seconds("optimization_time", time.time() - start_time)
            rr.log_image(f"prediction{dtype}", image_prediction.numpy(force=True))
            rr.log_scalar(f"loss{dtype}", loss.to(torch.float32).numpy(force=True))
