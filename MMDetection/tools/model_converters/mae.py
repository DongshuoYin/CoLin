import torch
import argparse
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='Hyperparams')
parser.add_argument('filename', nargs='?', type=str, default=None)

args = parser.parse_args()

ckpt = torch.load(args.filename, map_location=torch.device('cpu'))
ckpt = ckpt['model']

out_ckpt = {}
for k, v in ckpt.items():
    if "mask_token" in k:
        continue
    out_ckpt[k] = v

torch.save(out_ckpt, args.filename.replace(".pth", "_mmdet.pth"))