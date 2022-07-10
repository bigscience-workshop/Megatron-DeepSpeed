"""

python ckptavg.py --ckpts /gpfsscratch/rech/six/commun/checkpoints/tr11-176B-ml/checkpoints/main/global_step95200 /gpfsscratch/rech/six/commun/checkpoints/tr11-176B-ml/checkpoints/main/global_step95100 /gpfsscratch/rech/six/commun/checkpoints/tr11-176B-ml/checkpoints/main/global_step95000 /gpfsscratch/rech/six/commun/checkpoints/tr11-176B-ml/checkpoints/main/global_step94900 /gpfsscratch/rech/six/commun/checkpoints/tr11-176B-ml/checkpoints/main/global_step94800 --output ./

https://github.com/rwightman/pytorch-image-models/blob/master/avg_checkpoints.py
"""

import argparse
import os

import torch

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpts',
                        default=None,
                        nargs='+',
                        required=True,
                        help='Paths to ckpts to avg')
    parser.add_argument('--output', 
                        default='./', 
                        type=str, 
                        metavar='PATH',
                        help='output folder')
    args = parser.parse_args()
    print(f'args = {args}')
    return args


def main():
    args = parse_arguments()

    layer_files = []
    for f in os.listdir(args.ckpts[0]):
        if f.startswith("layer"):
            layer_files.append(f)

    for lf in layer_files:
        new_sd = {}
        for i, ckpt in enumerate(args.ckpts):
            sd = torch.load(os.path.join(ckpt, lf), map_location=torch.device('cpu'))
            if i == 0:
                for k in sd.keys():
                    new_sd[k] = sd[k].clone().to(dtype=torch.float64)
            else:
                for k in sd.keys():
                    new_sd[k] += sd[k].to(dtype=torch.float64)
        for k, v in new_sd.items():
            v.div_(len(args.ckpts))

        bfinfo = torch.finfo(torch.bfloat16)
        final_state_dict = {}
        for k, v in new_sd.items():
            v = v.clamp(bfinfo.min, bfinfo.max)
            final_state_dict[k] = v.to(dtype=torch.bfloat16)

        torch.save(final_state_dict, os.path.join(args.output, lf))

if __name__ == "__main__":
    main()
