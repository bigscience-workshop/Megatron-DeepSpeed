
# usage:
# python compare_tp_weights.py input_layernorm.weight 40 2 .


# 13B
# cd /gpfsdsstore/projects/rech/six/commun/checkpoints/tr1-13B/tr1-13B-with-optim/global_step168000
# python ~/compare_tp_weights.py input_layernorm.weight 40 2 .

# 104B
# cd /gpfsssd/scratch/rech/six/commun/checkpoints/tr8b-104B/checkpoints/emb-norm/global_step16800
# python ~/compare_tp_weights.py input_layernorm.weight 64 4 .


import torch, sys



key, nlayers, tp_size, checkpoint_dir = sys.argv[1:5]

print(f"checking key={key}")
matched, mismatched = 0, 0
for layer_id in range(int(nlayers)):
    for tp in range(int(tp_size)-1):
        f1 = f"{checkpoint_dir}/layer_{3+layer_id:02d}-model_{tp:02d}-model_states.pt"
        f2 = f"{checkpoint_dir}/layer_{3+layer_id:02d}-model_{tp+1:02d}-model_states.pt"
        c1 = torch.load(f1, map_location=torch.device('cpu'))
        c2 = torch.load(f2, map_location=torch.device('cpu'))
        # print(f1)
        # print(f2)
        header = f"layer_id={layer_id}: {tp}-{tp+1}"
        try:
            torch.testing.assert_close(c1[key], c2[key], rtol=0.0, atol=0.0, check_device=False)
            print(f"✓ {header}")
            matched += 1
        except:
            print(f"✗ {header}")
            mismatched += 1
            #raise

print(f"Matched   : {matched}")
print(f"Mismatched: {mismatched}")
