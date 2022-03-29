
# usage:
# python compare_tp_weights.py input_layernorm.weight 40 2 .

# input_layernorm.weight
# input_layernorm.bias
# post_attention_layernorm.weight
# post_attention_layernorm.bias

# one liner for just 2 weights comparison
# python -c 'import torch, sys; k=sys.argv[1]; a,b = map(torch.load, sys.argv[2:4]); print("Exact match" if torch.testing.assert_close(a[k], b[k], rtol=0.0, atol=0.0, check_device=False) is None else "Mismatch")' input_layernorm.weight layer_03-model_00-model_states.pt layer_03-model_01-model_states.pt

# 13B
# cd /gpfsdsstore/projects/rech/six/commun/checkpoints/tr1-13B/tr1-13B-with-optim/global_step168000
# python ~/compare_tp_weights.py input_layernorm.weight 40 2 .

# 104B
# cd /gpfsssd/scratch/rech/six/commun/checkpoints/tr8b-104B/checkpoints/emb-norm/global_step16800
#
# python ~/compare_tp_weights.py input_layernorm.weight 64 4 .          > ~/104B.input_layernorm.weight.txt
# python ~/compare_tp_weights.py post_attention_layernorm.weight 64 4 . > ~/104B.post_attention_layernorm.weight.txt
# python ~/compare_tp_weights.py input_layernorm.bias 64 4 .            > ~/104B.input_layernorm.bias.txt
# python ~/compare_tp_weights.py post_attention_layernorm.bias 64 4 .   > ~/104B.post_attention_layernorm.bias.txt

# other 104B checkpoints:

# cd /gpfsssd/scratch/rech/six/commun/checkpoints/tr8b-104B/to-back-up/tr8b-104B/checkpoints/cl-exp-02/global_step10500
# mismatched 68
#
# cd /gpfsssd/scratch/rech/six/commun/checkpoints/tr8-104B-wide/experiment11/global_step15660
# mismatched
#
# cd /gpfsssd/scratch/rech/six/commun/checkpoints/tr8-104B-wide/experiment06/global_step5100
# python ~/compare_tp_weights.py input_layernorm.weight 32 4
# **all matched**
#
# python ~/compare_tp_weights.py post_attention_layernorm.weight 32 4
# not matched



# # 104B/176B embed-norm check
# python -c 'import torch, sys; k=sys.argv[1]; a,b = map(torch.load, sys.argv[2:4]); print("Exact match" if torch.testing.assert_close(a[k], b[k], rtol=0.0, atol=0.0, check_device=False) is None else "Mismatch")' word_embeddings.norm.weight layer_01-model_00-model_states.pt layer_01-model_01-model_states.pt
# python -c 'import torch, sys; k=sys.argv[1]; a,b = map(torch.load, sys.argv[2:4]); print("Exact match" if torch.testing.assert_close(a[k], b[k], rtol=0.0, atol=0.0, check_device=False) is None else "Mismatch")' word_embeddings.norm.weight layer_01-model_01-model_states.pt layer_01-model_02-model_states.pt
# python -c 'import torch, sys; k=sys.argv[1]; a,b = map(torch.load, sys.argv[2:4]); print("Exact match" if torch.testing.assert_close(a[k], b[k], rtol=0.0, atol=0.0, check_device=False) is None else "Mismatch")' word_embeddings.norm.weight layer_01-model_02-model_states.pt layer_01-model_03-model_states.pt


# # 176B
# cd /gpfsssd/scratch/rech/six/commun/checkpoints/tr11-176B-ml/checkpoints/main/global_step16400
# python ~/compare_tp_weights.py input_layernorm.weight 70 4 .          > ~/176B.input_layernorm.weight.txt
# python ~/compare_tp_weights.py post_attention_layernorm.weight 70 4 . > ~/176B.post_attention_layernorm.weight.txt
# python ~/compare_tp_weights.py input_layernorm.bias 70 4 .            > ~/176B.input_layernorm.bias.txt
# python ~/compare_tp_weights.py post_attention_layernorm.bias 70 4 .   > ~/176B.post_attention_layernorm.bias.txt


import torch, sys



key, nlayers, tp_size, checkpoint_dir = sys.argv[1:5]

print(f"checking key={key}")
matched, mismatched = 0, 0
for layer_id in range(int(nlayers)):
    for tp in range(int(tp_size)-1):
        f1 = f"{checkpoint_dir}/layer_{3+layer_id:02d}-model_{tp:02d}-model_states.pt"
        f2 = f"{checkpoint_dir}/layer_{3+layer_id:02d}-model_{tp+1:02d}-model_states.pt"
        c1 = torch.load(f1)
        c2 = torch.load(f2)
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
