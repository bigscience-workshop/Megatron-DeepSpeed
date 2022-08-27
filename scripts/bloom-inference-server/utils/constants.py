from pickle import TRUE


HF_ACCELERATE = "hf_accelerate"
DS_INFERENCE = "ds_inference"
DS_ZERO = "ds_zero"

BIGSCIENCE_BLOOM = "bigscience/bloom"
DS_INFERENCE_BLOOM_FP16 = "microsoft/bloom-deepspeed-inference-fp16"
DS_INFERENCE_BLOOM_INT8 = "microsoft/bloom-deepspeed-inference-int8"

BF16 = "bf16"
FP16 = "fp16"
INT8 = "int8"

FRAMEWORK_MODEL_DTYPE_ALLOWED = {
    HF_ACCELERATE: {
        BIGSCIENCE_BLOOM: {
            BF16,
            FP16
        }
    },
    DS_INFERENCE: {
        BIGSCIENCE_BLOOM: {
            FP16
        },
        DS_INFERENCE_BLOOM_FP16: {
            FP16
        },
        DS_INFERENCE_BLOOM_INT8: {
            INT8
        }
    },
    DS_ZERO: {
        BIGSCIENCE_BLOOM: {
            BF16,
            FP16
        }
    }
}
