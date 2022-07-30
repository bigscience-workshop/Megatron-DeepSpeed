# Inference scripts for BLOOM

To run a server using HuggingFace (requires accelerate to be installed):
```
python scripts/inference/bloom-accelerate-server.py --model_name bigscience/bloom --dtype bf16 --log_file data.log --host $ADDRESS --port $PORT
```

To run a server using deepspeed:
```
deepspeed --num_gpus 8 scripts/inference/bloom-ds-server.py --model_name bigscience/bloom --dtype fp16 --log_file data.log --host $ADDRESS --port $PORT
```

Usage:
Currently, the script supports 3 method:
1. The main generate method
```
curl -H "Content-Type: application/json" -X POST -d '{ "input_text": "India is a country of", "top_k": "5", "top_p": "0.9", "temperature": "0.7", "min_length": "1", "max_new_tokens": "40", "return_type": "output_only" }' http://$ADDRESS:$PORT/generate/
```
returns
```
{"num_output_tokens":40,"output_text":" many languages and cultures. The country is a melting pot of different cultures and languages. The country is a home to more than 1.2 billion people. The country is a home to more than 22","query_id":8,"throughput":"2.066 tokens/s","total_time_taken":"19.358 s"}
```
2. Method that returns the model description
```
curl -H "Content-Type: application/json" -X GET http://$ADDRESS:$PORT/about/
```
returns
```
Please don't send any personal information to this endpoint. We are logging your data.

Usage:
A request object should look like:
{
    input_text: "Hello, I'm a model",
    "top_k": 5,
    "top_p": 0.9,
    "temperature": 0.7,
    "min_length": 1,
    "max_new_tokens": 40,
    "return_type": "output_only"
}

Default values (use if not provided in request object):
top_k = 50
top_p = 1
temperature = 1
min_length = 1
max_new_tokens = 40
return_type = "both_input_output"
```
3. Method to check GPU usage
```
curl -H "Content-Type: application/json" -X GET http://$ADDRESS:$PORT/gpu/
```
returns the nvidia-smi output