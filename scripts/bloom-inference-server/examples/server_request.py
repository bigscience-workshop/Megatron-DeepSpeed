import requests


def main():
    url = "http://127.0.0.1:5000/generate/"

    request_body = {
        "text": "DeepSpeed is a machine learning framework",
        "max_new_tokens": 40
    }
    response = requests.post(
        url=url,
        json=request_body,
        verify=False
    )
    print(response.json())
    # -------------------------------------------------------------------------
    # higher batch size
    request_body = {
        "text": [
            "DeepSpeed",
            "DeepSpeed is a",
            "DeepSpeed is a machine",
            "DeepSpeed is a machine learning framework",
        ],
        "max_new_tokens": 40
    }
    response = requests.post(
        url=url,
        json=request_body,
        verify=False
    )
    print(response.json())


if (__name__ == "__main__"):
    main()
