import argparse

import requests


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group(title="launch config")
    group.add_argument("--host", type=str, required=True, help="host address")
    group.add_argument("--port", type=int, required=True, help="port number")

    return parser.parse_args()


def generate(url: str) -> None:
    url = url + "/generate/"

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
    print(response.json(), "\n")


def tokenize(url: str) -> None:
    url = url + "/tokenize/"

    request_body = {
        "text": [
            "DeepSpeed is a",
            "DeepSpeed is a machine learning framework"
        ]
    }
    response = requests.post(
        url=url,
        json=request_body,
        verify=False
    )
    print(response.json(), "\n")


def query_id(url: str) -> None:
    url = url + "/query_id/"

    response = requests.get(
        url=url,
        verify=False
    )
    print(response.json(), "\n")


def main():
    args = get_args()
    url = "http://{}:{}".format(args.host, args.port)

    generate(url)
    tokenize(url)
    query_id(url)


if (__name__ == "__main__"):
    main()
