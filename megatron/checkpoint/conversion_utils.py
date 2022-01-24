import os


def validate_files(file_list):
    for file in file_list:
        if not os.path.isfile(file):
            print(f'Error: {file} is not existent')


def get_files(dir):
    file_list = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list


def partition_data(data_list, num_partitions):
    num_elems = len(data_list)
    assert num_elems % num_partitions == 0
    partition_size = num_elems // num_partitions
    partitions_list = [
        data_list[i:i + partition_size]
        for i in range(0, num_elems, partition_size)
    ]
    return partitions_list
