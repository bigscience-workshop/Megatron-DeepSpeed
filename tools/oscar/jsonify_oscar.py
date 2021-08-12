# srun -n320 python3 ./jsonify_oscar.py

import os
import json
import gzip
import numpy as np
from mpi4py import MPI

# FIXME
oscarpath = '/path/to/oscar'

def get_start_end(num, rank, num_ranks):
    num_per_rank = num // num_ranks
    remainder = num % num_ranks
    if rank < remainder:
        start = (num_per_rank + 1) * rank;
        end = start + (num_per_rank + 1)
    else:
        start = (num_per_rank + 1) * remainder + num_per_rank * (rank - remainder);
        end = start + num_per_rank
    return start, end

def mpi_create_file(filename, mpi, comm):
    """Create, truncate, and open a file shared by all ranks."""
    # Don't truncate file until all ranks reach this point
    comm.barrier()

    # Wait for rank 0 to open (and truncate) file,
    # then have all ranks open file for writing.
    rank = comm.Get_rank()
    if rank == 0:
        f = open(filename, 'wb')
    comm.barrier()
    if rank != 0:
        f = open(filename, 'r+b')

    # TODO: verify that all ranks successfully opened the file
    comm.barrier()

    return f

def mpi_get_offset(val, mpi, comm):
    """Compute preifx sum (exclusive scan) of val, and return offset of each rank."""
    insize = np.array([val], dtype=np.int64)
    outsize = np.zeros_like(insize)
    comm.Scan(insize, outsize, op=mpi.SUM)
    offset = outsize[0] - insize[0]
    return offset

def mpi_cat_files(outfile, infile, mpi, comm):
    """Concatenate per-rank binary files into a new file given by outfile"""
    import stat
    import shutil

    comm.barrier()

    # Create shared output file.
    f = mpi_create_file(outfile, mpi, comm)

    # get file size of binary file for this rank
    filesize = os.stat(infile)[stat.ST_SIZE]

    # compute offset this rank should start copying
    # its data into the merged file
    offset = mpi_get_offset(filesize, mpi, comm)

    # seek to appropriate offset and copy data
    f.seek(offset)
    with open(infile, "rb") as fsrc:
        shutil.copyfileobj(fsrc, f)

    f.close()

    # TODO: check that all ranks wrote successfully
    comm.barrier()

def generate_samples(filepaths):
    id_ = 0
    current_lines = []
    for filepath in filepaths:
        print(filepath, flush=True)
        with gzip.open(open(filepath, "rb"), "rt", encoding="utf-8") as f:
            for line in f:
                if len(line.strip()) > 0:
                    current_lines.append(line)
                elif current_lines:
                    feature = id_, {"id": id_, "text": "".join(current_lines).rstrip()}
                    yield feature
                    id_ += 1
                    current_lines = []
            # last paragraph
            if current_lines:
                feature = id_, {"id": id_, "text": "".join(current_lines).rstrip()}
                yield feature
                id_ += 1
                current_lines = []

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
ranks = comm.Get_size()

start, end = get_start_end(670, rank, ranks)
start += 1
end += 1

pathformat = os.path.join(oscarpath, 'download', 'en_part_{}.txt.gz')
files = [pathformat.format(i) for i in range(start, end)] 

rankfile = os.path.join(oscarpath, f"oscar_{rank}.json")
with open(rankfile, "w") as f:
  for id, sample in generate_samples(files):
    f.write(json.dumps(sample) + "\n")

jsonfile = os.path.join(oscarpath, "oscar.json")
mpi_cat_files(jsonfile, rankfile, MPI, comm)

comm.barrier()

os.remove(rankfile)

comm.barrier()
