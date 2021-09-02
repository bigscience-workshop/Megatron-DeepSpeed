import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir)))

import argparse
import time
import numpy as np

import torch

from mpi4py import MPI
from megatron.data.distdata_mpi import DistData
from indexed_json import IndexedJSON

def msg(msg, flush=False):
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"{timestamp}: {msg}", flush=flush)

def get_proc_counts(num, num_ranks):
    num_per_rank, remainder = divmod(num, num_ranks)
    return [num_per_rank + 1 if rank < remainder else num_per_rank for rank in range(num_ranks)]

def pack_sendbuf(shufbuf, sendindex, sendoffsets, sendsizes):
    # Allocate, prepare, and return send buffers.

    # The index list records the sample id within the batch.
    # The size list records the number of bytes of each sample.
    tensorindex = torch.tensor(sendindex, dtype=torch.int64)
    tensorsizes = torch.tensor(sendsizes, dtype=torch.int64)
    #sendmeta = torch.zeros(2 * len(sendindex), dtype=torch.int64)
    #sendmeta[0::2] = tensorindex[:]
    #sendmeta[1::2] = tensorsizes[:]

    # Allocate a buffer to hold bytes for all samples.
    numbytes = sum(sendsizes)
    sendbuf = np.zeros(numbytes, dtype=np.uint8)
    #sendbuf = torch.zeros(numbytes, dtype=torch.uint8)

    # Pack each sample from our shufbuf into the send buffer.
    # Each sample is concatenated to the previous one.
    packsize = 0
    for o, s in zip(sendoffsets, sendsizes):
        sendbuf[packsize : packsize + s] = shufbuf[o : o + s]
        packsize += s

    return tensorindex, tensorsizes, sendbuf

def register_pointers(pointers, buffers, src, recvindex, recvsizes, recvbuf):
    # Compute offsets into recieve buffer for the start of each sample.
    recvoffsets = torch.cumsum(recvsizes, dim=0)
    recvoffsets -= recvsizes

    # Hold a reference to the receive buffer for this process.
    #buffers[src] = recvbuf.numpy()
    buffers[src] = recvbuf

    # For each sample received, construct a (srcrank, offset, size) tuple to store
    # in our pointers dictionary indexed by the local sample id within our batch.
    for i in range(len(recvindex)):
        pointers[int(recvindex[i])] = (src, int(recvoffsets[i]), int(recvsizes[i]))

def shuffle_dset(args, dset):
    # TODO: perhaps determine dynamically based on max memory and average sample size
    # max number of samples to gather to each rank in each write phase
    batch_size = 100000

    # TODO: to use None, we could have rank 0 use None,
    # use that to generate a random seed, and then bcast that seed to all ranks
    # args.seed may be an int (to seed) or None (to not)
    rng = np.random.default_rng(args.seed)

    rank = args.distctx.rank
    numranks = args.distctx.numranks

    if rank == 0:
        msg(f"Computing offsets ...", flush=True)

    num_samples = len(dset)

    # Get a list of the number of elements each rank holds
    counts = get_proc_counts(num_samples, numranks)
    counts = np.array(counts, dtype=np.int64)

    # Computing starting sample index for each rank
    rank_offsets = np.cumsum(counts)
    rank_offsets -= counts

    # TODO: for reading offset/size values of a consecutive set of samples,
    #       it would be nice to extend distdata to return lists, it could grab
    #       those with a single read operation, too.
    # Determine file offset and size of each sample this rank is responsible for.
    # The offset here is the byte offset to the start of the line within the source file.
    # The size is the number of bytes of each line.
    time_start = time.time()
    offsets = []
    sizes = []
    numbytes = 0
    for i in range(rank_offsets[rank], rank_offsets[rank] + counts[rank]):
        sample_id = int(i)
        offset, size = dset.index(sample_id)
        offsets.append(offset)
        sizes.append(size)
        numbytes += size

    # Total up bytes across all ranks to get file size
    filesize = args.distctx.sum(numbytes)

    args.distctx.barrier()
    time_sizes = time.time()
    if rank == 0:
        msg(f"Seconds to compute offsets: {time_sizes - time_start}", flush=True)

    # TODO: for files where samples are contiguous, this could be read in a single
    #       read operation rather than read per sample.
    # Allocate a buffer to hold all bytes in our porition,
    # and read samples from source file into memory.
    # Construct an list of offsets to the start of each sample in memory.
    shufbuf = np.zeros(numbytes, dtype=np.uint8)
    shufbufoffsets = np.zeros(counts[rank], dtype=np.int64)
    sample_start = 0
    sample_idx = 0
    for o, s in zip(offsets, sizes):
        sample = dset.pread(o, s)
        shufbuf[sample_start : sample_start + s] = np.frombuffer(sample, dtype=np.uint8)
        shufbufoffsets[sample_idx] = sample_start
        sample_idx += 1
        sample_start += s

    args.distctx.barrier()
    time_shufbuf = time.time()
    if rank == 0:
        msg(f"Seconds to prepare shuffle buffer: {time_shufbuf - time_sizes}", flush=True)

    # Each process randomly shuffles its set of samples.
    # We do this by just shuffling a list of local index values.
    # Each rank executes the RNG for the same number of steps here
    # to keep the RNG in lock-step across all procs.
    samplelist = None
    for r in range(numranks):
        tmplist = np.arange(counts[r])
        rng.shuffle(tmplist)
        if r == rank:
            samplelist = tmplist

    args.distctx.barrier()
    time_shuffle = time.time()
    if rank == 0:
        msg(f"Seconds to shuffle: {time_shuffle - time_shufbuf}", flush=True)

    # Randomly pick ranks to draw each sample.
    # This creates a list with one entry for each sample a rank has.
    # Each entry records the rank number of the owning rank.
    # The list is then shuffled in order to pick from ranks in a random order.
    if rank == 0:
        ordered = np.zeros(len(dset), dtype=np.int64)
        rank_start = 0
        for r, c in enumerate(counts):
            ordered[rank_start : rank_start + c] = r
            rank_start += c
        rng.shuffle(ordered)

    # Open output file for shared writing, pre-truncate to its final size.
    # On some file systems, truncating in advance helps speed up the write.
    progress_interval = 10.0
    err = None
    with args.distctx.open(args.output, truncate=filesize) as fout:
        time_next = time.time() + progress_interval

        # We iterate over the final file writing in phases.
        # Each process writes a set of samples (a batch) within a contiguous section.
        # Rank 0 determines the set of samples for each batch.
        # It marches through the ordered list to determine the order in which ranks
        # are sampled to build each batch.
        # That information is broadcasted to all ranks.
        # Each rank assembles a send buffer for every other rank.
        # Processes execute an alltoallv-style communication pattern to gather its samples.
        # After collecting its batch, each process writes its batch to the final file.
        # The process completes until all samples are accounted for.
        next_sample = 0     # tracks local sample id to use on the current rank
        totalwritten = 0    # tracks byte offset into the final file for each phase
        samples_remaining = num_samples # global number of samples left to write
        while samples_remaining > 0:
            args.distctx.barrier()
            time_start_bcast = time.time()

            # TODO: if this becomes a bottleneck, it may be possible
            # to do this locally on each rank.
            # Broadcast list of ranks to sample from for each batch.
            # There is one batch per rank, as each rank will
            # collect a batch of samples to write to the final file.
            batches = []
            for r in range(numranks):
                maxsize = min(batch_size, samples_remaining)
                batch = np.zeros(maxsize, dtype=np.int64)
                if rank == 0:
                    sample_start = num_samples - samples_remaining
                    batch[:] = ordered[sample_start : sample_start + maxsize]
                MPI.COMM_WORLD.Bcast(batch, root=0)
                batches.append(batch)
                samples_remaining -= len(batch)

            args.distctx.barrier()
            time_end_bcast = time.time()

            # Each process steps through each batch to identify the data
            # it will send to each process.
            # We collect the sample index within its batch (used to sort
            # at the destination process), the offset within the memory
            # buffer holding the sample and the size of each sample.
            sendindex_all = []
            sendoffsets_all = []
            sendsizes_all = []
            for r in range(numranks):
                batch = batches[r]
                sendindex = []
                sendoffsets = []
                sendsizes = []
                send_index_list = np.argwhere(batch == rank)
                for i in send_index_list:
                    local_index = samplelist[next_sample]
                    next_sample += 1
                    sendindex.append(i[0])
                    sendoffsets.append(shufbufoffsets[local_index])
                    sendsizes.append(sizes[local_index])
                sendindex_all.append(sendindex)
                sendoffsets_all.append(sendoffsets)
                sendsizes_all.append(sendsizes)

            args.distctx.barrier()
            time_end_ranks = time.time()

            # Now that each process has identified which samples it will send
            # to each rank, we allocate and pack send buffers for each rank.
            tensorindex_all = []
            tensorsizes_all = []
            sendbuf_all = []
            for r in range(numranks):
                tensorindex, tensorsizes, sendbuf = pack_sendbuf(shufbuf, sendindex_all[r], sendoffsets_all[r], sendsizes_all[r])
                tensorindex_all.append(tensorindex)
                tensorsizes_all.append(tensorsizes)
                sendbuf_all.append(sendbuf)

            args.distctx.barrier()
            time_end_pack = time.time()

            # Processes exchange data using a ring-based alltoallv algorithm using non-blocking pt2pt calls.
            pointers = dict()
            buffers = dict()
            for dist in range(numranks):
                # This barrier is not strictly necessary, but it could help with flow control
                # by keeping procs in step with each other to prevent too many procs flooding
                # the same destination with messages.  One could also implement a go-ahead message
                # from the destination to each sender.
                #args.distctx.barrier()

                if dist == 0:
                    # Receive from ourself
                    register_pointers(pointers, buffers, rank, tensorindex_all[rank], tensorsizes_all[rank], sendbuf_all[rank])
                else:
                    # Otherwise in this step, we send to the rank dist hops to our left (lower)
                    # and receive from the rank dist hops to our right (higher)
                    lhs = (rank - dist + numranks) % numranks
                    rhs = (rank + dist + numranks) % numranks

                    # Send all data with isends.
                    # First, send a small message listing the number of samples and number of bytes we'll send.
                    # Second, send a list of the batch ids for each of our samples.
                    # Third, send a list of sizes for each of our samples.
                    # Fourth, send a packed buffer of all sample data.
                    sendcounts = torch.tensor([len(tensorindex_all[lhs]), len(sendbuf_all[lhs])], dtype=torch.int64)
                    sreq1 = MPI.COMM_WORLD.Isend(sendcounts.numpy(), dest=lhs)
                    sreq2 = MPI.COMM_WORLD.Isend(tensorindex_all[lhs].numpy(), dest=lhs)
                    sreq3 = MPI.COMM_WORLD.Isend(tensorsizes_all[lhs].numpy(), dest=lhs)
                    #sreq4 = MPI.COMM_WORLD.Isend(sendbuf_all[lhs].numpy(), dest=lhs)
                    sreq4 = MPI.COMM_WORLD.Isend(sendbuf_all[lhs], dest=lhs)

                    # Receive number of incoming samples and bytes in this step.
                    recvcounts = torch.zeros(2, dtype=torch.int64)
                    MPI.COMM_WORLD.Recv(recvcounts.numpy(), source=rhs)

                    # Allocate receive buffers for following messages.
                    recvindex = torch.zeros(recvcounts[0], dtype=torch.int64)
                    recvsizes = torch.zeros(recvcounts[0], dtype=torch.int64)
                    recvbuf = torch.zeros(recvcounts[1], dtype=torch.uint8)

                    # Post receives for incoming messages.
                    rreq1 = MPI.COMM_WORLD.Irecv(recvindex.numpy(), source=rhs)
                    rreq2 = MPI.COMM_WORLD.Irecv(recvsizes.numpy(), source=rhs)
                    rreq3 = MPI.COMM_WORLD.Irecv(recvbuf.numpy(), source=rhs)

                    # Wait for all communication to complete.
                    rreq1.wait()
                    rreq2.wait()
                    rreq3.wait()

                    sreq1.wait()
                    sreq2.wait()
                    sreq3.wait()
                    sreq4.wait()

                    # Process received data to prepare it for writing later.
                    # In particular, for each sample in our batch, we construct a tuple (srcrank, offset, size)
                    # that tells us which process sent the sample, its offset into our receive buffer, and its size.
                    # That tuple is stored in the pointers dict, indexed by its id within the batch.
                    register_pointers(pointers, buffers, rhs, recvindex, recvsizes, recvbuf.numpy())

            args.distctx.barrier()
            time_end_exchange = time.time()

            # Now we compute the offset at which each process will write into the final file.
            # Each process lists the number of bytes it sent to each other process.
            totalbytes_all = np.zeros(numranks, dtype=np.int64)
            for r in range(numranks):
                totalbytes_all[r] = len(sendbuf_all[r])

            # Execute an allreduce to compute the total number of bytes each process writes in its batch.
            outtotalbytes = np.empty_like(totalbytes_all)
            MPI.COMM_WORLD.Allreduce(totalbytes_all, outtotalbytes, MPI.SUM)

            # Compute the offset for each rank and the total number of bytes written by all ranks.
            rankoffsets = np.cumsum(outtotalbytes, axis=0)
            numbytes = rankoffsets[-1]
            rankoffsets -= outtotalbytes

            # Seek to the proper spot for this rank in the final file.
            fout.seek(totalwritten + rankoffsets[rank])
            totalwritten += numbytes

            # Each process writes its batch of samples to the final file.
            for i in range(len(batches[rank])):
                src = pointers[i][0]
                offset = pointers[i][1]
                size = pointers[i][2]
                fout.write(buffers[src][offset : offset + size].tobytes())

                # Print a progress message if enabled.
                if rank == 0 and progress_interval > 0.0:
                    time_now = time.time()
                    if time_now > time_next:
                        time_next = time_now + progress_interval
                        elapsed = time_now - time_sizes
                        estbytes = totalwritten
                        byterate = estbytes / elapsed / (1024.0 * 1024.0) if elapsed > 0.0 else 0.0
                        percent = totalwritten * 100.0 / filesize
                        msg(f"Wrote {totalwritten} of {filesize} bytes ({percent:0.2f}%) in {int(elapsed)} secs, {byterate:0.3f} MB/s", flush=True)

            # Wait for everyone one to finish writing,
            # and print some stats for this pass.
            args.distctx.barrier()
            time_end_write = time.time()

            if rank == 0 and progress_interval > 0.0:
                msg(f"  bcast {time_end_bcast    - time_start_bcast}")
                msg(f"  ident {time_end_ranks    - time_end_bcast}")
                msg(f"  pack  {time_end_pack     - time_end_ranks}")
                msg(f"  exch  {time_end_exchange - time_end_pack}")
                msg(f"  write {time_end_write    - time_end_exchange}", flush=True)

    if rank == 0 and progress_interval > 0.0:
        msg(f"Waiting for ranks to finish ...", flush=True)

    args.distctx.allraise_if(err)

    if rank == 0:
        time_end = time.time()
        msg(f"Seconds to write file: {time_end - time_sizes}", flush=True)

def get_args():
    parser = argparse.ArgumentParser()
    group = parser.add_argument_group(title='input data')
    group.add_argument('--input', type=str, required=True,
                       help='Dataset name')
    group.add_argument('--output', type=str, required=True,
                       help='Output file name')
    group.add_argument('--seed', type=int, required=True,
                       help='Seed to pass to random.seed for shuffle operations.')
    group.add_argument('--torch-backend', type=str, default='gloo', choices = ['gloo', 'mpi'],
                       help='Select torch.distributed backend.')
    group.add_argument('--local_rank', type=int, default=None,
                       help='Local rank of calling process on its node (from torch.distributed.launch).')
    args = parser.parse_args()

    args.distctx = DistData(backend=args.torch_backend, use_mpi4py=True)

    # Check that user doesn't clobber their input file with a simple typo
    if args.input == args.output:
        raise ValueError(f"--input {args.input} and --output {args.output} must be different files")

    return args

def main():
    args = get_args()

    # load the dataset
    # assume file is JSONL format
    dset = IndexedJSON(args.input, args.distctx)

    shuffle_dset(args, dset)

if __name__ == '__main__':
    main()
