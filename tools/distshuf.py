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

def get_num_samples(args, dset_size):
    # determine total number of samples that we'll read
    num_samples = dset_size
    if args.count is not None and args.count < dset_size:
        num_samples = args.count
    return num_samples

def select_sample_list(args, dset_size):
    """Given the total number of samples, select a list of sample index values"""
    # determine total number of samples that we'll read
    num_samples = get_num_samples(args, dset_size)

    # create sample index list on rank 0,
    # optionally shuffle the list,
    # and optionally limit the sample count
    idxlist = None
    time_select = time.time()
    if args.distctx.rank == 0:
        # generate a list of all index values
        idxlist = np.arange(dset_size, dtype=np.int64)

#        # optionally limit the sample count
#        if args.count is not None:
#            idxlist = idxlist[:args.count]

    # get a list of the number of elements each rank will hold
#    counts = get_proc_counts(num_samples, args.distctx.numranks)
    counts = get_proc_counts(dset_size, args.distctx.numranks)

    # scatter sample index values from rank 0 to all procs
    # based on distribution defined in counts list
    time_bcast = time.time()
    idx = args.distctx.scatterv_(idxlist, counts, root=0)

    args.distctx.barrier()
    time_end = time.time()
    if args.distctx.rank == 0:
        msg(f"Select index stats:")
#        msg(f"    Shuffle: {args.shuffle}")
        msg(f"    Seconds to select: {time_bcast - time_select}")
        msg(f"    Seconds to broadcast: {time_end - time_bcast}")
        msg(f"    Seconds total: {time_end - time_select}", flush=True)

    return idx

def pack_sendbuf(shufbuf, sendindex, sendoffsets, sendsizes):
    tensorindex = torch.tensor(sendindex, dtype=torch.int64)
    tensorsizes = torch.tensor(sendsizes, dtype=torch.int64)
    #sendmeta = torch.zeros(2 * len(sendindex), dtype=torch.int64)
    #sendmeta[0::2] = tensorindex[:]
    #sendmeta[1::2] = tensorsizes[:]

    numbytes = sum(sendsizes)
    sendbuf = np.zeros(numbytes, dtype=np.uint8)
    #sendbuf = torch.zeros(numbytes, dtype=torch.uint8)

    # Pack each sample from our shufbuf into the send buffer.
    # Each sample is concatenated to the previous.
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

def total_rec_sizes(args, dset, idx):
    rank = args.distctx.rank
    numranks = args.distctx.numranks

    if rank == 0:
        msg(f"Computing offsets ...", flush=True)

    idxnp = idx.numpy()

    # args.seed may be an int (to seed) or None (to not)
    rng = np.random.default_rng(args.seed)

    # Determine file offset and size of each sample this rank is responsible for.
    # The offset here is the byte offset to the start of the line within the source file.
    # The size is the number of bytes of each line.
    time_start = time.time()
    offsets = []
    sizes = []
    numbytes = 0
    for i in idxnp:
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

    # Allocate a buffer to hold all bytes in our porition,
    # and read samples from source file into memory.
    # Construct an list of offsets to the start of each sample in memory.
    shufbuf = np.zeros(numbytes, dtype=np.uint8)
    shufbufoffsets = np.zeros(len(idxnp), dtype=np.int64)
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
    samplelist = np.arange(len(idx))
    rng.shuffle(samplelist)

    args.distctx.barrier()
    time_shuffle = time.time()
    if rank == 0:
        msg(f"Seconds to shuffle: {time_shuffle - time_shufbuf}", flush=True)

    # TODO: fixme: this is redundant, and it must match how the index values were divided up.
    # Get a list of the number of elements each rank holds
    num_samples = len(dset)
    counts = get_proc_counts(num_samples, numranks)
    counts = np.array(counts, dtype=np.int64)

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
        batch_size = 100000 # max number of samples to gather to each rank in each phase
        next_sample = 0     # tracks local sample id to use on the current rank
        totalwritten = 0    # tracks byte offset into the final file for each phase
        samples_remaining = num_samples # global number of samples left to write
        while samples_remaining > 0:
            args.distctx.barrier()
            time_start_bcast = time.time()

            # Broadcast list of ranks to sample from for each batch.
            # There is one batch per rank, as each rank will
            # collect a batch of samples to write to the final file.
            batches = []
            for r in range(numranks):
                batch = None
                if rank == 0:
                    maxsize = min(batch_size, samples_remaining)
                    sample_start = num_samples - samples_remaining
                    batch = ordered[sample_start : sample_start + maxsize].tolist()
                batch = args.distctx.bcast_list(batch, root=0)
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
                for i, srcrank in enumerate(batch):
                    if srcrank == rank:
                        local_index = samplelist[next_sample]
                        next_sample += 1
                        sendindex.append(i)
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
                # This barrier is not strictly necessary, but it's cheap and it helps with flow control
                # by keeping procs in step with each other.
                args.distctx.barrier()

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
                        msg(f"{rank}: Wrote {totalwritten} of {filesize} bytes ({percent:0.2f}%) in {int(elapsed)} secs, {byterate:0.3f} MB/s", flush=True)

            # Wait for everyone one to finish writing,
            # and print some stats for this pass.
            args.distctx.barrier()
            time_end_write = time.time()

            if rank == 0:
                msg(f"bcast {time_end_bcast    - time_start_bcast}")
                msg(f"ranks {time_end_ranks    - time_end_bcast}")
                msg(f"pack  {time_end_pack     - time_end_ranks}")
                msg(f"exch  {time_end_exchange - time_end_pack}")
                msg(f"write {time_end_write    - time_end_exchange}", flush=True)

    if rank == 0 and progress_interval > 0.0:
        msg(f"{args.distctx.rank}: Waiting for ranks to finish ...", flush=True)

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
    group.add_argument('--count', type=int, default=None,
                       help='Limit the number of samples to select.')
    group.add_argument('--seed', type=int, default=None,
                       help='Seed to pass to random.seed for shuffle operations.')
    group.add_argument('--torch-backend', type=str, default='gloo', choices = ['gloo', 'mpi'],
                       help='Select torch.distributed backend.')
    group.add_argument('--local_rank', type=int, default=None,
                       help='Local rank of calling process on its node (from torch.distributed.launch).')
    args = parser.parse_args()

    args.distctx = DistData(backend=args.torch_backend, use_mpi4py=True)

    return args

def main():
    args = get_args()
    startup_start = time.time()

    # load the dataset
    # assume file is JSONL format
    dset = IndexedJSON(args.input, args.distctx)

    # create sample index list,
    # optionally shuffle the list,
    # and optionally limit the sample count
    idx = select_sample_list(args, len(dset))

    total_rec_sizes(args, dset, idx)

if __name__ == '__main__':
    main()
