import numpy as np
import torch
import torch.distributed as dist

class DistData(object):
    def __init__(self, mpi=None):
        self.MPI = mpi

        if self.MPI:
            self.comm = mpi.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.numranks = self.comm.Get_size()
        else:
            self.rank = dist.get_rank()
            self.numranks = dist.get_world_size()

    def barrier(self):
        """Globally synchronize all processes"""
        if self.MPI:
            self.comm.barrier()
        else:
            dist.barrier()

    def bcast(self, val, root):
        """Broadcast a scalar value from root to all ranks"""
        if self.MPI:
            return self.comm.bcast(val, root=root)
        else:
            vals = [val]
            dist.broadcast_object_list(vals, src=root)
            return vals[0]

    def bcast_list(self, vals, root=0):
        """Broadcast list of vals from root to all ranks, returns newly allocated list"""
        if self.MPI:
            return self.comm.bcast(vals, root=root)
        else:
            # broadcast length of vals list
            length = [len(vals)]
            dist.broadcast_object_list(length, src=root)

            # allocate a tensor of appropriate size
            # initialize tensor with list values on root
            if self.rank == root:
                tvals = torch.tensor(vals, dtype=torch.int64)
            else:
                tvals = torch.zeros(length[0], dtype=torch.int64)

            # broadcast tensor from root, and return as a new list
            dist.broadcast(tvals, src=root)
            return tvals.tolist()

    def alltrue(self, val):
        """Returns True if all procs input True, False otherwise"""
        if self.MPI:
            inval = np.array([val], dtype=np.bool_)
            outval = np.zeros_like(inval)
            self.comm.Allreduce(inval, outval, op=self.MPI.LAND)
            return bool(outval[0])
        else:
            tensor = torch.tensor([int(val)], dtype=torch.int32)
            dist.all_reduce(tensor, op=dist.ReduceOp.BAND)
            return bool(tensor[0])

    def sum(self, val):
        """Compute sum of val, and return total on all ranks."""
        if self.MPI:
            insize = np.array([val], dtype=np.int64)
            outsize = np.zeros_like(insize)
            self.comm.Allreduce(insize, outsize, op=self.MPI.SUM)
            return outsize[0]
        else:
            tensor = torch.tensor([val], dtype=torch.int64)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return tensor[0]

    def exscan(self, val):
        """Compute prefix sum (exclusive scan) of val, and return offset of each rank."""
        if self.MPI:
            insize = np.array([val], dtype=np.int64)
            outsize = np.zeros_like(insize)
            self.comm.Scan(insize, outsize, op=self.MPI.SUM)
            return outsize[0] - insize[0]
        else:
            # torch.distributed doesn't have a scan, so fallback to allreduce
            tensor = torch.zeros(self.numranks, dtype=torch.int64)
            tensor[self.rank:] = val
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return int(tensor[self.rank]) - val

    def min(self, val):
        """Return minimum val to all ranks."""
        if self.MPI:
            insize = np.array([val], dtype=np.int64)
            outsize = np.zeros_like(insize)
            self.comm.Allreduce(insize, outsize, op=self.MPI.MIN)
            return outsize[0]
        else:
            # torch.distributed doesn't have a scan, so fallback to allreduce
            tensor = torch.tensor([val], dtype=torch.int64)
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            return tensor[0]

    def minrank(self, cond):
        """Find first rank whose condition is True, return that rank if any, None otherwise."""
        minrank = self.numranks
        if cond:
            minrank = self.rank
        minrank = self.min(minrank)

        if minrank < self.numranks:
            return minrank
        return None

    def bcast_first(self, val):
        """Broadcast val from first rank where it is not None, return val if any, None otherwise"""
        # Find the first rank with a valid value.
        minrank = self.minrank(val is not None)

        # If there is no rank with a valid value, return None
        if minrank is None:
            return None

        # Otherwise broadcast the value from the first valid rank.
        val = self.bcast(val, root=minrank)
        return val

    def open(self, filename):
        """Create, truncate, and open a file shared by all ranks."""
        success = True
        err = None

        # Don't truncate existing file until all ranks reach this point
        self.barrier()

        # Rank 0 creates and truncates file.
        if self.rank == 0:
            try:
                f = open(filename, 'wb')
            except Exception as e:
                success = False
                err = e

        # Verify that rank 0 created the file
        success = self.alltrue(success)
        if not success:
            if err is not None:
                raise err
            return None

        # Wait for rank 0 to open (and truncate) file,
        # then have all ranks open file for writing.
        if self.rank != 0:
            try:
                f = open(filename, 'r+b')
            except Exception as e:
                success = False
                err = e

        # Verify that all ranks successfully opened the file
        success = self.alltrue(success)
        if not success:
            if err is not None:
                raise err
            return None

        return f
