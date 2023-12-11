import os
import stat
import shutil
import numpy as np

import torch
import torch.nn.functional as F
import torch.distributed as dist

class DistDataError(Exception):
    """Defines an empty exception to throw when some other rank hit a real exception."""
    pass

class DistData(object):
    def __init__(self, backend='gloo', use_mpi4py=False):
        # use mpi4py instead of torch.distributed if requested
        self.mpi4py = None
        if use_mpi4py:
            try:
                from mpi4py import MPI
                self.mpi4py = MPI
            except:
                #print(f"ERROR: mpi4py requested, but failed to import, falling back to torch.distributed.", flush=True)
                pass

        # lookup our process rank and the group size
        if self.mpi4py is not None:
            self.comm = self.mpi4py.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.numranks = self.comm.Get_size()
        else:
            assert backend in ['gloo', 'mpi'], f"torch.distributed backend '{backend}' is not supported, valid options are 'gloo' or 'mpi'"
            if 'OMPI_COMM_WORLD_RANK' in os.environ:
                os.environ["RANK"] = os.environ['OMPI_COMM_WORLD_RANK']
            if 'OMPI_COMM_WORLD_SIZE' in os.environ:
                os.environ["WORLD_SIZE"] = os.environ['OMPI_COMM_WORLD_SIZE']
            dist.init_process_group(backend, init_method="env://")
            self.rank = dist.get_rank()
            self.numranks = dist.get_world_size()

    def allassert(self, cond, msg):
        """Check that cond is True on all ranks, assert with msg everywhere if not.

        To prevent deadlocks in cases where an assertion might only fail on one rank,
        this executes an allreduce to ensure that if any rank finds that an assertion
        has been violated, all ranks fail an assertion check.
        The condition must be true on all ranks for this not to assert.
        """
        alltrue = self.alltrue(cond)
        assert alltrue, msg

    def allraise_if(self, err):
        """Raise exception if err is not None on any rank.

        Similarly to allassert, this raises an exception on all ranks if err
        is set to an exception on any rank.  Rank(s) where err is not None
        re-raise err as exception, and ranks where err is None raise DistDataError.
        Thus all ranks raise an exception if any rank has an active exception,
        which helps avoid deadlocks in cases where an exception may be raised
        on a subset of ranks.
        """
        alltrue = self.alltrue(err is None)
        if not alltrue:
            # At least one rank raised an exception.
            # Re-raise the actual exception if this rank threw one.
            if err is not None:
                raise err

            # TODO: is there a better exception to use here?
            # On other ranks, raise an "empty" exception to indicate
            # that we're only failing because someone else did.
            raise DistDataError

    def barrier(self):
        """Globally synchronize all processes"""
        if self.mpi4py is not None:
            self.comm.barrier()
        else:
            dist.barrier()

    def bcast(self, val, root):
        """Broadcast a scalar value from root to all ranks"""
        if self.mpi4py is not None:
            return self.comm.bcast(val, root=root)
        else:
            vals = [val]
            dist.broadcast_object_list(vals, src=root)
            return vals[0]

    def bcast_list(self, vals, root=0):
        """Broadcast list of vals from root to all ranks, returns newly allocated list"""
        if self.mpi4py is not None:
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

    def scatterv_(self, invals: np.array, counts: list, root:int=0):
        """Scatter int64 values from invals according to counts array, return received portion in a new tensor"""

        self.allassert(len(counts) == self.numranks,
            f"Length of counts list {len(counts)} does not match number of ranks {self.numranks}")

        if self.mpi4py is not None:
            counts = np.array(counts)
            displs = np.cumsum(counts) - counts
            outval = np.zeros(counts[self.rank], dtype=np.int64)
            self.comm.Scatterv([invals, counts, displs, self.mpi4py.INT64_T], outval, root=root)
            return torch.from_numpy(outval)
        else:
            # Define list of tensors to scatter on the root.
            # torch.distributed.scatter requires each tensor to be the same shape,
            # so find the max size across all count values and pad.
            max_size = max(counts)
            scatterlist = None
            if self.rank == root:
                slices = list(torch.split(torch.from_numpy(invals), counts))
                scatterlist = [F.pad(s, (0, max_size - len(s))) for s in slices]

            # Receive a tensor of the max count size from the root,
            # then copy values into output numpy array, which may be smaller.
            recvtensor = torch.zeros(max_size, dtype=torch.int64)
            dist.scatter(recvtensor, scatterlist, src=root)
            return recvtensor[:counts[self.rank]]

    def alltrue(self, val):
        """Returns True if all procs input True, False otherwise"""
        if self.mpi4py is not None:
            inval = np.array([val], dtype=np.bool_)
            outval = np.zeros_like(inval)
            self.comm.Allreduce(inval, outval, op=self.mpi4py.LAND)
            return bool(outval[0])
        else:
            # torch.dist does not support reductions with bool types
            # so we cast to int and cast the result back to bool
            tensor = torch.tensor([int(val)], dtype=torch.int32)
            dist.all_reduce(tensor, op=dist.ReduceOp.BAND)
            return bool(tensor[0])

    def sum(self, val):
        """Compute sum of a scalar val, and return total on all ranks."""
        if self.mpi4py is not None:
            insize = np.array([val], dtype=np.int64)
            outsize = np.zeros_like(insize)
            self.comm.Allreduce(insize, outsize, op=self.mpi4py.SUM)
            return outsize[0]
        else:
            tensor = torch.tensor([val])
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return tensor[0]

    def exscan(self, val: int):
        """Compute prefix sum (exclusive scan) of int64 val, and return offset of each rank."""
        if self.mpi4py is not None:
            insize = np.array([val], dtype=np.int64)
            outsize = np.zeros_like(insize)
            self.comm.Scan(insize, outsize, op=self.mpi4py.SUM)
            return outsize[0] - insize[0]
        else:
            # torch.distributed doesn't have a scan, so fallback to allreduce
            tensor = torch.zeros(self.numranks, dtype=torch.int64)
            tensor[self.rank:] = val
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            return int(tensor[self.rank]) - val

    def min(self, val):
        """Return minimum of scalar val to all ranks."""
        if self.mpi4py is not None:
            insize = np.array([val], dtype=np.int64)
            outsize = np.zeros_like(insize)
            self.comm.Allreduce(insize, outsize, op=self.mpi4py.MIN)
            return outsize[0]
        else:
            tensor = torch.tensor([val])
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

    def all_sum_(self, vals: np.array):
        """Sums values in numpy array vals element-wise and update vals in place with final result on all ranks"""
        if self.mpi4py is not None:
            outval = np.zeros_like(vals)
            self.comm.Allreduce(vals, outval, op=self.mpi4py.SUM)
            vals[:] = outval
        else:
            # Builds torch.tensor with from_numpy to use same underlying memory as numpy array.
            tensor = torch.from_numpy(vals)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

    def open(self, filename, truncate=None):
        """Create, truncate, and open a file for writing shared by all ranks."""
        f = None

        # Don't truncate existing file until all ranks reach this point
        self.barrier()

        # We'll capture any exception in this variable
        err = None

        # Rank 0 creates and truncates file.
        if self.rank == 0:
            try:
                f = open(filename, 'wb')

                # Some file systems like GPFS deliver faster write speed
                # if the file size is known before data is written to the file.
                if truncate is not None:
                    f.truncate(truncate)

            except Exception as e:
                err = e
                if f is not None:
                    f.close()

        # Verify that rank 0 created the file
        self.allraise_if(err)

        # Wait for rank 0 to open (and truncate) file,
        # then have all ranks open file for writing.
        if self.rank != 0:
            try:
                f = open(filename, 'r+b')
            except Exception as e:
                err = e

        # Verify that all ranks successfully opened the file
        if not self.alltrue(err is None):
            # Someone failed to open the file.
            # If we succeeded, close our file.
            if f is not None:
                f.close()

        # All raise an exception if anyone did
        self.allraise_if(err)

        return f

    def openread(self, filename):
        """Open a shared file for reading by all ranks."""
        f = None

        # Don't attempt to open until all ranks are ready.
        self.barrier()

        # Open the file for reading on all ranks.
        # Catch exception if the rank fails.
        err = None
        try:
            f = open(filename, 'rb')
        except Exception as e:
            err = e

        # Verify that all ranks successfully opened the file
        if not self.alltrue(err is None):
            # Someone failed to open the file.
            # If we succeeded, close our file.
            if f is not None:
                f.close()

        # All raise an exception if anyone did
        self.allraise_if(err)

        return f

    def remove(self, filename):
        """Remove a shared file."""

        # Don't remove the file until all are ready
        self.barrier()

        # We'll capture any exception in this variable
        err = None

        # Rank 0 removes the file if it exists.
        if self.rank == 0:
            try:
                if os.path.exists(filename):
                    os.remove(filename)
            except Exception as e:
                err = e

        # Verify that rank 0 successfully removed the file.
        self.allraise_if(err)

    def rename(self, srcfile, destfile):
        """Rename a shared file."""

        # Don't rename until all are ready
        self.barrier()

        # We'll capture any exception in this variable
        err = None

        # Rank 0 renames the file.
        if self.rank == 0:
            try:
                if os.path.exists(srcfile):
                    os.rename(srcfile, destfile)
            except Exception as e:
                err = e

        # Verify that the rename succeeded
        self.allraise_if(err)

    def exists(self, filename):
        """Test whether file exists and broadcast result to all ranks."""
        # We'll capture any exception in this variable
        err = None

        # Rank 0 executes the existence check
        exists = False
        if self.rank == 0:
            try:
                exists = os.path.exists(filename)
            except Exception as e:
                err = e

        # Verify that the check succeeded
        self.allraise_if(err)

        # Get value from rank 0
        exists = self.bcast(exists, root=0)
        return exists

    def stat(self, filename, field):
        """Lookup field from stat on file and broadcast to all ranks."""
        # We'll capture any exception in this variable
        err = None

        # Rank 0 does the actual stat call
        val = None
        if self.rank == 0:
            try:
                val = os.stat(filename)[field]
            except Exception as e:
                err = e

        # Verify that the stat succeeded
        self.allraise_if(err)

        # Get value from rank 0
        val = self.bcast(val, root=0)
        return val

    def filesize(self, filename):
        """Lookup filesize and broadcast to all ranks."""
        return self.stat(filename, stat.ST_SIZE)

    def mtime(self, filename):
        """Lookup file mtime and broadcast to all ranks."""
        return self.stat(filename, stat.ST_MTIME)

    # We stat each file to determine its size, execute a scan to compute
    # the byte offset where the calling rank should write its data, seek to proper
    # spot, and copy each file.
    def concat_files_gather(self, outfile, filelist):
        """Concatenate files in filelist into a new file given by outfile"""
        # We first write to a temporary file name.  We rename to the final name
        # if successful or delete the temporary file if not.
        # This way if the final name appears, the user knows it's a valid file.
        tmpfile = outfile + ".tmp"

        # First delete the final file if it already exists
        self.remove(outfile)

        # Lookup size of each of our files
        filesizes = [os.stat(f)[stat.ST_SIZE] for f in filelist]

        # Compute total bytes of the final file and the offset
        # at which this rank will write data from its files.
        numbytes = sum(filesizes)
        count = self.sum(numbytes)
        offset = self.exscan(numbytes)

        # Catch I/O errors from any process
        err = None
        try:
            # Create shared output file and pre-truncate to its final size.
            with self.open(tmpfile, truncate=count) as fout:
                # Seek to appropriate starting offset in the merged file.
                fout.seek(offset)

                # Copy in contents of each of our files.
                for f in filelist:
                    with open(f, "rb") as fsrc:
                        shutil.copyfileobj(fsrc, fout)

        except Exception as e:
            err = e

        # Check that all ranks wrote successfully.
        # This will raise an exception all on ranks if we detect
        # an exception on any rank.
        self.allraise_if(err)

        # Everyone wrote their part successfully.
        # Rename the temporary file to the final file.
        self.rename(tmpfile, outfile)
