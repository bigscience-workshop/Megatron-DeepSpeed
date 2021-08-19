import os
import stat
import json
import time
import numpy as np

class IndexedJSON(object):
    def __init__(self, filename, distctx, bufsize=16*1024*1024):
        self.filename = filename # JSON file name
        self.distctx = distctx   # distributed environment for collective ops
        self.bufsize = bufsize   # buffer size used while building index
        self.numsamples = 0      # number of records in JSON file
        self.fh_idx = None       # file handle to JSON index file
        self.fh_json = None      # file handle to JSON file
        self.time_index = 0      # record cost to construct index

        # given a JSON file name, compute the name of its index file
        filename_idx = self.index_filename(filename)

        # check for index file and create it if it does not exist
        exists = False
        if self.distctx.rank == 0:
            exists = os.path.exists(filename_idx)
        exists = self.distctx.bcast(exists, root=0)
        if not exists:
            self.create_index(filename, self.bufsize)

        # Identify number of samples in JSON file.
        # For now, we can do that using the size of index file.
        filesize_idx = self.get_filesize(filename_idx)
        self.numsamples = int(filesize_idx / 16)

        # Open the index and the json files for reading.
        # Disable buffering to avoid reading extra bytes we won't use.
        self.fh_idx = open(filename_idx, "rb", buffering=0)
        self.fh_json = open(filename, "rb", buffering=0)
#        self.fh_idx = open(filename_idx, "rb")
#        self.fh_json = open(filename, "rb")

    def get_filesize(self, filename):
        """Lookup filesize and broadcast to all ranks."""
        filesize = 0
        if self.distctx.rank == 0:
            filesize = os.stat(filename)[stat.ST_SIZE]
        filesize = self.distctx.bcast(filesize, root=0)
        return filesize

    def index_filename(self, filename):
        """Given the name of a JSON file, return the name of its index file."""
        return filename + '.idx'

    def get_start_end(self, num):
        """Given num items, compute and return [start,end) range on each rank."""
        rank = self.distctx.rank
        num_ranks = self.distctx.numranks

        num_per_rank = num // num_ranks
        remainder = num % num_ranks
        if rank < remainder:
            start = (num_per_rank + 1) * rank;
            end = start + (num_per_rank + 1)
        else:
            start = (num_per_rank + 1) * remainder + num_per_rank * (rank - remainder);
            end = start + num_per_rank
        return start, end

    def create_index(self, filename, bufsize):
        """Given a JSON file named dataset.json, write index to dataset.json.idx."""

        # To compute this index, ranks collective scan the JSON
        # file and record the byte offset of newline characters.
        # Those byte offsets are accumulated in a temporary index file
        # until the entire JSON file has been scanned.  The processes
        # then read back those byte locations from the temporary file
        # to compute the length of each record.  Finally for each
        # record an (offset,length) pair of int64 types is written into
        # the index file to specify the starting offset and length of
        # each record in the JSON file.

        time_start = time.time()
        rank = self.distctx.rank
        numranks = self.distctx.numranks

        # define file names for the index and the temporary index file
        filename_idx = self.index_filename(filename)
        filename_tmp = filename_idx + 'tmp'

        # lookup and broadcast size of JSON file to all ranks
        filesize = self.get_filesize(filename)

        # create the temporary index file, shared across all ranks
        with self.distctx.open(filename_tmp) as ftmp:
            # open and scan the JSON file
            recstart = 0
            with open(filename, "rb") as f:
                curpos = 0
                while curpos < filesize:
                    # each rank reads a section of the file
                    offset = curpos + bufsize * rank
                    f.seek(offset)
                    data = f.read(bufsize)

                    # scan section for newline chars, and record offset
                    # of byte immediately following each newline (start of new record)
                    newlines = []
                    pos = 0
                    length = len(data)
                    while pos < length:
                        found = data.find(b'\n', pos)
                        if found >= 0:
                            # We actually store the byte offset to the start
                            # of the record that would follow the newline char.
                            newlines.append(offset + found + 1)

                            # Update our buffer position and keep scanning.
                            pos = found + 1
                        else:
                            # No newlines in the remainder of the buffer
                            break

                    # Count number of newline chars we found,
                    # and compute sum and offset of newlines across ranks.
                    numrecs = len(newlines)
                    reccount = self.distctx.sum(numrecs)
                    recoffset = self.distctx.exscan(numrecs)

                    # Store offsets as int64
                    vals = np.array(newlines, dtype=np.int64)

                    # Write offsets into temporary index file
                    pos = (recstart + recoffset) * 8
                    ftmp.seek(pos)
                    ftmp.write(vals.tobytes(order='C'))

                    # Bump up to next slot in the temporary index file.
                    recstart += reccount

                    # Move on to the next section of the JSON file.
                    curpos += bufsize * numranks

        # Wait for all ranks to close the file.
        self.distctx.barrier()

        # Create the actual index file.
        with self.distctx.open(filename_idx) as fidx:
            # We'll read the offsets back from the temporary index file.
            with open(filename_tmp, "rb") as ftmp:
                # Compute the [start,end) range for this rank within the list of offsets.
                start, end = self.get_start_end(recstart)

                # Determine how many records this rank is responsible for.
                readcount = end - start
                if readcount > 0:
                    # We'll read all offsets in our portion,
                    # plus one offset that comes immediately before our section.
                    readcount += 1
                    if start > 0:
                        pos = (start - 1) * 8
                        ftmp.seek(pos)

                    # Allocate a buffer and read in the offsets
                    recoffsets = np.zeros(readcount, dtype=np.int64)
                    if start > 0:
                        ftmp.readinto(recoffsets)
                    else:
                        # We leave the first entry as 0 on rank 0
                        ftmp.readinto(recoffsets[1:])

                    # Compute length of each record by computing the difference
                    # between consecutive offset values.  Also ignore the first
                    # offset when writing our (offset,length) pairs.
                    lengths = recoffsets[1:] - recoffsets[:-1]
                    offsets = recoffsets[:-1]

                    # Prepare list of (offset,length) pairs for writing.
                    # Store are int64 types.
                    vals = np.zeros((readcount - 1, 2), dtype=np.int64)
                    vals[:,0] = offsets
                    vals[:,1] = lengths

                # Write our portion of the index values.
                if readcount > 0:
                    pos = start * 16
                    fidx.seek(pos)
                    fidx.write(vals.tobytes(order='C'))

        # Wait for all ranks to finish writing to the index file.
        self.distctx.barrier()

        # Can now delete the temporary index file.
        if rank == 0:
            os.remove(filename_tmp)

        # Wait for everyone again and record how long it took.
        self.distctx.barrier()
        time_end = time.time()
        self.time_index = time_end - time_start

    def __str__(self):
        return (f"IndexedJSON (\n"
                f"  file: {self.filename}\n"
                f"  rows: {self.numsamples}\n"
                f"  secs: {self.time_index})")

    def __len__(self):
        """Return number of samples (lines) in the JSON file."""
        return self.numsamples

    def __getitem__(self, idx):
        """Given a sample id, return the sample as a python object parsed from JSON string."""
        # read offset and length of record from the index
        # seek to offset in JSON file and read the record
        buf = self.read(idx)

        # convert json record into a python dictionary
        try:
            #entry = json.loads(buf.decode("utf-8").strip())
            entry = json.loads(buf)
            return entry
        except:
            # TODO: throw exception instead?
            return None

    def __get__(self, idx):
        return self.getitem(idx)

    def index(self, idx):
        """Given an sample id, return (offset, size) tuple of location of sample in the JSON file."""
        assert idx < self.numsamples

        # seek to the right spot in the index file for the given sample id
        offset_idx = idx * 16
        self.fh_idx.seek(offset_idx)

        # read offset and length of record from the index
        vals = np.zeros(2, dtype=np.int64)
        self.fh_idx.readinto(vals)
        offset = vals[0]
        size = vals[1]

        return offset, size

    def pread(self, offset, size):
        """Read size bytes at the given offset in the JSON file and return as a buffer."""
        # seek to offset in JSON file and read the record
        self.fh_json.seek(offset)
        buf = self.fh_json.read(size)
        return buf

    def read(self, idx):
        """Given a sample id, read sample from the file and return as a buffer."""
        # read offset and length of record from the index
        # seek to offset in JSON file and read the record
        offset, size = self.index(idx)
        return self.pread(offset, size)

    def size(self, idx):
        """Given a sample id, return the number of bytes of that sample as stored in the JSON file."""
        offset, size = self.index(idx)
        return size
