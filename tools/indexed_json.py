import os
import stat
import json
import struct
import time
import numpy as np

class IndexedJSON(object):
    def __init__(self, filename, distctx, bufsize=16*1024*1024, progress=10.0):
        self.filename = filename # JSON file name
        self.distctx = distctx   # distributed environment for collective ops
        self.bufsize = bufsize   # buffer size used while building index
        self.numsamples = 0      # number of records in JSON file
        self.fh_idx = None       # file handle to JSON index file
        self.fh_json = None      # file handle to JSON file
        self.time_index = 0      # record cost to construct index
        self.progress = progress # number of secs between progress msgs (0 to disable)

        # given a JSON file name, compute the name of its index file
        self.filename_idx = self.index_filename(self.filename)

        # determine whether we need to create the index
        create_index = False
        exists = self.distctx.exists(self.filename_idx)
        if not exists:
            # index file does not exist
            create_index = True
        else:
            # index exists, but rebuild the index if the original file
            # has been modified since the index was built
            mtime = self.distctx.mtime(self.filename)
            mtime_idx = self.distctx.mtime(self.filename_idx)
            if mtime > mtime_idx:
                # original file may have changed, rebuild the index
                create_index = True
        if create_index:
            self.create_index(self.filename, self.bufsize)

        # Open the index and the json files for reading.
        # Disable buffering to avoid reading extra bytes we won't use.
        self.fh_idx = open(self.filename_idx, "rb", buffering=0)
        self.fh_json = open(self.filename, "rb", buffering=0)
#        self.fh_idx = open(self.filename_idx, "rb")
#        self.fh_json = open(self.filename, "rb")

        # Read the header from the index file.
        # This verifies the file format and sets self.idx_version.
        self.read_index_header()

        # Identify number of samples in JSON file.
        # For now, we can do that using the size of index file.
        self.numsamples = None
        if self.idx_version == 1:
            # version 1 has a 16-byte header
            # followed by a list of (offset, length) pairs of uint64
            header_size = 16
            filesize_idx = self.distctx.filesize(self.filename_idx)
            self.numsamples = int((filesize_idx - header_size) / 16)

    def read_index_header(self):
        """Read header from index file and check its version."""
        # Rank 0 reads the header, and bcasts its version
        version = None
        if self.distctx.rank == 0:
            try:
                # Seek to the front of the file
                self.fh_idx.seek(0)

                # Read the magic valud and check that it matches what we expect
                magic = self.fh_idx.read(8)
                if magic == b'INDXJSON':
                    # Good magic value, now read file format version number
                    buf = self.fh_idx.read(8)
                    if len(buf) == 8:
                        version = struct.unpack(">Q", buf)[0]
            except Exception as e:
                pass

        # Get version from rank 0 (should be None on any error)
        self.idx_version = self.distctx.bcast(version, root=0)

        # Check that we have a version number that we support
        if self.idx_version != 1:
            raise ValueError("Unknown index file format version '{self.idx_version}' in file '{self.filename_idx}'")

    def index_filename(self, filename):
        """Given the name of a JSON file, return the name of its index file."""
        return filename + '.idx'

    def get_start_end(self, num):
        """Given num items, compute and return [start,end) range on calling rank."""
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
        """Given a JSON file named dataset.jsonl, write index to dataset.jsonl.idx."""

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
        filesize = self.distctx.filesize(filename)

        # if progress messages are enabled, print a header about what we're doing
        if rank == 0 and self.progress > 0.0:
            print(f"Indexing '{filename}' of {filesize} bytes ...", flush=True)

        # create the temporary index file, shared across all ranks
        with self.distctx.open(filename_tmp) as ftmp:
            # open and scan the JSON file
            time_next = time_start + self.progress
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

                    # this can take a while, so print progress messages if enabled
                    if rank == 0 and self.progress > 0.0:
                        time_now = time.time()
                        if time_now > time_next:
                            time_next = time_now + self.progress
                            elapsed = time_now - time_start
                            percent = curpos * 100.0 / filesize if filesize > 0 else 0.0
                            byterate = curpos / elapsed / (1024.0 * 1024.0) if elapsed > 0.0 else 0.0
                            remaining = (100.0 - percent) * elapsed / percent if percent > 0.0 else 0.0
                            print(f"Scanned {curpos} of {filesize} bytes ({percent:0.2f}%) in {int(elapsed)} secs, "
                                  f"{byterate:0.3f} MB/s, {int(remaining)} secs left ...", flush=True)

        # Wait for all ranks to close the file.
        self.distctx.barrier()

        # Create the actual index file.
        with self.distctx.open(filename_idx) as fidx:
            # Rank 0 writes the index file header.
            if self.distctx.rank == 0:
                fidx.write(b'INDXJSON')          # use 8-byte magic value of "INDXJSON"
                fidx.write(struct.pack(">Q", 1)) # file format version number in network byte order
            data_offset = 16

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
                # We write values to the index file in network byte order,
                # so that the file can be read correctly on any system.
                if readcount > 0:
                    pos = data_offset + start * 16
                    fidx.seek(pos)
                    fidx.write(vals.astype(">i8").tobytes(order='C'))

        # Wait for all ranks to finish writing to the index file.
        self.distctx.barrier()

        # Can now delete the temporary index file.
        self.distctx.remove(filename_tmp)

        # Wait for everyone again and record how long it took.
        self.distctx.barrier()
        time_end = time.time()
        self.time_index = time_end - time_start

        # if progress messages are enabled, print a summary
        if rank == 0 and self.progress > 0.0:
            print(f"Indexed '{filename}' in {int(self.time_index)} seconds", flush=True)

    def __str__(self):
        return (f"IndexedJSON (\n"
                f"  file: {self.filename}\n"
                f"  rows: {self.numsamples}\n"
                f")")

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

        if self.idx_version == 1:
            # Version 1 has a 16-byte header followed by a
            # list of (offset, length) pairs of uint64

            # Seek to the right spot in the index file for the given sample id.
            header_size = 16
            offset_idx = header_size + idx * 16
            self.fh_idx.seek(offset_idx)

            # Read offset and length of record from the index.
            # Values in the index file are stored in network byte order.
            vals = np.zeros(2, dtype=">i8")
            self.fh_idx.readinto(vals)
            offset = vals[0]
            size = vals[1]

            return offset, size

    def pread(self, offset, size):
        """Read size bytes at the given offset in the JSON file and return as a buffer."""
        # seek to offset in JSON file and read the record
        self.fh_json.seek(offset)
        return self.fh_json.read(size)

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
