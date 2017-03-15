# Algorithms

Index of algorithms provided by Gloo and their semantics.

Variables used:
* **P**: Number of processes/machines
* **N**: Number of buffers per process
* **S**: Size of buffer

Terms used:
* **Communication steps**: number of communication steps. Every
  communication step has some latency, depending on the transport.
  Therefore, the fewer steps an algorithm uses, the better it is
  suited towards higher latency transports. Lower latency transports
  better tolerate more communication steps.
* **Bytes on the wire**: total number of bytes transmitted per
  participating process. The higher this number, the sooner an
  algorithm will be bound by the network bandwidth.

## Allreduce

Compute sum of N arrays per process across P processes. This
computation happens in place; all input arrays contain the resulting
sum after the algorithm completes.

There's 3 phases to each implementation of this algorithm:
1. Local reduction of N buffers
2. Allreduce between processes
3. Broadcast result back to N buffers

### allreduce_ring

* Communication steps: P-1
* Bytes on the wire: P\*S

Phase 2 is implemented as follows:
1. Transmit local result to right side neighbor
2. Receive buffer from left side neighbor and reduce into local result
3. Transmit incoming buffer to right side neighbor
4. Repeat 2-3 until process has seen all data

### allreduce_ring_chunked

* Communication steps: 4\*P
* Bytes on the wire: 2\*S

Phase 2 is implemented in 2 sub-phases:
1. First, the algorithm iterates over the local reduction,
   transmitting chunks of the buffer and reducing at every step. The
   number of chunks is equal to 2\*P, allowing double buffering to be
   used. This means there is always one chunk in flight while
   reduction is done on another chunk concurrently. At the end of this
   phase, every process P holds 1/P of the reduced result.
2. Second, the algorithm iterates over the local reduction again, now
   broadcasting the local results.

With 2\*P chunks and two sub-phases, we arrive at 4\*P communication
steps.

These sub-phases are implemented as followed (roughly):

First:
1. Compute offset into local reduction buffer based on process rank
2. Transmit chunk at offset to right side neighbor
3. Receive chunk at offset-1 from left side neighbor and reduce into
   local result
4. Subtract 1 from offset, wrapping when needed
5. Repeat 2-4 until process has walked entire buffer

Second:
1. Transmit chunk at offset+1 (containing the global reduction) to
   right side neighbor
2. Receive chunk at offset from left side neighbor and copy into local
   result
3. Subtract 1 from offset, wrapping when needed
4. Repeat 1-3 until process has walked entire buffer

### cuda_allreduce_ring

CUDA-aware implementation of `allreduce_ring`. GPU side buffers are
copied to system memory in parallel, prior to running local reduction
on CPU. After phase 2 completes, CPU side result is copied back to GPU
side buffers in parallel.

### cuda_allreduce_ring_chunked

CUDA-aware implementation of `allreduce_ring_chunked`. GPU side
buffers are reduced into GPU buffer 0 (using NCCL). The result is
copied to system memory asynchronously. After phase 2 completes, the
CPU side result is copied back to GPU buffer 0, and then broadcast to
other GPU buffers in parallel (using NCCL).

Both local reduction in phase 1 and broadcast in phase 3 is pipelined
with the communication steps where this data is needed or becomes
available.

## Barrier

Synchronization point between processes.

### barrier_all_to_all

* Communication steps: 1
* Bytes on the wire: P

Every process sends a notification to every other process.
Then, it waits for a notification from every other process.

### barrier_all_to_one

* Communication steps: 2
* Bytes on the wire: 1 for non-root, P for root

_Non-root processes_: send notification to root, wait for notification
from root.

_Root process_: wait for notification from P-1 processes, send
notification to P-1 processes.

## Broadcast

Broadcast contents of buffer on one process to other P-1 processes.

### broadcast_one_to_all

* Communication steps: 1
* Bytes on the wire: P\*S

_Non-root processes_: receive buffer from root.

_Root process_: send buffer to P-1 processes.
