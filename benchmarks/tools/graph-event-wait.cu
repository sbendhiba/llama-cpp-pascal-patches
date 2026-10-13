// Can an external (non-captured) stream wait on an event that a CUDA graph records, and can that
// wait fire EARLY - at a node inside the graph rather than at graph completion?
//
// Why this decides the tensor-parallel exchange overlap: the producing matmul lives inside the
// device's captured subgraph (patch 50 is on by default), the AllReduce runs as ordinary
// host-enqueued work between subgraphs, and the wire transfer of chunk c must start when chunk c's
// kernels finish. That is exactly "wait from outside on an event recorded inside".
//
// Graph in every mode: [writer W] -> [spin ~300 ms], with the event recorded
//   mode 0: cudaEventRecord(ev, s1) during the capture (between W and the spin)
//   mode 1: cudaGraphAddEventRecordNode appended after capture, depending on the LAST node
//   mode 2: cudaGraphAddEventRecordNode appended after capture, depending on W  <- the primitive we need
//
// Then: cudaGraphLaunch(exec, s1); cudaStreamWaitEvent(s2, ev); reader on s2 snapshots W's flag and
// the spin's done flag before spinning ~150 ms itself.
//
//   wait rejected                        -> route closed
//   flag == 0                            -> the wait did not track W at all
//   flag == 1, spin_done == 0            -> EARLY: overlap primitive works
//   flag == 1, spin_done == 1            -> LATE: graph completion only
//
// build: nvcc -O2 -arch=sm_60 p2pev2.cu -o p2pev2

#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>

#define SPIN_TAIL_MS 300.0
#define SPIN_READ_MS 150.0

__global__ void writer(int * flag) {
    *flag = 1;
}

__global__ void spin_kernel(long long cycles, int * done) {
    long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
    if (done) {
        *done = 1;
    }
}

__global__ void reader(const int * flag, const int * spin_done, long long * out, long long cycles) {
    out[0] = (long long) (*flag);
    out[1] = (long long) (*spin_done);
    long long t0 = clock64();
    while (clock64() - t0 < cycles) {}
}

static double spin_cycles_ms(long long cycles, cudaStream_t s) {
    auto c0 = std::chrono::steady_clock::now();
    spin_kernel<<<1, 1, 0, s>>>(cycles, nullptr);
    cudaStreamSynchronize(s);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
}

static cudaGraphNode_t find_node(cudaGraph_t graph, size_t n_nodes, cudaGraphNode_t * nodes,
                                 const void * func, bool last_of_chain) {
    cudaGraphNodeType t = {};
    cudaKernelNodeParams p = {};
    for (size_t i = 0; i < n_nodes; ++i) {
        cudaGraphNodeGetType(nodes[i], &t);
        if (t != cudaGraphNodeTypeKernel) {
            continue;
        }
        if (cudaGraphKernelNodeGetParams(nodes[i], &p) != cudaSuccess) {
            continue;
        }
        if (p.func == func) {
            return nodes[i];
        }
    }
    if (last_of_chain && n_nodes > 0) {
        return nodes[n_nodes - 1];
    }
    return nullptr;
}

static void run_mode(int mode, int * flag, int * spin_done, long long * out, long long G, long long R) {
    cudaStream_t s1 = nullptr, s2 = nullptr;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);

    cudaEvent_t ev = nullptr;
    cudaEventCreateWithFlags(&ev, cudaEventDisableTiming);

    cudaMemset(flag, 0, sizeof(int));
    cudaMemset(spin_done, 0, sizeof(int));
    cudaMemset(out, 0, 2 * sizeof(long long));

    cudaError_t rc = cudaStreamBeginCapture(s1, cudaStreamCaptureModeRelaxed);
    if (rc != cudaSuccess) { printf("  begin capture: %s\n", cudaGetErrorString(rc)); return; }
    writer<<<1, 1, 0, s1>>>(flag);
    if (mode == 0) {
        rc = cudaEventRecord(ev, s1);
        if (rc != cudaSuccess) { printf("  cudaEventRecord inside capture: %s\n", cudaGetErrorString(rc)); }
    }
    spin_kernel<<<1, 1, 0, s1>>>(G, spin_done);
    cudaGraph_t graph = nullptr;
    rc = cudaStreamEndCapture(s1, &graph);
    if (rc != cudaSuccess || graph == nullptr) { printf("  end capture: %s\n", cudaGetErrorString(rc)); return; }

    if (mode != 0) {
        size_t nn = 0;
        cudaGraphGetNodes(graph, nullptr, &nn);
        cudaGraphNode_t nodes[16] = {};
        size_t want = nn < 16 ? nn : 16;
        cudaGraphGetNodes(graph, nodes, &want);
        cudaGraphNode_t dep = find_node(graph, want, nodes, mode == 2 ? (const void *) writer
                                                                      : (const void *) spin_kernel,
                                        /*last_of_chain*/ true);
        if (dep == nullptr) { printf("  could not find the dependency node\n"); cudaGraphDestroy(graph); return; }
        cudaGraphNode_t rec = nullptr;
        rc = cudaGraphAddEventRecordNode(&rec, graph, &dep, 1, ev);
        if (rc != cudaSuccess) { printf("  cudaGraphAddEventRecordNode: %s\n", cudaGetErrorString(rc)); cudaGraphDestroy(graph); return; }
        nn = want + 1;
        printf("  graph nodes %zu, record node depends on the %s node\n", nn, mode == 2 ? "writer" : "spin");
    } else {
        size_t nn = 0;
        cudaGraphGetNodes(graph, nullptr, &nn);
        printf("  graph nodes %zu (the in-capture record left no node)\n", nn);
    }

    cudaGraphExec_t exec = nullptr;
    rc = cudaGraphInstantiateWithFlags(&exec, graph, 0);
    if (rc != cudaSuccess) { printf("  instantiate: %s\n", cudaGetErrorString(rc)); cudaGraphDestroy(graph); return; }

    auto t0 = std::chrono::steady_clock::now();
    rc = cudaGraphLaunch(exec, s1);
    if (rc != cudaSuccess) { printf("  graph launch: %s\n", cudaGetErrorString(rc)); return; }

    rc = cudaStreamWaitEvent(s2, ev, 0);
    if (rc != cudaSuccess) {
        printf("  external cudaStreamWaitEvent on the graph's event: %s\n", cudaGetErrorString(rc));
        printf("  => ROUTE CLOSED: an ordinary stream cannot wait on this event\n");
        cudaDeviceSynchronize();
        return;
    }
    reader<<<1, 1, 0, s2>>>(flag, spin_done, out, R);
    rc = cudaGetLastError();
    if (rc != cudaSuccess) { printf("  reader launch: %s\n", cudaGetErrorString(rc)); return; }
    cudaDeviceSynchronize();
    double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    long long host[2] = {};
    cudaMemcpy(host, out, 2 * sizeof(long long), cudaMemcpyDeviceToHost);
    printf("  combined %.1f ms (serialised would be %.0f ms)\n", ms, SPIN_TAIL_MS + SPIN_READ_MS);
    printf("  reader saw flag=%lld spin_done=%lld  => %s\n", host[0], host[1],
           host[0] == 0 ? "the wait did not track the writer"
                        : (host[1] == 0 ? "EARLY - overlap primitive works" : "LATE - graph completion only"));

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaEventDestroy(ev);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
}

int main() {
    cudaSetDevice(0);
    int * flag = nullptr, * spin_done = nullptr;
    long long * out = nullptr;
    cudaMalloc(&flag, sizeof(int));
    cudaMalloc(&spin_done, sizeof(int));
    cudaMalloc(&out, 2 * sizeof(long long));

    cudaStream_t s0 = nullptr;
    cudaStreamCreate(&s0);
    const long long cal = 100LL * 1000 * 1000;
    const double ms100 = spin_cycles_ms(cal, s0);
    const long long G = (long long) (cal * SPIN_TAIL_MS / ms100);
    const long long R = (long long) (cal * SPIN_READ_MS / ms100);
    printf("calibration %.1f ms per 100M cycles; tail %lld cyc (~%.0f ms), reader %lld cyc (~%.0f ms)\n",
           ms100, G, SPIN_TAIL_MS, R, SPIN_READ_MS);
    printf("control, no graph: tail+reader serialised should be ~%.0f ms\n\n", SPIN_TAIL_MS + SPIN_READ_MS);

    printf("mode 0: cudaEventRecord inside the capture, between writer and spin\n");
    run_mode(0, flag, spin_done, out, G, R);
    printf("\nmode 1: cudaGraphAddEventRecordNode after the spin (fires at graph end)\n");
    run_mode(1, flag, spin_done, out, G, R);
    printf("\nmode 2: cudaGraphAddEventRecordNode after the writer (fires mid-graph)\n");
    run_mode(2, flag, spin_done, out, G, R);

    cudaFree(flag);
    cudaFree(spin_done);
    cudaFree(out);
    cudaStreamDestroy(s0);
    return 0;
}
