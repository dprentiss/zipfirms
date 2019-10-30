#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

static const unsigned int NUM_AGENTS = 1 << 20;
static const unsigned int MAX_FIRMS =  1 << 15;

unsigned int *firmEmployees;
curandState *states;

    __global__
void init(unsigned int *firmEmployees, curandState *states, unsigned long seed)
{
    int idx = threadIdx.x;

    // init random states
    curand_init(seed, idx, 0, &states[idx]);

    // distribute employees as evenly as possible
    // first disribute the evenly divisible portion
    firmEmployees[idx] = NUM_AGENTS / NUM_FIRMS;
    // then disribute the remainder
    if (idx < NUM_AGENTS % NUM_FIRMS) firmEmployees[idx] += 1;
}

    __global__
void flow(unsigned int *firmEmployees, curandState *states, unsigned int N)
{
    int idx = threadIdx.x;

    // activate one agent with firm-size-weighted uniform probability
    // randomly select another firm
    // decide to move or not with probabilities p and q
    // move with atomicAdd
    __syncthreads();
}

__global__
void stats(unsigned int *transactionPrice, unsigned int numTrades, unsigned int price) {
    int idx = threadIdx.x;
    unsigned int traded = 1;

    if (transactionPrice == 0) traded = 0;
}

int main()
{
    unsigned long int seed = 0;
    size_t uintSize = NUM_FIRMS*sizeof(unsigned int); // size of firm array
    size_t stateSize = NUM_BUYERS*sizeof(curandState); // size of state array   

    // allocate managed memeory on device
    // TODO implement error handling
    cudaMallocManaged(&firmEmployees, uintSize);
    cudaMallocManaged(&states, stateSize);

    init<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(firmEmployees, states, seed);

    cudaDeviceSynchronize();

    flow<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(firmEmployees, N);

    cudaDeviceSynchronize();

    // free memory
    cudaFree(firmEmployees);

    return EXIT_SUCCESS;
}
