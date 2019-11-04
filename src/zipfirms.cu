#include <stdio.h>
#include <curand.h>
#include <curand_kernel.h>

static const unsigned int NUM_AGENTS = 1 << 14;
//static const unsigned int NUM_AGENTS = 2000000;
static const unsigned int NUM_FIRMS = 1 << 6;
//static const unsigned int NUM_FIRMS = 10000;
static const unsigned int NUM_ITER = 1 << 0;
//static const unsigned int NUM_ITER = 10000;

static const float Q = 0.5;
static const float BIAS = 0.5;
static const float P = Q + BIAS;
static const int THREADS_PER_BLOCK = 1 << 10;
static const int NUM_BLOCKS = ceil(NUM_AGENTS / THREADS_PER_BLOCK);

unsigned int *firms;
unsigned int *firms_tmp;
unsigned int *agents;
curandState *states;

    __global__
void init(unsigned int *firms, unsigned int *firms_tmp,
        unsigned int *agents, curandState *states, unsigned long seed)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx < NUM_AGENTS) {
        curandState state;

        // init random states
        curand_init(seed, (unsigned long long)idx, 0, &state);

        // randomly select an initial firm for agent
        agents[idx] = curand(&state) % NUM_FIRMS;

        // tally agents assigned to firms
        atomicAdd(&firms_tmp[agents[idx]], 1);

        // copy local state to global state array
        states[idx] = state;
    }
}

    __global__
void move(unsigned int *firms, unsigned int *firms_tmp,
        unsigned int *agents, curandState *states, unsigned int N)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    __shared__ int s_firms[NUM_FIRMS];

    if (idx < NUM_AGENTS) {

        curandState state = states[idx];
        unsigned int firm = agents[idx];
        unsigned int firmSize = firms[firm];
        unsigned int newFirm;
        unsigned int newFirmSize;
        float p;

        for (int i = 0; i < N; i++) {
            // reset local tally to 0
            for (int j = threadIdx.x; j < NUM_FIRMS; j += blockDim.x) {
                s_firms[j] = 0;
            }
            __syncthreads();
            // randomly select another firm
            newFirm = curand(&state) % NUM_FIRMS;
            newFirmSize = firms[newFirm];
            // compare firms to get probabiliy of moving
            p = newFirmSize > firmSize ? P : Q;
            if (curand_uniform(&state) < p) { // if moving
                // decrement local tally at old firm
                atomicSub(&s_firms[firm], 1);
                firm = newFirm;
                firmSize = newFirmSize;
                // increment local tally at new firm
                atomicAdd(&s_firms[firm], 1);
            }
            __syncthreads();
            for (int j =  threadIdx.x; j < NUM_FIRMS; j += blockDim.x) {
                atomicAdd(&firms_tmp[j], s_firms[j]);
            }
            __syncthreads();
        }
        agents[idx] = firm;
        states[idx] = state;
    }
}

/*
   __global__
   void stats(unsigned int *transactionPrice, unsigned int numTrades, unsigned int price) {
   }
 */

int main()
{
    int sum = 0;
    //unsigned long int seed = 0;
    //unsigned long int seed = (unsigned long int) time(NULL);
    unsigned long int seed = 1572534477;
    size_t firmSize = NUM_FIRMS*sizeof(unsigned int); // size of firm array
    size_t agentSize = NUM_AGENTS*sizeof(unsigned int); // size of firm array
    size_t stateSize = NUM_AGENTS*sizeof(curandState); // size of state array

    printf("Seed: %lu, Agents: %u, Firms: %u, Blocks: %i, Threads per block: %i, Threads: %i, Iterations: %u\n", seed, NUM_AGENTS, NUM_FIRMS, NUM_BLOCKS, THREADS_PER_BLOCK, NUM_BLOCKS * THREADS_PER_BLOCK, NUM_ITER);

    // allocate memeory
    // TODO implement error handling
    cudaMallocManaged(&firms, firmSize);
    cudaMallocManaged(&firms_tmp, firmSize);
    cudaMallocManaged(&agents, agentSize);
    cudaMallocManaged(&states, stateSize);

    init<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(firms, firms_tmp, agents, states, seed);
    cudaDeviceSynchronize();

    // copy firms_tmp to firms
    for (int i = 0; i < NUM_FIRMS; i++) {
        firms[i] = firms_tmp[i];
    }

    for (int i = 0; i < NUM_FIRMS; i++) {
        printf("%5u", firms[i]);
    }
    printf("\n");
    sum = 0;
    for (int i = 0; i < NUM_FIRMS; i++) {
        sum += firms[i];
    }
    //printf("%d\n", sum);

    // outer loop
    for (int i = 0; i < 100000 ; i++) {
        move<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(firms, firms_tmp, agents, states, NUM_ITER);
        cudaDeviceSynchronize();

        // copy firms_tmp to firms
        for (int i = 0; i < NUM_FIRMS; i++) {
            firms[i] = firms_tmp[i];
        }

        /*
        for (int i = 0; i < NUM_FIRMS; i++) {
            printf("%5u", firms[i]);
        }
        printf("\n");
        printf("\n");
        sum = 0;
        for (int i = 0; i < NUM_FIRMS; i++) {
            sum += firms[i];
        }
        printf("Agent Count: %d\n", sum);
        */
    }

    for (int i = 0; i < NUM_FIRMS; i++) {
        printf("%5u", firms[i]);
    }
    printf("\n");
    printf("\n");
    sum = 0;
    for (int i = 0; i < NUM_FIRMS; i++) {
        sum += firms[i];
    }
    printf("Agent Count: %d\n", sum);

    // free memory
    cudaFree(firms);
    cudaFree(agents);
    cudaFree(states);

    return EXIT_SUCCESS;
}
