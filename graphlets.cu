#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>

// Number of threads in each block
// Should be a multiple of GPU warp size (32 in recent architectures)
// TODO: Tune this parameter based upon empirical performance
const int blocksize = 128;

// Struct containing edge outputs computed in parallel
// One produced per thread, aggregated after GPU computation
struct EDGE_OUTPUT {
    // 3-node graphlet counts
    unsigned long long g31;
    unsigned long long g32;
    unsigned long long g33;

    // 4-node cliques and cycles
    unsigned long long g41;
    unsigned long long g44;

    // Unrestricted counts for 4-node connected graphlets
    unsigned long long T_T;
    unsigned long long Su_Sv;
    unsigned long long T_SuVSv;
    unsigned long long S_S;
    
    // Unrestricted counts for 4-node disconnected graphlets
    unsigned long long T_I;
    unsigned long long SuVSv_I;
    unsigned long long I_I;
    unsigned long long I_I_1;
};

struct GRAPHLET_COUNTS {
    // 3-node graphlet counts
    unsigned long long g31;
    unsigned long long g32;
    unsigned long long g33;
    unsigned long long g34;

    // 4-node graphlet counts
    unsigned long long g41;
    unsigned long long g42;
    unsigned long long g43;
    unsigned long long g44;
    unsigned long long g45;
    unsigned long long g46;
    unsigned long long g47;
    unsigned long long g48;
    unsigned long long g49;
    unsigned long long g410;
    unsigned long long g411;
};

// CUDA kernel used to count graphlets of size k=3
// Each thread processes a single edge
__global__
void graphlets(int* V, unsigned long long V_num, int* E, unsigned long long E_num, int* E_u, int* E_v, EDGE_OUTPUT* outputs)
{
	// Calculate global thread index in 1D grid of 1D blocks
    // Used as the edge number to compute
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Return immediately if thread index is greater than maximum edge index
    if (i >= E_num) return;

    int ei = E[i];

    // Lookup the endpoints of the current edge
    int u = E_u[ei];
    int v = E_v[ei];

    int u_len = V[u + 1] - V[u] - 1;
    int** ptrs = new int*[u_len];
    int* lens = new int[u_len];
    int* inds = new int[u_len];

    int prev_w = -1;
    for (int edge_i = V[u], arr_i = 0; edge_i < V[u + 1]; ++edge_i && ++arr_i) {
        int w = E_v[edge_i];
        
        if (w == v) {
            arr_i--;
            continue;
        }

        ptrs[arr_i] = &E_v[V[w]];
        lens[arr_i] = V[w + 1] - V[w];
        inds[arr_i] = 0;

        while (inds[arr_i] < lens[arr_i] && ptrs[arr_i][inds[arr_i]] <= prev_w) {
            inds[arr_i]++;
        }

        prev_w = w;
    }

    int tri_e = 0, star_u = 0, star_v = 0, cliq_e = 0, cyc_e = 0;
    int iu = V[u], iv = V[v];
    while (iu < V[u + 1] || iv < V[v + 1]) {
        if (iv < V[v + 1]) {
            for (int i = 0; i < u_len; i++) {
                while (inds[i] < lens[i] && ptrs[i][inds[i]] < E_v[iv]) {
                    inds[i]++;
                }
            }
        }

        int *count_ptr = NULL;
        if (iu < V[u + 1] && (iv == V[v + 1] || E_v[iu] < E_v[iv])) {
            if (E_v[iu] == v) {
                iu++;
                continue;
            }

            star_u++;
            count_ptr = NULL;
        } else if (iu == V[u + 1] || E_v[iv] < E_v[iu]) {
            if (E_v[iv] == u) {
                iv++;
                continue;
            }

            star_v++;
            count_ptr = &cyc_e;
        } else {
            tri_e++;
            count_ptr = &cliq_e;
        }

        if (count_ptr != NULL) {
            for (int i = 0; i < u_len; i++) {
                if (inds[i] < lens[i] && ptrs[i][inds[i]] == E_v[iv]) {
                    (*count_ptr)++;
                }
            }
        }

        if (iu < V[u + 1] && (iv == V[v + 1] || E_v[iu] < E_v[iv])) {
            iu++;
        } else if (iu == V[u + 1] || E_v[iv] < E_v[iu]) {
            iv++;
        } else {
            iu++;
            iv++;
        }
    }

    outputs[i].g31 = tri_e;
    outputs[i].g32 = star_u + star_v;
    outputs[i].g33 = V_num - (tri_e + star_u + star_v + 2);
    outputs[i].g41 = cliq_e;
    outputs[i].g44 = cyc_e;

    outputs[i].T_T = (tri_e * (tri_e - 1)) / 2;
    outputs[i].Su_Sv = star_u * star_v;
    outputs[i].T_SuVSv = tri_e * (star_u + star_v);
    outputs[i].S_S = ((star_u * (star_u - 1)) / 2) + ((star_v * (star_v - 1)) / 2);

    outputs[i].T_I = tri_e * outputs[i].g33;
    outputs[i].SuVSv_I = (star_u + star_v) * outputs[i].g33;
    outputs[i].I_I = (outputs[i].g33 * (outputs[i].g33 - 1)) / 2;
    outputs[i].I_I_1 = E_num - (V[u + 1] - V[u] - 1) - (V[v + 1] - V[v] - 1) - 1;
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cout << "usage: " << argv[0] << " <input_file>" << std::endl;
        return EXIT_FAILURE;
    }

    // Nested vector used to store read file into adjacency list
    std::vector< std::vector<int> > adj_list;
    
    // Input network file provided by user
    // File format assume to be edge list
    std::ifstream infile(argv[1]);

    std::string su, sv;
	int u, v, edge_count = 0, max = -1;
	while (getline(infile, su, '\t') && getline(infile, sv)) {
        // Node ids assumed to be 1-indexed and decremented to be 0-indexed
        u = std::stoi(su) - 1;
        v = std::stoi(sv) - 1;

        // Dynamically add empty vectors as nodes are found in edge list
        int new_max = u > v ? u : v;
        if (new_max > max) {
            for (int i = max + 1; i <= new_max; i++) {
                adj_list.push_back(std::vector<int>(0));
            }
            max = new_max;
        }

        // Add both directions of edge to adjacency list
        adj_list[u].push_back(v);
        adj_list[v].push_back(u);

        edge_count++;
    }

    unsigned long long V_num = adj_list.size();
    unsigned long long E_num = edge_count;

    std::cout << "Graph found with " << V_num << " nodes and " << E_num << " edges" << std::endl;

    // Value of V[i] is index of E_u and E_v where vertex i's edges begin
    // Value of V[i+1] is index of E_u and E_v where vertex i's edges end
    std::vector<int> V;

    std::vector<int> E;

    // Value of E_u[i] is the source vertex id (as used in V) associated with edge i
    std::vector<int> E_u;

    // Value of E_v[i] is the destination vertex id of edge i
    std::vector<int> E_v;

    V.reserve(V_num + 1);
    E.reserve(E_num);
    E_u.reserve(E_num * 2);
    E_v.reserve(E_num * 2);

    // Build V, E_u, and E_v from adjacency list representation
    int edge_index = 0;
    for (int i = 0; i < adj_list.size(); i++) {
        V.push_back(edge_index);

        for (int j = 0; j < adj_list[i].size(); j++) {
            E_u.push_back(i);
            E_v.push_back(adj_list[i][j]);

            if (i < adj_list[i][j]) {
                E.push_back(E_u.size() - 1);
            }
        }

        edge_index += adj_list[i].size();
    }
    V.push_back(edge_index);

    // Create and initialize CUDA thread output structs
    std::vector<EDGE_OUTPUT> outputs(E_num);
    outputs.resize(E_num, {0});

    // Pointers of V, E_u, E_v, and edge outputs in GPU memory
    int* V_ptr;
    int* E_ptr;
    int* E_u_ptr;
    int* E_v_ptr;
    EDGE_OUTPUT* outputs_ptr;

    int V_size = (V_num + 1) * sizeof(int);
    int E_size = E_num * sizeof(int);
    int outputs_size = E_num * sizeof(EDGE_OUTPUT);
    
    // Malloc GPU memory and store location in pointers
    cudaMalloc((void**)&V_ptr, V_size);
    cudaMalloc((void**)&E_ptr, E_size);
    cudaMalloc((void**)&E_u_ptr, E_size * 2);
    cudaMalloc((void**)&E_v_ptr, E_size * 2);
    cudaMalloc((void**)&outputs_ptr, outputs_size);

    // Copy data structures from main memory to allocated GPU memory
    cudaMemcpy(V_ptr, V.data(), V_size, cudaMemcpyHostToDevice);
    cudaMemcpy(E_ptr, E.data(), E_size, cudaMemcpyHostToDevice);
    cudaMemcpy(E_u_ptr, E_u.data(), E_size * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(E_v_ptr, E_v.data(), E_size * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(outputs_ptr, outputs.data(), outputs_size, cudaMemcpyHostToDevice);

    // Calculate number of blocks in CUDA grid based upon number of edges in graph
    int gridsize = E_num / blocksize;
    if (E_num % blocksize > 0) {
        gridsize++;
    }

    // Create one-dimensional blocks and grids based upon blocksize and gridsize
    dim3 dimBlock(blocksize, 1);
    dim3 dimGrid(gridsize, 1);

    // Execute CUDA kernel
    graphlets<<<dimGrid, dimBlock>>>(V_ptr, V_num, E_ptr, E_num, E_u_ptr, E_v_ptr, outputs_ptr);

    // Copy output data from GPU memory back into main memory
    cudaMemcpy(outputs.data(), outputs_ptr, outputs_size, cudaMemcpyDeviceToHost);

    // Free memory in GPU
    cudaFree(V_ptr);
    cudaFree(E_ptr);
    cudaFree(E_u_ptr);
    cudaFree(E_v_ptr);
    cudaFree(outputs_ptr);

    // Compute graphlet counts based upon edge outputs
    EDGE_OUTPUT aggregates = {0};

    for (int i = 0; i < E_num; i++) {
        aggregates.g31 += outputs[i].g31;
        aggregates.g32 += outputs[i].g32;
        aggregates.g33 += outputs[i].g33;

        aggregates.g41 += outputs[i].g41;
        aggregates.g44 += outputs[i].g44;

        aggregates.T_T += outputs[i].T_T;
        aggregates.Su_Sv += outputs[i].Su_Sv;
        aggregates.T_SuVSv += outputs[i].T_SuVSv;
        aggregates.S_S += outputs[i].S_S;

        aggregates.T_I += outputs[i].T_I;
        aggregates.SuVSv_I += outputs[i].SuVSv_I;
        aggregates.I_I += outputs[i].I_I;
        aggregates.I_I_1 += outputs[i].I_I_1;
    }

    GRAPHLET_COUNTS counts = {0};

    counts.g31 = aggregates.g31 / 3;
    counts.g32 = aggregates.g32 / 2;
    counts.g33 = aggregates.g33;
    counts.g34 = ((V_num * (V_num - 1) * (V_num - 2)) / (3 * 2)) - counts.g31 - counts.g32 - counts.g33;

    counts.g41 = aggregates.g41 / 6;
    counts.g42 = aggregates.T_T - 6 * counts.g41;
    counts.g43 = (aggregates.T_SuVSv - 4 * counts.g42) / 2;
    counts.g44 = aggregates.g44 / 4;
    counts.g45 = (aggregates.S_S - counts.g43) / 3;
    counts.g46 = aggregates.Su_Sv - 4 * counts.g44;
    counts.g47 = (aggregates.T_I - counts.g43) / 3;
    counts.g48 = (aggregates.SuVSv_I - 2 * counts.g46) / 2;
    counts.g49 = (aggregates.I_I_1 - (6 * counts.g41) - (4 * counts.g42) - (2 * counts.g43) - (4 * counts.g44) - (2 * counts.g46)) / 2;
    counts.g410 = aggregates.I_I - 2 * counts.g49;
    counts.g411 = ((V_num * (V_num - 1) * (V_num - 2) * (V_num - 3)) / (4 * 3 * 2)) - counts.g41 - counts.g42 - counts.g43 - counts.g44 - counts.g45 - counts.g46 - counts.g47 - counts.g48 - counts.g49 - counts.g410;

    std::cout << std::endl;
    std::cout << "      Graphlet Counts" << std::endl;
    std::cout << "===========================" << std::endl;
    std::cout << std::endl;
    std::cout << "k=4 Connected Graphlets" << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "4-clique           (g41)  : " << counts.g41 << std::endl;
    std::cout << "4-chordalcycle     (g42)  : " << counts.g42 << std::endl;
    std::cout << "4-tailedtriangle   (g43)  : " << counts.g43 << std::endl;
    std::cout << "4-cycle            (g44)  : " << counts.g44 << std::endl;
    std::cout << "3-star             (g45)  : " << counts.g45 << std::endl;
    std::cout << "4-path             (g46)  : " << counts.g46 << std::endl;
    std::cout << std::endl;
    std::cout << "k=4 Disconnected Graphlets" << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "4-node-1-triangle  (g47)  : " << counts.g47 << std::endl;
    std::cout << "4-node-2-star      (g48)  : " << counts.g48 << std::endl;
    std::cout << "4-node-2-edge      (g49)  : " << counts.g49 << std::endl;
    std::cout << "4-node-1-edge      (g410) : " << counts.g410 << std::endl;
    std::cout << "4-node-independent (g411) : " << counts.g411 << std::endl;
    std::cout << std::endl;
    std::cout << "k=3 Graphlets" << std::endl;
    std::cout << "---------------------------" << std::endl;
    std::cout << "triangle           (g31)  : " << counts.g31 << std::endl;
    std::cout << "2-star             (g32)  : " << counts.g32 << std::endl;
    std::cout << "3-node-1-edge      (g33)  : " << counts.g33 << std::endl;
    std::cout << "3-node-independent (g34)  : " << counts.g34 << std::endl;

    return EXIT_SUCCESS;
}
