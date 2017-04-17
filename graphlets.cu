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
    int tri_e;
    int star_e;
};

// CUDA kernel used to count graphlets of size k=3
// Each thread processes a single edge
__global__
void graphlets(int* V, int V_num, int* E_u, int* E_v, int E_num, EDGE_OUTPUT* outputs)
{
	// Calculate global thread index in 1D grid of 1D blocks
    // Used as the edge number to compute
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Return immediately if thread index is greater than maximum edge index
    if (i >= E_num) return;

    // Lookup the endpoints of the current edge
    int u = E_u[i];
    int v = E_v[i];

    // Lookup starting and ending indices for u and v in the edge list array
    int u_start = V[u], u_end = V[u + 1], v_start = V[v], v_end = V[v + 1];
    int iu = u_start, iv = v_start;
    int tri_e = 0, star_e = 0;

    while (iu < u_end || iv < v_end) {
        if (E_v[iu] == v) {
            // If current neighbor of u is v, skip by incrementing iu
            iu++;
        } else if (E_v[iv] == u) {
            // If current neighbor of v is u, skip by incrementing iv
            iv++;
        } else if (E_v[iu] < E_v[iv]) {
            // If current neighbor of u is less than current neighbor of v, star is found
            star_e++;
            iu++;
        } else if (E_v[iv] < E_v[iu]) {
            // If current neighbor of v is less than current neighbor of u, star is found
            star_e++;
            iv++;
        } else if (E_v[iu] == E_v[iv]) {
            // If u and v have the same current neighbor, triangle is found
            tri_e++;
            iu++;
            iv++;
        }
    }

    outputs[i].tri_e = tri_e;
    outputs[i].star_e = star_e;
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

        edge_count += 2;
	}

    int V_num = adj_list.size();
    int E_num = edge_count;

    std::cout << "Graph found with " << V_num << " nodes and " << E_num << " edges" << std::endl;

    // Value of V[i] is index of E_u and E_v where vertex i's edges begin
    // Value of V[i+1] is index of E_u and E_v where vertex i's edges end
    std::vector<int> V;

    // Value of E_u[i] is the source vertex id (as used in V) associated with edge i
    std::vector<int> E_u;

    // Value of E_v[i] is the destination vertex id of edge i
    std::vector<int> E_v;

    V.reserve(V_num + 1);
    E_u.reserve(E_num);
    E_v.reserve(E_num);

    // Build V, E_u, and E_v from adjacency list representation
    int edge_index = 0;
    for (int i = 0; i < adj_list.size(); i++) {
        V.push_back(edge_index);

        for (int j = 0; j < adj_list[i].size(); j++) {
            E_u.push_back(i);
            E_v.push_back(adj_list[i][j]);
        }

        edge_index += adj_list[i].size();
    }
    V.push_back(edge_index);

    // Create and initialize CUDA thread output structs
    std::vector<EDGE_OUTPUT> outputs(E_num);
    outputs.resize(E_num, {0, 0});

    // Pointers of V, E_u, E_v, and edge outputs in GPU memory
    int* V_ptr;
    int* E_u_ptr;
    int* E_v_ptr;
    EDGE_OUTPUT* outputs_ptr;

    int V_size = (V_num + 1) * sizeof(int);
    int E_size = E_num * sizeof(int);
    int outputs_size = E_num * sizeof(EDGE_OUTPUT);
    
    // Malloc GPU memory and store location in pointers
    cudaMalloc((void**)&V_ptr, V_size);
	cudaMalloc((void**)&E_u_ptr, E_size);
    cudaMalloc((void**)&E_v_ptr, E_size);
    cudaMalloc((void**)&outputs_ptr, outputs_size);

    // Copy data structures from main memory to allocated GPU memory
    cudaMemcpy(V_ptr, V.data(), V_size, cudaMemcpyHostToDevice);
	cudaMemcpy(E_u_ptr, E_u.data(), E_size, cudaMemcpyHostToDevice);
    cudaMemcpy(E_v_ptr, E_v.data(), E_size, cudaMemcpyHostToDevice);
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
    graphlets<<<dimGrid, dimBlock>>>(V_ptr, V_num, E_u_ptr, E_v_ptr, E_num, outputs_ptr);

    // Copy output data from GPU memory back into main memory
    cudaMemcpy(outputs.data(), outputs_ptr, outputs_size, cudaMemcpyDeviceToHost);

    // Free memory in GPU
	cudaFree(V_ptr);
	cudaFree(E_u_ptr);
    cudaFree(E_v_ptr);
    cudaFree(outputs_ptr);

    // Compute graphlet counts based upon edge outputs
    int g31 = 0, g32 = 0, g33 = 0, g34;

    for (int i = 0; i < E_num; i++) {
        g31 += outputs[i].tri_e;
        g32 += outputs[i].star_e;
        g33 += V_num - (outputs[i].tri_e + outputs[i].star_e + 2);
    }

    g31 /= 3;
    g32 /= 2;

    int V_num_choose_3 = (V_num * (V_num - 1) * (V_num - 2)) / (3 * 2);
    g34 = V_num_choose_3 - g31 - g32 - g33;

    std::cout << std::endl;
    std::cout << "     Graphlet Counts" << std::endl;
    std::cout << "--------------------------" << std::endl;
    std::cout << "triangle           (g31) : " << g31 << std::endl;
    std::cout << "2-star             (g32) : " << g32 << std::endl;
    std::cout << "3-node-1-edge      (g33) : " << g33 << std::endl;
    std::cout << "3-node-independent (g34) : " << g34 << std::endl;

	return EXIT_SUCCESS;
}
