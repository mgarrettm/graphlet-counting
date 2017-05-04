#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <climits>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

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

	// 4-node connected graphlet counts
	unsigned long long g41;
	unsigned long long g42;
	unsigned long long g43;
	unsigned long long g44;
	unsigned long long g45;
	unsigned long long g46;

	// 4-node disconnected graphlet counts
	unsigned long long g47;
	unsigned long long g48;
	unsigned long long g49;
	unsigned long long g410;
	unsigned long long g411;
};

// CUDA kernel used to count graphlets of size k=4
// Each thread processes a single edge
__global__
void graphlets(int* V, unsigned long long V_num, int* E, unsigned long long E_num, int* E_u, int* E_v, EDGE_OUTPUT* outputs)
{
	// Calculate global thread index in 1D grid of 1D blocks
	// Used as the undirected edge number to compute
	int edge = blockIdx.x * blockDim.x + threadIdx.x;

	// Return immediately if thread index is greater than maximum edge index
	if (edge >= E_num) return;

	// Using thread number, look up directed edge number in array of undirected edges
	int ei = E[edge];

	// Lookup the endpoints of the current edge
	// TODO: Dynamically choose u to be the node with smallest neighborhood
	int u = E_u[ei];
	int v = E_v[ei];

	// Array length is 1 less because v is omitted
	int arr_len = V[u + 1] - V[u] - 1;
	// Array holding current index of edges in E_v of each of u's neighbors
	int* inds = new int[arr_len];
	// Array holding maximum index of edges in E_v of each of u's neighbors
	int* ends = new int[arr_len];
	// Array indicating whether each of u's neighbors neighbors v
	bool* neighbors_v = new bool[arr_len];

	// To count graphlets, nodes are advanced in ascending order concurrently across u, v,
	// and all of u's neighbors. Through this walk, counts can be gathered by checking
	// which nodes have identical neighbors after each step
	int tri_e = 0, star_u = 0, star_v = 0;
	int iu = V[u], iv = V[v], arr_i = 0;
	while (iu < V[u + 1] || iv < V[v + 1]) {
		int cu = iu < V[u + 1] ? E_v[iu] : INT_MAX;
		int cv = iv < V[v + 1] ? E_v[iv] : INT_MAX;

		if (cu < cv) {
			if (cu != v) {
				// A star with u is found when the current node in the walk is a neighbor of u
				// but not a neighbor of v
				star_u++;

				inds[arr_i] = V[cu];
				ends[arr_i] = V[cu + 1];
				neighbors_v[arr_i] = false;
				arr_i++;
			}

			iu++;
		}
		else if (cv < cu) {
			if (cv != u) {
				// A star with v is found when the current node in the walk is a neighbor of v
				// but not a neighbor of u
				star_v++;
			}

			iv++;
		}
		else {
			// A triangle is found when the current node in the walk is both a neighbor of
			// u and a neighbor of v
			tri_e++;

			inds[arr_i] = V[cu];
			ends[arr_i] = V[cu + 1];
			neighbors_v[arr_i] = true;
			arr_i++;

			iu++;
			iv++;
		}
	}

	int cliq_e = 0, cyc_e = 0;
	iu = V[u];
	iv = V[v];
	while (iu < V[u + 1] || iv < V[v + 1]) {
		int cu = iu < V[u + 1] ? E_v[iu] : INT_MAX;
		int cv = iv < V[v + 1] ? E_v[iv] : INT_MAX;

		// Cycles and cliques can only occur when current node is in N(v) \ {u}
		if (cv <= cu && cv != u) {
			for (int arr_i = 0; arr_i < arr_len; arr_i++) {
				// Before checking for cliques or cycles, the edge index is advanced to the current
				// location in the walk
				while (inds[arr_i] < ends[arr_i] && E_v[inds[arr_i]] < cv) {
					inds[arr_i]++;
				}

				// If u's neighbor neighbors v's neighbor, a clique or cycle may be found
				if (inds[arr_i] < ends[arr_i] && E_v[inds[arr_i]] == cv) {
					if (cu == cv && neighbors_v[arr_i]) {
						// If u's neighbor and v's neighbor form triangles with e, a clique is found
						cliq_e++;
					}
					else if (cu != cv && !neighbors_v[arr_i]) {
						// If neither u's neighbor or v's neighbor form triangles with e, a cycle is found
						cyc_e++;
					}
				}
			}
		}

		if (cu <= cv) iu++;
		if (cv <= cu) iv++;
	}

    delete(inds);
    delete(ends);
    delete(neighbors_v);

	// 3-node graphlet and 4-node unrestricted counts calculated as described
	// in http://nesreenahmed.com/publications/ahmed-et-al-icdm2015.pdf
	outputs[edge].g31 = tri_e;
	outputs[edge].g32 = star_u + star_v;
	outputs[edge].g33 = V_num - (tri_e + star_u + star_v + 2);
	outputs[edge].g41 = cliq_e / 2;
	outputs[edge].g44 = cyc_e;

	outputs[edge].T_T = (tri_e * (tri_e - 1)) / 2;
	outputs[edge].Su_Sv = star_u * star_v;
	outputs[edge].T_SuVSv = tri_e * (star_u + star_v);
	outputs[edge].S_S = ((star_u * (star_u - 1)) / 2) + ((star_v * (star_v - 1)) / 2);

	outputs[edge].T_I = tri_e * outputs[edge].g33;
	outputs[edge].SuVSv_I = (star_u + star_v) * outputs[edge].g33;
	outputs[edge].I_I = (outputs[edge].g33 * (outputs[edge].g33 - 1)) / 2;
	outputs[edge].I_I_1 = E_num - (V[u + 1] - V[u] - 1) - (V[v + 1] - V[v] - 1) - 1;
}

int main(int argc, char *argv[])
{
	if (argc != 3) {
		std::cout << "usage: " << argv[0] << " <input_file> <block_size>" << std::endl;
		return EXIT_FAILURE;
	}

	// Number of threads in each block
	// Should be a multiple of GPU warp size (32 in recent architectures)
	int blocksize = atoi(argv[2]);

	// Nested vector used to store read file into adjacency list
	std::vector< std::vector<int> > adj_list;

	// Input network file provided by user
	// File format assume to be edge list
	std::ifstream infile(argv[1]);

	std::string su, sv;
	int u, v, edge_count = 0, max = -1;
	while (getline(infile, su, '\t') && getline(infile, sv)) {
		// Node ids assumed to be 1-indexed and decremented to be 0-indexed
		u = std::atoi(su.c_str()) - 1;
		v = std::atoi(sv.c_str()) - 1;

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

	int V_num = adj_list.size();
	int E_num = edge_count;

	std::cout << "Graph found with " << V_num << " nodes and " << E_num << " edges" << std::endl;

	// Value of V[i] is index of E_u and E_v where vertex i's edges begin
	// Value of V[i+1] is index of E_u and E_v where vertex i's edges end
	std::vector<int> V;

	// Value of E[i] is the directed edge index in E_u and E_v associated with
	// the undirected edge at index i in E
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
	for (int i = 0; i < (int) adj_list.size(); i++) {
		V.push_back(edge_index);

		for (int j = 0; j < (int) adj_list[i].size(); j++) {
			E_u.push_back(i);
			E_v.push_back(adj_list[i][j]);

			if (i < adj_list[i][j]) {
				E.push_back(E_u.size() - 1);
			}
		}

		edge_index += adj_list[i].size();
	}
	V.push_back(edge_index);

	int max_degree = 0;
	for (int i = 1; i < (int) E.size(); i++) {
		int degree = E[i] - E[i - 1];
		if (degree > max_degree) {
			max_degree = degree;
		}
	}

	// Create and initialize CUDA thread output structs
	std::vector<EDGE_OUTPUT> outputs(E_num);
	outputs.resize(E_num);

	// Pointers of V, E_u, E_v, and edge outputs in GPU memory
	int* V_ptr;
	int* E_ptr;
	int* E_u_ptr;
	int* E_v_ptr;
	EDGE_OUTPUT* outputs_ptr;

	int V_size = (V_num + 1) * sizeof(int);
	int E_size = E_num * sizeof(int);
	int outputs_size = E_num * sizeof(EDGE_OUTPUT);

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	// Malloc GPU memory and store location in pointers
	gpuErrchk(cudaMalloc((void**)&V_ptr, V_size));
	gpuErrchk(cudaMalloc((void**)&E_ptr, E_size));
	gpuErrchk(cudaMalloc((void**)&E_u_ptr, E_size * 2));
	gpuErrchk(cudaMalloc((void**)&E_v_ptr, E_size * 2));
	gpuErrchk(cudaMalloc((void**)&outputs_ptr, outputs_size));

	std::chrono::steady_clock::time_point malloc_end = std::chrono::steady_clock::now();

	// Copy data structures from main memory to allocated GPU memory
	gpuErrchk(cudaMemcpy(V_ptr, V.data(), V_size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(E_ptr, E.data(), E_size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(E_u_ptr, E_u.data(), E_size * 2, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(E_v_ptr, E_v.data(), E_size * 2, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(outputs_ptr, outputs.data(), outputs_size, cudaMemcpyHostToDevice));

	std::chrono::steady_clock::time_point memcpy_input_end = std::chrono::steady_clock::now();

	// Calculate number of blocks in CUDA grid based upon number of edges in graph
	int gridsize = E_num / blocksize;
	if (E_num % blocksize > 0) {
		gridsize++;
	}

	// Create one-dimensional blocks and grids based upon blocksize and gridsize
	// TODO: Increase dimensionality in order to support larger networks
	dim3 dimBlock(blocksize, 1);
	dim3 dimGrid(gridsize, 1);

	int heap_size = (2 * sizeof(int) + sizeof(bool)) * max_degree * E_num;
	gpuErrchk(cudaDeviceSetLimit(cudaLimitMallocHeapSize, heap_size));

	std::chrono::steady_clock::time_point kernel_begin = std::chrono::steady_clock::now();

	// Execute CUDA kernel
	// TODO: Add timing to kernel execution and count aggregation below
	graphlets<<<dimGrid, dimBlock>>>(V_ptr, V_num, E_ptr, E_num, E_u_ptr, E_v_ptr, outputs_ptr);
	gpuErrchk(cudaGetLastError());

	std::chrono::steady_clock::time_point kernel_end = std::chrono::steady_clock::now();

	// Copy output data from GPU memory back into main memory
	gpuErrchk(cudaMemcpy(outputs.data(), outputs_ptr, outputs_size, cudaMemcpyDeviceToHost));

	std::chrono::steady_clock::time_point memcpy_output_end = std::chrono::steady_clock::now();

	// Free memory in GPU
	gpuErrchk(cudaFree(V_ptr));
	gpuErrchk(cudaFree(E_ptr));
	gpuErrchk(cudaFree(E_u_ptr));
	gpuErrchk(cudaFree(E_v_ptr));
	gpuErrchk(cudaFree(outputs_ptr));

	std::chrono::steady_clock::time_point free_end = std::chrono::steady_clock::now();

	// Compute aggregate outputs based upon individual edge outputs
	EDGE_OUTPUT aggregates = { 0 };

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

	// 3-nodeand 4-node graphlet counts calculated as described
	// in http://nesreenahmed.com/publications/ahmed-et-al-icdm2015.pdf
	GRAPHLET_COUNTS counts = { 0 };

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

	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

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

	std::cout << std::endl;
	std::cout << "    Timing (us)" << std::endl;
	std::cout << "====================" << std::endl;
	std::cout << "total elapsed      : " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << std::endl;
	std::cout << "cudaMalloc         : " << std::chrono::duration_cast<std::chrono::microseconds>(malloc_end - begin).count() << std::endl;
	std::cout << "cudaMemcpy (input) : " << std::chrono::duration_cast<std::chrono::microseconds>(memcpy_input_end - malloc_end).count() << std::endl;
	std::cout << "kernel (graphlets) : " << std::chrono::duration_cast<std::chrono::microseconds>(kernel_end - kernel_begin).count() << std::endl;
	std::cout << "cudaMemcpy (output): " << std::chrono::duration_cast<std::chrono::microseconds>(memcpy_output_end - kernel_end).count() << std::endl;
	std::cout << "cudaFree           : " << std::chrono::duration_cast<std::chrono::microseconds>(free_end - memcpy_output_end).count() << std::endl;
	std::cout << "aggregate          : " << std::chrono::duration_cast<std::chrono::microseconds>(end - memcpy_output_end).count() << std::endl;

	return EXIT_SUCCESS;
}
