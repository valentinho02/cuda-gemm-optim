template<typename Kernel>
void run_benchmark(Kernel kernel, dim3 grid, dim3 block, const float* A, const float* B, float* C, int N, const std::string& name);