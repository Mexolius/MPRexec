#include <omp.h>
#include <iostream>
#include <random>
#include <string>
#include <memory>

const int64_t SIZE = 100000000;

using namespace std;

void print_arr(int *arr, const int size) {
	for (int64_t i = 0; i < size; i++) {
		cout << arr[i] << " ";
	}
}

void randomize_static(float *values, size_t size, int thread_num, int chunk_size) {
	std::random_device rd;
	#pragma omp parallel num_threads(thread_num) shared(values)
	{
		std::mt19937 generator(rd());
		std::uniform_real_distribution<float> distribution(0, 200);

		#pragma omp for schedule(static, chunk_size)
		for (int64_t i = 0; i < size; i++) {
			values[i] = distribution(generator);
		}
	}
}

void randomize_dynamic(float *values, size_t size, int thread_num, int chunk_size) {
#pragma omp parallel num_threads(thread_num) shared(values)
	{
		std::random_device rd;
		std::uniform_real_distribution<float> distribution(0, 200);
		std::mt19937 generator(rd());

#pragma omp for schedule(dynamic, chunk_size)
		for (int64_t i = 0; i < size; i++) {
			values[i] = distribution(generator);
		}
	}
}

void randomize_guided(float *values, size_t size, int thread_num, int chunk_size) {
#pragma omp parallel num_threads(thread_num) shared(values)
	{
		std::random_device rd;
		std::uniform_real_distribution<float> distribution(0, 200);
		std::mt19937 generator(rd());

#pragma omp for schedule(guided, chunk_size)
		for (int64_t i = 0; i < size; i++) {
			values[i] = distribution(generator);
		}
	}
}

void randomize_runtime(float *values, size_t size, int thread_num, int _) {
#pragma omp parallel num_threads(thread_num) shared(values)
	{
		std::random_device rd;
		std::uniform_real_distribution<float> distribution(0, 200);
		std::mt19937 generator(rd());

#pragma omp for schedule(runtime)
		for (int64_t i = 0; i < size; i++) {
			values[i] = distribution(generator);
		}
	}
}

typedef void(*fun_t)(float*, size_t, int, int);

struct cheat
{
	fun_t fun;
	bool use_chunks;
	string name;
	cheat(fun_t f, string name, bool use_chunks = true) : use_chunks(use_chunks), name(name) { fun = f; }
	void operator ()(float* v0, size_t v1, int v2, int v3)
	{
		fun(v0, v1, v2, v3);
	}
};

int main() {
	double start, end;

	const int size_exponents[] = { 16,17,18,19,20,21,22,23 };
	const int chunk_size_exponents[] = { 2,3,4,5,6 };
	cout << "Schedule" << ',' << "Threads" << ',' << "Size" << ',' << "Chunk size" << ',' << "Time" << endl;

	const int repeats = 25;

	cheat schedules[] = {
		{randomize_static,"static"},
		{randomize_dynamic,"dynamic"},
		{randomize_guided,"guided"},
		{randomize_runtime, "runtime", false}
	};
	for (const int &size_exp : size_exponents) {
		size_t size = pow(2, size_exp);
		unique_ptr<float[]> values = std::make_unique<float[]>(size);

		for (auto &schedule : schedules) {
			for (int threads = 1; threads < 5; threads++) {
				if (schedule.use_chunks) {
					for (const auto &chunk_exp : chunk_size_exponents) {
						int chunk_size = pow(2, chunk_exp);

						start = omp_get_wtime();
						for(int i=0;i<repeats;i++)
							schedule(values.get(), size, threads, chunk_size);
						end = omp_get_wtime();
						cout << schedule.name << ',' << threads << ',' << size << ',' << chunk_size << ',' << (end - start)/repeats << endl;
					}
				}
				else {
					start = omp_get_wtime();
					for (int i = 0; i < repeats; i++)
						schedule(values.get(), size, threads, 0);
					end = omp_get_wtime();
					cout << schedule.name << ',' << threads << ',' << size << ',' << 0 << ',' << (end - start) / repeats << endl;
				}
			}
			
		}
	}

	//print_arr(values, SIZE);
}

