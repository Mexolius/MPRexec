#include <omp.h>
#include <iostream>
#include <random>

const int64_t SIZE = 100000000;

using namespace std;

void print_arr(int *arr, const int size) {
	for (int64_t i = 0; i < size; i++) {
		cout << arr[i] << " ";
	}
}

int main() {
	int* values;
	values = new int[SIZE]();
	/*# pragma omp parallel 
	{
		printf("Thread rank: %d\n", omp_get_thread_num());
	}*/

	double start, end;

	
	for (int threads = 1; threads < 5; threads++) {
		start = omp_get_wtime();
		#pragma omp parallel num_threads(threads) shared(values)
		{
			std::random_device rd;
			std::uniform_int_distribution<int> distribution(0, 200);
			std::mt19937 generator(rd());
		
			#pragma omp for schedule(runtime)
			for (int64_t i = 0; i < SIZE; i++) {
				values[i] = distribution(generator);
			}
		}
		end = omp_get_wtime();
		cout <<"Threads: "<< threads << " - " << end - start << "s" << endl;
	}

	//print_arr(values, SIZE);
}

