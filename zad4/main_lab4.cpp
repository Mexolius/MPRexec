#include <omp.h>
#include <iostream>
#include <random>
#include <string>
#include <memory>
#include <deque>
#include <numeric>
#include <vector>

const size_t SIZE = 100;
const float MAX = 200.0f;
size_t THREAD_COUNT = 4;
size_t BUCKETS_PER_THREAD = 4;

void print_arr(float *arr, const int size) {
	std::cout << "[";
	for (int64_t i = 0; i < size; i++) {
		std::cout << arr[i] << ", ";
	}
	std::cout << "]" << std::endl;
}

void randomize_static(float *values, size_t size, int thread_num, int chunk_size) {
	std::random_device rd;
	#pragma omp parallel num_threads(thread_num) shared(values)
	{
		std::mt19937 generator(rd());
		std::uniform_real_distribution<float> distribution(0, MAX);

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
		std::uniform_real_distribution<float> distribution(0, MAX);
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
		std::uniform_real_distribution<float> distribution(0, MAX);
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
		std::uniform_real_distribution<float> distribution(0, MAX);
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
	std::string name;
	cheat(fun_t f, std::string name, bool use_chunks = true) : use_chunks(use_chunks), name(name) { fun = f; }
	void operator ()(float* v0, size_t v1, int v2, int v3)
	{
		fun(v0, v1, v2, v3);
	}
};

void test_generate() {
	double start, end;

	const int size_exponents[] = { 16,17,18,19,20,21,22,23 };
	const int chunk_size_exponents[] = { 2,3,4,5,6 };
	std::cout << "Schedule" << ',' << "Threads" << ',' << "Size" << ',' << "Chunk size" << ',' << "Time" << std::endl;

	const int repeats = 25;

	cheat schedules[] = {
		{randomize_static,"static"},
		{randomize_dynamic,"dynamic"},
		{randomize_guided,"guided"},
		{randomize_runtime, "runtime", false}
	};
	for (const int &size_exp : size_exponents) {
		size_t size = pow(2, size_exp);
		std::unique_ptr<float[]> values = std::make_unique<float[]>(size);

		for (auto &schedule : schedules) {
			for (int threads = 1; threads < 5; threads++) {
				if (schedule.use_chunks) {
					for (const auto &chunk_exp : chunk_size_exponents) {
						int chunk_size = pow(2, chunk_exp);

						start = omp_get_wtime();
						for (int i = 0; i < repeats; i++)
							schedule(values.get(), size, threads, chunk_size);
						end = omp_get_wtime();
						std::cout << schedule.name << ',' << threads << ',' << size << ',' << chunk_size << ',' << (end - start) / repeats << std::endl;
					}
				}
				else {
					start = omp_get_wtime();
					for (int i = 0; i < repeats; i++)
						schedule(values.get(), size, threads, 0);
					end = omp_get_wtime();
					std::cout << schedule.name << ',' << threads << ',' << size << ',' << 0 << ',' << (end - start) / repeats << std::endl;
				}
			}

		}
	}
}

class bucket_t
{
private:
	std::unique_ptr<float[]> values = nullptr;
	size_t _capacity = 0;
	size_t next = 0;
public:
	bucket_t() = default;
	bucket_t(size_t capacity)
		: values(std::make_unique<float[]>(capacity))
		, _capacity(capacity)
	{ }
	bucket_t(const bucket_t& other)
		: values(other._capacity != 0 ? std::make_unique<float[]>(other._capacity) : nullptr)
		, _capacity(other._capacity)
		, next(other.next)
	{
		if (other._capacity != 0)
			std::copy_n(other.values.get(), other._capacity, values.get());
	}
	bucket_t(bucket_t&& other) noexcept
		: values(std::move(other.values))
		, _capacity(std::move(other._capacity))
		, next(std::move(other.next))
	{
		other._capacity = 0;
		other.next = 0;
	}
public:
	bucket_t& operator=(const bucket_t& other)
	{
		values = other._capacity != 0 ? std::make_unique<float[]>(other._capacity) : nullptr;
		_capacity = other._capacity;
		next = other.next;

		if (other._capacity != 0)
			std::copy_n(other.values.get(), other._capacity, values.get());

		return *this;
	}
	bucket_t& operator=(bucket_t&& other)
	{
		values = std::move(other.values);
		_capacity = std::move(other._capacity);
		next = std::move(other.next);

		other._capacity = 0;
		other.next = 0;

		return *this;
	}
public:
	size_t capacity() { return _capacity; }
	size_t size() { return next; }

	const float* begin() const { return values.get(); }
	const float* end() const { return values.get() + next; }
public:
	void insert(float value)
	{
		if (next == _capacity)
			throw std::runtime_error("Bucket full!");
		values[next++] = value;
	}

	void sort() {
		int j;
		float key;
		float* arr = this->values.get();
		size_t size = next;
		for (size_t i = 0; i < size; i++)
		{
			key = arr[i];
			j = i - 1;

			/* Move elements of arr[0..i-1], that are
			greater than key, to one position ahead
			of their current position */
			while (j >= 0 && arr[j] > key)
			{
				arr[j + 1] = arr[j];
				j = j - 1;
			}
			arr[j + 1] = key;
		}
	}
public:
	friend std::ostream& operator<<(std::ostream& o, const bucket_t& bucket);
};

std::ostream& operator<<(std::ostream& o, const bucket_t& bucket)
{
	o << '[';
	for (auto v : bucket) o << v << ", ";
	o << ']';
	return o;
}

class buckets_t
{
private:
	std::unique_ptr<bucket_t[]> buckets;
	size_t bucket_count;
	float bucket_size;
	size_t total_size = 0;
public:
	buckets_t(size_t bucket_count, size_t bucket_capacity, float bucket_size)
		: buckets(std::make_unique<bucket_t[]>(bucket_count))
		, bucket_count(bucket_count)
		, bucket_size(bucket_size)
	{
		std::generate_n(buckets.get(), bucket_count, [bucket_capacity]() { return bucket_t(bucket_capacity); });
	}
public:
	template<typename BucketSelector>
	void insert(float value, BucketSelector&& bucket_selector)
	{
		size_t bucket_index = std::forward<BucketSelector>(bucket_selector)(value, bucket_size);
		if (bucket_index >= bucket_count)
			throw std::runtime_error("Bucket index out of range!");
		buckets[bucket_index].insert(value);
		total_size++;
	}

	void sort() {
		for (size_t b = 0; b < size(); b++) {
			buckets[b].sort();
		}
	}

	size_t get_total_size() const {
		return total_size;
	}

	std::unique_ptr<float[]> merge() {
		auto ret = std::make_unique<float[]>(total_size);
		auto ptr = ret.get();
		size_t i = 0;
		for (size_t b = 0; b < size(); b++) {
			for (auto &v : buckets[b]) {
				ptr[i] = v;
				i++;
			}
		}

		return ret;
	}

public:
	size_t size() { return bucket_count; }

	const bucket_t* begin() const { return buckets.get(); }
	const bucket_t* end() const { return buckets.get() + bucket_count; }
public:
	friend std::ostream& operator<<(std::ostream& o, const buckets_t& buckets);
};

std::ostream& operator<<(std::ostream& o, const buckets_t& buckets)
{
	size_t i = 0;
	for (auto b : buckets)
	{
		o << '{' << i << " [" << buckets.get_total_size() << "] " << '}' << " -> " << b << '\n';
		i++;
	}
	return o;
}

double calc_time_whole(size_t thread_count, size_t size, size_t buckets_per_thread) {
	std::unique_ptr<float[]> values = std::make_unique<float[]>(size);
	float bucket_size = static_cast<float>(200.0f / thread_count);

	std::unique_ptr<size_t[]> sizes = std::make_unique<size_t[]>(thread_count);
	std::unique_ptr<size_t[]> start_points = std::make_unique<size_t[]>(thread_count);

	double start = omp_get_wtime();

	randomize_guided(values.get(), size, thread_count, 64);

#pragma omp parallel num_threads(thread_count)
	{
		int thread_num = omp_get_thread_num();
		float bucket_start = static_cast<float>(bucket_size * thread_num);
		float bucket_end = static_cast<float>(bucket_size * (1 + thread_num));
		float thread_bucket_size = static_cast<float>(bucket_size / buckets_per_thread);
		size_t start_point = static_cast<size_t>(size * thread_num / thread_count);

		auto bucket_selector = [&bucket_start](float v, float bucket_size)
		{
			return static_cast<size_t>((v - bucket_start) / bucket_size);
		};

		size_t buck_size = 2*size / buckets_per_thread;

		buckets_t buckets{ buckets_per_thread, buck_size, thread_bucket_size };

		for (int j = 0; j < size; j++) {
			size_t pos = (j + start_point) % size;
			if (bucket_start < values[pos] && values[pos] < bucket_end) {
				buckets.insert(values[pos], bucket_selector);
			}
		}

		buckets.sort();

		std::unique_ptr<float[]> result = buckets.merge();

		sizes[thread_num] = buckets.get_total_size();

#pragma omp barrier
#pragma omp single
		{
			start_points[0] = 0;
			for (int i = 1; i < thread_count; i++) {
				start_points[i] = start_points[i - 1] + sizes[i - 1];
			}
		}

		size_t from = start_points[thread_num];

		for (size_t i = 0; i < buckets.get_total_size(); i++) {
			values[i + from] = result[i];
		}
	}
	double end = omp_get_wtime();

	//print_arr(values.get(), size);
	return end - start;
}

void calc_time_partitioned(size_t thread_count, size_t size, size_t buckets_per_thread, double* result_times) {
	std::unique_ptr<float[]> values = std::make_unique<float[]>(size);
	float bucket_size = static_cast<float>(200.0f / thread_count);

	std::unique_ptr<float[]> result_arr_uni = std::make_unique<float[]>(size);
	std::unique_ptr<size_t[]> sizes = std::make_unique<size_t[]>(thread_count);
	std::unique_ptr<size_t[]> start_points = std::make_unique<size_t[]>(thread_count);

	double start = omp_get_wtime();

	randomize_guided(values.get(), size, thread_count, 64);

	double randomized = omp_get_wtime();

	double assorted, sorted, in_result, merged;

#pragma omp parallel num_threads(thread_count)
	{
		int thread_num = omp_get_thread_num();
		float bucket_start = static_cast<float>(bucket_size * thread_num);
		float bucket_end = static_cast<float>(bucket_size * (1 + thread_num));
		float thread_bucket_size = static_cast<float>(bucket_size / buckets_per_thread);
		size_t start_point = static_cast<size_t>(size * thread_num / thread_count);

		auto bucket_selector = [&bucket_start](float v, float bucket_size)
		{
			return static_cast<size_t>((v - bucket_start) / bucket_size);
		};

		size_t buck_size = 2*size/ buckets_per_thread;


		buckets_t buckets{ buckets_per_thread, buck_size, thread_bucket_size };

		for (int j = 0; j < size; j++) {
			size_t pos = (j + start_point) % size;
			if (bucket_start < values[pos] && values[pos] < bucket_end) {
				buckets.insert(values[pos], bucket_selector);
			}
		}
#pragma omp barrier
#pragma omp single
		{
			assorted = omp_get_wtime();
		}

		buckets.sort();
#pragma omp barrier
#pragma omp single
		{
			sorted = omp_get_wtime();
		}

		std::unique_ptr<float[]> result = buckets.merge();

		sizes[thread_num] = buckets.get_total_size();

#pragma omp barrier
#pragma omp single
		{
			merged = omp_get_wtime();
			start_points[0] = 0;
			for (int i = 1; i < thread_count; i++) {
				start_points[i] = start_points[i - 1] + sizes[i - 1];
			}
		}

		size_t from = start_points[thread_num];

		for (size_t i = 0; i < buckets.get_total_size(); i++) {
			result_arr_uni[i + from] = result[i];
		}
	}

	//print_arr(values.get(), size);
	double end = omp_get_wtime();

	result_times[0] = randomized - start;
	result_times[1] = assorted - randomized;
	result_times[2] = sorted - assorted;
	result_times[3] = merged - sorted;
	result_times[4] = end - merged;
}

void test_whole_buckets(int threads,int size, int start, int step, int max, int reps = 10) {
	// SINGLE Thread TESTS
	for (int i = start; i < max; i*= step) {
		
		std::vector<double> times(reps);
		for (int r = 0; r < reps; r++) {
			times[r] = calc_time_whole(threads, size, i);
		}
		double sum = std::accumulate(times.begin(), times.end(), 0.0);
		double mean = sum / reps;

		double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
		double stdev = std::sqrt(sq_sum / times.size() - mean * mean);

		std::cout << threads << "," << size << "," << i*threads << "," << mean << "," << stdev << std::endl;
	}
}

void test_whole_size(int threads, int buckets, int start, int step, int max, int reps = 10) {
	// SINGLE Thread TESTS
	for (int i = start; i < max; i *= step) {

		std::vector<double> times(reps);
		for (int r = 0; r < reps; r++) {
			times[r] = calc_time_whole(threads, i, buckets/threads);
		}
		double sum = std::accumulate(times.begin(), times.end(), 0.0);
		double mean = sum / reps;

		double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
		double stdev = std::sqrt(sq_sum / times.size() - mean * mean);

		std::cout << threads << "," << i << "," << buckets << "," << mean << "," << stdev << std::endl;
	}
}

void test_partial(int threads, int buckets, int start, int step, int max, int reps = 10) {
	// SINGLE Thread TESTS
	for (int i = start; i < max; i *= step) {

		std::vector<std::unique_ptr<double[]>> times(reps);
		for (int r = 0; r < reps; r++) {
			times[r] = std::make_unique<double[]>(5);
			calc_time_partitioned(threads, i, buckets/threads, times[r].get());
		}

		double means[5] = { 0.0 };
		for (auto &v : times) {
			for (int i = 0; i < 5; i++) {
				means[i] += v[i];
			}
		}

		for (int i = 0; i < 5; i++) {
			means[i] /= reps;
		}

		std::cout << threads << "," << i << "," << buckets << ",";
		for (int i = 0; i < 5; i++) {
			std::cout << means[i] << ",";
		}
		std::cout << std::endl;
	}
}

int main() {
	//test_whole_buckets(1, 100000, 10, 2, 5000);
	//test_whole_buckets(1, 1000000, 100, 2, 20000);
	//test_whole_buckets(1, 6000000, 1000, 2, 40000);

	int thread_counts[] = { 1,2,3,4 };

	for (auto &tc : thread_counts) {
		//test_whole_size(tc, 1920, 50000, 2, 6400001);
	}

	for (auto &tc : thread_counts) {
		test_partial(tc, 1920, 50000, 2, 6400001);
	}

	return 0;
}

