#include <omp.h>
#include <iostream>
#include <random>
#include <string>
#include <memory>
#include <deque>

const size_t SIZE = 8;
const float MAX = 200.0f;
size_t THREAD_COUNT = 2;
size_t BUCKETS_PER_THREAD = 2;

void print_arr(float *arr, const int size) {
	for (int64_t i = 0; i < size; i++) {
		std::cout << arr[i] << " ";
	}
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
	}

	void sort() {
		for (size_t b = 0; b < size(); b++) {
			buckets[b].sort();
		}
	}

	std::unique_ptr<float[]> merge() {
		auto ret = std::make_unique<float[]>(SIZE);
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
		//if(b.size()>1)
			o << '{' << i << '}' << " -> " << b << '\n';
		i++;
	}
	return o;
}

int main() {
	std::unique_ptr<float[]> values = std::make_unique<float[]>(SIZE);

	randomize_guided(values.get(), SIZE, 4, 64);

	float bucket_size;
	size_t thread_count = THREAD_COUNT;

	std::unique_ptr<float[]> shared_result = std::make_unique<float[]>(SIZE);
	float* values_ptr = values.get();;
#pragma omp parallel num_threads(thread_count)
	{

#pragma omp single
		{
			bucket_size = static_cast<float>(200.0f / thread_count);
		}

		int thread_num = omp_get_thread_num();
		float bucket_start = static_cast<float>(bucket_size * thread_num);
		float bucket_end = static_cast<float>(bucket_size * (1+thread_num));
		float thread_bucket_size = static_cast<float>(bucket_size / BUCKETS_PER_THREAD);
		size_t start_point = static_cast<size_t>(SIZE * thread_num / thread_count);

		auto bucket_selector = [&bucket_start](float v, float bucket_size)
		{
			return static_cast<size_t>((v - bucket_start) / bucket_size);
		};

		buckets_t buckets{ BUCKETS_PER_THREAD, SIZE, thread_bucket_size };


		size_t i = start_point;
#pragma omp for
		for (int j = 0; j < SIZE; j++) {
			if( values_ptr[i] > bucket_start && values_ptr[i] < bucket_end)
				buckets.insert(values_ptr[i], bucket_selector);
			i = (i + 1) % SIZE;
		}

		buckets.sort();
	
		std::unique_ptr<float[]> result = buckets.merge();

		
		
#pragma omp for
		for (int i = 0; i < buckets.size(); i++) {
			shared_result[i + start_point] = result[i];
		}

		std::cout << "Thread " << thread_num << " done" << std::endl;

#pragma omp single
		{
			print_arr(result.get(), SIZE);
		}
	}

	return 0;
}

