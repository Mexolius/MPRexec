#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define root 0

struct point {
	float x;
	float y;
} point;

struct point random_point() {
	struct point p = {
		(float)(rand() / (RAND_MAX + 1.0)),
		(float)(rand() / (RAND_MAX + 1.0))
	};
	return p;
}

int is_in_circle(const struct point *p) {
	return (p->x*p->x + p->y*p->y) < 1;
}

uint64_t calculate_inside_points(uint64_t num_points) {
	uint64_t inside = 0;
	for (uint64_t i = 0; i < num_points; i++) {
		struct point p = random_point();
		if (is_in_circle(&p)) inside++;
	}
	return inside;
}

int main(int argc, char* argv[]) {
	if (argc <= 1)
	{
		fprintf(stderr, "Specify number of points to generate [1;1e12]");
		return -1;
	}

	MPI_Init(NULL, NULL);

	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	uint64_t num_points = strtoul(argv[1], NULL, 10);
	struct timespec tstamp = { 0,0 };
	struct timespec trecv = { 0,0 };
	timespec_get(&trecv, TIME_UTC);


	uint64_t local_points = num_points / world_size;
	MPI_Barrier(MPI_COMM_WORLD);
	srand(time(NULL)+world_rank);
	uint64_t local_inside = calculate_inside_points(local_points);
	double local_pi = 4 * ((double)local_inside / (double)local_points);

	printf("Local PI for process %ld is %lf | npoints = %ld\n", world_rank, local_pi, local_points);

	uint64_t global_inside;
	MPI_Reduce(&local_inside, &global_inside, 1, MPI_UINT64_T, MPI_SUM, 0,
		MPI_COMM_WORLD);

	if (world_rank == root) {
		timespec_get(&tstamp, TIME_UTC);
		double global_pi = 4 * ((double)global_inside / (double)num_points);
		double s = ((double)tstamp.tv_sec + 1.0e-9 * tstamp.tv_nsec) - ((double)trecv.tv_sec + 1.0e-9 * trecv.tv_nsec);
		printf("Global PI is %lf | calculated in %fs\n", global_pi, s);
	}

	MPI_Finalize();
	return 0;
}