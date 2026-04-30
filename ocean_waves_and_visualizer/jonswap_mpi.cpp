#include <mpi.h>
#include <chrono>
#include <cstdio>
#include <vector>
#include "jonswap.hpp"

static constexpr int GRID = 1000;
static constexpr double EXTENT = 50.0;
static constexpr double T = 0.0;

static void build_coords(
    int row_start, int row_end,
    std::vector<double> &xs, std::vector<double> &ys)
{
  int n_rows = row_end - row_start;
  xs.resize(n_rows * GRID);
  ys.resize(n_rows * GRID);

  for (int r = 0; r < n_rows; ++r)
  {
    int global_row = row_start + r;
    double y = -EXTENT + (2.0 * EXTENT * global_row) / (GRID - 1);
    for (int c = 0; c < GRID; ++c)
    {
      double x = -EXTENT + (2.0 * EXTENT * c) / (GRID - 1);
      xs[r * GRID + c] = x;
      ys[r * GRID + c] = y;
    }
  }
}

static constexpr int WAVE_DOUBLES = 6;

static std::vector<double> pack_waves(const std::vector<Wave> &waves)
{
  std::vector<double> buf(waves.size() * WAVE_DOUBLES);
  for (size_t i = 0; i < waves.size(); ++i)
  {
    buf[i * WAVE_DOUBLES + 0] = waves[i].amplitude;
    buf[i * WAVE_DOUBLES + 1] = waves[i].wavelength;
    buf[i * WAVE_DOUBLES + 2] = waves[i].angular_freq;
    buf[i * WAVE_DOUBLES + 3] = waves[i].phase;
    buf[i * WAVE_DOUBLES + 4] = waves[i].dir_x;
    buf[i * WAVE_DOUBLES + 5] = waves[i].dir_y;
  }
  return buf;
}

static std::vector<Wave> unpack_waves(const std::vector<double> &buf)
{
  size_t n = buf.size() / WAVE_DOUBLES;
  std::vector<Wave> waves(n);
  for (size_t i = 0; i < n; ++i)
  {
    waves[i].amplitude = buf[i * WAVE_DOUBLES + 0];
    waves[i].wavelength = buf[i * WAVE_DOUBLES + 1];
    waves[i].angular_freq = buf[i * WAVE_DOUBLES + 2];
    waves[i].phase = buf[i * WAVE_DOUBLES + 3];
    waves[i].dir_x = buf[i * WAVE_DOUBLES + 4];
    waves[i].dir_y = buf[i * WAVE_DOUBLES + 5];
  }
  return waves;
}

static void run_serial(const std::vector<Wave> &waves)
{
  std::vector<double> xs, ys;
  build_coords(0, GRID, xs, ys);

  auto t0 = std::chrono::high_resolution_clock::now();
  std::vector<double> Z = height_grid(waves, xs, ys, T);
  auto t1 = std::chrono::high_resolution_clock::now();

  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  printf("[SERIAL]  %dx%d grid, %zu waves → %.2f ms\n",
         GRID, GRID, waves.size(), ms);
}

static void run_mpi(int rank, int nprocs, const std::vector<Wave> &waves)
{
  int n_waves = (int)waves.size();
  MPI_Bcast(&n_waves, 1, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<double> wave_buf;
  if (rank == 0)
    wave_buf = pack_waves(waves);
  else
    wave_buf.resize(n_waves * WAVE_DOUBLES);

  MPI_Bcast(wave_buf.data(), (int)wave_buf.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  std::vector<Wave> local_waves = unpack_waves(wave_buf);

  int rows_per_rank = GRID / nprocs;
  int remainder = GRID % nprocs;

  int my_row_start = rank * rows_per_rank;
  int my_row_end = my_row_start + rows_per_rank + (rank == nprocs - 1 ? remainder : 0);
  int my_rows = my_row_end - my_row_start;

  std::vector<double> xs, ys;
  build_coords(my_row_start, my_row_end, xs, ys);

  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();

  std::vector<double> local_Z = height_grid(local_waves, xs, ys, (double)T);

  MPI_Barrier(MPI_COMM_WORLD);
  double t1 = MPI_Wtime();

  std::vector<int> recv_counts(nprocs), displs(nprocs);
  for (int r = 0; r < nprocs; ++r)
  {
    int r_start = r * rows_per_rank;
    int r_rows = rows_per_rank + (r == nprocs - 1 ? remainder : 0);
    recv_counts[r] = r_rows * GRID;
    displs[r] = r_start * GRID;
  }

  std::vector<double> Z;
  if (rank == 0)
    Z.resize(GRID * GRID);

  MPI_Gatherv(local_Z.data(), (int)local_Z.size(), MPI_DOUBLE,
              Z.data(), recv_counts.data(), displs.data(), MPI_DOUBLE,
              0, MPI_COMM_WORLD);

  if (rank == 0)
  {
    double ms = (t1 - t0) * 1000.0;
    printf("[MPI x%d] %dx%d grid, %zu waves → %.2f ms\n",
           nprocs, GRID, GRID, local_waves.size(), ms);
  }
}

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  std::vector<Wave> waves;
  if (rank == 0)
  {
    waves = generate_waves(JonswapConditions::STORMY, -1.0, -1.0, NUM_WAVES, 420u);
    printf("Generated %zu waves\n", waves.size());

    run_serial(waves);
  }

  run_mpi(rank, nprocs, waves);

  MPI_Finalize();
  return 0;
}
