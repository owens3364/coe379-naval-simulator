#include <mpi.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "jonswap.hpp"

static constexpr int GRID = 500;
static constexpr float FPS = 20.0f;
static constexpr float T_END = 60.0f;
static constexpr int N_FRAMES = (int)(T_END * FPS);
static constexpr float DT = 1.0f / FPS;
static constexpr double EXTENT = 50.0;
static constexpr double T = 0.0;

struct FileHeader
{
  uint32_t magic;
  uint32_t version;
  uint32_t rows;
  uint32_t cols;
  uint32_t n_frames;
  float fps;
  float extent;
  uint8_t padding[36];
};
static constexpr MPI_Offset HEADER_SIZE = 64;
static constexpr MPI_Offset FRAME_SIZE = sizeof(float) * GRID * GRID + sizeof(float);

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

  MPI_File fh;
  MPI_File_open(MPI_COMM_WORLD, "ocean.bin",
                MPI_MODE_CREATE | MPI_MODE_WRONLY,
                MPI_INFO_NULL, &fh);

  if (rank == 0)
  {
    FileHeader hdr = {};
    hdr.magic = 0x4E45434F;
    hdr.version = 1;
    hdr.rows = GRID;
    hdr.cols = GRID;
    hdr.n_frames = N_FRAMES;
    hdr.fps = FPS;
    hdr.extent = (float)EXTENT;
    MPI_File_write_at(fh, 0, &hdr, sizeof(FileHeader), MPI_BYTE, MPI_STATUS_IGNORE);
    printf("[MPI x%d] Writing %d frames at %.0ffps, %dx%d grid → ocean.bin\n",
           nprocs, N_FRAMES, FPS, GRID, GRID);
  }

  MPI_Offset my_heights_offset_in_frame = sizeof(float) + my_row_start * GRID * sizeof(float);

  std::vector<float> local_Z(my_rows * GRID);
  std::vector<double> xs, ys;
  build_coords(my_row_start, my_row_end, xs, ys);

  MPI_Barrier(MPI_COMM_WORLD);
  double t0 = MPI_Wtime();

  for (int frame = 0; frame < N_FRAMES; ++frame)
  {
    double t = frame * DT;

    std::vector<double> local_Zd = height_grid(local_waves, xs, ys, t);
    for (int i = 0; i < (int)local_Zd.size(); ++i)
      local_Z[i] = (float)local_Zd[i];

    MPI_Offset frame_start = HEADER_SIZE + frame * FRAME_SIZE;

    if (rank == 0)
    {
      float ts = (float)t;
      MPI_File_write_at(fh, frame_start, &ts, 1, MPI_FLOAT, MPI_STATUS_IGNORE);
    }

    MPI_Offset offset = frame_start + my_heights_offset_in_frame;
    MPI_File_write_at(fh, offset,
                      local_Z.data(), (int)local_Z.size(),
                      MPI_FLOAT, MPI_STATUS_IGNORE);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  double t1 = MPI_Wtime();

  MPI_File_close(&fh);

  if (rank == 0)
    printf("Done in %.2f ms\n", (t1 - t0) * 1000.0);
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
    if (argc < 2)
    {
      fprintf(stderr, "Usage: %s <waves.txt>\n", argv[0]);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    FILE *f = fopen(argv[1], "r");
    if (!f)
    {
      fprintf(stderr, "Failed to open %s\n", argv[1]);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int n;
    fscanf(f, "%d", &n);
    waves.resize(n);
    for (auto &w : waves)
      fscanf(f, "%lf %lf %lf %lf %lf %lf",
             &w.amplitude, &w.wavelength, &w.angular_freq,
             &w.phase, &w.dir_x, &w.dir_y);
    fclose(f);

    printf("Loaded %zu waves from %s\n", waves.size(), argv[1]);
    run_serial(waves);
  }

  run_mpi(rank, nprocs, waves);

  MPI_Finalize();
  return 0;
}
