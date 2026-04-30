#include <omp.h>
#include <json.hpp>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include "jonswap.hpp"

using json = nlohmann::json;

static constexpr int SURGE = 0;
static constexpr int SWAY = 1;
static constexpr int HEAVE = 2;
static constexpr int ROLL = 3;
static constexpr int PITCH = 4;
static constexpr int YAW = 5;
static const char *DOF_NAMES[] = {"Surge", "Sway", "Heave", "Roll", "Pitch", "Yaw"};

struct RaoTable
{
  int n_waves;
  std::vector<std::vector<double>> mag;
  std::vector<std::vector<double>> phase;

  RaoTable(int n) : n_waves(n), mag(6, std::vector<double>(n, 0.0)),
                    phase(6, std::vector<double>(n, 0.0)) {}
};

static std::vector<Wave> load_waves(const std::string &path)
{
  FILE *f = fopen(path.c_str(), "r");
  if (!f)
  {
    fprintf(stderr, "Cannot open %s\n", path.c_str());
    exit(1);
  }
  int n;
  fscanf(f, "%d", &n);
  std::vector<Wave> waves(n);
  for (auto &w : waves)
    fscanf(f, "%lf %lf %lf %lf %lf %lf",
           &w.amplitude, &w.wavelength, &w.angular_freq,
           &w.phase, &w.dir_x, &w.dir_y);
  fclose(f);
  return waves;
}

static int nearest_idx(const std::vector<double> &arr, double val)
{
  int best = 0;
  double best_dist = std::abs(arr[0] - val);
  for (int i = 1; i < (int)arr.size(); ++i)
  {
    double d = std::abs(arr[i] - val);
    if (d < best_dist)
    {
      best_dist = d;
      best = i;
    }
  }
  return best;
}

static RaoTable load_rao(const std::string &path, const std::vector<Wave> &waves)
{
  std::ifstream f(path);
  if (!f)
  {
    fprintf(stderr, "Cannot open %s\n", path.c_str());
    exit(1);
  }
  json root;
  f >> root;

  auto &rao_json = root["rao"];
  auto &real_data = rao_json["data"]["real"];
  auto &imag_data = rao_json["data"]["imag"];
  auto &coords = rao_json["coords"];

  std::vector<double> omegas, dirs;
  for (auto &v : coords["omega"])
    omegas.push_back(v.get<double>());
  for (auto &v : coords["wave_direction"])
    dirs.push_back(v.get<double>());

  RaoTable table((int)waves.size());

  for (int n = 0; n < (int)waves.size(); ++n)
  {
    int oi = nearest_idx(omegas, waves[n].angular_freq);

    double wave_dir = std::atan2(waves[n].dir_y, waves[n].dir_x);
    int di = nearest_idx(dirs, wave_dir);

    for (int k = 0; k < 6; ++k)
    {
      double re = real_data[oi][di][k].get<double>();
      double im = imag_data[oi][di][k].get<double>();
      table.mag[k][n] = std::sqrt(re * re + im * im);
      table.phase[k][n] = std::atan2(im, re);
    }
  }

  return table;
}

static void compute_motion(
    const std::vector<Wave> &waves,
    const RaoTable &rao,
    double t,
    double motion[6])
{
  for (int k = 0; k < 6; ++k)
    motion[k] = 0.0;

#pragma omp parallel for reduction(+ : motion[ : 6]) schedule(static)
  for (int n = 0; n < (int)waves.size(); ++n)
  {
    double phase_total = waves[n].angular_freq * t + waves[n].phase;
    for (int k = 0; k < 6; ++k)
    {
      motion[k] += waves[n].amplitude * rao.mag[k][n] * std::cos(phase_total + rao.phase[k][n]);
    }
  }
}

int main(int argc, char **argv)
{
  if (argc < 3)
  {
    fprintf(stderr, "Usage: %s <waves.txt> <output.json>\n", argv[0]);
    return 1;
  }

  auto waves = load_waves(argv[1]);
  printf("Loaded %zu waves\n", waves.size());

  printf("Loading RAO from %s (this may take a moment for large JSON)...\n", argv[2]);
  auto rao = load_rao(argv[2], waves);
  printf("RAO table built\n");

  double dt = 0.05;
  double t_end = 60.0;
  int n_steps = (int)(t_end / dt);

  printf("\nSimulating %.1fs of ship motion (dt=%.3fs, %d steps)\n\n",
         t_end, dt, n_steps);
  printf("%-10s %-12s %-12s %-12s %-12s %-12s %-12s\n",
         "t(s)", "Surge(m)", "Sway(m)", "Heave(m)",
         "Roll(rad)", "Pitch(rad)", "Yaw(rad)");
  printf("%-10s %-12s %-12s %-12s %-12s %-12s %-12s\n",
         "----------", "------------", "------------", "------------",
         "------------", "------------", "------------");

  auto wall_start = std::chrono::high_resolution_clock::now();

  for (int step = 0; step <= n_steps; step += 10)
  {
    double t = step * dt;
    double motion[6];
    compute_motion(waves, rao, t, motion);

    printf("%-10.2f %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f %-12.4f\n",
           t,
           motion[SURGE], motion[SWAY], motion[HEAVE],
           motion[ROLL], motion[PITCH], motion[YAW]);
  }

  auto wall_end = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(wall_end - wall_start).count();
  printf("\nSimulated %d steps in %.2f ms (%.3f ms/step)\n",
         n_steps, ms, ms / n_steps);

  return 0;
}
