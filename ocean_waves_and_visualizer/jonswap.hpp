#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

static constexpr double G = 9.81;
static constexpr double TWO_PI = 2.0 * static_cast<double>(M_PI);
static constexpr double LOWEST_WAVE_FREQ_HZ = 0.05;
static constexpr double HIGHEST_WAVE_FREQ_HZ = 2.0;
static constexpr int NUM_WAVES = 100;

namespace jonswap_detail
{
  inline std::vector<double> logspace(double lo, double hi, int n)
  {
    std::vector<double> v(n);
    double log_lo = std::log10(lo), log_hi = std::log10(hi);
    for (int i = 0; i < n; ++i)
      v[i] = std::pow(10.0, log_lo + (log_hi - log_lo) * i / (n - 1));
    return v;
  }

  inline std::vector<double> jonswap_spectrum(
      const std::vector<double> &freqs,
      double fp, double alpha, double gamma_val)
  {
    std::vector<double> S(freqs.size());
    for (size_t i = 0; i < freqs.size(); ++i)
    {
      double f = freqs[i];
      double sig = (f <= fp) ? 0.07 : 0.09;
      double r = std::exp(-((f - fp) * (f - fp)) / (2.0 * sig * sig * fp * fp));
      S[i] = static_cast<double>(
          alpha * G * G * std::pow(f, -5.0) * std::pow(TWO_PI, -4.0) * std::exp(-1.25 * std::pow(fp / f, 4.0)) * std::pow(gamma_val, r));
      if (i == 0)
      {
        printf("f=%e\n", f);
        printf("alpha*G*G=%e\n", alpha * G * G);
        printf("pow(f,-5)=%e\n", std::pow(f, -5.0));
        printf("pow(TWO_PI,-4)=%e\n", std::pow(TWO_PI, -4.0));
        printf("exp term=%e\n", std::exp(-1.25 * std::pow(fp / f, 4.0)));
        printf("gamma^r=%e\n", std::pow(gamma_val, r));
        printf("r=%e\n", r);
      }
    }
    return S;
  }

  inline std::vector<double> direction_spreads(
      const std::vector<double> &freqs,
      double conditions_factor)
  {
    double spread_min = 40.0 * conditions_factor;
    double spread_max = (140.0 / 1.5) * conditions_factor;
    double sig_min = std::sqrt(LOWEST_WAVE_FREQ_HZ);
    double sig_max = std::sqrt(HIGHEST_WAVE_FREQ_HZ);

    std::vector<double> out(freqs.size());
    for (size_t i = 0; i < freqs.size(); ++i)
    {
      double t = (std::sqrt(freqs[i]) - sig_min) / (sig_max - sig_min);
      t = std::clamp(t, 0.0, 1.0);
      out[i] = spread_min + t * (spread_max - spread_min);
    }
    return out;
  }

  template <typename RNG>
  inline double cosine_deviation_deg(double spread_deg, RNG &rng)
  {
    double sigma_rad = spread_deg / 2.0 * static_cast<double>(M_PI) / 180.0;
    double s = (2.0 / sigma_rad) * (2.0 / sigma_rad);

    std::uniform_real_distribution<double> theta_dist(-static_cast<double>(M_PI) / 2.0,
                                                      static_cast<double>(M_PI) / 2.0);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    while (true)
    {
      double theta = theta_dist(rng);
      double prob = std::pow(std::cos(theta), s);
      if (unit(rng) < prob)
        return theta * 180.0 / static_cast<double>(M_PI);
    }
  }

}

enum class JonswapConditions
{
  CALM,
  MODERATE,
  STORMY
};

inline double peak_frequency(JonswapConditions c)
{
  switch (c)
  {
  case JonswapConditions::CALM:
    return 0.9;
  case JonswapConditions::MODERATE:
    return 0.5;
  case JonswapConditions::STORMY:
    return 0.1;
  }
  return 0.5;
}

inline double alpha(JonswapConditions c)
{
  switch (c)
  {
  case JonswapConditions::CALM:
    return 0.6;
  case JonswapConditions::MODERATE:
    return 1.1;
  case JonswapConditions::STORMY:
    return 0.1;
  }
  return 1.0;
}

inline double gamma_factor(JonswapConditions c)
{
  switch (c)
  {
  case JonswapConditions::CALM:
    return 1.7;
  case JonswapConditions::MODERATE:
    return 2.5;
  case JonswapConditions::STORMY:
    return 3.3;
  }
  return 2.0;
}

inline double direction_spread_factor(JonswapConditions c)
{
  switch (c)
  {
  case JonswapConditions::CALM:
    return 0.5;
  case JonswapConditions::MODERATE:
    return 1.0;
  case JonswapConditions::STORMY:
    return 1.5;
  }
  return 1.0;
}

struct Wave
{
  double amplitude;
  double wavelength;
  double angular_freq;
  double phase;
  double dir_x;
  double dir_y;
};

inline std::vector<Wave> generate_waves(
    JonswapConditions conditions,
    double dominant_dir_x = -1.0,
    double dominant_dir_y = -1.0,
    int n_waves = NUM_WAVES,
    uint32_t seed = std::random_device{}())
{
  using namespace jonswap_detail;

  double len = std::sqrt(dominant_dir_x * dominant_dir_x + dominant_dir_y * dominant_dir_y);
  if (len < 1e-6)
    throw std::invalid_argument("dominant direction vector cannot be zero");
  dominant_dir_x /= len;
  dominant_dir_y /= len;
  double dominant_angle_deg = std::atan2(dominant_dir_y, dominant_dir_x) * 180.0 / static_cast<double>(M_PI);

  auto freqs = logspace(LOWEST_WAVE_FREQ_HZ, HIGHEST_WAVE_FREQ_HZ, n_waves);

  auto S = jonswap_spectrum(freqs, (double)peak_frequency(conditions),
                            (double)alpha(conditions), (double)gamma_factor(conditions));

  std::vector<double> df(n_waves);
  for (int i = 0; i < n_waves - 1; ++i)
    df[i] = freqs[i + 1] - freqs[i];
  df[n_waves - 1] = df[n_waves - 2];

  auto spreads = direction_spreads(freqs, direction_spread_factor(conditions));

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> phase_dist(0.0, TWO_PI);

  std::vector<Wave> waves(n_waves);
  for (int i = 0; i < n_waves; ++i)
  {
    double f = freqs[i];
    double amplitude = std::sqrt(2.0 * S[i] * df[i]);
    double wavelength = G / (TWO_PI * f * f);
    double k = TWO_PI / wavelength;
    double ang_freq = std::sqrt(G * k);
    double phase = phase_dist(rng);
    double dev_deg = cosine_deviation_deg(spreads[i], rng);
    double dir_deg = dev_deg + dominant_angle_deg;
    double dir_rad = dir_deg * static_cast<double>(M_PI) / 180.0;

    waves[i] = Wave{
        amplitude,
        wavelength,
        ang_freq,
        phase,
        std::cos(dir_rad),
        std::sin(dir_rad),
    };
    std::cout << "MAGNITUDE" << "\n";
    std::cout << amplitude << "\n";
    std::cout << "WAVELENGTH" << "\n";
    std::cout << wavelength << "\n";
    std::cout << "DIRECTION" << "\n";
    std::cout << "[" << std::cos(dir_rad) << ", " << std::sin(dir_rad) << "]" << "\n";
    std::cout << "ANGULAR FREQUENCY" << "\n";
    std::cout << ang_freq << "\n";
    std::cout << "PHASE SHIFT" << "\n";
    std::cout << phase << "\n";
  }
  return waves;
}

inline double height_at(const std::vector<Wave> &waves, double x, double y, double t)
{
  double total = 0.0;
  for (const auto &w : waves)
  {
    double dot = (w.dir_x * x + w.dir_y * y) / w.wavelength;
    double angle = TWO_PI * dot - w.angular_freq * t + w.phase;
    total += w.amplitude * std::sin(angle);
  }
  return total;
}

inline std::vector<double> height_grid(
    const std::vector<Wave> &waves,
    const std::vector<double> &xs,
    const std::vector<double> &ys,
    double t)
{
  assert(xs.size() == ys.size());
  std::vector<double> result(xs.size(), 0.0);

  for (const auto &w : waves)
  {
    for (size_t j = 0; j < xs.size(); ++j)
    {
      double dot = (w.dir_x * xs[j] + w.dir_y * ys[j]) / w.wavelength;
      double angle = TWO_PI * dot - w.angular_freq * t + w.phase;
      result[j] += w.amplitude * std::sin(angle);
    }
  }
  return result;
}
