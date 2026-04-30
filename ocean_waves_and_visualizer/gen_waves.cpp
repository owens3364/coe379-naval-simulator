#include <cstdio>
#include "jonswap.hpp"

int main()
{
  auto waves = generate_waves(JonswapConditions::STORMY, -1.0, -1.0, NUM_WAVES, 420u);

  FILE *f = fopen("waves.txt", "w");
  if (!f)
  {
    fprintf(stderr, "Failed to open waves.txt\n");
    return 1;
  }

  fprintf(f, "%zu\n", waves.size());

  // one wave per line: amplitude wavelength angular_freq phase dir_x dir_y
  for (const auto &w : waves)
    fprintf(f, "%.17g %.17g %.17g %.17g %.17g %.17g\n",
            w.amplitude, w.wavelength, w.angular_freq,
            w.phase, w.dir_x, w.dir_y);

  fclose(f);
  printf("Wrote %zu waves to waves.txt\n", waves.size());
  return 0;
}
