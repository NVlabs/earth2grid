#include <iostream>
#include <stdint.h>
#include <vector>

#include "third_party/healpy_bare/healpix_bare.h"
#include "regridding/interpolation.h"

// Test for interpolation routine
int main(int argc, char *argv[]) {
  constexpr bool nest = false;
  double lat = 0., lon = 23.;
  t_ang ptg = latlon2ang(lat, lon);
  std::array<int64_t, 4> pix;
  std::array<double, 4> weight;
  int64_t nside = 8;

  interpolation_weights<nest>(ptg, pix, weight, nside);

  std::cout << "lat = " << lat << std::endl;
  std::cout << "lon = " << lon << std::endl;
  std::cout << "nside = " << nside << std::endl;

  std::cout << "pix: " << std::endl;
  for (int i = 0; i < 4; ++i)
    std::cout << pix[i] << " ";
  std::cout << std::endl;

  std::cout << "weight: " << std::endl;
  for (int i = 0; i < 4; ++i)
    std::cout << weight[i] << " ";
  std::cout << std::endl;

  // Test batched version
  std::vector<double> lats = {0., 12., 67.};
  std::vector<double> lons = {23., 84., -23.};
  std::vector<int64_t> pixs(4 * lats.size(), 0);
  std::vector<double> weights(4 * lats.size(), 0.);
  bool lonlat = true;

  assert(lats.size() == lons.size());
  get_interp_weights<nest>(nside, lons.data(), lats.data(), lats.size(), lonlat,
                           pixs.data(), weights.data());

  std::cout << "lats: " << std::endl;
  for (size_t i = 0; i < lats.size(); ++i)
    std::cout << lats[i] << " ";
  std::cout << std::endl;

  std::cout << "lons: " << std::endl;
  for (size_t i = 0; i < lons.size(); ++i)
    std::cout << lons[i] << " ";
  std::cout << std::endl;

  std::cout << "pixs: " << std::endl;
  for (int i = 0; i < 4; ++i) {
    for (size_t j = 0; j < lats.size(); ++j) {
      std::cout << pixs[i + 4 * j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout << "weights: " << std::endl;
  for (int i = 0; i < 4; ++i) {
    for (size_t j = 0; j < lats.size(); ++j) {
      std::cout << weights[i + 4 * j] << " ";
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  return 0;
}
