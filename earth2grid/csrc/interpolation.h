// This is NVIDIA code
#pragma once

#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <numbers>
#include <stdint.h>

#include "healpix_bare.h"

void hpx_info(int64_t nside, int64_t &npface, int64_t &ncap, int64_t &npix,
              double &fact1, double &fact2) {
  npix = nside2npix(nside);
  npface = npix / 12;
  ncap = (npface - nside) << 1;
  fact2 = 4. / npix;
  fact1 = (nside << 1) * fact2;
}

int64_t ring_above(double z, int64_t nside) {
  double az = std::abs(z);
  if (az <= 2. / 3.)
    return int64_t(nside * (2 - 1.5 * z));
  int64_t iring = int64_t(nside * std::sqrt(3 * (1 - az)));
  return (z > 0) ? iring : 4 * nside - iring - 1;
}

void ring_info(int64_t ring, int64_t &startpix, int64_t &ringpix, double &theta,
               bool &shifted, int64_t nside) {
  int64_t northring = (ring > 2 * nside) ? 4 * nside - ring : ring;
  int64_t npface, ncap, npix;
  double fact1, fact2;
  hpx_info(nside, npface, ncap, npix, fact1, fact2);
  if (northring < nside) {
    double tmp = northring * northring * fact2;
    double costheta = 1 - tmp;
    double sintheta = std::sqrt(tmp * (2 - tmp));
    theta = std::atan2(sintheta, costheta);
    ringpix = 4 * northring;
    shifted = true;
    startpix = 2 * northring * (northring - 1);
  } else {
    theta = acos((2 * nside - northring) * fact1);
    ringpix = 4 * nside;
    shifted = ((northring - nside) & 1) == 0;
    startpix = ncap + (northring - nside) * ringpix;
  }
  if (northring != ring) // southern hemisphere
  {
    theta = std::numbers::pi - theta;
    startpix = npix - startpix - ringpix;
  }
}

// TODO (asubramaniam): switch from std::array to double* pointers to avoid
// copies in the batched case
template <bool NEST = false>
void interpolation_weights(const t_ang &ptg, std::array<int64_t, 4> &pix,
                           std::array<double, 4> &wgt, int64_t nside) {
  assert((ptg.theta >= 0) && (ptg.theta <= std::numbers::pi));
  double z = std::cos(ptg.theta);
  int64_t npix = nside2npix(nside);

  // Do everything in ring ordering first
  // Can convert indices to nest ordering at the end if needed
  int64_t ir1 = ring_above(z, nside);
  int64_t ir2 = ir1 + 1;
  double theta1, theta2, w1, tmp, dphi;
  int64_t sp, nr;
  bool shift;
  int64_t i1, i2;
  if (ir1 > 0) {
    ring_info(ir1, sp, nr, theta1, shift, nside);
    dphi = 2. * std::numbers::pi / nr;
    tmp = (ptg.phi / dphi - .5 * shift);
    i1 = (tmp < 0) ? int64_t(tmp) - 1 : int64_t(tmp);
    w1 = (ptg.phi - (i1 + .5 * shift) * dphi) / dphi;
    if (i1 < 0) {
      i1 += nr;
    }
    i2 = i1 + 1;
    if (i2 >= nr) {
      i2 -= nr;
    }
    pix[0] = sp + i1;
    pix[1] = sp + i2;
    wgt[0] = 1 - w1;
    wgt[1] = w1;
  }
  if (ir2 < (4 * nside)) {
    ring_info(ir2, sp, nr, theta2, shift, nside);
    dphi = 2. * std::numbers::pi / nr;
    tmp = (ptg.phi / dphi - .5 * shift);
    i1 = (tmp < 0) ? int64_t(tmp) - 1 : int64_t(tmp);
    w1 = (ptg.phi - (i1 + .5 * shift) * dphi) / dphi;
    if (i1 < 0)
      i1 += nr;
    i2 = i1 + 1;
    if (i2 >= nr)
      i2 -= nr;
    pix[2] = sp + i1;
    pix[3] = sp + i2;
    wgt[2] = 1 - w1;
    wgt[3] = w1;
  }

  if (ir1 == 0) {
    double wtheta = ptg.theta / theta2;
    wgt[2] *= wtheta;
    wgt[3] *= wtheta;
    double fac = (1 - wtheta) * 0.25;
    wgt[0] = fac;
    wgt[1] = fac;
    wgt[2] += fac;
    wgt[3] += fac;
    pix[0] = (pix[2] + 2) & 3;
    pix[1] = (pix[3] + 2) & 3;
  } else if (ir2 == 4 * nside) {
    double wtheta = (ptg.theta - theta1) / (std::numbers::pi - theta1);
    wgt[0] *= (1 - wtheta);
    wgt[1] *= (1 - wtheta);
    double fac = wtheta * 0.25;
    wgt[0] += fac;
    wgt[1] += fac;
    wgt[2] = fac;
    wgt[3] = fac;
    pix[2] = ((pix[0] + 2) & 3) + npix - 4;
    pix[3] = ((pix[1] + 2) & 3) + npix - 4;
  } else {
    double wtheta = (ptg.theta - theta1) / (theta2 - theta1);
    wgt[0] *= (1 - wtheta);
    wgt[1] *= (1 - wtheta);
    wgt[2] *= wtheta;
    wgt[3] *= wtheta;
  }

  // Convert indices from ring to nest format if needed
  if (NEST)
    for (size_t m = 0; m < pix.size(); ++m)
      pix[m] = ring2nest(pix[m], nside);
}

double degrees2radians(double theta) { return theta * std::numbers::pi / 180.; }

t_ang latlon2ang(double lat, double lon) {
  double theta = std::numbers::pi / 2. - degrees2radians(lat);
  double phi = degrees2radians(lon);
  t_ang ang = {theta, phi};
  return ang;
}
