/* -----------------------------------------------------------------------------
 *
 *  Copyright (C) 1997-2019 Krzysztof M. Gorski, Eric Hivon, Martin Reinecke,
 *                          Benjamin D. Wandelt, Anthony J. Banday,
 *                          Matthias Bartelmann,
 *                          Reza Ansari & Kenneth M. Ganga
 *
 *  Test for the Healpix bare bones C library
 *
 *  Licensed under a 3-clause BSD style license - see LICENSE
 *
 *  For more information on HEALPix and additional software packages, see
 *  https://healpix.sourceforge.io/
 *
 *  If you are using this code in your own packages, please consider citing
 *  the original paper in your publications:
 *  K.M. Gorski et al., 2005, Ap.J., 622, p.759
 *  (http://adsabs.harvard.edu/abs/2005ApJ...622..759G)
 *
 *----------------------------------------------------------------------------*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <inttypes.h>

#include "healpix_bare.h"

static void error_check(int64_t nside, int64_t ipix, int64_t ip1)
  {
  if (ip1 != ipix) {
    printf("Error0: %" PRId64 " %" PRId64 " %" PRId64 "\n", nside, ipix, ip1);
    abort();
    }
  }

static void test2_helper1 (int64_t nside, int64_t dpix, int64_t epix)
  {
  /* Find the number of pixels in the full map */
  for (int64_t ipix = 0; ipix < epix; ipix +=dpix)
    error_check(nside, ipix, ring2nest(nside, nest2ring(nside, ipix)));
  for (int64_t ipix = 0; ipix < epix; ipix +=dpix)
    error_check(nside, ipix, ang2nest(nside, nest2ang(nside, ipix)));
  for (int64_t ipix = 0; ipix < epix; ipix +=dpix)
    error_check(nside, ipix, vec2nest(nside, nest2vec(nside, ipix)));
  }

static void test2_helper2 (int64_t nside, int64_t dpix)
  {
  /* Find the number of pixels in the full map */
  int64_t npix = nside2npix(nside);
  for (int64_t ipix = 0; ipix < npix; ipix +=dpix)
    error_check(nside, ipix, ang2ring(nside, ring2ang(nside, ipix)));
  for (int64_t ipix = 0; ipix < npix; ipix +=dpix)
    error_check(nside, ipix, vec2ring(nside, ring2vec(nside, ipix)));
  }

static void test2(void) {
  printf("Starting basic C Healpix pixel routines test\n");

  for (int i=0; i<=29; ++i)
    {
    int64_t nside = 1LL<<i;
    int64_t npix = nside2npix(nside);
    int64_t epix = 2000;
    if (epix>12*nside*nside) epix = 12*nside*nside;
    printf("Nside: %" PRId64 "\n",nside);
    test2_helper1(nside,nside*nside/123456+1, npix);
    test2_helper1(nside, 1, epix);
    test2_helper2(nside,nside*nside/123456+1);
    }

  printf("test completed\n\n");
}

static void test3(void) {
  printf("Starting nontrivial C Healpix pixel routines test\n");

  int64_t nside = 1024;
  int64_t dpix = 23;

  /* Find the number of pixels in the full map */
  int64_t npix = nside2npix(nside);
  printf("Number of pixels in full map: %ld\n", npix);

  printf("dpix: %ld\n", dpix);
  printf("Nest -> ang -> vec -> ang -> Ring -> Nest\n");
  for (int64_t ipix = 0; ipix < npix; ipix +=dpix)
    error_check(nside, ipix,
      ring2nest(nside, ang2ring(nside, vec2ang(ang2vec(nest2ang(nside, ipix))))));
  printf("Ring -> ang -> Nest -> Ring\n");
  for (int64_t ipix = 0; ipix < npix; ipix +=dpix)
    error_check(nside, ipix,
      nest2ring(nside,ang2nest(nside, ring2ang(nside, ipix))));

  printf("Nest -> vec -> Ring -> Nest\n");
  for (int64_t ipix = 0; ipix < npix; ipix +=dpix)
    error_check(nside, ipix,
      ring2nest(nside,vec2ring(nside,nest2vec(nside, ipix))));
  printf("Ring -> vec -> Nest -> Ring\n");
  for (int64_t ipix = 0; ipix < npix; ipix +=dpix)
    error_check(nside, ipix,
      nest2ring(nside,vec2nest(nside,ring2vec(nside, ipix))));

  printf("%" PRId64 "\n", nside);
  printf("test completed\n\n");
}

static void test_ang(void) {
  printf("Starting vector angle test\n");
  for (double i=0; i<10; i+=0.01) {
    t_vec v1 = { i,10-i,sin(i) }, v2 = { 3-1.5*i, sqrt(i), i*0.3 };
    double dot = v1.x*v2.x + v1.y*v2.y + v1.z*v2.z,
           l1  = sqrt(v1.x*v1.x+v1.y*v1.y+v1.z*v1.z),
           l2  = sqrt(v2.x*v2.x+v2.y*v2.y+v2.z*v2.z);
    double ang = acos(dot/(l1*l2));
    if (fabs(ang - vec_angle(v1,v2))>1e-13)
      printf("error in vector angle test: %e %e\n", ang, ang-vec_angle(v1, v2));
    }
  printf("test completed\n\n");
}

int main(void) {
  test2();
  test3();
  test_ang();
  return 0;
}
