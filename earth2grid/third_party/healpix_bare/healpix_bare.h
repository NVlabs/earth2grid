/* -----------------------------------------------------------------------------
 *
 *  Copyright (C) 1997-2019 Krzysztof M. Gorski, Eric Hivon, Martin Reinecke,
 *                          Benjamin D. Wandelt, Anthony J. Banday,
 *                          Matthias Bartelmann,
 *                          Reza Ansari & Kenneth M. Ganga
 *
 *  Public interface of the Healpix bare bones C library
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

#ifndef HEALPIX_BARE_H
#define HEALPIX_BARE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* This code is written in C99; you may have to adjust your compiler flags. */

/* Continuous coordinate systems */

/* Admissible values for theta (definition see below)
   0 <= theta <= pi

   Admissible values for phi (definition see below)
   In principle unconstrained, but best accuracy is obtained for
   -2*pi <= phi <= 2*pi */

/*! A structure describing a location on the sphere. \a Theta is the co-latitude
    in radians (0 at the North Pole, increasing to pi at the South Pole.
    \a Phi is the azimuth in radians. */
typedef struct { double theta, phi; } t_ang;
/*! A structure describing a 3-vector with coordinates \a x, \a y and \a z.*/
typedef struct { double x, y, z; } t_vec;

/*! Returns a normalized 3-vector pointing in the same direction as \a ang. */
t_vec ang2vec(t_ang ang);
/*! Returns a t_ang describing the same direction as the 3-vector \a vec.
    \a vec need not be normalized. */
t_ang vec2ang(t_vec vec);


/* Discrete coordinate systems */

/* Admissible values for nside parameters:
   any integer power of 2 with 1 <= nside <= 1<<29

   Admissible values for pixel indices:
   0 <= idx < 12*nside*nside */

/*! Returns the RING pixel index of pixel \a ipnest at resolution \a nside.
    On error, returns -1. */
int64_t nest2ring(int64_t nside, int64_t ipnest);
/*! Returns the NEST pixel index of pixel \a ipring at resolution \a nside.
    On error, returns -1. */
int64_t ring2nest(int64_t nside, int64_t ipring);


/* Conversions between continuous and discrete coordinate systems */

/*! Returns the pixel number in NEST scheme at resolution \a nside,
    which contains the position \a ang. */
int64_t ang2nest(int64_t nside, t_ang ang);
/*! Returns the pixel number in RING scheme at resolution \a nside,
    which contains the position \a ang. */
int64_t ang2ring(int64_t nside, t_ang ang);

/*! Returns a t_ang corresponding to the angular position of the center of
    pixel \a ipix in NEST scheme at resolution \a nside. */
t_ang nest2ang(int64_t nside, int64_t ipix);
/*! Returns a t_ang corresponding to the angular position of the center of
    pixel \a ipix in RING scheme at resolution \a nside. */
t_ang ring2ang(int64_t nside, int64_t ipix);

/*! Returns the pixel number in NEST scheme at resolution \a nside,
    which contains the direction described the 3-vector \a vec. */
int64_t vec2nest(int64_t nside, t_vec vec);
/*! Returns the pixel number in RING scheme at resolution \a nside,
    which contains the direction described the 3-vector \a vec. */
int64_t vec2ring(int64_t nside, t_vec vec);

/*! Returns a normalized 3-vector pointing in the direction of the center
    of pixel \a ipix in NEST scheme at resolution \a nside. */
t_vec nest2vec(int64_t nside, int64_t ipix);
/*! Returns a normalized 3-vector pointing in the direction of the center
    of pixel \a ipix in RING scheme at resolution \a nside. */
t_vec ring2vec(int64_t nside, int64_t ipix);


/* Miscellaneous utility routines */

/*! Returns \a 12*nside*nside. */
int64_t nside2npix(int64_t nside);
/*! Returns \a sqrt(npix/12) if this is an integer number, otherwise \a -1. */
int64_t npix2nside(int64_t npix);

/*! Returns the angle (in radians) between the vectors \a v1 and \a v2.
    The result is accurate even for angles close to 0 and pi. */
double vec_angle(t_vec v1, t_vec v2);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* HEALPIX_REFERENCE_H */
