/* -----------------------------------------------------------------------------
 *
 *  Copyright (C) 1997-2019 Krzysztof M. Gorski, Eric Hivon, Martin Reinecke,
 *                          Benjamin D. Wandelt, Anthony J. Banday,
 *                          Matthias Bartelmann,
 *                          Reza Ansari & Kenneth M. Ganga
 *
 *  Implementation of the Healpix bare bones C library
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
#include "healpix_bare.h"

#define pi 3.141592653589793238462643383279502884197

static const int jrll[] = { 2,2,2,2,3,3,3,3,4,4,4,4 };
static const int jpll[] = { 1,3,5,7,0,2,4,6,1,3,5,7 };

/* conversions between continuous coordinate systems */

typedef struct { double z, s, phi; } tloc;

/*! A structure describing the continuous Healpix coordinate system.
    \a f takes values in [0;11], \a x and \a y lie in [0.0; 1.0]. */
typedef struct { double x, y; int32_t f; } t_hpc;

static t_hpc loc2hpc (tloc loc)
  {
  double za = fabs(loc.z);
  double x = loc.phi*(1./(2.*pi));
  if (x<0.)
    x += (int64_t)x + 1.;
  else if (x>=1.)
    x -= (int64_t)x;
  double tt = 4.*x;

  if (za<=2./3.) /* Equatorial region */
    {
    double temp1 = 0.5+tt; // [0.5; 4.5)
    double temp2 = loc.z*0.75; // [-0.5; +0.5]
    double jp = temp1-temp2; /* index of  ascending edge line */ // [0; 5)
    double jm = temp1+temp2; /* index of descending edge line */ // [0; 5)
    int ifp = (int)jp;  /* in {0,4} */
    int ifm = (int)jm;
    return (t_hpc){jm-ifm, 1+ifp - jp,
                  (ifp==ifm) ? (ifp|4) : ((ifp<ifm) ? ifp : (ifm+8))};
    }
  int64_t ntt = (int64_t)tt;
  if (ntt>=4) ntt=3;
  double tp = tt-ntt; // [0;1)
  double tmp = loc.s/sqrt((1.+za)*(1./3.)); // FIXME optimize!

  double jp = tp*tmp; /* increasing edge line index */
  double jm = (1.0-tp)*tmp; /* decreasing edge line index */
  if (jp>1.) jp = 1.; /* for points too close to the boundary */
  if (jm>1.) jm = 1.;
  return (loc.z >= 0) ? (t_hpc){1.-jm, 1.-jp, ntt}
                      : (t_hpc){jp, jm, ntt+8};
  }

static tloc hpc2loc (t_hpc hpc)
  {
  double jr = jrll[hpc.f] - hpc.x - hpc.y;
  if (jr<1.)
    {
    double tmp = jr*jr*(1./3.);
    double z = 1. - tmp;
    double s = sqrt(tmp*(2.-tmp));
    double phi = (pi*0.25)*(jpll[hpc.f] + (hpc.x-hpc.y)/jr);
    return (tloc){z,s,phi};
    }
  else if (jr>3.)
    {
    jr = 4.-jr;
    double tmp = jr*jr*(1./3.);
    double z = tmp - 1.;
    double s = sqrt(tmp*(2.-tmp));
    double phi = (pi*0.25)*(jpll[hpc.f] + (hpc.x-hpc.y)/jr);
    return (tloc){z,s,phi};
    }
  else
    {
    double z = (2.-jr)*(2./3.);
    double s = sqrt((1.+z)*(1.-z));
    double phi = (pi*0.25)*(jpll[hpc.f] + hpc.x - hpc.y);
    return (tloc){z,s,phi};
    }
  }

static tloc ang2loc(t_ang ang)
  {
  double cth=cos(ang.theta), sth=sin(ang.theta);
  if (sth<0) { sth=-sth; ang.phi+=pi; }
  return (tloc){cth, sth, ang.phi};
  }

static t_ang loc2ang(tloc loc)
  { return (t_ang){atan2(loc.s,loc.z), loc.phi}; }

static tloc vec2loc(t_vec vec)
  {
  double vlen=sqrt(vec.x*vec.x+vec.y*vec.y+vec.z*vec.z);
  double cth = vec.z/vlen;
  double sth = sqrt(vec.x*vec.x+vec.y*vec.y)/vlen;
  return (tloc){cth,sth,atan2(vec.y,vec.x)};
  }

static t_vec loc2vec(tloc loc)
  { return (t_vec){loc.s*cos(loc.phi), loc.s*sin(loc.phi), loc.z}; }

t_vec ang2vec(t_ang ang)
  { return loc2vec(ang2loc(ang)); }

t_ang vec2ang(t_vec vec)
  {
  return (t_ang) {atan2(sqrt(vec.x*vec.x+vec.y*vec.y),vec.z),
                     atan2(vec.y,vec.x)};
  }

/* conversions between discrete coordinate systems */

static int64_t isqrt(int64_t v)
  {
  int64_t res = sqrt(v+0.5);
  if (v<((int64_t)(1)<<50)) return res;
  if (res*res>v)
    --res;
  else if ((res+1)*(res+1)<=v)
    ++res;
  return res;
  }

static int64_t spread_bits (int64_t v)
  {
  int64_t res = v & 0xffffffff;
  res = (res^(res<<16)) & 0x0000ffff0000ffff;
  res = (res^(res<< 8)) & 0x00ff00ff00ff00ff;
  res = (res^(res<< 4)) & 0x0f0f0f0f0f0f0f0f;
  res = (res^(res<< 2)) & 0x3333333333333333;
  res = (res^(res<< 1)) & 0x5555555555555555;
  return res;
  }

static int64_t compress_bits (int64_t v)
  {
  int64_t res = v & 0x5555555555555555;
  res = (res^(res>> 1)) & 0x3333333333333333;
  res = (res^(res>> 2)) & 0x0f0f0f0f0f0f0f0f;
  res = (res^(res>> 4)) & 0x00ff00ff00ff00ff;
  res = (res^(res>> 8)) & 0x0000ffff0000ffff;
  res = (res^(res>>16)) & 0x00000000ffffffff;
  return res;
  }

/*! A structure describing the discrete Healpix coordinate system.
    \a f takes values in [0;11], \a x and \a y lie in [0; nside[. */
typedef struct { int64_t x, y; int32_t f; } t_hpd;

static int64_t hpd2nest (int64_t nside, t_hpd hpd)
  { return (hpd.f*nside*nside) + spread_bits(hpd.x) + (spread_bits(hpd.y)<<1); }

static t_hpd nest2hpd (int64_t nside, int64_t pix)
  {
  int64_t npface_=nside*nside, p2=pix&(npface_-1);
  return (t_hpd){compress_bits(p2), compress_bits(p2>>1), pix/npface_};
  }

static int64_t hpd2ring (int64_t nside_, t_hpd hpd)
  {
  int64_t nl4 = 4*nside_;
  int64_t jr = (jrll[hpd.f]*nside_) - hpd.x - hpd.y - 1;

  if (jr<nside_)
    {
    int64_t jp = (jpll[hpd.f]*jr + hpd.x - hpd.y + 1) / 2;
    jp = (jp>nl4) ? jp-nl4 : ((jp<1) ? jp+nl4 : jp);
    return 2*jr*(jr-1) + jp - 1;
    }
  else if (jr > 3*nside_)
    {
    jr = nl4-jr;
    int64_t jp = (jpll[hpd.f]*jr + hpd.x - hpd.y + 1) / 2;
    jp = (jp>nl4) ? jp-nl4 : ((jp<1) ? jp+nl4 : jp);
    return 12*nside_*nside_ - 2*(jr+1)*jr + jp - 1;
    }
  else
    {
    int64_t jp = (jpll[hpd.f]*nside_ + hpd.x - hpd.y + 1 + ((jr-nside_)&1)) / 2;
    jp = (jp>nl4) ? jp-nl4 : ((jp<1) ? jp+nl4 : jp);
    return 2*nside_*(nside_-1) + (jr-nside_)*nl4 + jp - 1;
    }
  }

static t_hpd ring2hpd (int64_t nside_, int64_t pix)
  {
  int64_t ncap_=2*nside_*(nside_-1);
  int64_t npix_=12*nside_*nside_;

  if (pix<ncap_) /* North Polar cap */
    {
    int64_t iring = (1+isqrt(1+2*pix))>>1; /* counted from North pole */
    int64_t iphi  = (pix+1) - 2*iring*(iring-1);
    int64_t face = (iphi-1)/iring;
    int64_t irt = iring - (jrll[face]*nside_) + 1;
    int64_t ipt = 2*iphi- jpll[face]*iring -1;
    if (ipt>=2*nside_) ipt-=8*nside_;
    return (t_hpd) {(ipt-irt)>>1, (-(ipt+irt))>>1, face};
    }
  else if (pix<(npix_-ncap_)) /* Equatorial region */
    {
    int64_t ip = pix - ncap_;
    int64_t iring = (ip/(4*nside_)) + nside_; /* counted from North pole */
    int64_t iphi  = (ip%(4*nside_)) + 1;
    int64_t kshift = (iring+nside_)&1;
    int64_t ire = iring-nside_+1;
    int64_t irm = 2*nside_+2-ire;
    int64_t ifm = (iphi - ire/2 + nside_ -1) / nside_;
    int64_t ifp = (iphi - irm/2 + nside_ -1) / nside_;
    int64_t face = (ifp==ifm) ? (ifp|4) : ((ifp<ifm) ? ifp : (ifm+8));
    int64_t irt = iring - (jrll[face]*nside_) + 1;
    int64_t ipt = 2*iphi- jpll[face]*nside_ - kshift -1;
    if (ipt>=2*nside_) ipt-=8*nside_;
    return (t_hpd) {(ipt-irt)>>1, (-(ipt+irt))>>1, face};
    }
  else /* South Polar cap */
    {
    int64_t ip = npix_ - pix;
    int64_t iring = (1+isqrt(2*ip-1))>>1; /* counted from South pole */
    int64_t iphi  = 4*iring + 1 - (ip - 2*iring*(iring-1));
    int64_t face=8+(iphi-1)/iring;
    int64_t irt = 4*nside_ - iring - (jrll[face]*nside_) + 1;
    int64_t ipt = 2*iphi- jpll[face]*iring -1;
    if (ipt>=2*nside_) ipt-=8*nside_;
    return (t_hpd) {(ipt-irt)>>1, (-(ipt+irt))>>1, face};
    }
  }

int64_t nest2ring(int64_t nside, int64_t ipnest)
  {
  if ((nside&(nside-1))!=0) return -1;
  return hpd2ring (nside, nest2hpd (nside, ipnest));
  }

int64_t ring2nest(int64_t nside, int64_t ipring)
  {
  if ((nside&(nside-1))!=0) return -1;
  return hpd2nest (nside, ring2hpd (nside, ipring));
  }

/* mixed conversions */

static t_hpd loc2hpd (int64_t nside_, tloc loc)
  {
  t_hpc tmp = loc2hpc(loc);
  return (t_hpd){(tmp.x*nside_), (tmp.y*nside_), tmp.f};
  }

static tloc hpd2loc (int64_t nside_, t_hpd hpd)
  {
  double xns = 1./nside_;
  t_hpc tmp = (t_hpc){(hpd.x+0.5)*xns,(hpd.y+0.5)*xns,hpd.f};
  return hpc2loc(tmp);
  }

int64_t npix2nside(int64_t npix)
  {
  int64_t res = isqrt(npix/12);
  return (res*res*12==npix) ? res : -1;
  }

int64_t nside2npix(int64_t nside)
  { return 12*nside*nside; }

double vec_angle(t_vec v1, t_vec v2)
  {
  t_vec cross = { v1.y*v2.z - v1.z*v2.y,
                  v1.z*v2.x - v1.x*v2.z,
                  v1.x*v2.y - v1.y*v2.x };
  double len_cross = sqrt(cross.x*cross.x + cross.y*cross.y + cross.z*cross.z);
  double dot = v1.x*v2.x + v1.y*v2.y + v1.z*v2.z;
  return atan2 (len_cross, dot);
  }

int64_t ang2ring(int64_t nside, t_ang ang)
  { return hpd2ring(nside, loc2hpd(nside, ang2loc(ang))); }
int64_t ang2nest(int64_t nside, t_ang ang)
  { return hpd2nest(nside, loc2hpd(nside, ang2loc(ang))); }
int64_t vec2ring(int64_t nside, t_vec vec)
  { return hpd2ring(nside, loc2hpd(nside, vec2loc(vec))); }
int64_t vec2nest(int64_t nside, t_vec vec)
  { return hpd2nest(nside, loc2hpd(nside, vec2loc(vec))); }
t_ang ring2ang(int64_t nside, int64_t ipix)
  { return loc2ang(hpd2loc(nside, ring2hpd(nside, ipix))); }
t_ang nest2ang(int64_t nside, int64_t ipix)
  { return loc2ang(hpd2loc(nside, nest2hpd(nside, ipix))); }
t_vec ring2vec(int64_t nside, int64_t ipix)
  { return loc2vec(hpd2loc(nside, ring2hpd(nside, ipix))); }
t_vec nest2vec(int64_t nside, int64_t ipix)
  { return loc2vec(hpd2loc(nside, nest2hpd(nside, ipix))); }
