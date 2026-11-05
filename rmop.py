#!/usr/bin/env python3
"""
rmop.py - emulation of the NCEP "Relative Measure of Predictability" (RMOP)
          product, computed from the GFS ensemble (GEFS).

WHAT RMOP IS
------------
RMOP was developed at NCEP/EMC (Z. Toth, Y. Zhu, T. Marchok) and was widely
used by forecasters at CPC/HPC/WFOs.  For 500 hPa height it answers "is this
ensemble forecast more or less predictable than usual?" using only the
agreement among ensemble members, normalised for location, season, lead time
and the position of the forecast within the climate distribution.

ALGORITHM (as documented in the literature - see REFERENCES)
------------------------------------------------------------
 1. CLIMATE BINS.  At every grid point and for every calendar month, split the
    reanalysis climatology of Z500 into 10 equally likely bins (deciles).
    Bin edges are linearly interpolated in time between months.     [T01 s5]
 2. ENSEMBLE-MEAN BIN.  Find the climate bin that contains the ensemble mean.
 3. SUPPORTING MEMBERS.  Count how many individual members fall in that same
    bin.  Few => low predictability, many => high predictability.   [T01 s5]
    Because the bins are climatologically equiprobable they are wide in the
    tails and narrow near the median, so the count is comparable regardless of
    where in the climate distribution the forecast sits.
 4. RELATIVE (PERCENTILE) SCALE.  The count is converted to a percentile of the
    distribution of counts from the PREVIOUS 30 DAYS of forecasts at the same
    lead time.  "Shading at 90% means at least 9 of 10 ensemble forecasts in
    the past 30 days had fewer members in the ensemble-mean bin than the
    present forecast."                                              [EMC GEFS page]
    The 2001 paper pooled the distribution over the NH extratropics and
    optionally weighted days with a recursive filter (0.97 old / 0.03 new).
 5. DISPLAY.  Shade the 0-100 percentile in 10% steps (blue = least predictable
    10% ... orange = most predictable 10%), contour the ensemble-mean Z500.
    The default output re-creates the original GrADS graphic: Hershey Roman Simplex
    text, 2 px contours with horizontal labels in white boxes, a dotted grid, the
    triangle-ended colour bar and the original 14-colour palette.  It is written as
    a 300 DPI PNG at 5x the original resolution (3475 x 2725 px = the 11 x 8.5 in
    page at print size); --scale 1 --out x.gif gives the original 695 x 545 GIF87a.
    --style mpl gives a smooth matplotlib version with the same layout.
 6. VERIFICATION NUMBERS.  For past forecasts that can already be verified,
    compute how often the verifying analysis fell in the same climate bin as
    the ensemble mean, stratified by predictability class (the "blue numbers"
    printed on the original graphic).

WHAT IS AN APPROXIMATION HERE (the original code is not public)
---------------------------------------------------------------
 * Original: 10 (later 20/21) members, 2.5 deg grid, 1x1 reanalysis deciles.
   Here: GEFS (control + 30 perturbed members by default), on the native 0.5 deg
   grid.  0.5 deg is the finest PUBLIC GEFS grid that carries 500 hPa height: the
   0.25 deg GEFS product ("pgrb2s", NCO inventory gec00.t00z.pgrb2s.0p25) is a
   select-parameters file of surface/near-surface fields only (2 m T, 10 m wind,
   CAPE, ...) with no pressure-level heights, and the model's own ~25 km cubed-sphere
   grid is not distributed for pressure levels.  --stride 5 thins to the original 2.5 deg.
 * Tie handling for the percentile is not documented.  The count is a whole
   number, and at short leads a large share of points have EVERY member in the
   mean's bin, so those points all tie.  Default here (--ties spread): each count
   keeps the percentile range the history gives it, but the tied points are spread
   across that range ordered by normalised ensemble spread (tighter = higher), so
   the colours use the full 0-100 scale as the EMC FAQ says they should.  This is
   my own addition.  --ties mid puts all tied points at the middle of the range
   (flat colour blocks at short leads); --ties strict is the literal EMC wording
   (fraction of history with strictly fewer members).
 * The "Probability" row follows the paper: observed hit frequency for each bin
   count, averaged over the points of the current map that make up each colour.
 * Climatology: default recipe builds monthly deciles from NCEP/NCAR
   Reanalysis-1 daily Z500, 1991-2020 (the 2001 paper used the NCEP
   reanalysis, Kalnay et al. 1996).  Any NetCDF with daily Z500 works.
 * Whether the 30-day distribution is pooled over the domain or kept per grid
   point is ambiguous in the text.  Default: pooled over the domain (as in the
   paper); --pool point gives a per-grid-point history.

REFERENCES
----------
 [T01] Toth, Z., Y. Zhu, and T. Marchok, 2001: The use of ensembles to
       identify forecasts with small and large uncertainty.  Wea. Forecasting,
       16, 463-477.  https://journals.ametsoc.org/view/journals/wefo/16/4/
       1520-0434_2001_016_0463_tuoeti_2_0_co_2.xml
       Preprint ("On the ability of ensembles to distinguish between forecasts
       with small and large uncertainty"), Section 5 defines RMOP:
       https://emc.ncep.noaa.gov/gmb/ens/papers/probpaperrevweb/probpaperrevweb.html
 [EMC] NCEP/EMC GEFS page (current RMOP description, "previous 30 days",
       10% shading increments, verification numbers):
       https://emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gefs.php
 [FAQ] NCEP/EMC GEFS FAQ, "Ensemble Relative Measure of Predictability":
       https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gefs/faq.php
 [Z05] Zhu, Y., 2005: Ensemble forecast: a new approach to uncertainty and
       predictability.  Adv. Atmos. Sci., 22, 781-788.
       https://emc.ncep.noaa.gov/gmb/yzhu/gif/pub/AAS200512_Zhu.pdf
 [TZ03] Toth, Z. et al., 2003 ("Overview of predictability related work at
       NCEP"): https://www.emc.ncep.noaa.gov/gmb/ens/Toth1.pdf
 [K96] Kalnay, E. et al., 1996: The NCEP/NCAR 40-year reanalysis project.
       Bull. Amer. Meteor. Soc., 77, 437-471.
 [COMET] COMET/MetEd media gallery, description of the RMOP graphic:
       https://www.meted.ucar.edu/nwp/fcstoutcomes/media_gallery.php

DEPENDENCIES
------------
 required : numpy, scipy (only for the synthetic demo), Pillow, matplotlib
            (matplotlib supplies contourpy, used for the contour lines)
 for real data : requests, and either `eccodes` or `pygrib` (GRIB2 decoding),
                 xarray + netCDF4/h5netcdf (only for building the climatology)
 optional : cartopy (coastlines)

USAGE
-----
  # 1) one-off: build decile climatology from daily Z500 NetCDF files
  python rmop.py climo --download ./r1 --years 1991 2020 --out climo_z500.npz
  python rmop.py climo --files "era5_z500_*.nc" --var z --out climo_z500.npz

  # 2) make the product
  python rmop.py run --init 2026110200 --lead 120 --climo climo_z500.npz \
                     --out rmop_f120.png

  # 3) no network / no data needed: synthetic ensemble demo
  python rmop.py demo --out rmop_demo.png

  # 4) web package: PNG images + dashboard with an hour slider and a 7-day archive
  python rmop.py run --init 2026110300 --lead-range 12 240 12 --climo climo_z500.npz \
                     --site rmop_site
  #    rmop_site/index.html    dashboard (hover/tap/drag/keyboard hour slider, day + cycle picker;
  #                            opens straight from disk)
  #    rmop_site/manifest.json index of every run and hour in the archive
  #    rmop_site/img/<init>/   rmop_fHHH.png  (300 DPI, 256-colour palette PNG, roughly 0.3-0.5 MB each)
  #    Run it again for the next cycle: runs already in rmop_site are kept for --keep-days (7) and
  #    older ones are deleted, so the folder is always the latest week.
  python rmop.py serve rmop_site --open     # optional local preview over HTTP
"""
from __future__ import annotations

import argparse
import base64
import glob
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

log = logging.getLogger("rmop")

N_BINS = 10                      # climatological deciles
N_EDGES = N_BINS - 1             # 9 interior edges (10th ... 90th percentile)
DECILES = np.arange(1, N_BINS) / N_BINS
N_CLASSES = 10                   # 10% shading increments


# ======================================================================
# 1. Climatology: monthly deciles, interpolated in time, regridded
# ======================================================================
@dataclass
class Climatology:
    """Monthly decile edges of Z500 (m) on a regular lat/lon grid."""
    lat: np.ndarray              # (nlat,) ascending
    lon: np.ndarray              # (nlon,) 0..360
    edges: np.ndarray            # (12, 9, nlat, nlon)
    label: str = ""

    def save(self, path: str) -> None:
        np.savez_compressed(path, lat=self.lat, lon=self.lon,
                            edges=self.edges.astype(np.float32),
                            label=np.array(self.label))
        log.info("wrote climatology %s", path)

    @staticmethod
    def load(path: str) -> "Climatology":
        z = np.load(path, allow_pickle=False)
        return Climatology(z["lat"], z["lon"], z["edges"], str(z["label"]))

    def signature(self) -> str:
        h = hashlib.md5()
        h.update(np.ascontiguousarray(self.edges[::3, ::4, ::8, ::8]).tobytes())
        return h.hexdigest()[:10]


def month_interp(d: datetime):
    """Mid-month (16th) anchored linear time interpolation between months.
    Returns (m0, m1, w): edges(d) = (1-w)*edges[m0] + w*edges[m1]  (0-based)."""
    y, m = d.year, d.month
    mid = datetime(y, m, 16)
    if d >= mid:
        m0 = m
        m1 = m % 12 + 1
        start, end = mid, datetime(y + (m == 12), m1, 16)
    else:
        m1 = m
        m0 = (m - 2) % 12 + 1
        start, end = datetime(y - (m == 1), m0, 16), mid
    w = (d - start) / (end - start)
    return m0 - 1, m1 - 1, float(w)


class ClimoOnGrid:
    """Climatology bilinearly regridded to the target grid (weights built once)."""

    def __init__(self, clim: Climatology, lat: np.ndarray, lon: np.ndarray):
        self.clim = clim
        la = np.clip(lat, clim.lat[0], clim.lat[-1])
        i1 = np.clip(np.searchsorted(clim.lat, la), 1, len(clim.lat) - 1)
        i0 = i1 - 1
        self.wl = ((la - clim.lat[i0]) / (clim.lat[i1] - clim.lat[i0]))[:, None]
        self.i0, self.i1 = i0[:, None], i1[:, None]

        nlon = len(clim.lon)
        dlon = 360.0 / nlon
        pos = (np.mod(lon, 360.0) - clim.lon[0]) / dlon
        j0 = np.floor(pos).astype(int)
        self.wo = (pos - j0)[None, :]
        self.j0 = (j0 % nlon)[None, :]
        self.j1 = ((j0 + 1) % nlon)[None, :]

    def edges(self, d: datetime) -> np.ndarray:
        m0, m1, w = month_interp(d)
        E = (1 - w) * self.clim.edges[m0] + w * self.clim.edges[m1]   # (9,nlat,nlon)
        wl, wo = self.wl, self.wo
        return ((1 - wl) * (1 - wo) * E[:, self.i0, self.j0]
                + (1 - wl) * wo * E[:, self.i0, self.j1]
                + wl * (1 - wo) * E[:, self.i1, self.j0]
                + wl * wo * E[:, self.i1, self.j1]).astype(np.float32)


def build_climatology(files, var: Optional[str] = None, level: float = 500.0,
                      years=(1991, 2020)) -> Climatology:
    """Monthly decile edges from daily Z500 NetCDF files (one or many years)."""
    import xarray as xr
    buckets = [[] for _ in range(12)]
    lat = lon = None
    for f in sorted(files):
        ds = xr.open_dataset(f)
        name = var or next((v for v in ("hgt", "z", "gh", "Z500", "z500")
                            if v in ds.data_vars), None)
        if name is None:
            raise SystemExit(f"{f}: cannot find Z500 variable; pass --var")
        da = ds[name]
        for ldim in ("level", "plev", "pressure_level", "isobaricInhPa"):
            if ldim in da.dims:
                da = da.sel({ldim: level}, method="nearest")
        tdim = next(d for d in ("time", "valid_time") if d in da.dims)
        latn = next(d for d in ("lat", "latitude") if d in da.dims)
        lonn = next(d for d in ("lon", "longitude") if d in da.dims)
        da = da.transpose(tdim, latn, lonn)
        da = da.sel({tdim: slice(f"{years[0]}-01-01", f"{years[1]}-12-31")})
        if da.sizes[tdim] == 0:
            continue
        if name == "z" or str(da.attrs.get("units", "")).startswith("m**2"):
            da = da / 9.80665                       # geopotential -> height
        la = da[latn].values
        lo = np.mod(da[lonn].values, 360.0)
        vals = da.values.astype(np.float32)
        o_lat, o_lon = np.argsort(la), np.argsort(lo)
        vals = vals[:, o_lat][:, :, o_lon]
        lat, lon = la[o_lat], lo[o_lon]
        months = da[tdim].dt.month.values
        for m in range(12):
            sel = months == m + 1
            if sel.any():
                buckets[m].append(vals[sel])
        log.info("read %s (%d days)", os.path.basename(f), vals.shape[0])
    if lat is None:
        raise SystemExit("no data read for the requested years")
    edges = np.empty((12, N_EDGES, len(lat), len(lon)), np.float32)
    for m in range(12):
        allm = np.concatenate(buckets[m], axis=0)
        edges[m] = np.nanquantile(allm, DECILES, axis=0)
        log.info("month %2d: %d samples", m + 1, allm.shape[0])
    return Climatology(lat, lon, edges, f"decile climatology {years[0]}-{years[1]}")


def download_r1(outdir: str, years) -> list:
    """Fetch NCEP/NCAR Reanalysis-1 daily-mean height files from NOAA PSL.
    NOTE: URL pattern written from documentation; unverified from my sandbox."""
    import requests
    os.makedirs(outdir, exist_ok=True)
    base = "https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis.dailyavgs/pressure"
    out = []
    for y in range(years[0], years[1] + 1):
        dst = os.path.join(outdir, f"hgt.{y}.nc")
        if not os.path.exists(dst):
            log.info("downloading %s", f"{base}/hgt.{y}.nc")
            with requests.get(f"{base}/hgt.{y}.nc", stream=True, timeout=120) as r:
                r.raise_for_status()
                with open(dst + ".part", "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
            os.replace(dst + ".part", dst)
        out.append(dst)
    return out


# ======================================================================
# 2. Core RMOP mathematics
# ======================================================================
def bin_index(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Climate-decile index 0..9 of x given edges (9, ...) - memory-light."""
    idx = np.zeros(x.shape, np.int8)
    for k in range(edges.shape[0]):
        idx += (x > edges[k]).astype(np.int8)
    return idx


def local_width(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Width (m) of the climatological decile at height x, varying CONTINUOUSLY with x.

    The 8 interior deciles have their own widths; the two open-ended tail bins borrow the
    width of their interior neighbour.  The width is linearly interpolated between bin
    centres so that it has no jumps at the decile edges."""
    nb = edges.shape[0] + 1                                   # 10 bins
    w = edges[1:] - edges[:-1]                                # (8,...) interior widths
    W = np.concatenate([w[:1], w, w[-1:]]).astype(np.float64)                   # (10,...)
    C = np.concatenate([(edges[:1] - 0.5 * W[:1]), 0.5 * (edges[1:] + edges[:-1]),
                        (edges[-1:] + 0.5 * W[-1:])]).astype(np.float64)        # (10,...)
    b = bin_index(x, edges).astype(np.int64)
    take = lambda A, i: np.take_along_axis(A, i[None], 0)[0]
    cb, wb = take(C, b), take(W, b)
    n = np.clip(np.where(x >= cb, b + 1, b - 1), 0, nb - 1)
    cn, wn = take(C, n), take(W, n)
    den = cn - cb
    t = np.where(n == b, 0.0, (x - cb) / np.where(den == 0, 1.0, den))
    return (wb + (wn - wb) * np.clip(t, 0.0, 1.0)).astype(np.float32)


def supporting_members(members: np.ndarray, edges: np.ndarray, with_spread: bool = False,
                       bins: str = "fixed"):
    """members (M,ny,nx), edges (9,ny,nx) ->
       ensemble mean, reference, number of supporting members
       (and, if with_spread, the member std-dev as a fraction of the mean
       interior decile width, used only to break ties between equal counts).

    bins = "fixed"   : supporting = members in the SAME climatological decile as the ensemble
                       mean (Toth et al. 2001).  `reference` is that decile's index (int8).
    bins = "sliding" : supporting = members within half a decile-width of the ensemble mean
                       (the window is centred on the mean).  `reference` is a (2,ny,nx) float32
                       stack [mean, half-width].  This removes the dependence of the count on
                       where the mean happens to sit inside its fixed decile - the cause of the
                       thin ripple bands that follow the height isolines at short leads."""
    mean = members.mean(axis=0)
    count = np.zeros(mean.shape, np.int16)
    if bins == "sliding":
        half = 0.5 * local_width(mean, edges)
        for m in members:
            count += (np.abs(m - mean) <= half)
        ref = np.stack([mean, half]).astype(np.float32)
    else:
        ref = bin_index(mean, edges)
        for m in members:
            count += (bin_index(m, edges) == ref)
    if not with_spread:
        return mean, ref, count
    width = (edges[-1] - edges[0]) / (N_EDGES - 1)          # mean width of the 8 interior bins
    sdn = members.std(axis=0) / np.maximum(width, 1e-6)
    return mean, ref, count, sdn.astype(np.float32)


def is_hit(analysis: np.ndarray, edges: np.ndarray, ref: np.ndarray, bins: str = "fixed") -> np.ndarray:
    """1 where the verifying analysis lies in the same decile (fixed) / the same window
    around the ensemble mean (sliding) that the members were counted in."""
    if bins == "sliding":
        return (np.abs(analysis - ref[0]) <= ref[1]).astype(np.int8)
    return (bin_index(analysis, edges) == ref).astype(np.int8)


def despike(a: np.ndarray, thresh: float, wrap_lon: bool, passes: int = 2) -> np.ndarray:
    """Remove ONE-cell-wide ridges/pits (terrain-locked noise in the GFS pressure-level
    heights over steep mountains - Front Range, Sierra, Sierra Madre, Alaska Range ...).

    A point is a spike along an axis when it departs from the mean of its two neighbours by
    more than `thresh` metres AND by nearly as much from the mean of the neighbours two
    cells away (a genuine trough or ridge curves smoothly, so its 1-cell departure is only
    about a quarter of its 2-cell departure).  Spikes are replaced by the neighbour mean.
    Real troughs/ridges, however sharp, are left alone.  Only the contoured field is treated."""
    if thresh <= 0:
        return a
    out = a.astype(np.float64)
    for _ in range(passes):
        for axis in (1, 0):
            wrap = wrap_lon and axis == 1
            def sh(k):
                if wrap:
                    return np.roll(out, -k, axis)
                P = np.pad(out, [(abs(k), abs(k)) if ax == axis else (0, 0) for ax in (0, 1)], mode="edge")
                idx = np.arange(out.shape[axis]) + abs(k) + k
                return np.take(P, idx, axis=axis)
            m1, p1, m2, p2 = sh(-1), sh(1), sh(-2), sh(2)
            d1 = out - 0.5 * (m1 + p1)
            d2 = out - 0.5 * (m2 + p2)
            bad = (np.abs(d1) > thresh) & (np.abs(d1) > 0.6 * np.abs(d2)) & (np.sign(d1) == np.sign(d2))
            out = np.where(bad, 0.5 * (m1 + p1), out)
    return out.astype(a.dtype)


def smooth_field(a: np.ndarray, sigma_cells: float, wrap_lon: bool) -> np.ndarray:
    """Gaussian smoothing (separable, numpy only) used for the CONTOURED height field.
    Removes one-gridpoint noise that the GFS leaves over steep terrain (e.g. the Front Range)
    without touching the shading or the saved fields."""
    if sigma_cells < 0.3:
        return a
    r = int(math.ceil(3 * sigma_cells))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma_cells) ** 2)
    k /= k.sum()
    out = a.astype(np.float64)
    for axis, wrap in ((0, False), (1, wrap_lon)):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        P = np.pad(out, pad, mode="wrap" if wrap else "edge")
        out = sum(k[i] * np.take(P, np.arange(i, i + out.shape[axis]), axis=axis)
                  for i in range(len(k)))
    return out.astype(a.dtype)


@dataclass
class Case:
    init: datetime
    counts: np.ndarray                   # (ny,nx) int16
    mbin: np.ndarray                     # reference: decile index (ny,nx) int8, or (2,ny,nx) [mean, half-width] when sliding
    hit: Optional[np.ndarray] = None     # (ny,nx) int8: 1 hit, 0 miss, -1 unknown
    weight: float = 1.0


@dataclass
class History:
    cases: list
    nmem: int
    area_w: Optional[np.ndarray] = None  # (ny,nx) or None
    ties: str = "mid"
    pool: str = "domain"                 # "domain" | "point"

    def pooled_hist(self) -> np.ndarray:
        h = np.zeros(self.nmem + 1)
        for c in self.cases:
            aw = None if self.area_w is None else self.area_w.ravel()
            h += c.weight * np.bincount(c.counts.ravel(), weights=aw,
                                        minlength=self.nmem + 1)[:self.nmem + 1]
        return h

    def percentile(self, counts: np.ndarray, tie_key: Optional[np.ndarray] = None) -> np.ndarray:
        """Percentile (0-100) of `counts` within the history.

        ties = "mid"    : points sharing a count all get the middle of that count's range
        ties = "strict" : fraction of history with strictly fewer members (EMC wording)
        ties = "spread" : the percentile RANGE of a count is the same as for "mid", but the
                          points sharing that count are spread across the range, ordered by
                          `tie_key` (normalised ensemble spread: tighter => higher).  This
                          lets the colours use the whole 0-100 scale at short leads, where a
                          large share of points have every member in the mean's bin."""
        if not self.cases:
            raise ValueError("no history available")
        if self.pool == "domain":
            h = self.pooled_hist()
            tot = h.sum()
            cum = np.concatenate([[0.0], np.cumsum(h)])
            below, at = cum[counts], h[counts]
        else:
            w = np.array([c.weight for c in self.cases])[:, None, None]
            H = np.stack([c.counts for c in self.cases])
            tot = w.sum()
            below = (w * (H < counts[None])).sum(0)
            at = (w * (H == counts[None])).sum(0)
        if self.ties == "spread" and tie_key is not None:
            lo, wd = (below / tot).ravel(), (at / tot).ravel()
            c, k = counts.ravel(), tie_key.ravel()
            out = np.empty(c.shape, float)
            for v in np.unique(c):
                idx = np.where(c == v)[0]
                order = np.argsort(-k[idx], kind="stable")      # largest spread -> lowest percentile
                rank = np.empty(len(idx))
                rank[order] = np.arange(len(idx))
                out[idx] = lo[idx] + wd[idx] * (rank + 0.5) / len(idx)
            return 100.0 * out.reshape(counts.shape)
        mid = self.ties in ("mid", "spread")                  # "spread" without a key -> mid
        return 100.0 * (below + (0.5 * at if mid else 0.0)) / tot

    def hit_rate_by_count(self) -> np.ndarray:
        """P(verifying analysis is in the ensemble-mean bin | k members in that bin),
        k = 0..N, from the verified history (NaN where no verified cases)."""
        n = self.nmem + 1
        hits, tot = np.zeros(n), np.zeros(n)
        for c in self.cases:
            if c.hit is None or not (c.hit >= 0).any():
                continue
            ok = c.hit >= 0
            aw = 1.0 if self.area_w is None else self.area_w
            w = c.weight * np.broadcast_to(aw, c.counts.shape)
            tot += np.bincount(c.counts[ok], weights=w[ok], minlength=n)[:n]
            hit = ok & (c.hit == 1)
            hits += np.bincount(c.counts[hit], weights=w[hit], minlength=n)[:n]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(tot > 0, hits / tot, np.nan)

    def class_probabilities(self, counts: np.ndarray, cls: np.ndarray) -> np.ndarray:
        """Calibrated probability (%) for each colour class: the observed frequency for each
        bin count, averaged over the points of the CURRENT map that make up that colour,
        weighted by number of points (Toth et al. 2001, section 5c)."""
        hc = self.hit_rate_by_count()
        w_all = np.ones(counts.shape) if self.area_w is None else self.area_w
        out = np.full(N_CLASSES, np.nan)
        for k in range(N_CLASSES):
            m = cls == k
            if not m.any():
                continue
            h, w = hc[counts[m]], w_all[m]
            ok = np.isfinite(h)
            if ok.any():
                out[k] = 100.0 * (h[ok] * w[ok]).sum() / w[ok].sum()
        return out


def to_class(pct: np.ndarray) -> np.ndarray:
    return np.clip((pct / (100.0 / N_CLASSES)).astype(int), 0, N_CLASSES - 1)


def day_weights(n: int, scheme: str) -> np.ndarray:
    """Weights for lags 1..n days back.  'recursive' = F <- 0.97 F + 0.03 f_new."""
    k = np.arange(n)
    return np.ones(n) if scheme == "equal" else 0.97 ** k


# ======================================================================
# 3. Grid helper
# ======================================================================
class Geom:
    """Domain selection / thinning shared by every field."""

    def __init__(self, lat, lon, lat_min, lat_max, stride):
        self.ix_lat = np.where((lat >= lat_min) & (lat <= lat_max))[0][::stride]
        self.ix_lon = np.arange(0, len(lon), stride)
        self.lat, self.lon = lat[self.ix_lat], lon[self.ix_lon]

    def apply(self, a: np.ndarray) -> np.ndarray:
        return a[..., self.ix_lat[:, None], self.ix_lon[None, :]]

    def area_weights(self) -> np.ndarray:
        w = np.cos(np.deg2rad(self.lat))[:, None] * np.ones(len(self.lon))[None, :]
        return w / w.mean()


# ======================================================================
# 4. Data sources
# ======================================================================
def decode_grib_message(raw: bytes):
    """Decode ONE GRIB message -> (lat_asc, lon_0_360, values[nlat,nlon])."""
    try:
        import eccodes as ec
        gid = ec.codes_new_from_message(raw)
        try:
            ni, nj = ec.codes_get(gid, "Ni"), ec.codes_get(gid, "Nj")
            vals = ec.codes_get_values(gid).reshape(nj, ni)
            lats = ec.codes_get_array(gid, "latitudes").reshape(nj, ni)[:, 0]
            lons = ec.codes_get_array(gid, "longitudes").reshape(nj, ni)[0]
        finally:
            ec.codes_release(gid)
    except ImportError:
        import pygrib
        g = pygrib.fromstring(raw)
        vals = g.values
        la, lo = g.latlons()
        lats, lons = la[:, 0], lo[0]
    lons = np.mod(lons, 360.0)
    ol, oo = np.argsort(lats), np.argsort(lons)
    return lats[ol], lons[oo], np.asarray(vals)[ol][:, oo].astype(np.float32)


class GEFSAws:
    """GEFS v12-style files on the AWS open-data bucket (0.5 deg, pgrb2a).

      https://noaa-gefs-pds.s3.amazonaws.com/gefs.YYYYMMDD/HH/atmos/pgrb2ap5/
              gep01.tHHz.pgrb2a.0p50.fFFF      (perturbed 01..30)
              gec00.tHHz.pgrb2a.0p50.fFFF      (control)
    Only the HGT:500 mb GRIB message is fetched, using the .idx byte offsets.
    Check that the member count / layout still match the bucket you use."""

    BASE = "https://noaa-gefs-pds.s3.amazonaws.com"

    def __init__(self, n_perturbed: int = 30, include_control: bool = True,
                 workers: int = 8):
        import requests
        self.sess = requests.Session()
        self.members_list = ([0] if include_control else []) + \
                            list(range(1, n_perturbed + 1))
        self.n_members = len(self.members_list)
        self.workers = workers
        self.lat = self.lon = None

    def _url(self, init: datetime, mem: int, lead: int) -> str:
        hh = f"{init:%H}"
        name = "gec00" if mem == 0 else f"gep{mem:02d}"
        return (f"{self.BASE}/gefs.{init:%Y%m%d}/{hh}/atmos/pgrb2ap5/"
                f"{name}.t{hh}z.pgrb2a.0p50.f{lead:03d}")

    def _fetch(self, url: str, retries: int = 3) -> bytes:
        import time
        last = None
        for a in range(retries):
            try:
                r = self.sess.get(url + ".idx", timeout=60)
                r.raise_for_status()
                lines = [l for l in r.text.splitlines() if l.strip()]
                for i, l in enumerate(lines):
                    p = l.split(":")
                    if p[3] == "HGT" and p[4].strip() == "500 mb":
                        start = int(p[1])
                        end = int(lines[i + 1].split(":")[1]) - 1 \
                            if i + 1 < len(lines) else ""
                        rr = self.sess.get(url, headers={"Range": f"bytes={start}-{end}"},
                                           timeout=120)
                        rr.raise_for_status()
                        return rr.content
                raise RuntimeError(f"HGT:500 mb not in {url}.idx")
            except Exception as e:                     # noqa: BLE001
                last = e
                time.sleep(2 ** a)
        raise RuntimeError(f"failed: {url}: {last}")

    def _grid(self, lat, lon):
        if self.lat is None:
            self.lat, self.lon = lat, lon

    @property
    def grid(self):
        if self.lat is None:                       # probe: nominal 0.5 deg
            self.lat = np.linspace(-90, 90, 361)
            self.lon = np.arange(720) * 0.5
        return self.lat, self.lon

    def _read(self, url: str) -> np.ndarray:
        lat, lon, v = decode_grib_message(self._fetch(url))
        self._grid(lat, lon)
        return v

    def members(self, init: datetime, lead: int) -> np.ndarray:
        urls = [self._url(init, m, lead) for m in self.members_list]
        with ThreadPoolExecutor(self.workers) as ex:
            raws = list(ex.map(self._fetch, urls))
        out = []
        for raw in raws:
            lat, lon, v = decode_grib_message(raw)
            self._grid(lat, lon)
            out.append(v)
        return np.stack(out)

    def analysis(self, valid: datetime) -> Optional[np.ndarray]:
        """Control-run f000 of the cycle valid at `valid` (None if unavailable)."""
        if valid.hour % 6:
            return None
        try:
            return self._read(self._url(valid, 0, 0))
        except Exception as e:                         # noqa: BLE001
            log.warning("analysis %s unavailable: %s", valid, e)
            return None


class SyntheticSource:
    """Toy ensemble with flow-dependent spread, for testing and demos only."""

    def __init__(self, n_members: int = 31, dlat: float = 2.5, tscale: float = 1.0):
        from scipy.stats import norm            # noqa: F401  (used in climatology)
        self.n_members = n_members
        self.tscale = tscale                    # >1 => slower error growth (tighter short-lead spread)
        self.lat = np.arange(-90, 90 + 1e-6, dlat)
        self.lon = np.arange(0, 360, dlat)
        ny, nx = len(self.lat), len(self.lon)
        ky = np.fft.fftfreq(ny)[:, None]
        kx = np.fft.rfftfreq(nx)[None, :]
        scale = 6.0                                  # grid points (~15 deg)
        self._h = np.exp(-0.5 * (kx ** 2 + ky ** 2) * (2 * np.pi * scale) ** 2)
        self._sd = self._smooth(np.random.default_rng(0), 8).std()
        self.sd_lat = (45 + 0.9 * np.abs(self.lat))[:, None]        # climate sd (m)

    # -- helpers
    def _smooth(self, rng, n):
        ny, nx = len(self.lat), len(self.lon)
        w = rng.standard_normal((n, ny, nx))
        f = np.fft.irfft2(np.fft.rfft2(w) * self._h, s=(ny, nx))
        return f if not hasattr(self, "_sd") else f / self._sd

    def clim_mean(self, d: datetime):
        doy = d.timetuple().tm_yday
        seas = -90 * np.cos(2 * np.pi * (doy - 20) / 365.0)   # lower in winter (NH)
        return (5900 - 6.0 * np.maximum(np.abs(self.lat) - 20, 0))[:, None] \
            + seas * np.sign(self.lat + 1e-9)[:, None] * 0.5

    def climatology(self) -> Climatology:
        from scipy.stats import norm
        z = norm.ppf(DECILES)
        edges = np.empty((12, N_EDGES, len(self.lat), len(self.lon)), np.float32)
        for m in range(12):
            mu = self.clim_mean(datetime(2001, m + 1, 16))
            edges[m] = (mu[None] + z[:, None, None] * self.sd_lat[None]) \
                * np.ones((1, 1, len(self.lon)))
        return Climatology(self.lat, self.lon, edges, "synthetic gaussian climate")

    def _truth_anom(self, valid: datetime):
        rng = np.random.default_rng([int(valid.timestamp()) % (2 ** 31), 99])
        return self._smooth(rng, 1)[0]

    def members(self, init: datetime, lead: int) -> np.ndarray:
        valid = init + timedelta(hours=lead)
        rho = np.exp(-lead / (144.0 * self.tscale))
        e = np.sqrt(1 - rho ** 2)
        rng = np.random.default_rng([int(init.timestamp()) % (2 ** 31), lead, 1])
        # flow-dependent spread factor g, normalised so E[g^2] = 1
        g = np.clip(np.exp(0.4 * self._smooth(rng, 1)[0] - 0.16), 0.3, 2.5)
        b = self._smooth(rng, 1)[0]
        # ensemble-mean anomaly: corr(M, truth) = rho and var(M) = rho^2, as for a
        # statistically consistent ensemble; members scatter around M with sd e*g.
        M = rho ** 2 * self._truth_anom(valid) + rho * e * g * b
        eps = self._smooth(rng, self.n_members)
        anom = M[None] + e * g[None] * eps
        mu = self.clim_mean(valid)
        return (mu[None] + self.sd_lat[None] * anom).astype(np.float32)

    def analysis(self, valid: datetime):
        return (self.clim_mean(valid) + self.sd_lat * self._truth_anom(valid)
                ).astype(np.float32)

    @property
    def grid(self):
        return self.lat, self.lon


# ======================================================================
# 5. Case computation, caching, history
# ======================================================================
class Engine:
    def __init__(self, source, clim: Climatology, lat_min=20.0, lat_max=80.0,
                 stride=1, cache_dir: Optional[str] = None, bins: str = "fixed"):
        lat, lon = source.grid
        self.src = source
        self.bins = bins
        self.geom = Geom(lat, lon, lat_min, lat_max, stride)
        self.cog = ClimoOnGrid(clim, self.geom.lat, self.geom.lon)
        self.cache_dir = cache_dir
        self.sig = hashlib.md5(
            f"{lat_min}|{lat_max}|{stride}|{source.n_members}|{clim.signature()}"
            f"{'|sliding' if bins == 'sliding' else ''}"          # fixed keeps the old cache names
            .encode()).hexdigest()[:10]
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    # ---- single forecast
    def current(self, init: datetime, lead: int, spread: bool = False):
        mem = self.geom.apply(self.src.members(init, lead))
        edges = self.cog.edges(init + timedelta(hours=lead))
        return supporting_members(mem, edges, with_spread=spread, bins=self.bins)

    def _cpath(self, init, lead):
        return os.path.join(self.cache_dir, f"{init:%Y%m%d%H}_f{lead:03d}_{self.sig}.npz")

    def historical(self, init: datetime, lead: int, verify: bool,
                   latest: datetime) -> Optional[Case]:
        path = self._cpath(init, lead) if self.cache_dir else None
        case, dirty = None, False
        if path and os.path.exists(path):
            z = np.load(path)
            case = Case(init, z["counts"], z["mbin"], z["hit"])
        if case is None:
            _, mbin, cnt = self.current(init, lead)
            case = Case(init, cnt, mbin, np.full(cnt.shape, -1, np.int8))
            dirty = True
        valid = init + timedelta(hours=lead)
        if verify and valid <= latest and not (case.hit >= 0).any():
            an = self.src.analysis(valid)
            if an is not None:
                an = self.geom.apply(an)
                case.hit = is_hit(an, self.cog.edges(valid), case.mbin, self.bins)
                dirty = True
        if path and dirty:                         # only rewrite a cache file when something changed
            done = bool((case.hit >= 0).any())     # verified: the mean/bin reference is never needed again
            np.savez_compressed(path, counts=case.counts, hit=case.hit,
                                mbin=np.zeros(0, case.mbin.dtype) if done else case.mbin)
        return case

    def history(self, init: datetime, lead: int, days: int, verify=True,
                weighting="recursive", ties="spread", pool="domain", area_weight=False):
        cases = []
        w = day_weights(days, weighting)
        for k in range(1, days + 1):
            t = init - timedelta(days=k)
            try:
                c = self.historical(t, lead, verify, latest=init)
            except Exception as e:                    # noqa: BLE001
                log.warning("skipping %s: %s", t, e)
                continue
            c.weight = float(w[k - 1])
            cases.append(c)
            log.info("history %s f%03d ok (%d/%d)", f"{t:%Y-%m-%d %HZ}", lead,
                     len(cases), days)
        return History(cases, self.src.n_members,
                       self.geom.area_weights() if area_weight else None, ties, pool)


# ======================================================================
# 6. Plot - styled after the NCEP/EMC GrADS "RMOP" graphic
# ======================================================================
# Colours sampled from the reference graphic (class 0-10% ... 90-100%).
RMOP_COLORS = [(80, 165, 245), (120, 185, 250), (150, 210, 250), (180, 240, 250),
               (225, 255, 255), (255, 250, 170), (255, 232, 120), (255, 192, 60),
               (255, 160, 0), (255, 96, 0)]
TXT_BLUE = "#1f3dfa"
GRID_GREY = "#7d7d7d"
IMG_W, IMG_H = 695, 545                 # reference image size in pixels
FRAME = (31.0, 91.0, 662.5, 445.5)      # map frame x0, y0, x1, y1 in reference pixels
BAR_EDGES = [231.5 + 44.4 * j for j in range(9)]   # x of the 10 ... 90 ticks
BAR_TIP_L, BAR_TIP_R, BAR_Y0, BAR_Y1 = 187.0, 632.0, 496.0, 506.0


def lon_label(l: float) -> str:
    v = ((l + 180.0) % 360.0) - 180.0
    if abs(abs(v) - 180.0) < 1e-6:
        return "180"
    if abs(v) < 1e-6:
        return "0"
    return f"{abs(v):g}{'E' if v > 0 else 'W'}"


def lat_label(l: float) -> str:
    return "EQ" if abs(l) < 1e-6 else f"{abs(l):g}{'N' if l > 0 else 'S'}"


def split_at_wrap(xy: np.ndarray):
    """Split a lon/lat polyline (lon already in 0..360) where it jumps across the seam."""
    brk = np.where(np.abs(np.diff(xy[:, 0])) > 180)[0] + 1
    return [p for p in np.split(xy, brk) if len(p) > 1]


def coast_segments():
    """Natural Earth 110m coastlines + land borders as lon(0-360)/lat polylines.
    Needs cartopy and (first time) network access; raises on failure."""
    import cartopy.io.shapereader as shp
    segs = []
    for cat, name in (("physical", "coastline"), ("cultural", "admin_0_boundary_lines_land")):
        path = shp.natural_earth(resolution="110m", category=cat, name=name)
        for geom in shp.Reader(path).geometries():
            for ls in getattr(geom, "geoms", [geom]):
                xy = np.asarray(ls.coords)[:, :2].copy()
                xy[:, 0] = np.mod(xy[:, 0], 360.0)
                segs += split_at_wrap(xy)
    return segs


def plot_rmop_mpl(path, lat, lon, pct, mean_z, init, valid, lead, hit_rates,
                  plot_lon=(170.0, 300.0), plot_lat=(20.0, 80.0), coast=True, dpi=200,
                  credit=""):
    """RMOP map laid out like the 2001 EMC graphic: three-line title, plain lat/lon
    map with dotted grid and thick black height contours (metres, 40 m interval),
    shading in ten 10% classes (blue = low ... orange = high), a 'Probability (%)'
    row and a triangle-ended colour bar underneath.  Layout is in reference-image
    pixels (695 x 545) and scales with --dpi."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Polygon

    lo0, lo1 = plot_lon
    la0, la1 = plot_lat
    if not (0 <= lo0 < lo1 <= 360):
        raise ValueError("--plot-lon must satisfy 0 <= lon0 < lon1 <= 360 (degrees east)")
    jm = np.where((lon >= lo0 - 1e-6) & (lon <= lo1 + 1e-6))[0]
    im = np.where((lat >= la0 - 1e-6) & (lat <= la1 + 1e-6))[0]
    if len(jm) < 2 or len(im) < 2:
        raise ValueError("plot sector does not overlap the computed domain")
    lo, la = lon[jm], lat[im]
    sel = np.ix_(im, jm)
    pc, zc = pct[sel], mean_z[sel]

    cols = np.array(RMOP_COLORS) / 255.0
    fig = plt.figure(figsize=(IMG_W / 100, IMG_H / 100), dpi=100)
    fx0, fy0, fx1, fy1 = FRAME
    ax = fig.add_axes([fx0 / IMG_W, 1 - fy1 / IMG_H, (fx1 - fx0) / IMG_W, (fy1 - fy0) / IMG_H])
    ax.set_xlim(lo0, lo1)
    ax.set_ylim(la0, la1)

    levels = np.linspace(0, 100, N_CLASSES + 1)
    ax.contourf(lo, la, np.clip(pc, 0, 100 - 1e-6), levels=levels,
                colors=cols, zorder=1)

    xt = np.arange(np.ceil(lo0 / (10 if lo1 - lo0 <= 180 else 30)) * (10 if lo1 - lo0 <= 180 else 30),
                   lo1 + 1e-6, 10 if lo1 - lo0 <= 180 else 30)
    yst = 5 if la1 - la0 <= 60 else 10
    yt = np.arange(np.ceil(la0 / yst) * yst, la1 + 1e-6, yst)
    for x in xt:
        ax.axvline(x, color=GRID_GREY, lw=0.5, ls=(0, (1, 3)), zorder=2)
    for y in yt:
        ax.axhline(y, color=GRID_GREY, lw=0.5, ls=(0, (1, 3)), zorder=2)

    if coast:
        try:
            segs = coast_segments()
            ax.add_collection(LineCollection(segs, colors="k", linewidths=0.45, zorder=3))
        except Exception as e:                            # noqa: BLE001
            log.warning("coastlines unavailable (%s); drawing without them", e)

    zmin, zmax = np.nanmin(zc), np.nanmax(zc)
    zl = np.arange(np.floor(zmin / 40.0) * 40.0, zmax + 40.0, 40.0)
    cs = ax.contour(lo, la, zc, levels=zl, colors="k", linewidths=1.5, zorder=4)
    for t in ax.clabel(cs, fmt="%d", fontsize=6.5, inline=True, inline_spacing=2):
        t.set_fontweight("bold")
        t.set_bbox(dict(facecolor="white", edgecolor="none", pad=0.6))
        t.set_zorder(5)

    ax.set_xticks(xt)
    ax.set_xticklabels([lon_label(x) for x in xt], fontsize=6.5, fontweight="bold")
    ax.set_yticks(yt)
    ax.set_yticklabels([lat_label(y) for y in yt], fontsize=6.5, fontweight="bold")
    ax.tick_params(length=3, width=1.0, pad=1.5)
    for sp in ax.spines.values():
        sp.set_linewidth(1.5)
        sp.set_zorder(6)

    # ---- overlay in reference-pixel coordinates (y down)
    ov = fig.add_axes([0, 0, 1, 1], facecolor="none")
    ov.set_xlim(0, IMG_W)
    ov.set_ylim(IMG_H, 0)
    ov.axis("off")
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()

    def put(x, y, s, width=None, ha="center", size=8, color="k", weight="normal"):
        t = ov.text(x, y, s, ha=ha, va="center", fontsize=size, color=color,
                    fontweight=weight)
        if width:                                         # fit to measured pixel width
            w = t.get_window_extent(rend).width
            t.set_fontsize(size * width / w)
        return t

    put(IMG_W / 2, 25.5, "Relative measure of predictability (colors)", 465, size=14)
    put(IMG_W / 2, 46.0, "for ensemble mean forecast (contours) of 500 hPa height", 663, size=14)
    put(IMG_W / 2, 69.0, f"ini: {init:%Y%m%d%H} valid: {valid:%Y%m%d%H} fcst: {lead} hours",
        428, size=10, weight="bold")

    cx = [(BAR_TIP_L + BAR_EDGES[0]) / 2] + \
         [(BAR_EDGES[j] + BAR_EDGES[j + 1]) / 2 for j in range(8)] + \
         [(BAR_EDGES[-1] + BAR_TIP_R) / 2]
    e = BAR_EDGES
    polys = [[(BAR_TIP_L, 501), (e[0], BAR_Y0), (e[0], BAR_Y1)]]
    polys += [[(e[j], BAR_Y0), (e[j + 1], BAR_Y0), (e[j + 1], BAR_Y1), (e[j], BAR_Y1)]
              for j in range(8)]
    polys += [[(e[-1], BAR_Y0), (BAR_TIP_R, 501), (e[-1], BAR_Y1)]]
    for pgon, c in zip(polys, cols):
        ov.add_patch(Polygon(pgon, closed=True, facecolor=c, edgecolor="k", lw=0.7))
    for j, x in enumerate(e):
        put(x, 511.5, f"{(j + 1) * 10}", size=6.5, weight="bold")
    put(14, 510.5, "Measure of predictability (%)", 173, ha="left", size=7, weight="bold")
    put(57, 487.5, "Probability (%)", 86, ha="left", size=8, color=TXT_BLUE, weight="bold")
    for k in range(N_CLASSES):
        v = np.nan if hit_rates is None else hit_rates[k]
        put(cx[k], 487.5, "--" if not np.isfinite(v) else f"{v:.0f}", size=9,
            color=TXT_BLUE, weight="bold")
    put(680, 537.5, credit, 150, ha="right", size=6, color=TXT_BLUE, weight="bold")

    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    log.info("wrote %s", path)


# ======================================================================
# 6b. Pixel-level GrADS-style renderer (default)
# ======================================================================
# The reference RMOP graphics were made with GrADS: a 695 x 545 non-antialiased GIF
# using GrADS's default stroke font (Hershey Roman Simplex), libgd thick lines, a
# 14-entry palette, a dotted grid and a fixed colour-bar graphic.  Everything below
# was measured from, and fitted to, the pixels of a real product (init 2001061200,
# f168): text is Hershey Roman Simplex at 2-3 px stroke thickness (fits within 1 px
# on 98-100 % of the text pixels), lines use libgd's gdImageLine thick-line algorithm,
# and the static pieces (dot pattern, colour-bar graphic) are reproduced exactly.
#
# scale = 1 reproduces the original 695 x 545 pixels.  scale = S > 1 is a true
# re-render at S times the resolution (same layout, strokes S times as thick, smooth
# diagonals and class boundaries), NOT a pixel enlargement.  The original page is
# 11 x 8.5 in at ~63 px/in, so scale 5 at 300 DPI is the page at print size.
#
# The Hershey Fonts were originally created by Dr. A. V. Hershey while working at the
# U. S. National Bureau of Standards.  The format of the Font data in this distribution
# was originally created by James Hurt, Cognition, Inc., 900 Technology Park Drive,
# Billerica, MA 01821.  (Public-domain distribution; glyph data embedded below.)
_ASSET_B85 = """
c-qZc2b&x@5&bK$Wax%g?lw8+oLMh-N3snEHa6)n=C|`+1+{v1>^pP&-TJdVU79YHs!FedG=6$<@6ER!-
G6d%=jp}!F78~;>$FanS69=;`#CC8nJzDvX%2B(rg$}hZWw`SnYKxv?TY+&FZ%S$Im-+_3g+lY2ITwz8N)J#0M>wab(-
rGDma^r79HTrIYHi>IA~MC(n%zI&?PKfEUJh&1v2W$29USGgc(akRmmRu-7N3<*{A})EuwYe-
$X+Tw1IMyE?+TTA!?(bjod5eR?+H1q7_z-&}!3lwrQjCd0%{y$0}>|C7XP1uFxh7RIqkO`us2jk&@-
J$v3hX)+tbpddG=W1J4GPw~4!=RK~aRRy6*wXq@0H?!-c!_?f$Dg$v|KyqiTZcV{*0DNuGC3O7-
w$Y<ggfv<=h3Royg{2lmiQRCunTyO`dGqU{Udk9mfH!Q|Kf$IlH-
A9fn)%74vAsgIZvZMfa*1h^D=QRku6m(XBM;;VwvO$rb;`lZRU?DTZ6G5eXY?m@;N8v6<#E29M=`v;!GIuE--vh@ZZ6Roifj7y{n
sYP&J~3|GqV@^Ugurw%O}lSAU@dZ+L~fuP7s3@^2NgVzf)CO@IePjzIp9+=J7{EY>X6ONTn<4S4trk(@al7VY382eC8I#rAaYkhL
KFRG!UA)0kBF#SufPt6wX#T$hDyQ3CT#J*Ry_@@nje;~o&q&`8VkXIF1)pCD+yyJP@64~4hs|!P;HNOLY5uY2y2#SD*j~IphKRil
O!rlqIr~@)P9EcDp^{;NTx*@dCQ&v<D@Cpl3uhREmQ76|1v7oahZCWc@^4R&4PXP)g}1KWZ8Z>?02=qt8SYlHG4+!JT10}7*V5*Y
BXPU(;C82>o8fb)sek4hyjKPP13SkflqN(6(CYv5jD?L26vBc&0I}ug=*^CFSfI|oGn(O>2~5<2BPE@iC_s}^GfyfRUFQ0i9tCGn
~@yKtPP?CGkBSN8xEFe5Z^iE+p?$keOt<lx`a=^v-
$5hUwvk*xvN!9RtQD@XHU`@fnfc_#_+Iy;&Y>vS;R=~axtIhdlsLY$PxluDO7LMQGLOc&l<KewQ5z-
L}NWfc7%EaXRSI!YnFxp6m9O4dcYf?m0jyh>+>|^Gu<911y2axBv|p^>wZV=wbUNOmFia?CwWNeN_<r;T6)eW_O$LdW&-F|HI{-
P6uOC%zPN9VWtdfBT?b0fW$1an<TXlSzIo;x@`IY8CROjU#8T!X&)5u4n-
jG~5HvqGD|kux^4R@K(Hol7bnsvr8>8xdYPL42N=NN!Q+$@X&nh=g*89ra+Iub;G~Ej>#C5n#&kAnISIrU+3#})cb?!aU>M(YP)u
BFq%|xs(A%gS1{@yu}iAO4`q##-S8)KVubg5UEtwW=^&fnxL#_-
<esT01n_kVs;%2xEoJe+$&UM&&G>~BARfBQf@D86vVf5#ly&N*I=3c~i?)5N$3n2ogWyiR-S{ND3@HIL5PY0mooVOBg%DE-
W~PTJfBO<NwIc0h@THbUE%RblPrHlHuIKX?~O!q8L<u$prfI6u5avshG#o}jAy=myiE=&Yq|-
uLjwCw<s^cM$){^#j$}_V1_Xv6)7PhqLXdOZeH)7@Us>a{c@eMPYHv_d0bLC>{0zW&GmMs$P$ro|S)jMy2^wFH1?kI@2n*>%NlVp
0qEAQ*Y8kUa}jT{VGO=`c_WHI+T%}2k+h;<oI&t*Q@<izt5?5u3;rodD%v@5B(iQ{Cc>K^4@!sF5J%6Njp3{U@zJ@s)er&b63J+^
{{z;JX3q|)V6xFSeM?&sa0mZp?k77@0x1uS_O(n?}h!CVclHS8kJ6K)Q|#Xf#1Ba6vyF0#ZGvB^oSBuT)|6xI2IgV;kP@iZcgpr9
sMGQ<C}HN?+=}OFl!~9=sE7)FYyP3MC)hkG~7LPSZOqr{TB4zbxcVArT0rk5HB6KhF7^!x7;5VpUaqnOm6JAcF7K#S^egn!Gpb<n
a4=j>EstOlOFXvxc+#0u^fi75M<)gE0*9@1cst{Ww<!|lUlJ}Gon}pDSGJHwSMz?J=mWmfC^W+y!3{VH>)B+16-?eb*B%9*I-
+`=!K=9>ZM=P17bft>2iNL@qc+)_dgs3{9ib*N*1PbsFF1vog|ggujy|psr>a!4AdJ1`eQ95)oNAgS3z>$5VbP;I>jx2qWQlkyK<
^Z9#qL%V1wPe>wR?{JCrcA@2b|{_MG83EYij4;p0rY`1?7;l%Xrsql?^;ss~F4%Rh#4Ry`Dr#{=M>hhAtkD>o0`k<Q*qD9PFX+8s
8RLxD47lj*G&tD(Q(NU#{v<cy{c`}bi-4&fpIC_30rE>BN82>My~FXY~R#{QeZs=p9e#+fa7Sxno!H2k_?KX635k;w+9B-
e3lp7W$(&@Zwq1qeq^Ohk?Lwsr|bCn%ounk#GSi|&a395a~xSjIQchIVQ+SuNf#r<Q3q*WeuN{%yBc%^j?@1Q@41_JiL*&Owg>b}
c$wW=TQVP1f?T_w#zM7wv+HG355n-QfQon&)l!acS=N*V%upp@qR*A3g5`z3$UFVKBe|VB3`}SM4Fo%fV-
Q+)uC+XD#rdYrw4lq7<nPaKoVl07N#$$pjZwSQIh<a`YSs4hAgSYnB1lcY1_RHo7#4H<SDa_$2UV@Mabv@D=bC;-
8*iR|{(zo(;SWyeoKD@NRf+;NQT%;i$jBnuSkT0?$Y8eL!N2cqaHc!QKS#WISO{%;1GZv4FSm3Em1`m=zoN8+cdn!nC-
87ska6yc?g`js>2BeTl#O1wP@$h&_gVDWb;EbFeWbBsbWYGWg-XRKQ=r4?9x@KkQ5u{EeOle*-
`4Oe^@|%>+Bs27cI?7wqxEC$@z5C9pFG><avuVP}pg0<bkFWGL8~Gk9TVF5rco8Fps)G{eqZ!3#Tc1224<VQXH&4_osF{*6x@6Bb
l6*jg5BJ$zbVYl%qPu(d=aUHG-
Y)&iIm*jfOTLYNRIu(JRu1$LIkf8YaJ3T!OEOIi5@??y}kPQXlE_(U1u>6?u)FjL{v3L9%gnum=QxT(NQ1!gL2tiVl$jkPp8n{N3
G0HXmAV1l3kK?=ZP0Av(a17TDU!eI(R_FmwJvjQ9y;HUsc1vn}Y5O7k2lOmiHnYDOwgM(t^6X1b^BES&gpa>vDfFQy-5g>?gP6P-
dKo9};2oS`^Cu9Kx5nzu1K?K+%K#%}}1h6N7AOY+NAV}Dl1hA(Fpa1{~o00(bgiT2Rd%~t9fIR^O2^*6D_JoZ|0DD3ZCu~l_#w36
}VPg_DM!^T5z>gVz6yU*+3OonhYM{#ox`n{wvFY){C+sFJC0zNlE{GVMNShKlG^R({j^I>ToLZgInL^1_p=zhL)u<FZRXVl$8nzk
?HOZh%GAWhZn(;+AQ?(LGv=VA{O6LkCbA_s%60Jz7Rzj6dt-gk>#{8y|TXxB$R5B>#ON6iRky1XPgpZW*wcvc9jE|J^wP1WX`Ua6
wKB0t<l<`&H4^qZgeLqMkUkUDAQpzWk@R2e;z)rm@lz3OD)hU}xO6QWQof7YoQtt{?I<@*5x>^nH-
5T7xLWyKj<YR;Sh)1PMLY`n?es20(Rn&3F@_%h|wxL|k<g}0S;axertA}^(@NPZ4+Yax7c8pAkqD*m4Z7$l*c?+6_byO|2!B0VtH
Mgf*Y@=>rJLne6t?HKQVN30uzz)g!R#nTUjeL7}c^k?&gu<)jnt0f?R}fGqV1K0o(KUSq^H)!;uR@;pqT@vft(DfO*Ph&e;}P$8c
>>6*efjY5gL@Bn#TKNAFIUh02N@X(rT
"""
_A_CACHE = {}


def _assets():
    if not _A_CACHE:
        raw = zlib.decompress(base64.b85decode("".join(_ASSET_B85.split())))
        _A_CACHE.update(json.loads(raw))
    return _A_CACHE


# palette indices, exactly as in the reference GIF
G_WHITE, G_BLACK, G_GREY, G_BLUE = 0, 11, 12, 13
G_PALETTE = [(255, 255, 255)] + RMOP_COLORS + [(0, 0, 0), (125, 125, 125), (31, 61, 250)]
G_W, G_H = 695, 545
G_FX0, G_FY0, G_FX1, G_FY1 = 31, 91, 663, 446            # map frame (pixel coordinates)
G_XORG, G_XSPAN = 31.05, 631.7                           # lon -> x mapping (reproduces the ticks)
G_YORG, G_YSPAN = 91.2, 355.2                            # lat -> y mapping
_ri = lambda v: int(math.floor(v + 0.5))


# ---- libgd thick lines ------------------------------------------------
def _vstroke(img, x, ws, wid, c):
    h, w = img.shape
    if 0 <= x < w:
        a, b = max(ws, 0), min(ws + wid, h)
        if a < b:
            img[a:b, x] = c


def _hstroke(img, y, ws, wid, c):
    h, w = img.shape
    if 0 <= y < h:
        a, b = max(ws, 0), min(ws + wid, w)
        if a < b:
            img[y, a:b] = c


def _gd_hline(img, y, x1, x2, c, thick):
    if x1 > x2:
        x1, x2 = x2, x1
    half = thick >> 1
    h, w = img.shape
    img[max(y - half, 0):max(min(y + thick - half, h), 0), max(x1, 0):max(min(x2 + 1, w), 0)] = c


def _gd_vline(img, x, y1, y2, c, thick):
    if y1 > y2:
        y1, y2 = y2, y1
    half = thick >> 1
    h, w = img.shape
    img[max(y1, 0):max(min(y2 + 1, h), 0), max(x - half, 0):max(min(x + thick - half, w), 0)] = c


def _gd_line(img, x1, y1, x2, y2, c, thick=1):
    """Port of libgd gdImageLine (Bresenham with a thick stroke)."""
    dx, dy = abs(x2 - x1), abs(y2 - y1)
    if dx == 0 and dy == 0:
        _gd_hline(img, y1, x1, x1, c, thick)
        return
    if dx == 0:
        _gd_vline(img, x1, y1, y2, c, thick)
        return
    if dy == 0:
        _gd_hline(img, y1, x1, x2, c, thick)
        return
    if dy <= dx:                                          # more horizontal: vertical stroke
        wid = max(int(thick / math.cos(math.atan2(dy, dx))), 1)
        d, i1, i2 = 2 * dy - dx, 2 * dy, 2 * (dy - dx)
        if x1 > x2:
            x, y, ydir, xend = x2, y2, -1, x1
        else:
            x, y, ydir, xend = x1, y1, 1, x2
        up = ((y2 - y1) * ydir) > 0
        _vstroke(img, x, y - wid // 2, wid, c)
        while x < xend:
            x += 1
            if d < 0:
                d += i1
            else:
                y += 1 if up else -1
                d += i2
            _vstroke(img, x, y - wid // 2, wid, c)
    else:                                                 # more vertical: horizontal stroke
        wid = max(int(thick / math.sin(math.atan2(dy, dx))), 1)
        d, i1, i2 = 2 * dx - dy, 2 * dx, 2 * (dx - dy)
        if y1 > y2:
            y, x, xdir, yend = y2, x2, -1, y1
        else:
            y, x, xdir, yend = y1, x1, 1, y2
        right = ((x2 - x1) * xdir) > 0
        _hstroke(img, y, x - wid // 2, wid, c)
        while y < yend:
            y += 1
            if d < 0:
                d += i1
            else:
                x += 1 if right else -1
                d += i2
            _hstroke(img, y, x - wid // 2, wid, c)


def _emit(img, S, x1, y1, x2, y2, col, T):
    """Draw a line given in ORIGINAL-pixel float coordinates at scale S.  At S = 1 the
    vertices are rounded like GrADS; at S > 1 the same line is re-drawn S times larger
    with its (asymmetric libgd) centre line preserved."""
    if S == 1:
        _gd_line(img, _ri(x1), _ri(y1), _ri(x2), _ri(y2), col, T)
        return
    wid = T * S
    sh = 0.5 if T % 2 else 0.0
    off = (wid % 2) * 0.5
    f = lambda c: int(round((c + sh) * S - off))
    _gd_line(img, f(x1), f(y1), f(x2), f(y2), col, wid)


def _line_e(img, S, x1, y1, x2, y2, col, thick):
    """Line in continuous EDGE coordinates (pixel p spans [p, p+1)) of the original grid."""
    wid = max(int(round(thick * S)), 1)
    off = (wid % 2) * 0.5
    f = lambda c: int(round(c * S - off))
    _gd_line(img, f(x1), f(y1), f(x2), f(y2), col, wid)


def _blk(img, S, x0, x1, y0, y1, c):
    """Fill the inclusive original-pixel box [x0..x1] x [y0..y1] at scale S."""
    img[max(y0 * S, 0):min((y1 + 1) * S, img.shape[0]),
        max(x0 * S, 0):min((x1 + 1) * S, img.shape[1])] = c


def _fill_tri(img, S, pts, c):
    """Scanline fill of a triangle given in edge coordinates of the original grid."""
    P = [(x * S, y * S) for x, y in pts]
    ys = [p[1] for p in P]
    for row in range(max(int(math.floor(min(ys))), 0), min(int(math.ceil(max(ys))), img.shape[0] - 1) + 1):
        yc, xs = row + 0.5, []
        for (a, b), (p, q) in zip(P, P[1:] + P[:1]):
            if (b <= yc < q) or (q <= yc < b):
                xs.append(a + (p - a) * (yc - b) / (q - b))
        if len(xs) >= 2:
            img[row, max(_ri(min(xs)), 0):max(_ri(max(xs)), 0)] = c


def _clip_seg(x1, y1, x2, y2, r):
    """Liang-Barsky clip of a float segment to rectangle r=(xa, ya, xb, yb)."""
    xa, ya, xb, yb = r
    t0, t1, dx, dy = 0.0, 1.0, x2 - x1, y2 - y1
    for p, q in ((-dx, x1 - xa), (dx, xb - x1), (-dy, y1 - ya), (dy, yb - y1)):
        if p == 0:
            if q < 0:
                return None
        else:
            t = q / p
            if p < 0:
                if t > t1:
                    return None
                t0 = max(t0, t)
            else:
                if t < t0:
                    return None
                t1 = min(t1, t)
    return x1 + t0 * dx, y1 + t0 * dy, x1 + t1 * dx, y1 + t1 * dy


def _gd_poly(img, pts, c, thick, rect, S=1):
    """Draw a float-pixel polyline clipped to rect (original-pixel coordinates)."""
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        s = _clip_seg(xa, ya, xb, yb, rect)
        if s is not None:
            _emit(img, S, s[0], s[1], s[2], s[3], c, thick)


# ---- Hershey Roman Simplex text --------------------------------------
def _text_segments(text):
    G, cum, segs = _assets()["glyphs"], 0, []
    for ch in text:
        g = G.get(ch)
        if g is None:
            continue
        left, right, strokes = g
        for st in strokes:
            for i in range(0, len(st) - 2, 2):
                segs.append((cum + st[i] - left, st[i + 1], cum + st[i + 2] - left, st[i + 3]))
        cum += right - left
    return segs


def _draw_text(img, text, sx, sy, t, col, y0, x0=None, cx=None, rx=None, S=1):
    """Draw text; sx, sy are pixels per Hershey unit, y0 the baseline row.  Anchor with
    x0 (left), cx (centre) or rx (right) of the stroke extents.  Returns (left, right)."""
    segs = _text_segments(text)
    if not segs:
        return 0.0, 0.0
    xs = [s[0] for s in segs] + [s[2] for s in segs]
    xmin, w = min(xs), sx * (max(xs) - min(xs))
    if x0 is None:
        x0 = cx - w / 2.0 if cx is not None else rx - w
    for ax, ay, bx, by in segs:
        _emit(img, S, x0 + sx * (ax - xmin), y0 + sy * (ay - 9),
              x0 + sx * (bx - xmin), y0 + sy * (by - 9), col, t)
    return x0, x0 + w


def _text_bbox(text, sx, sy, t, y0, x0):
    """Pixel bounding box (x0, y0, x1, y1) a text will occupy at scale 1."""
    ix, iy = math.floor(x0), math.floor(y0)
    scratch = np.full((60, 220), G_WHITE, np.uint8)
    _draw_text(scratch, text, sx, sy, t, G_BLACK, 30 + (y0 - iy), x0=20 + (x0 - ix))
    ys, xs = np.where(scratch == G_BLACK)
    return xs.min() + ix - 20, ys.min() + iy - 30, xs.max() + ix - 20, ys.max() + iy - 30


# ---- text styles fitted to the reference -----------------------------------
_ST_TITLE = dict(sx=0.7055, sy=0.626, t=2)               # title lines 1-2 (one GrADS size)
_ST_TITLE3 = dict(sx=0.513, sy=0.426, t=3)               # bold third line
_ST_AXIS_X = dict(sx=0.282, sy=0.284, t=1)               # lon labels
_ST_AXIS_Y = dict(sx=0.2765, sy=0.287, t=1)              # lat labels
_ST_TICK = dict(sx=0.30, sy=0.3075, t=2)                 # colour-bar numbers
_ST_PROB = dict(sx=0.386, sy=0.335, t=2)                 # probability numbers
_ST_CLAB = dict(sx=0.318, sy=0.278, t=2)                 # contour labels
_PROB_CX = [211.0, 255.5, 298.5, 344.0, 388.0, 432.5, 476.5, 521.0, 564.5, 608.5]
_CB_E = [_ri(231 + 44.5 * k) for k in range(9)]          # colour-bar divider pixels


def _grads_tick_positions(lo, la):
    span_lon, span_lat = lo[1] - lo[0], la[1] - la[0]
    dstep = 10 if span_lon <= 180 else 30
    lstep = 5 if span_lat <= 60 else 10
    lons = np.arange(np.ceil(lo[0] / dstep) * dstep, lo[1] + 1e-6, dstep)
    lats = np.arange(np.ceil(la[0] / lstep) * lstep, la[1] + 1e-6, lstep)
    X = lambda v: G_XORG + (v - lo[0]) * G_XSPAN / span_lon
    Y = lambda v: G_YORG + (la[1] - v) * G_YSPAN / span_lat
    return lons, lats, X, Y


def _bilinear_classes(lat, lon, pct, lo, la, S=1):
    """Colour index of every pixel of the map frame at scale S (bilinear field, 10% classes)."""
    span_lon, span_lat = lo[1] - lo[0], la[1] - la[0]
    px = np.arange(G_FX0 * S, (G_FX1 + 1) * S)
    py = np.arange(G_FY0 * S, (G_FY1 + 1) * S)
    xi = (px + 0.5) / S - 0.5                             # equivalent original-pixel index
    yi = (py + 0.5) / S - 0.5
    lonq = np.clip(lo[0] + (xi - G_XORG) * span_lon / G_XSPAN, lon[0], lon[-1])
    latq = np.clip(la[1] - (yi - G_YORG) * span_lat / G_YSPAN, lat[0], lat[-1])
    j = np.clip(np.searchsorted(lon, lonq, side="right") - 1, 0, len(lon) - 2)
    wx = ((lonq - lon[j]) / (lon[j + 1] - lon[j]))[None, :].astype(np.float32)
    i = np.clip(np.searchsorted(lat, latq, side="right") - 1, 0, len(lat) - 2)
    wy = ((latq - lat[i]) / (lat[i + 1] - lat[i])).astype(np.float32)
    out = np.empty((len(py), len(px)), np.uint8)
    f = np.asarray(pct, np.float32)
    for a in range(0, len(py), 256):                      # row chunks keep memory modest
        ii = i[a:a + 256, None]
        w = wy[a:a + 256, None]
        v = ((1 - w) * ((1 - wx) * f[ii, j[None, :]] + wx * f[ii, j[None, :] + 1])
             + w * ((1 - wx) * f[ii + 1, j[None, :]] + wx * f[ii + 1, j[None, :] + 1]))
        out[a:a + 256] = np.clip(np.floor(v / 10.0), 0, N_CLASSES - 1).astype(np.uint8) + 1
    return out


def _draw_colorbar(img, S):
    """Colour-bar graphic.  Scale 1: the captured original pixels.  Scale > 1: the same
    shapes drawn as smooth geometry (boxes, 1 px outline, 2 px triangle edges)."""
    A = _assets()
    if S == 1:
        th, tw = A["tpl_shape"]
        flat = np.concatenate([np.full(n, v, np.uint8) for v, n in A["tpl"]])
        ox, oy = A["tpl_origin"]
        img[oy:oy + th, ox:ox + tw] = flat.reshape(th, tw)
        return
    e = [x + 0.5 for x in _CB_E]                          # divider centres (edge coordinates)
    yt, yb, ym = 495.5, 507.5, 501.5
    for k in range(8):
        img[int(round(yt * S)):int(round(yb * S)), int(round(e[k] * S)):int(round(e[k + 1] * S))] = 2 + k
    _fill_tri(img, S, [(187.5, ym), (e[0], yt), (e[0], yb)], 1)
    _fill_tri(img, S, [(634.5, ym), (e[8], yt), (e[8], yb)], 10)
    for y in (yt, yb):
        _line_e(img, S, e[0], y, e[8], y, G_BLACK, 1)
    for x in e:
        _line_e(img, S, x, yt, x, yb, G_BLACK, 1)
    for (xa, ya), (xb, yb2) in (((187.5, ym), (e[0], yt)), ((187.5, ym), (e[0], yb)),
                                ((e[8], yt), (634.5, ym)), ((e[8], yb), (634.5, ym))):
        _line_e(img, S, xa, ya, xb, yb2, G_BLACK, 2)


# ---- high-resolution (scale > 1) antialiased vector path ----------------------
# Effective stroke widths in ORIGINAL pixels for the antialiased output.  The original
# GrADS strokes were 2-3 px wide on glyphs only 5-13 px tall, which clogs small text; these
# are lighter (bold line kept bold) so every size stays crisp.  --stroke scales them all.
_LW = dict(title=1.8, title3=2.5, axis=1.0, tick=1.25, prob=1.4, label=1.5, credit=1.05,
           clab=1.15, contour=1.8, coast=0.9, bar=1.0, bar_edge=1.8)


def _text_strokes(text):
    """Hershey strokes of a string as polylines in glyph units."""
    G, cum, out = _assets()["glyphs"], 0, []
    for ch in text:
        g = G.get(ch)
        if g is None:
            continue
        left, right, strokes = g
        for st in strokes:
            out.append([(cum + st[i] - left, st[i + 1]) for i in range(0, len(st), 2)])
        cum += right - left
    return out


def _smooth_stroke(P, corner_deg=38.0, iters=3):
    """Chaikin corner cutting on the smooth stretches of a polyline; sharp corners and
    end points stay fixed.  Hershey draws curves as coarse polylines, which looks scalloped
    once the text is large and antialiased."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return P
    keep = [0] + [i for i in range(1, len(P)) if math.hypot(*(P[i] - P[i - 1])) > 1e-9]
    P = P[keep]
    if len(P) < 3:
        return P
    v = np.diff(P, axis=0)
    ang = np.arctan2(v[:, 1], v[:, 0])
    turn = np.abs((np.diff(ang) + np.pi) % (2 * np.pi) - np.pi)
    corners = sorted({0, len(P) - 1} | {i + 1 for i, t in enumerate(turn)
                                         if t > math.radians(corner_deg)})
    out = [P[0]]
    for a, b in zip(corners[:-1], corners[1:]):
        Q = P[a:b + 1]
        for _ in range(iters):
            if len(Q) < 3:
                break
            R = [Q[0]]
            for i in range(len(Q) - 1):
                R.append(0.75 * Q[i] + 0.25 * Q[i + 1])
                R.append(0.25 * Q[i] + 0.75 * Q[i + 1])
            R.append(Q[-1])
            Q = np.array(R)
        out.extend(Q[1:])
    return np.array(out)


def _text_polys(text, sx, sy, y0, x0=None, cx=None, rx=None):
    """Smoothed stroke polylines of a text in original-pixel EDGE coordinates (same anchoring
    as _draw_text)."""
    st = _text_strokes(text)
    if not st:
        return []
    xs = [p[0] for s_ in st for p in s_]
    xmin, w = min(xs), sx * (max(xs) - min(xs))
    if x0 is None:
        x0 = cx - w / 2.0 if cx is not None else rx - w
    return [_smooth_stroke([(x0 + sx * (a - xmin) + 0.5, y0 + sy * (b - 9) + 0.5)
                            for a, b in s_]) for s_ in st]


def _clip_polyline(pts, rect):
    """Clip a polyline to rect, returning the visible runs as polylines."""
    runs, cur = [], []
    for (xa, ya), (xb, yb) in zip(pts[:-1], pts[1:]):
        sg = _clip_seg(xa, ya, xb, yb, rect)
        if sg is None:
            if len(cur) > 1:
                runs.append(cur)
            cur = []
            continue
        p0, p1 = (sg[0], sg[1]), (sg[2], sg[3])
        if cur and abs(cur[-1][0] - p0[0]) < 1e-9 and abs(cur[-1][1] - p0[1]) < 1e-9:
            cur.append(p1)
        else:
            if len(cur) > 1:
                runs.append(cur)
            cur = [p0, p1]
    if len(cur) > 1:
        runs.append(cur)
    return runs


def _contour_polys(slon, slat, z, X, Y):
    """40 m height contours as (level, Nx2 array) in original-pixel index coordinates."""
    import contourpy
    z = np.asarray(z, float)
    levels = np.arange(np.floor(np.nanmin(z) / 40.0) * 40.0, np.nanmax(z) + 40.0, 40.0)
    gen = contourpy.contour_generator(slon, slat, z, line_type=contourpy.LineType.Separate)
    return [(lev, np.asarray([(X(a), Y(b)) for a, b in ln]))
            for lev in levels for ln in gen.lines(lev)]


def _plan_labels(polys):
    """Horizontal contour labels: [(text, x0, y0, box)] in original-pixel coordinates."""
    boxes, out, cl = [], [], _ST_CLAB
    for lev, pl in polys:
        seg = np.hypot(np.diff(pl[:, 0]), np.diff(pl[:, 1]))
        s = np.r_[0.0, np.cumsum(seg)]
        L = s[-1]
        if L < 45:
            continue
        txt = str(int(round(lev)))
        n = max(1, int(L // 330))
        for q in range(n):
            k = min(np.searchsorted(s, (q + 0.5) * L / n), len(pl) - 1)
            xc, yc = pl[k]
            bb = _text_bbox(txt, cl["sx"], cl["sy"], cl["t"], 0.0, 0.0)
            tw, th = bb[2] - bb[0] + 1, bb[3] - bb[1] + 1
            x0, y0 = xc - tw / 2.0 - bb[0], yc - th / 2.0 - bb[1]
            real = _text_bbox(txt, cl["sx"], cl["sy"], cl["t"], y0, x0)
            box = (real[0] - 1, real[1] - 1, real[2] + 1, real[3] + 1)
            if (box[0] < G_FX0 + 3 or box[2] > G_FX1 - 3 or box[1] < G_FY0 + 3
                    or box[3] > G_FY1 - 3):
                continue
            if any(not (box[2] + 4 < b[0] or box[0] - 4 > b[2] or box[3] + 4 < b[1]
                        or box[1] - 4 > b[3]) for b in boxes):
                continue
            boxes.append(box)
            out.append((txt, x0, y0, box))
    return out


def _grid_mask(lons, lats, X, Y, lo, la):
    A = _assets()
    mask = np.zeros((G_H, G_W), bool)
    hx = [x for x in A["H"] if G_FX0 < x < G_FX1]
    vy = [y for y in A["V"] if G_FY0 < y < G_FY1]
    gcols = [_ri(X(v)) for v in lons if lo[0] < v < lo[1]]
    grows = [_ri(Y(v)) for v in lats if la[0] < v < la[1]]
    for y in grows:
        mask[y, hx] = True
    for x in gcols:
        mask[vy, x] = True
    cross = {tuple(c) for c in A["cross"]}
    for x in gcols:
        for y in grows:
            if (x, y) in cross:
                mask[y, x] = True
    return mask


def _plot_grads_vec(path, slat, slon, pct_sel, z_sel, init, valid, lead, hit_rates, lo, la,
                    coast, S, credit, dpi, title_lines, stroke):
    """Antialiased re-render: the same layout and Hershey lettering as the pixel version, with
    smooth round-capped strokes (matplotlib/Agg) over exact-pixel shading, grid and frame."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.collections import LineCollection
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon, Rectangle

    pal = np.array(G_PALETTE, np.uint8)
    rgbf = lambda i: tuple(c / 255.0 for c in G_PALETTE[i])
    fig = Figure(figsize=(G_W * S / 100.0, G_H * S / 100.0), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    pt = 0.72 * S                                         # points per original pixel
    lons, lats, X, Y = _grads_tick_positions(lo, la)
    ext = (G_FX0, G_FX1 + 1, G_FY1 + 1, G_FY0)
    rect = (G_FX0, G_FY0, G_FX1 + 1, G_FY1 + 1)

    ax.imshow(pal[_bilinear_classes(slat, slon, pct_sel, lo, la, S)], extent=ext,
              interpolation="nearest", aspect="auto", zorder=0)

    def lines(runs, lw, z, col="k"):
        if runs:
            ax.add_collection(LineCollection(runs, linewidths=lw * pt * stroke, colors=col,
                                             capstyle="round", joinstyle="round", zorder=z))

    if coast:
        try:
            runs = []
            for sg in coast_segments():
                if ((sg[:, 0] >= lo[0] - 5) & (sg[:, 0] <= lo[1] + 5)).any():
                    runs += _clip_polyline([(X(a) + 0.5, Y(b) + 0.5) for a, b in sg], rect)
            lines(runs, _LW["coast"], 1)
        except Exception as e:                            # noqa: BLE001
            log.warning("coastlines unavailable (%s); drawing without them", e)

    if z_sel is not None:
        polys = _contour_polys(slon, slat, z_sel, X, Y)
        runs = []
        for _, pl in polys:
            runs += _clip_polyline([(a + 0.5, b + 0.5) for a, b in pl], rect)
        lines(runs, _LW["contour"], 2)
        for txt, x0, y0, bx in _plan_labels(polys):
            ax.add_patch(Rectangle((bx[0], bx[1]), bx[2] - bx[0] + 1, bx[3] - bx[1] + 1,
                                   fc="white", ec="none", lw=0, antialiased=False, zorder=3))
            lines(_text_polys(txt, _ST_CLAB["sx"], _ST_CLAB["sy"], y0, x0=x0), _LW["clab"], 4)

    sub = _grid_mask(lons, lats, X, Y, lo, la)[G_FY0:G_FY1 + 1, G_FX0:G_FX1 + 1]
    big = np.kron(sub, np.ones((S, S), bool))
    rgba = np.zeros(big.shape + (4,), np.uint8)
    rgba[big] = G_PALETTE[G_GREY] + (255,)
    ax.imshow(rgba, extent=ext, interpolation="nearest", aspect="auto", zorder=5)

    box = lambda x0, x1, y0, y1: ax.add_patch(Rectangle(
        (x0, y0), x1 - x0, y1 - y0, fc="black", ec="none", lw=0, antialiased=False, zorder=6))
    box(G_FX0 - 1, G_FX1 + 1, G_FY0 - 1, G_FY0 + 1)       # frame (2 px)
    box(G_FX0, G_FX1 + 1, G_FY1 - 1, G_FY1 + 1)
    box(G_FX0 - 1, G_FX0 + 1, G_FY0, G_FY1 + 1)
    box(G_FX1 - 1, G_FX1 + 1, G_FY0, G_FY1 + 1)
    for v in lons:
        x = _ri(X(v))
        box(x, x + 1, G_FY1 + 1, G_FY1 + 4)
        lines(_text_polys(lon_label(v), cx=x + 0.22, y0=457.8, **{k: _ST_AXIS_X[k] for k in ("sx", "sy")}),
              _LW["axis"], 7)
    for v in lats:
        y = _ri(Y(v))
        box(G_FX0 - 3, G_FX0 - 1, y, y + 1)
        lines(_text_polys(lat_label(v), rx=24.375, y0=y + 3.075, **{k: _ST_AXIS_Y[k] for k in ("sx", "sy")}),
              _LW["axis"], 7)

    # colour bar: boxes and triangles as smooth shapes
    e = [x + 0.5 for x in _CB_E]
    yt, yb, ym = 495.5, 507.5, 501.5
    for k in range(8):
        ax.add_patch(Rectangle((e[k], yt), e[k + 1] - e[k], yb - yt, fc=rgbf(2 + k), ec="none",
                               lw=0, zorder=8))
    ax.add_patch(Polygon([(187.5, ym), (e[0], yt), (e[0], yb)], fc=rgbf(1), ec="none", zorder=8))
    ax.add_patch(Polygon([(634.5, ym), (e[8], yt), (e[8], yb)], fc=rgbf(10), ec="none", zorder=8))
    ax.add_collection(LineCollection(
        [[(e[0], y), (e[8], y)] for y in (yt, yb)] + [[(x, yt), (x, yb)] for x in e],
        linewidths=_LW["bar"] * pt * stroke, colors="k", capstyle="butt", zorder=9))
    for tri in ([(e[0], yt), (187.5, ym), (e[0], yb)], [(e[8], yt), (634.5, ym), (e[8], yb)]):
        ax.add_line(Line2D([q[0] for q in tri], [q[1] for q in tri],
                           lw=_LW["bar_edge"] * pt * stroke, color="k", solid_joinstyle="miter",
                           solid_capstyle="butt", zorder=9))

    def put(text, st, lw, col, **pos):
        lines(_text_polys(text, st["sx"], st["sy"], **pos), lw, 10, col)

    blue = rgbf(G_BLUE)
    for j in range(9):
        put(f"{(j + 1) * 10}", _ST_TICK, _LW["tick"], "k", cx=231.5 + 44.4 * j + 0.76, y0=515.325)
    put("Measure of predictability (%)", dict(sx=0.3842, sy=0.354), _LW["label"], "k", x0=14.23, y0=513.6)
    put("Probability (%)", dict(sx=0.3855, sy=0.3415), _LW["label"], blue, x0=57.52, y0=491.0)
    for k in range(N_CLASSES):
        v = np.nan if hit_rates is None else hit_rates[k]
        put("--" if not np.isfinite(v) else f"{v:.0f}", _ST_PROB, _LW["prob"], blue,
            cx=_PROB_CX[k] + 0.25, y0=491.0)
    put(credit, dict(sx=0.2548, sy=0.234), _LW["credit"], blue, rx=680.8, y0=539.9)
    l1, l2 = title_lines or ("Relative measure of predictability (colors)",
                             "for ensemble mean forecast (contours) of 500 hPa height")
    put(l1, _ST_TITLE, _LW["title"], "k", cx=347.08, y0=32.6)
    put(l2, _ST_TITLE, _LW["title"], "k", cx=346.66, y0=51.6)
    put(f"ini: {init:%Y%m%d%H} valid: {valid:%Y%m%d%H} fcst: {lead} hours", _ST_TITLE3,
        _LW["title3"], "k", cx=341.9, y0=73.0)

    ax.set_xlim(0, G_W)
    ax.set_ylim(G_H, 0)
    ax.axis("off")
    fig.canvas.draw()
    return _save_rgb(path, np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy(), dpi)


PNG_COLORS = 256          # palette size of the saved PNGs (0 = full-colour RGB); set by --png-colors
PNG_MAX_ERR = 24         # fall back to full colour if any pixel would shift by more than this (0-255)


def _save_rgb(path, rgb, dpi=300):
    """Save the antialiased image.  The graphic only uses flat colours plus antialiasing blends, so a
    256-colour palette (no dithering) is visually identical but about 60 % smaller than RGB; the
    worst-case pixel error is checked and the file falls back to full colour if it is not tiny."""
    from PIL import Image
    im = Image.fromarray(rgb, "RGB")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gif":
        im.quantize(256).save(path, format="GIF")
    else:
        out = im
        if PNG_COLORS and PNG_COLORS <= 256:
            q = im.quantize(colors=PNG_COLORS, method=Image.Quantize.MAXCOVERAGE, dither=Image.Dither.NONE)
            err = int(np.abs(np.asarray(q.convert("RGB"), np.int16) - rgb.astype(np.int16)).max())
            if err <= PNG_MAX_ERR:
                out = q
            else:
                log.warning("%s: palette error %d > %d, keeping full colour", path, err, PNG_MAX_ERR)
        out.save(path, format="PNG", dpi=(dpi, dpi), optimize=True)
    log.info("wrote %s (%dx%d px, %d dpi)", path, rgb.shape[1], rgb.shape[0], dpi)
    return rgb


def _plot_grads(path, lat, lon, pct, mean_z, init, valid, lead, hit_rates,
                plot_lon=(170.0, 300.0), plot_lat=(20.0, 80.0), coast=True, scale=1,
                credit="", title_lines=None, dpi=300,
                stroke=1.0):
    S = max(int(scale), 1)
    lo, la = tuple(plot_lon), tuple(plot_lat)
    if not (0 <= lo[0] < lo[1] <= 360):
        raise ValueError("--plot-lon must satisfy 0 <= lon0 < lon1 <= 360 (degrees east)")
    jm = np.where((lon >= lo[0] - 1e-6) & (lon <= lo[1] + 1e-6))[0]
    im = np.where((lat >= la[0] - 1e-6) & (lat <= la[1] + 1e-6))[0]
    if len(jm) < 2 or len(im) < 2:
        raise ValueError("plot sector does not overlap the computed domain")
    slon, slat = lon[jm], lat[im]
    sel = np.ix_(im, jm)
    if S > 1:                                             # antialiased high-resolution path
        return _plot_grads_vec(path, slat, slon, pct[sel], None if mean_z is None else mean_z[sel],
                               init, valid, lead, hit_rates, lo, la, coast, S, credit, dpi,
                               title_lines, stroke)
    A = _assets()
    img = np.full((G_H * S, G_W * S), G_WHITE, np.uint8)

    # shading
    img[G_FY0 * S:(G_FY1 + 1) * S, G_FX0 * S:(G_FX1 + 1) * S] = _bilinear_classes(
        slat, slon, pct[sel], lo, la, S)
    lons, lats, X, Y = _grads_tick_positions(lo, la)
    inner = (G_FX0, G_FY0, G_FX1, G_FY1)

    # coastlines / borders
    if coast:
        try:
            for seg in coast_segments():
                m = (seg[:, 0] >= lo[0] - 5) & (seg[:, 0] <= lo[1] + 5)
                if not m.any():
                    continue
                _gd_poly(img, [(X(a), Y(b)) for a, b in seg], G_BLACK, 1, inner, S)
        except Exception as e:                            # noqa: BLE001
            log.warning("coastlines unavailable (%s); drawing without them", e)

    # height contours (40 m) and horizontal labels in white boxes
    if mean_z is not None:
        polys = _contour_polys(slon, slat, np.asarray(mean_z[sel], float), X, Y)
        for _, pl in polys:
            _gd_poly(img, [tuple(q) for q in pl], G_BLACK, 2, inner, S)
        cl = _ST_CLAB
        for txt, x0, y0, bx in _plan_labels(polys):
            _blk(img, S, bx[0], bx[2], bx[1], bx[3], G_WHITE)
            _draw_text(img, txt, cl["sx"], cl["sy"], cl["t"], G_BLACK, y0, x0=x0, S=S)

    # dotted grid (drawn over everything inside the map)
    mask = _grid_mask(lons, lats, X, Y, lo, la)
    img[np.kron(mask, np.ones((S, S), bool)) if S > 1 else mask] = G_GREY

    # frame (2 px, libgd thick-line convention), ticks, tick labels
    _blk(img, S, G_FX0 - 1, G_FX1, G_FY0 - 1, G_FY0, G_BLACK)
    _blk(img, S, G_FX0, G_FX1, G_FY1 - 1, G_FY1, G_BLACK)
    _blk(img, S, G_FX0 - 1, G_FX0, G_FY0, G_FY1, G_BLACK)
    _blk(img, S, G_FX1 - 1, G_FX1, G_FY0, G_FY1, G_BLACK)
    for v in lons:
        x = _ri(X(v))
        _blk(img, S, x, x, G_FY1 + 1, G_FY1 + 3, G_BLACK)
        _draw_text(img, lon_label(v), cx=x + 0.22, y0=457.8, col=G_BLACK, S=S, **_ST_AXIS_X)
    for v in lats:
        y = _ri(Y(v))
        _blk(img, S, G_FX0 - 3, G_FX0 - 2, y, y, G_BLACK)
        _draw_text(img, lat_label(v), rx=24.375, y0=y + 3.075, col=G_BLACK, S=S, **_ST_AXIS_Y)

    # colour bar graphic, numbers, probability row
    _draw_colorbar(img, S)
    for j in range(9):
        _draw_text(img, f"{(j + 1) * 10}", cx=231.5 + 44.4 * j + 0.76, y0=515.325,
                   col=G_BLACK, S=S, **_ST_TICK)
    _draw_text(img, "Measure of predictability (%)", 0.3842, 0.354, 2, G_BLACK, 513.6,
               x0=14.23, S=S)
    _draw_text(img, "Probability (%)", 0.3855, 0.3415, 2, G_BLUE, 491.0, x0=57.52, S=S)
    for k in range(N_CLASSES):
        v = np.nan if hit_rates is None else hit_rates[k]
        _draw_text(img, "--" if not np.isfinite(v) else f"{v:.0f}", cx=_PROB_CX[k] + 0.25,
                   y0=491.0, col=G_BLUE, S=S, **_ST_PROB)
    _draw_text(img, credit, 0.2548, 0.234, 2, G_BLUE, 539.9, rx=680.8, S=S)

    # title
    l1, l2 = title_lines or ("Relative measure of predictability (colors)",
                             "for ensemble mean forecast (contours) of 500 hPa height")
    _draw_text(img, l1, cx=347.08, y0=32.6, col=G_BLACK, S=S, **_ST_TITLE)
    _draw_text(img, l2, cx=346.66, y0=51.6, col=G_BLACK, S=S, **_ST_TITLE)
    _draw_text(img, f"ini: {init:%Y%m%d%H} valid: {valid:%Y%m%d%H} fcst: {lead} hours",
               cx=341.9, y0=73.0, col=G_BLACK, S=S, **_ST_TITLE3)
    return _save_indexed(path, img, dpi)


def _save_indexed(path, img, dpi=300):
    from PIL import Image
    pil = Image.fromarray(img, "P")
    flat = [c for rgb in G_PALETTE for c in rgb]
    pil.putpalette(flat + [0] * (768 - len(flat)))
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gif":
        pil.save(path, format="GIF")
    else:
        pil.save(path, format="PNG", dpi=(dpi, dpi))
    log.info("wrote %s (%dx%d px, %d dpi)", path, img.shape[1], img.shape[0], dpi)
    return img


def plot_rmop(path, lat, lon, pct, mean_z, init, valid, lead, hit_rates,
              plot_lon=(170.0, 300.0), plot_lat=(20.0, 80.0), coast=True, dpi=300,
              scale=5, style="grads", credit="", stroke=1.0):
    """style 'grads' (default): re-render of the original GrADS graphic.  scale 1 gives the
    original 695 x 545 pixels; scale 5 at 300 DPI is the 11 x 8.5 in page at print size.
    style 'mpl': smooth matplotlib version with the same layout."""
    if style == "mpl":
        return plot_rmop_mpl(path, lat, lon, pct, mean_z, init, valid, lead, hit_rates,
                             plot_lon, plot_lat, coast, dpi, credit)
    return _plot_grads(path, lat, lon, pct, mean_z, init, valid, lead, hit_rates,
                       plot_lon, plot_lat, coast, scale, credit, dpi=dpi, stroke=stroke)


# ======================================================================
# 8. Site export: PNG images, rolling archive and the dashboard viewer
# ======================================================================
ISO = "%Y-%m-%dT%H:%M:%SZ"


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(rgb)


def init_id(init: datetime) -> str:
    return f"{init:%Y%m%d%H}"


def export_lead(site, res, args):
    """Manifest entry for one lead; its PNG is already in the site folder."""
    hr = res["hr"]
    return {"lead": res["lead"], "valid": res["valid"].strftime(ISO),
            "png": os.path.relpath(res["png"], site).replace(os.sep, "/"),
            "probability": None if hr is None else [None if not np.isfinite(v) else round(float(v), 1)
                                                    for v in hr]}


_VIEWER_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="color-scheme" content="light dark">
<link rel="icon" href="data:,">
<title>__TITLE__</title>
<style>
:root{
  --bg:#f3f5f9; --card:#ffffff; --ink:#0c1424; --muted:#5a6a84; --line:rgba(12,20,36,.09);
  --accent:#2563eb; --accent-2:#1d4ed8; --on-accent:#fff;
  --shadow:0 1px 2px rgba(12,20,36,.06),0 18px 40px -18px rgba(12,20,36,.28);
  --radius:20px; --hover:rgba(37,99,235,.09);
  --chrome:222px;        /* page height used by everything except the image */
  --chrome-x:128px;     /* same, when the viewer is expanded to the full window */
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Inter,Roboto,"Helvetica Neue",Arial,sans-serif;
}
@media (prefers-color-scheme:dark){
  :root{--bg:#0a0f1a;--card:#101827;--ink:#e8eefb;--muted:#8fa2c2;--line:rgba(255,255,255,.09);
    --accent:#5b9dff;--accent-2:#79b0ff;--on-accent:#06101f;
    --shadow:0 1px 2px rgba(0,0,0,.5),0 22px 48px -20px rgba(0,0,0,.7);--hover:rgba(91,157,255,.14)}
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--ink);line-height:1.45;
  background-image:radial-gradient(1200px 500px at 50% -10%,color-mix(in srgb,var(--accent) 10%,transparent),transparent 70%)}
.app{max-width:1180px;margin:0 auto;padding:clamp(10px,1.6vw,18px) clamp(10px,2.4vw,24px) 32px}
.top{display:flex;flex-wrap:wrap;gap:8px 24px;align-items:flex-end;justify-content:space-between;margin-bottom:10px}
.top h1{margin:0;font-size:clamp(19px,2.8vw,27px);letter-spacing:-.02em;font-weight:720}
.top p{margin:3px 0 0;color:var(--muted);font-size:13.5px}
.runs{display:flex;flex-wrap:wrap;gap:8px}
.seg{display:inline-flex;flex-wrap:wrap;padding:3px;border-radius:13px;background:color-mix(in srgb,var(--ink) 7%,transparent);gap:2px}
.seg button{all:unset;cursor:pointer;padding:7px 12px;border-radius:10px;font-size:13.5px;font-weight:600;color:var(--muted);
  font-variant-numeric:tabular-nums;white-space:nowrap;transition:background .18s,color .18s,box-shadow .18s}
.seg button:hover:not(:disabled){color:var(--ink)}
.seg button[aria-pressed=true]{background:var(--card);color:var(--ink);box-shadow:0 1px 3px rgba(0,0,0,.18)}
.seg button:disabled{opacity:.35;cursor:default}
.seg button:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.card{background:var(--card);border-radius:var(--radius);box-shadow:var(--shadow);border:1px solid var(--line)}
.stage{background:color-mix(in srgb,var(--ink) 4%,var(--card));border-top:1px solid var(--line);border-bottom:1px solid var(--line)}
/* the image is as large as fits in the window height (never taller than the screen) and never wider than the card */
.viewport{position:relative;overflow:hidden;background:#fff;margin:0 auto;isolation:isolate;aspect-ratio:var(--ar,1.275);
  width:min(100%,calc(max(480px,100vh - var(--chrome)) * var(--ar,1.275)));
  width:min(100%,calc(max(480px,100dvh - var(--chrome)) * var(--ar,1.275)))}
.layer{position:absolute;inset:0;width:100%;height:100%;object-fit:contain;opacity:0;transition:opacity .2s ease;
  user-select:none;-webkit-user-drag:none}
.layer.on{opacity:1}
.skeleton{position:absolute;inset:0;z-index:3;background:linear-gradient(100deg,transparent 30%,rgba(125,140,165,.22) 50%,transparent 70%) 0 0/200% 100%,
  color-mix(in srgb,var(--ink) 5%,var(--card));animation:sh 1.3s linear infinite;transition:opacity .3s}
.skeleton.done{opacity:0;pointer-events:none}
@keyframes sh{to{background-position:-200% 0}}
.bar{display:flex;align-items:center;gap:12px;padding:8px clamp(10px,1.6vw,16px);flex-wrap:wrap}
.now{flex:1;min-width:150px;display:flex;flex-wrap:wrap;align-items:baseline;gap:0 12px}
.now b{font-size:clamp(18px,2.8vw,23px);letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.now span{color:var(--muted);font-size:14px;font-weight:550}
.tag{font-size:12.5px;font-weight:650;color:var(--muted);padding:5px 10px;border-radius:999px;background:color-mix(in srgb,var(--ink) 6%,transparent);white-space:nowrap}
.tag.latest{color:var(--accent-2);background:color-mix(in srgb,var(--accent) 13%,transparent)}
.tag.warn{color:#8a5a00;background:rgba(245,166,35,.2)}
@media (prefers-color-scheme:dark){.tag.warn{color:#ffc766;background:rgba(245,166,35,.18)}}
.update{all:unset;cursor:pointer;padding:5px 12px;border-radius:999px;background:var(--accent);color:var(--on-accent);
  font-size:12.5px;font-weight:700;white-space:nowrap;box-shadow:0 6px 14px -6px var(--accent)}
.update[hidden]{display:none}
.update:hover{filter:brightness(1.08)}.update:focus-visible{outline:2px solid var(--ink);outline-offset:3px}
.controls{padding:8px clamp(10px,1.6vw,16px) 12px}
.play{all:unset;cursor:pointer;flex:none;width:40px;height:40px;border-radius:50%;display:grid;place-items:center;background:var(--accent);color:var(--on-accent);
  box-shadow:0 8px 18px -8px var(--accent);transition:transform .15s,filter .15s}
.play:hover{filter:brightness(1.08)}.play:active{transform:scale(.94)}
.play:focus-visible{outline:2px solid var(--accent);outline-offset:3px}
.play svg{width:18px;height:18px;fill:currentColor}
.play .i-pause{display:none}.play[aria-pressed=true] .i-play{display:none}.play[aria-pressed=true] .i-pause{display:block}
.progress{height:3px;border-radius:3px;background:color-mix(in srgb,var(--ink) 9%,transparent);margin:0 0 6px;overflow:hidden}
.progress i{display:block;height:100%;background:var(--accent);transform-origin:left;transform:scaleX(var(--p,0));transition:transform .25s cubic-bezier(.2,.8,.2,1)}
.rail{display:flex;gap:0;touch-action:none;user-select:none;-webkit-user-select:none;border-radius:11px;overflow:hidden;
  background:color-mix(in srgb,var(--ink) 4%,transparent)}
.rail button{all:unset;box-sizing:border-box;flex:1 1 0;min-width:0;cursor:pointer;text-align:center;padding:6px 0 5px;display:grid;gap:0;position:relative;
  transition:background .12s,color .12s;-webkit-tap-highlight-color:transparent}
.rail button b{font-size:13px;line-height:1.25;letter-spacing:-.01em;font-variant-numeric:tabular-nums}
.rail button small{font-size:10px;line-height:1.2;color:var(--muted);font-variant-numeric:tabular-nums;white-space:nowrap}
.rail button.day::before{content:"";position:absolute;left:0;top:14%;bottom:14%;width:1px;background:color-mix(in srgb,var(--ink) 20%,transparent)}
.rail button:hover{background:var(--hover)}
.rail button[aria-selected=true]{background:var(--accent);color:var(--on-accent)}
.rail button[aria-selected=true] small{color:inherit;opacity:.88}
.rail button[aria-selected=true]::before{display:none}
.rail button:focus-visible{outline:2px solid var(--accent);outline-offset:-2px}
.icon{all:unset;cursor:pointer;flex:none;width:34px;height:34px;display:grid;place-items:center;border-radius:10px;color:var(--muted)}
.icon:hover{background:var(--hover);color:var(--ink)}.icon:focus-visible{outline:2px solid var(--accent);outline-offset:2px}
.icon svg{width:18px;height:18px}
.zoomer{position:absolute;inset:0;transform-origin:0 0;will-change:transform}
.viewport{touch-action:pan-y;cursor:zoom-in}
.viewport.zoomed{touch-action:none;cursor:grab}.viewport.zoomed:active{cursor:grabbing}
.zoomchip{position:absolute;right:8px;bottom:8px;z-index:4;pointer-events:none;font-size:12px;font-weight:650;padding:4px 10px;border-radius:999px;
  background:rgba(12,20,36,.72);color:#fff}
.zoomchip[hidden]{display:none}
body.noscroll{overflow:hidden}
.card.expanded{position:fixed;inset:0;z-index:50;border-radius:0;border:0;display:flex;flex-direction:column;overflow:hidden}
.card.expanded .stage{flex:1;min-height:0;display:flex;align-items:center;justify-content:center}
.card.expanded .viewport{width:min(100%,max(300px,calc((100vh - var(--chrome-x)) * var(--ar,1.275))));
  width:min(100%,max(300px,calc((100dvh - var(--chrome-x)) * var(--ar,1.275))));cursor:default}
.card.expanded .viewport.zoomed{cursor:grab}
.card.expanded .viewport{touch-action:none}
.grid{display:grid;gap:16px;margin-top:18px;grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}
.tile{padding:16px 18px}
.tile h2{margin:0 0 8px;font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--muted);font-weight:700}
.tile p{margin:0;font-size:14.5px}.tile ul{margin:0;padding:0;list-style:none;display:grid;gap:6px}
.tile a{color:var(--accent-2);text-decoration:none;font-weight:600;font-size:14px}.tile a:hover{text-decoration:underline}
.tile small{color:var(--muted)}
.legend{display:grid;grid-template-columns:repeat(auto-fill,minmax(84px,1fr));gap:6px 10px;font-size:12.5px;color:var(--muted);white-space:nowrap}
.legend div{display:flex;align-items:center;gap:8px}.legend i{width:16px;height:16px;border-radius:5px;border:1px solid var(--line);flex:none}
footer{margin-top:22px;color:var(--muted);font-size:12.5px;text-align:center}
kbd{font:inherit;font-size:11.5px;padding:1px 6px;border-radius:6px;border:1px solid var(--line);background:color-mix(in srgb,var(--ink) 5%,transparent)}
@media (max-width:900px){
  .rail{flex-wrap:wrap}.rail button{flex:1 0 10%}.rail button.day::before{display:none}
  :root{--chrome:252px;--chrome-x:168px}
}
@media (max-width:640px){
  .rail button{flex:1 0 10%}.rail button b{font-size:12.5px}.rail button small{font-size:9.5px}.rail .wd,.rail .pm{display:none}
  .bar{gap:10px}.runs{width:100%}
  :root{--chrome:356px;--chrome-x:170px}
}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}}
</style>
</head>
<body>
<main class="app">
  <header class="top">
    <div><h1 id="title"></h1><p id="subtitle"></p></div>
    <div class="runs" aria-label="Model run">
      <div class="seg" id="days" role="group" aria-label="Day (UTC)"></div>
      <div class="seg" id="cycles" role="group" aria-label="Cycle (UTC)"></div>
    </div>
  </header>

  <section class="card" id="card" aria-label="Forecast viewer">
    <div class="bar">
      <button class="play" id="play" aria-pressed="false" aria-label="Play forecast loop" title="Play / pause (Space)">
        <svg class="i-play" viewBox="0 0 24 24"><path d="M8 5.5v13a1 1 0 0 0 1.5.86l10.5-6.5a1 1 0 0 0 0-1.72L9.5 4.64A1 1 0 0 0 8 5.5Z"/></svg>
        <svg class="i-pause" viewBox="0 0 24 24"><rect x="6.5" y="5" width="4" height="14" rx="1.2"/><rect x="13.5" y="5" width="4" height="14" rx="1.2"/></svg>
      </button>
      <div class="now" aria-live="polite"><b id="chipLead"></b><span id="chipValid"></span></div>
      <button class="update" id="update" hidden></button>
      <span class="tag" id="runTag"></span>
      <button class="icon" id="expand" aria-pressed="false" aria-label="Expand image" title="Expand (F)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5"/></svg>
      </button>
    </div>
    <div class="stage">
      <div class="viewport" id="viewport">
        <div class="zoomer" id="zoomer">
          <img id="imgA" class="layer" alt="" draggable="false" decoding="async">
          <img id="imgB" class="layer" alt="" draggable="false" decoding="async">
        </div>
        <div class="skeleton" id="skeleton"></div>
        <div class="zoomchip" id="zoomchip" hidden></div>
      </div>
    </div>
    <div class="controls">
      <div class="progress" aria-hidden="true"><i id="prog"></i></div>
      <div class="rail" id="rail" role="listbox" aria-label="Forecast hour"></div>
    </div>
  </section>

  <div class="grid">
    <section class="card tile"><h2>Download this image</h2><ul id="links"></ul></section>
    <section class="card tile"><h2>RMOP scale</h2><div class="legend" id="legend"></div>
      <p style="margin-top:10px"><small>Colours show how many ensemble members agree with the ensemble mean, relative to recent forecasts.</small></p></section>
    <section class="card tile"><h2>About</h2><p id="about"></p>
      <p style="margin-top:8px"><small>Hover or tap an hour. <kbd>←</kbd> <kbd>→</kbd> step hours, <kbd>[</kbd> <kbd>]</kbd> older / newer run, <kbd>Space</kbd> plays. Double-click or <kbd>Ctrl</kbd>+scroll to zoom, drag to pan, <kbd>F</kbd> expands, <kbd>Esc</kbd> closes.</small></p></section>
  </div>
  <footer id="credit"></footer>
</main>
<noscript><p style="padding:20px">This viewer needs JavaScript.</p></noscript>

<script type="application/json" id="manifest">__MANIFEST__</script>
<script>
"use strict";
let M = JSON.parse(document.getElementById("manifest").textContent);
const $ = id => document.getElementById(id);
let runs = M.inits;                                     // newest first
const isComplete = r => r.complete !== false;           // every planned forecast hour was produced
const defaultIdx = rs => { const k = rs.findIndex(isComplete); return k < 0 ? 0 : k; };
let latestIdx = defaultIdx(runs);                       // newest COMPLETE run = what the page opens on
const fmt = (iso, o) => new Date(iso).toLocaleString("en-US", {timeZone: "UTC", ...o});
const dayName = iso => fmt(iso, {weekday: "short"});
const hourZ = iso => iso.slice(11, 13) + "Z";
const dayKey = iso => iso.slice(0, 10), cycKey = iso => iso.slice(11, 13);
const longValid = iso => `${fmt(iso, {weekday: "short", day: "2-digit", month: "short"})} ${hourZ(iso)}`;
const pad3 = n => String(n).padStart(3, "0");

let runIdx = latestIdx, items = runs[latestIdx].leads, run = runs[latestIdx], cur = -1, front, token = 0;

/* ---------- static content ---------- */
$("title").textContent = M.title;
$("credit").textContent = M.credit;
document.documentElement.style.setProperty("--ar", M.image.w / M.image.h);
M.colors.forEach(c => { const d = document.createElement("div"); d.innerHTML = `<i style="background:${c.color}"></i>${c.min}–${c.max}%`; $("legend").appendChild(d); });

/* ---------- run picker (day + cycle) ---------- */
const findRun = (d, c) => runs.findIndex(r => dayKey(r.init) === d && cycKey(r.init) === c);
function buildPickers() {
  const days = [...new Set(runs.map(r => dayKey(r.init)))].sort();          // oldest -> newest
  const cycles = [...new Set(runs.map(r => cycKey(r.init)))].sort();
  $("days").textContent = ""; $("cycles").textContent = "";
  days.forEach(d => {
    const b = document.createElement("button"), iso = d + "T00:00:00Z";
    b.type = "button"; b.dataset.day = d; b.textContent = `${dayName(iso)} ${fmt(iso, {day: "numeric"})}`;
    b.title = fmt(iso, {weekday: "long", day: "numeric", month: "long", year: "numeric"}) + " (UTC)";
    $("days").appendChild(b);
  });
  cycles.forEach(c => {
    const b = document.createElement("button");
    b.type = "button"; b.dataset.cyc = c; b.textContent = c + "Z"; b.title = `${c}Z model cycle`;
    $("cycles").appendChild(b);
  });
}
buildPickers();
$("days").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b) return;
  const d = b.dataset.day;
  let k = findRun(d, cycKey(run.init));                                    // same cycle if that day has it ...
  if (k < 0) k = runs.findIndex(r => dayKey(r.init) === d);                // ... else its newest run
  if (k >= 0) { pausePlay(); selectRun(k); }
});
$("cycles").addEventListener("click", e => {
  const b = e.target.closest("button"); if (!b || b.disabled) return;
  const k = findRun(dayKey(run.init), b.dataset.cyc);
  if (k >= 0) { pausePlay(); selectRun(k); }
});

/* ---------- forecast-hour rail ---------- */
const rail = $("rail");
let btns = [];
function buildRail() {
  rail.textContent = "";
  items.forEach((it, i) => {
    const b = document.createElement("button");
    b.type = "button"; b.dataset.i = i; b.setAttribute("role", "option");
    b.innerHTML = `<b><span class="pm">+</span>${it.lead}<span class="pm">h</span></b><small><span class="wd">${dayName(it.valid)} </span>${hourZ(it.valid)}</small>`;
    if (i > 0 && dayKey(it.valid) !== dayKey(items[i - 1].valid)) b.classList.add("day");
    b.title = `+${it.lead} h — valid ${longValid(it.valid)}`;
    rail.appendChild(b);
  });
  btns = [...rail.children];
}

/* ---------- image layers (two-layer crossfade, decode before swap) ---------- */
const A = $("imgA"), B = $("imgB");
front = A;
async function showImage(i) {
  const t = ++token, next = front === A ? B : A;
  next.alt = `RMOP ${items[i].lead}-hour forecast, run ${run.id}`;
  next.src = items[i].png;
  try { await next.decode(); } catch (e) { /* missing file: show nothing rather than throw */ }
  if (t !== token) return;                              // a newer hover won the race
  next.classList.add("on"); front.classList.remove("on"); front = next;
  $("skeleton").classList.add("done");
}
const warmed = new Set();
function warm(url) { if (warmed.has(url)) return; warmed.add(url);
  const im = new Image(); im.decoding = "async"; im.src = url; }
function warmAround(i) {                                // neighbours now, the rest of this run while idle
  const order = items.map((_, k) => k).sort((a, b) => Math.abs(a - i) - Math.abs(b - i));
  order.slice(0, 3).forEach(k => warm(items[k].png));
  if (navigator.connection && navigator.connection.saveData) return;
  const rest = order.slice(3), step = () => { const k = rest.shift(); if (k === undefined) return; warm(items[k].png);
    (window.requestIdleCallback || (f => setTimeout(f, 80)))(step); };
  (window.requestIdleCallback || (f => setTimeout(f, 250)))(step);
}

/* ---------- selection ---------- */
const hashFor = it => (runIdx === latestIdx ? "" : run.id + "-") + "f" + pad3(it.lead);
function select(i, opts = {}) {
  i = Math.max(0, Math.min(items.length - 1, i));
  if (i === cur && !opts.force) return;
  cur = i; const it = items[i];
  btns.forEach((b, k) => { const on = k === i; b.setAttribute("aria-selected", on); b.tabIndex = on ? 0 : -1; });
  $("prog").style.setProperty("--p", items.length > 1 ? i / (items.length - 1) : 1);
  $("chipLead").textContent = `+${it.lead} h`;
  $("chipValid").textContent = `valid ${longValid(it.valid)}`;
  $("links").innerHTML = `<li><a href="${it.png}" download="rmop_${run.id}_f${pad3(it.lead)}.png">PNG <small>(+${it.lead} h, run ${run.id})</small></a></li>`;
  showImage(i);
  if (!opts.noHash) history.replaceState(null, "", "#" + hashFor(it));
  if (!opts.noWarm) warmAround(i);
}
function selectRun(k, opts = {}) {
  k = Math.max(0, Math.min(runs.length - 1, k));
  const prevLead = cur >= 0 ? items[cur].lead : null;
  runIdx = k; run = runs[k]; items = run.leads;
  const d = dayKey(run.init), c = cycKey(run.init);
  [...$("days").children].forEach(b => b.setAttribute("aria-pressed", b.dataset.day === d));
  [...$("cycles").children].forEach(b => { const j = findRun(d, b.dataset.cyc);
    b.setAttribute("aria-pressed", b.dataset.cyc === c); b.disabled = j < 0;
    b.title = `${b.dataset.cyc}Z model cycle` + (j >= 0 && !isComplete(runs[j]) ? " (partial)" : ""); });
  $("subtitle").textContent = `${M.variable} · ${M.model} · initialized ${fmt(run.init, {weekday: "short", day: "2-digit", month: "short", year: "numeric"})} ${hourZ(run.init)}`;
  const tag = $("runTag"), nExp = (run.expected || []).length;
  let label = k === latestIdx ? "Latest run" : k < latestIdx ? "Newer run, still partial"
            : `Archive · ${fmt(run.init, {day: "2-digit", month: "short"})} ${hourZ(run.init)}`;
  if (!isComplete(run)) label += ` (${items.length}/${nExp} hrs)`;
  tag.textContent = label;
  tag.classList.toggle("latest", k === latestIdx && isComplete(run));
  tag.classList.toggle("warn", !isComplete(run));
  $("about").textContent = `${run.members || M.members} ensemble members on a ${M.grid_deg}° grid. ${M.method}. ` +
    `Showing the ${runs.length} most recent runs (past ${M.keep_days} days).`;
  buildRail();
  let i = opts.leadIndex;
  if (i == null) {                                      // keep the same forecast hour when the new run has it
    i = 0;
    if (prevLead != null) { let best = Infinity; items.forEach((it, j) => { const dd = Math.abs(it.lead - prevLead); if (dd < best) { best = dd; i = j; } }); }
  }
  cur = -1;
  select(i, {force: true, noHash: opts.noHash});
}

/* ---------- hover / tap / drag / keyboard ---------- */
let scrubbing = false;
const pick = e => { const el = document.elementFromPoint(e.clientX, e.clientY); return el && el.closest && el.closest("#rail button"); };
rail.addEventListener("pointerover", e => { if (e.pointerType === "touch") return;
  const b = e.target.closest("button"); if (b) { pausePlay(); select(+b.dataset.i); } });
rail.addEventListener("pointerdown", e => { if (e.pointerType !== "touch") return;
  scrubbing = true; pausePlay(); try { rail.setPointerCapture(e.pointerId); } catch (_) {}
  const b = pick(e); if (b) select(+b.dataset.i); });
rail.addEventListener("pointermove", e => { if (!scrubbing) return; const b = pick(e); if (b) select(+b.dataset.i); });
["pointerup", "pointercancel"].forEach(n => rail.addEventListener(n, () => { scrubbing = false; }));
rail.addEventListener("click", e => { const b = e.target.closest("button"); if (b) { pausePlay(); select(+b.dataset.i); } });
rail.addEventListener("focusin", e => { const b = e.target.closest("button"); if (b) select(+b.dataset.i); });
document.addEventListener("keydown", e => {
  if (e.metaKey || e.ctrlKey || e.altKey || /^(INPUT|SELECT|TEXTAREA)$/.test(e.target.tagName)) return;
  const k = e.key;
  if (k === "ArrowRight" || k === "ArrowDown") { e.preventDefault(); pausePlay(); select(cur + 1); btns[cur].focus({preventScroll: true}); }
  else if (k === "ArrowLeft" || k === "ArrowUp") { e.preventDefault(); pausePlay(); select(cur - 1); btns[cur].focus({preventScroll: true}); }
  else if (k === "Home") { e.preventDefault(); select(0); btns[0].focus({preventScroll: true}); }
  else if (k === "End") { e.preventDefault(); select(items.length - 1); btns[cur].focus({preventScroll: true}); }
  else if (k === "[") { e.preventDefault(); pausePlay(); selectRun(runIdx + 1); }      // older run
  else if (k === "]") { e.preventDefault(); pausePlay(); selectRun(runIdx - 1); }      // newer run (may be partial)
  else if (k === " " && e.target.tagName !== "BUTTON") { e.preventDefault(); togglePlay(); }
  else if (k === "f" || k === "F") { e.preventDefault(); setExpanded(!expanded); }
  else if (k === "Escape" && expanded) { e.preventDefault(); setExpanded(false); }
  else if (k === "0") { zs = 1; zx = zy = 0; applyZoom(); }
});

/* ---------- zoom, pan and expand ---------- */
const vp = $("viewport"), zoomer = $("zoomer"), ZMAX = 6;
let zs = 1, zx = 0, zy = 0, expanded = false;
function applyZoom() {
  const r = vp.getBoundingClientRect();
  zs = Math.min(ZMAX, Math.max(1, zs));
  zx = zs === 1 ? 0 : Math.min(0, Math.max(r.width * (1 - zs), zx));        // the image always covers the viewport
  zy = zs === 1 ? 0 : Math.min(0, Math.max(r.height * (1 - zs), zy));
  zoomer.style.transform = zs === 1 ? "" : `translate(${zx}px,${zy}px) scale(${zs})`;
  vp.classList.toggle("zoomed", zs > 1);
  $("zoomchip").hidden = zs === 1;
  $("zoomchip").textContent = `${zs.toFixed(1)}× · double-click to reset`;
}
function zoomAt(cx, cy, ns) {                           // cx, cy: point under the cursor, relative to the viewport
  ns = Math.min(ZMAX, Math.max(1, ns));
  const k = ns / zs;
  zx = cx - (cx - zx) * k; zy = cy - (cy - zy) * k; zs = ns; applyZoom();
}
function setExpanded(on) {
  expanded = on;
  $("card").classList.toggle("expanded", on); document.body.classList.toggle("noscroll", on);
  $("expand").setAttribute("aria-pressed", on); $("expand").title = on ? "Close (Esc)" : "Expand (F)";
  zs = 1; zx = zy = 0; applyZoom();
}
$("expand").addEventListener("click", () => setExpanded(!expanded));
vp.addEventListener("wheel", e => {                     // Ctrl/Cmd + scroll (also trackpad pinch); plain scroll when expanded
  if (!(e.ctrlKey || e.metaKey || expanded)) return;
  e.preventDefault();
  const r = vp.getBoundingClientRect();
  const gain = e.ctrlKey && Math.abs(e.deltaY) < 40 ? 0.012 : 0.0022;     // trackpad pinch sends small deltas, a wheel notch ~100
  zoomAt(e.clientX - r.left, e.clientY - r.top, zs * Math.exp(-e.deltaY * gain));
}, {passive: false});
const ptrs = new Map(); let pinch = null, lastTap = {t: 0, x: 0, y: 0}, moved = 0;
vp.addEventListener("pointerdown", e => {
  ptrs.set(e.pointerId, {x: e.clientX, y: e.clientY}); moved = 0;
  try { vp.setPointerCapture(e.pointerId); } catch (_) {}
  if (ptrs.size === 2) { const [a, b] = [...ptrs.values()]; pinch = {d: Math.hypot(a.x - b.x, a.y - b.y) || 1, s: zs}; }
});
vp.addEventListener("pointermove", e => {
  const p = ptrs.get(e.pointerId); if (!p) return;
  const dx = e.clientX - p.x, dy = e.clientY - p.y; p.x = e.clientX; p.y = e.clientY; moved += Math.abs(dx) + Math.abs(dy);
  if (ptrs.size === 2 && pinch) {
    const [a, b] = [...ptrs.values()], r = vp.getBoundingClientRect();
    zoomAt((a.x + b.x) / 2 - r.left, (a.y + b.y) / 2 - r.top, pinch.s * Math.hypot(a.x - b.x, a.y - b.y) / pinch.d);
  } else if (zs > 1 && ptrs.size === 1) { zx += dx; zy += dy; applyZoom(); }
});
const endPtr = e => {
  if (!ptrs.delete(e.pointerId)) return;
  if (ptrs.size < 2) pinch = null;
  if (e.type === "pointerup" && moved < 8 && ptrs.size === 0) {          // a click or tap; two in a row toggle the zoom
    const now = performance.now(), r = vp.getBoundingClientRect();
    if (now - lastTap.t < 320 && Math.hypot(e.clientX - lastTap.x, e.clientY - lastTap.y) < 24) {
      if (zs > 1) { zs = 1; zx = zy = 0; applyZoom(); } else zoomAt(e.clientX - r.left, e.clientY - r.top, 3);
      lastTap.t = 0;
    } else lastTap = {t: now, x: e.clientX, y: e.clientY};
  }
};
["pointerup", "pointercancel"].forEach(n => vp.addEventListener(n, endPtr));
window.addEventListener("resize", applyZoom);

/* ---------- autoplay ---------- */
let timer = null;
function togglePlay() { timer ? pausePlay() : startPlay(); }
function startPlay() { if (timer) return; $("play").setAttribute("aria-pressed", "true");
  timer = setInterval(() => select(cur >= items.length - 1 ? 0 : cur + 1), 800); }
function pausePlay() { if (!timer) return; clearInterval(timer); timer = null; $("play").setAttribute("aria-pressed", "false"); }
$("play").addEventListener("click", togglePlay);

/* ---------- boot ---------- */
// #f060 = latest run, +60 h        #2026110418-f060 = the 18Z run of 4 Nov 2026, +60 h
function fromHash() {
  const m = /^#(?:(\d{10})[-\/])?f?(\d+)$/.exec(location.hash);
  if (!m) return {run: latestIdx, lead: null};
  const k = m[1] ? runs.findIndex(r => r.id === m[1]) : latestIdx;
  return {run: k < 0 ? latestIdx : k, lead: +m[2]};
}
function applyHash(noHash) {
  const h = fromHash(), r = runs[h.run];
  const j = h.lead == null ? -1 : r.leads.findIndex(it => it.lead === h.lead);
  selectRun(h.run, {leadIndex: j < 0 ? 0 : j, noHash});
}
window.addEventListener("hashchange", () => { const h = fromHash(), r = runs[h.run];
  const j = h.lead == null ? 0 : Math.max(0, r.leads.findIndex(it => it.lead === h.lead));
  if (h.run !== runIdx) selectRun(h.run, {leadIndex: j, noHash: true}); else select(j, {noHash: true}); });
/* ---------- pick up new runs without a reload ---------- */
// A key that changes when a newer run appears, or the newest run gains hours / becomes the newest complete one.
const keyOf = rs => `${rs[0].id}:${rs[0].leads.length}|${rs[defaultIdx(rs)].id}`;
let pending = null;
async function checkForUpdate() {
  if (location.protocol === "file:") return;            // nothing to fetch when opened from disk
  try {
    const r = await fetch("manifest.json?t=" + Date.now(), {cache: "no-store"});
    if (!r.ok) return;
    const nm = await r.json();
    if (!nm || !Array.isArray(nm.inits) || !nm.inits.length || keyOf(nm.inits) === keyOf(runs)) return;
    pending = nm;
    const nr = nm.inits[defaultIdx(nm.inits)];
    $("update").textContent = `New run · ${fmt(nr.init, {weekday: "short", day: "2-digit", month: "short"})} ${hourZ(nr.init)} · load`;
    $("update").hidden = false;
  } catch (e) { /* offline or mid-deploy: try again later */ }
}
$("update").addEventListener("click", () => {
  if (!pending) return;
  M = pending; pending = null; runs = M.inits; latestIdx = defaultIdx(runs);
  document.documentElement.style.setProperty("--ar", M.image.w / M.image.h);
  $("credit").textContent = M.credit;
  buildPickers(); $("update").hidden = true; pausePlay();
  selectRun(latestIdx);
});
setInterval(() => { if (!document.hidden) checkForUpdate(); }, 10 * 60 * 1000);
document.addEventListener("visibilitychange", () => { if (!document.hidden) checkForUpdate(); });

window.rmopViewer = {select, selectRun, checkForUpdate, get runs() { return runs; }, get state() { return {runIdx, latestIdx, cur, playing: !!timer}; }};
applyHash(location.hash === "");
</script>
</body>
</html>
"""


def load_manifest(site):
    """Previous manifest in the site folder (restored from the cache), or None."""
    try:
        with open(os.path.join(site, "manifest.json"), encoding="utf-8") as f:
            m = json.load(f)
        return m if isinstance(m.get("inits"), list) else None
    except (OSError, ValueError):
        return None


def build_site(site, entries, args, init, credit_text, nmem, dlat, expected=None):
    """Merge this run into the site's archive, drop runs older than --keep-days, delete orphaned
    images, and write manifest.json + the self-contained viewer (index.html)."""
    import shutil
    from PIL import Image
    old = load_manifest(site) or {}
    runs = {r["id"]: r for r in old.get("inits", []) if "id" in r and "init" in r}
    rid = init_id(init)
    merged = {e["lead"]: e for e in runs.get(rid, {}).get("leads", [])}
    merged.update({e["lead"]: e for e in entries})                # leads from this run win
    want = sorted(set(runs.get(rid, {}).get("expected", [])) | set(expected or [e["lead"] for e in entries]))
    runs[rid] = {"id": rid, "init": init.strftime(ISO), "members": nmem, "expected": want,
                 "leads": list(merged.values())}

    # keep only runs inside the window, and only leads whose PNG is really there
    stamp = lambda r: datetime.strptime(r["init"], ISO)
    cutoff = max(stamp(r) for r in runs.values()) - timedelta(days=args.keep_days)
    kept = {}
    for k, r in runs.items():
        r["leads"] = sorted((e for e in r["leads"] if os.path.isfile(os.path.join(site, e["png"]))),
                            key=lambda e: e["lead"])
        have = {e["lead"] for e in r["leads"]}
        r["complete"] = (not r.get("expected")) or set(r["expected"]) <= have     # every planned hour is there
        if stamp(r) > cutoff and r["leads"]:
            kept[k] = r
    ordered = sorted(kept.values(), key=stamp, reverse=True)       # newest first

    for legacy in ("web", "geo", "data"):                          # folders from older builds: PNG only now
        shutil.rmtree(os.path.join(site, legacy), ignore_errors=True)
    img_root = os.path.join(site, "img")                           # delete what the manifest no longer lists
    listed = {os.path.normpath(os.path.join(site, e["png"])) for r in ordered for e in r["leads"]}
    for d in os.listdir(img_root):
        p = os.path.join(img_root, d)
        if os.path.isdir(p):
            if d not in kept:
                shutil.rmtree(p, ignore_errors=True)
                continue
            for fn in os.listdir(p):
                if os.path.normpath(os.path.join(p, fn)) not in listed:
                    os.remove(os.path.join(p, fn))
        elif d.endswith(".png"):                                   # flat files from an older layout
            os.remove(p)

    with Image.open(os.path.join(site, ordered[0]["leads"][0]["png"])) as im:
        iw, ih = im.size
    manifest = {
        "title": "Relative Measure of Predictability", "variable": "500 hPa height",
        "model": "NCEP GEFS", "members": nmem, "grid_deg": round(dlat, 4),
        "method": "", "credit": credit_text,
        "keep_days": args.keep_days, "latest": ordered[0]["init"],
        "image": {"w": iw, "h": ih, "dpi": args.dpi},
        "colors": [{"class": k, "min": 10 * k, "max": 10 * (k + 1), "color": _hex(RMOP_COLORS[k])}
                   for k in range(N_CLASSES)],
        "inits": ordered, "generator": "rmop.py"}
    with open(os.path.join(site, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, separators=(",", ":"))
    blob = json.dumps(manifest, separators=(",", ":")).replace("</", "<\\/")
    with open(os.path.join(site, "index.html"), "w", encoding="utf-8") as f:
        f.write(_VIEWER_HTML.replace("__TITLE__", "RMOP Redux - forecast viewer").replace("__MANIFEST__", blob))
    n_part = sum(not r["complete"] for r in ordered)
    n_png = sum(len(r["leads"]) for r in ordered)
    size = sum(os.path.getsize(os.path.join(site, e["png"])) for r in ordered for e in r["leads"])
    print(f"archive: {len(ordered)} runs, {n_png} images, {size / 1e6:.0f} MB "
          f"(window {args.keep_days} days)" + (f", {n_part} partial" if n_part else ""))
    return manifest


def serve_site(root, host="127.0.0.1", port=8000, open_browser=False):
    """Tiny static server for the site folder (the viewer also opens straight from disk)."""
    import functools
    import http.server

    class Handler(http.server.SimpleHTTPRequestHandler):
        def end_headers(self):
            fresh = self.path.split("?")[0].endswith((".html", ".json", "/"))
            self.send_header("Cache-Control", "no-cache" if fresh else "public, max-age=3600")
            super().end_headers()

        def log_message(self, fmt, *a):               # keep the terminal quiet
            pass

    if not os.path.isfile(os.path.join(root, "index.html")):
        raise SystemExit(f"{root} has no index.html - build it first with: rmop.py run ... --site {root}")
    handler = functools.partial(Handler, directory=root)
    with http.server.ThreadingHTTPServer((host, port), handler) as srv:
        url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}/"
        print(f"serving {os.path.abspath(root)}\n  viewer: {url}\n  Ctrl+C to stop")
        if open_browser:
            import webbrowser
            webbrowser.open(url)
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")


# ======================================================================
# 7. High-level driver + CLI
# ======================================================================
def run_product(src, clim, init, lead, args, eng=None):
    eng = eng or Engine(src, clim, args.lat_min, args.lat_max, args.stride, args.cache, args.bins)
    log.info("computing current forecast ...")
    mean, mbin, cnt, sdn = eng.current(init, lead, spread=True)
    hist = eng.history(init, lead, args.history_days, verify=not args.no_verify,
                       weighting=args.weighting, ties=args.ties, pool=args.pool,
                       area_weight=args.area_weight)
    if len(hist.cases) < 5:
        raise SystemExit("fewer than 5 historical forecasts available; cannot rank")
    pct = hist.percentile(cnt, tie_key=sdn)
    cls = to_class(pct)
    hr = None if args.no_verify else hist.class_probabilities(cnt, cls)
    valid = init + timedelta(hours=lead)
    dlat = float(np.median(np.diff(eng.geom.lat))) if len(eng.geom.lat) > 1 else float("nan")
    credit = args.credit or ("RMOP REDUX" + (f" BY {args.credit_name.upper()}" if args.credit_name else "")
                             + ", AFTER TOTH ET AL. 2001")

    dlon = float(np.median(np.diff(eng.geom.lon))) if len(eng.geom.lon) > 1 else float("nan")
    wrap = bool(np.isfinite(dlon) and (eng.geom.lon[-1] - eng.geom.lon[0] + dlon) >= 359.9)
    mean_c = mean
    if np.isfinite(dlat):
        mean_c = despike(mean_c, args.contour_despike, wrap)
        if args.contour_smooth > 0:
            mean_c = smooth_field(mean_c, args.contour_smooth / dlat, wrap)   # contours only

    print(f"\nRMOP  init {init:%Y-%m-%d %HZ}  f{lead:03d}  valid {valid:%Y-%m-%d %HZ}")
    print(f"history: {len(hist.cases)} forecasts, members: {src.n_members}, "
          f"grid: {len(eng.geom.lat)}x{len(eng.geom.lon)} points at {dlat:g} deg spacing")
    if args.stride != 1:
        print(f"note: grid thinned by --stride {args.stride}; omit --stride for the native grid")
    print(f"mean members in ensemble-mean bin: {cnt.mean():.1f} of {src.n_members}")
    print("class   frac_of_points  verified%")
    for k in range(N_CLASSES):
        v = "" if hr is None or not np.isfinite(hr[k]) else f"{hr[k]:6.1f}"
        print(f"{k*10:3d}-{k*10+10:<3d}  {np.mean(cls == k):10.3f}  {v}")

    plot_rmop(args.out, eng.geom.lat, eng.geom.lon, pct, mean_c, init, valid, lead, hr,
              plot_lon=tuple(args.plot_lon), plot_lat=tuple(args.plot_lat),
              coast=not args.no_coast, dpi=args.dpi, scale=args.scale, style=args.style,
              credit=credit, stroke=args.stroke)
    print(f"({len(hist.cases)} prior forecasts; bins={args.bins}; ties={args.ties}; pool={args.pool}; "
          f"weighting={args.weighting})")
    return dict(lead=lead, init=init, valid=valid, png=args.out, lat=eng.geom.lat, lon=eng.geom.lon,
                pct=pct, mean=mean_c, hr=hr, nmem=src.n_members, dlat=dlat, credit=credit)


def resolve_leads(a):
    """--lead-range START STOP STEP (inclusive) > --leads L1 L2 ... > --lead."""
    if a.lead_range:
        lo, hi, st = a.lead_range
        if st <= 0 or hi < lo:
            raise SystemExit("--lead-range needs START <= STOP and STEP > 0")
        leads = list(range(lo, hi + 1, st))
    elif a.leads:
        leads = list(a.leads)
    else:
        leads = [a.lead]
    bad = [l for l in leads if l < 0 or l % 3]
    if bad:
        raise SystemExit(f"GEFS lead times must be multiples of 3 h; bad: {bad}")
    odd = [l for l in leads if l % 6]
    if odd:
        log.warning("leads %s are not multiples of 6 h: no analysis exists at their "
                    "valid time, so the verification numbers will be blank", odd)
    return leads


def lead_path(out: str, lead: int, multi: bool) -> str:
    """One file per lead: use {lead} in --out, else insert _fFFF before the extension."""
    if "{lead" in out:
        return out.format(lead=lead)
    if not multi:
        return out
    root, ext = os.path.splitext(out)
    return f"{root}_f{lead:03d}{ext}"


def run_leads(src, clim, init, a):
    leads = resolve_leads(a)
    multi = len(leads) > 1
    eng = Engine(src, clim, a.lat_min, a.lat_max, a.stride, a.cache, a.bins)   # shared by all leads
    base_out, done, failed, entries, last = a.out, [], [], [], None
    site = getattr(a, "site", None)
    if site:
        os.makedirs(os.path.join(site, "img", init_id(init)), exist_ok=True)
        print(f"building site in {site}/  (PNG images + viewer, {a.keep_days}-day archive)")
    for L in leads:
        a.out = (os.path.join(site, "img", init_id(init), f"rmop_f{L:03d}.png") if site
                 else lead_path(base_out, L, multi))
        print(f"\n=== lead {L} h ({leads.index(L) + 1}/{len(leads)}) ===")
        try:
            res = run_product(src, clim, init, L, a, eng)
            done.append(a.out)
            if site:
                entries.append(export_lead(site, res, a))
                last = res
        except SystemExit as e:                           # e.g. too little history
            failed.append((L, str(e)))
        except Exception as e:                            # noqa: BLE001
            log.error("lead %d failed: %s", L, e)
            failed.append((L, str(e)))
    a.out = base_out
    print(f"\nfinished: {len(done)}/{len(leads)} leads written")
    for f in done:
        print("  ", f)
    for L, msg in failed:
        print(f"   FAILED f{L:03d}: {msg}")
    if site and entries:
        credit_text = (f"RMOP Redux" + (f" by {a.credit_name}" if a.credit_name else "")
                       + " \u00b7 after Toth, Zhu & Marchok (2001)") if not a.credit else a.credit
        build_site(site, entries, a, init, credit_text, last["nmem"], last["dlat"], leads)
        print(f"\nviewer ready: {os.path.join(site, 'index.html')}"
              f"\n  open it directly, or run:  python3 rmop.py serve {site}")
    if failed and (not done or not site):
        sys.exit(1)               # nothing usable was produced
    if failed and os.environ.get("GITHUB_ACTIONS"):
        print("::warning::" + "; ".join(f"f{L:03d} failed" for L, _ in failed))


def parse_init(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d%H")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("climo", help="build monthly decile climatology")
    c.add_argument("--files", help="glob of daily Z500 NetCDF files")
    c.add_argument("--download", metavar="DIR", help="download NCEP R1 hgt.YYYY.nc here")
    c.add_argument("--years", nargs=2, type=int, default=[1991, 2020])
    c.add_argument("--var")
    c.add_argument("--level", type=float, default=500.0)
    c.add_argument("--out", default="climo_z500.npz")

    def common(p):
        p.add_argument("--lead", type=int, default=120, help="a single forecast hour")
        p.add_argument("--leads", nargs="+", type=int, metavar="H",
                       help="several forecast hours, e.g. --leads 24 72 120 168")
        p.add_argument("--lead-range", nargs=3, type=int, metavar=("START", "STOP", "STEP"),
                       help="inclusive series, e.g. --lead-range 12 240 12 or 24 240 24")
        p.add_argument("--lat-min", type=float, default=20.0)
        p.add_argument("--lat-max", type=float, default=80.0)
        p.add_argument("--stride", type=int, default=1,
                       help="grid thinning of the GEFS grid: 1 = native 0.5 deg grid (default); "
                            "5 = the original product's 2.5 deg")
        p.add_argument("--history-days", type=int, default=30)
        p.add_argument("--weighting", choices=["equal", "recursive"], default="recursive")
        p.add_argument("--ties", choices=["spread", "mid", "strict"], default="spread")
        p.add_argument("--bins", choices=["sliding", "fixed"], default="sliding",
                       help="how members are counted: sliding (default) = within half a decile-width "
                            "of the ensemble mean, which removes the thin ripple bands that fixed "
                            "decile edges create at short leads; fixed = same decile as the mean "
                            "(the 2001 paper's definition)")
        p.add_argument("--contour-despike", type=float, default=2.0, metavar="M",
                       help="remove one-gridpoint terrain noise (spikes larger than M metres) from the "
                            "CONTOURED height field only; 0 = off.  Shading and saved fields are unaffected")
        p.add_argument("--contour-smooth", type=float, default=0.0, metavar="DEG",
                       help="optional extra Gaussian smoothing (deg) of the contoured field only")
        p.add_argument("--pool", choices=["domain", "point"], default="domain")
        p.add_argument("--area-weight", action="store_true")
        p.add_argument("--no-verify", action="store_true")
        p.add_argument("--plot-lon", nargs=2, type=float, default=[170.0, 300.0],
                       metavar=("W", "E"), help="map sector, deg east 0-360 (170E-60W)")
        p.add_argument("--plot-lat", nargs=2, type=float, default=[20.0, 80.0])
        p.add_argument("--credit-name", default="Aaron Perry",
                       help="name shown in the bottom credit line (empty string to omit)")
        p.add_argument("--credit", default=None,
                       help="full credit text, overriding --credit-name")
        p.add_argument("--no-coast", action="store_true", help="skip coastlines")
        p.add_argument("--scale", type=int, default=5,
                       help="resolution multiplier for the GrADS-style image: 5 (default) with "
                            "--dpi 300 is the original 11x8.5 in page at print size (3475x2725 px); "
                            "1 = the original 695x545 pixels")
        p.add_argument("--style", choices=["grads", "mpl"], default="grads",
                       help="grads = pixel-level emulation of the original graphic (default)")
        p.add_argument("--dpi", type=int, default=300, help="DPI written into the PNG (and used by --style mpl)")
        p.add_argument("--stroke", type=float, default=1.0,
                       help="line/lettering weight multiplier for the high-resolution output "
                            "(1 = default; 1.3 bolder, 0.8 lighter)")
        p.add_argument("--site", metavar="DIR", default=None,
                       help="build a web package in DIR: img/<init>/rmop_fHHH.png for every lead, a "
                            "manifest.json index and an index.html dashboard.  Runs already in DIR "
                            "are kept for --keep-days.  Overrides --out.")
        p.add_argument("--keep-days", type=int, default=7,
                       help="with --site: how many days of earlier runs stay in the archive (default 7)")
        p.add_argument("--png-colors", type=int, default=256, metavar="N",
                       help="PNG palette size (default 256 = visually lossless, ~60%% smaller files); "
                            "0 = full-colour RGB")
        p.add_argument("--cache", default="./rmop_cache")
        p.add_argument("--out", default="rmop.png", help=".png (default) or .gif")

    r = sub.add_parser("run", help="make RMOP from GEFS on AWS")
    r.add_argument("--init", required=True, type=parse_init, help="YYYYMMDDHH")
    r.add_argument("--climo", required=True)
    r.add_argument("--perturbed", type=int, default=30)
    r.add_argument("--no-control", action="store_true")
    common(r)

    d = sub.add_parser("demo", help="synthetic ensemble demo (no network)")
    d.add_argument("--init", type=parse_init, default=datetime(2026, 11, 2, 0))
    common(d)
    d.add_argument("--res", type=float, default=2.5, help="synthetic grid spacing (deg)")
    d.add_argument("--tscale", type=float, default=1.0, help="synthetic error-growth slowdown")
    d.set_defaults(stride=1)          # synthetic grid is already 2.5 deg

    v = sub.add_parser("serve", help="serve a --site folder on localhost")
    v.add_argument("dir", help="the folder written with --site")
    v.add_argument("--port", type=int, default=8000)
    v.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 to share on your network")
    v.add_argument("--open", action="store_true", help="open the viewer in your browser")

    a = ap.parse_args(argv)
    global PNG_COLORS
    PNG_COLORS = getattr(a, "png_colors", PNG_COLORS)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING,
                        format="%(levelname)s %(message)s")

    if a.cmd == "serve":
        serve_site(a.dir, a.host, a.port, a.open)
        return
    if a.cmd == "climo":
        files = []
        if a.download:
            files += download_r1(a.download, a.years)
        if a.files:
            files += glob.glob(a.files)
        if not files:
            ap.error("give --files and/or --download")
        build_climatology(files, a.var, a.level, tuple(a.years)).save(a.out)
    elif a.cmd == "run":
        clim = Climatology.load(a.climo)
        src = GEFSAws(a.perturbed, not a.no_control)
        run_leads(src, clim, a.init, a)
    else:
        src = SyntheticSource(dlat=a.res, tscale=a.tscale)
        a.cache = a.cache if a.cache != "./rmop_cache" else tempfile.mkdtemp(prefix="rmop_demo_")
        run_leads(src, src.climatology(), a.init, a)


if __name__ == "__main__":
    main()
