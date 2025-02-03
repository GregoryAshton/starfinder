import logging
import os.path
import warnings
from collections import deque

from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import EarthLocation, AltAz, SkyCoord
from astropy.wcs import WCS
from astropy import units
import numpy as np
import tqdm
from scipy.interpolate import RectBivariateSpline
import matplotlib.pyplot as plt
import matplotlib

from .cluster import Cluster

logger = logging.getLogger("starfinder")


class StarField(object):
    """Class to contain image data and lists of clusters and stars"""

    def __init__(self, fits_file, astronometry_api_key):
        self.fits_file = fits_file
        self.fits_file_basename = os.path.basename(self.fits_file)
        self.fits_file_dirname = os.path.dirname(self.fits_file)
        self.astronometry_api_key = astronometry_api_key

        self.read_file()
        self.get_coordinates_of_image_centre()

    def read_file(self):
        """Read the fits_file, store the contents, and initialise data"""
        logger.info(f"Reading file {self.fits_file}")

        with fits.open(self.fits_file) as hdu_list:
            # Convert the full image two a 2-D array with dtype int32
            self.hdu_list = hdu_list
            self.n_ij = hdu_list[0].data.astype(np.int32)
            self.header = hdu_list[0].header

        self.data_shape = self.n_ij.shape
        self.b_ij = np.empty(self.data_shape, dtype=np.float64)
        self.pixel_variance = np.empty(self.data_shape, dtype=np.float64)
        self.source_mark = np.empty(self.data_shape, dtype=bool)
        self.in_usable_region = np.ones(self.data_shape, dtype=bool)

        self.sigma_background = None
        self.mu_background = None

        self.cluster_number = np.empty(self.data_shape, dtype=np.int32)
        self.cluster_number.fill(-1)  # Initially no pixels belong to a cluster

        self.cluster_list = []
        self.star_list = []

    def get_coordinates_of_image_centre(self):
        warnings.simplefilter("ignore")  # suppress FITSFixedWarning
        if "GEO_LAT" in self.header and "GEO_LONG" in self.header:
            self.lat = self.header["GEO_LAT"]
            self.lon = self.header["GEO_LONG"]
        else:
            # use location of RHUL observatory
            self.lat = 51.42688295361113
            self.lon = -0.5632737048924312

        self.observation_date_str = self.header["DATE-OBS"]
        observation_date = Time(self.observation_date_str, format="isot", scale="utc")

        if "WCSAXES" in self.header:
            logger.info("Using WCS information from the fits file header")
            wcs_header = self.header
        else:
            try:
                logger.info("Attempting to extract  WCS information from astrometry.net")
                from astroquery.astrometry_net import AstrometryNet
                ast = AstrometryNet()
                ast.api_key = self.astronometry_api_key
                wcs_header = ast.solve_from_image(self.fits_file, verbose=False)
                if wcs_header:
                    logger.info("Succesfully extracted WCS information from astrometry.net")
                    logger.info("Updating fits file with WCS: overwriting")
                    self.header.update(wcs_header)
                    self.hdu_list.writeto(self.fits_file, overwrite=True)
                else:
                    logger.info("Failed to extract WCS information from astrometry.net")
            except Exception as e:
                logger.info(f"Failed to obtain data from astronometry.net due to {e}")
                wcs_header = {}

        if wcs_header:
            # Calculate the ra and dec
            numCol = self.header["NAXIS1"]
            numRow = self.header["NAXIS2"]
            self.ra, self.dec = WCS(wcs_header).all_pix2world(
                (numCol - 1) / 2.0, (numRow - 1) / 2.0, 1
            )
            self.ra = float(self.ra)
            self.dec = float(self.dec)

            # Calculate the alt and az
            sky_coord = SkyCoord(
                ra=self.ra * units.deg, dec=self.dec * units.deg, frame="icrs"
            )
            location = EarthLocation(lat=self.lat * units.deg, lon=self.lon * units.deg)
            altaz_frame = AltAz(obstime=observation_date, location=location)
            altaz = sky_coord.transform_to(altaz_frame)
            self.alt = altaz.alt.deg
            self.az = altaz.az.deg

            print(f"Coordinate data for {self.fits_file}:")
            print(f"Right Ascension (RA): {self.ra} degrees")
            print(f"Declination (Dec): {self.dec} degrees")
            print(f"Latitude: {self.lat} degrees")
            print(f"Longitude: {self.lon} degrees")
            print(f"Date and Time (UTC): {observation_date}")
            print(f"Altitude: {self.alt} degrees")
            print(f"Azimuth: {self.az} degrees")

        warnings.resetwarnings()

    def estimate_background(
        self,
        box=151,
        hole=0,
        background_algorithm="gauss",
        use_existing_background=True,
        background_dir=None,
    ):
        """Estimate the background and store it in a fits file"""

        if background_dir is None:
            background_dir = self.fits_file_dirname

        hb = int((box - 1) / 2)  # half of box-1
        hh = int(max((hole - 1) / 2, 0))  # half of hole-1 or zero
        background_file_name = "_".join(
            [self.fits_file_basename, str(box), str(hole), "bkg.fit"]
        )
        background_file_path = os.path.join(background_dir, background_file_name)

        if use_existing_background and os.path.exists(background_file_path):
            logger.info(f"Background file {background_file_path} exists: using this")
            with fits.open(background_file_path) as file:
                self.b_ij = file[0].data
        else:
            numRows, numCols = self.b_ij.shape
            delta = 10  # first calculate background on reduced grid
            rowsGrid = np.around(numRows / delta).astype(int)
            colsGrid = np.around(numCols / delta).astype(int)
            bGrid = np.empty([rowsGrid, colsGrid])
            for rowGrid in tqdm.tqdm(range(rowsGrid)):
                for colGrid in range(colsGrid):
                    i = rowGrid * delta
                    j = colGrid * delta
                    xlo = max(j - hb, 0)
                    xhi = min(j + hb + 1, numCols)
                    ylo = max(i - hb, 0)
                    yhi = min(i + hb + 1, numRows)
                    fitRegion = self.n_ij[ylo:yhi, xlo:xhi]
                    if hole > 0:
                        mask = np.full((yhi - ylo, xhi - xlo), True, dtype=bool)
                        mask[hb - hh : hb + hh + 1, hb - hh : hb + hh + 1] = False
                        fitRegion = np.extract(mask, fitRegion)  # exclude hole
                    if background_algorithm == "median":
                        bGrid[rowGrid, colGrid] = np.median(fitRegion)
                    elif background_algorithm == "gauss":
                        bGrid[rowGrid, colGrid], sigma = gaussian_background_fit(
                            fitRegion
                        )
                    else:
                        bGrid[rowGrid, colGrid] = 0

            # 2D spline to interpolate background from grid to full image
            grid_x = np.arange(0, numRows, delta)
            grid_y = np.arange(0, numCols, delta)
            spline_interpolator = RectBivariateSpline(grid_x, grid_y, bGrid)
            image_x = np.arange(numRows)
            image_y = np.arange(numCols)
            self.b_ij = spline_interpolator(image_x, image_y)

            # Store the background (using the fits_file as a basis)
            self.hdu_list[0].data = self.b_ij
            self.hdu_list.writeto(background_file_path, overwrite=True)

        #  Also initialise sigma2_n (same shape as image n) containing
        #  variance for individual pixel from CCD model.
        nCorr = (
            self.n_ij - self.b_ij + np.median(self.b_ij)
        )  #  correct for flatness
        self.mu_0, self.sigma_0 = gaussian_background_fit(nCorr[self.in_usuable_region])
        G = 2.39  # for SBIG ST-8XME camera
        self.pixel_variance = (
            np.maximum(self.n_ij - self.b_ij, 0) / G + self.sigma_0**2
        )

    def set_usable_region(self, border=[50, 50, 50, 50], corner_radius=300):
        """Define a region in which to accept stars using a boolean array

        This will create a 2D boolean array which omits a border and rounded corners

        """

        bL, bR, bT, bB = border  # border on Left, Right, Top and Bottom
        numRows, numCols = self.n_ij.shape
        rows, cols = np.ogrid[:numRows, :numCols]  # meshgrid of pixel coord
        TL_row, TL_col = bT + corner_radius, bL + corner_radius
        BL_row, BL_col = numRows - bB - corner_radius, bL + corner_radius
        BR_row, BR_col = numRows - bB - corner_radius, numCols - bR - corner_radius
        TR_row, TR_col = bT + corner_radius, numCols - bR - corner_radius
        in_TL = (
            (rows >= bT)
            & (rows < bT + corner_radius)
            & (cols >= bL)
            & (cols < bL + corner_radius)
        )
        in_BL = (
            (rows >= numRows - bB - corner_radius)
            & (rows < numRows - bB)
            & (cols >= bL)
            & (cols < bL + corner_radius)
        )
        in_BR = (
            (rows >= numRows - bB - corner_radius)
            & (rows < numRows - bB)
            & (cols >= numCols - bR - corner_radius)
            & (cols < numCols - bR)
        )
        in_TR = (
            (rows >= bT)
            & (rows < bT + corner_radius)
            & (cols >= numCols - bR - corner_radius)
            & (cols < numCols - bR)
        )
        TL_r = np.sqrt((rows - TL_row) ** 2 + (cols - TL_col) ** 2)
        BL_r = np.sqrt((rows - BL_row) ** 2 + (cols - BL_col) ** 2)
        BR_r = np.sqrt((rows - BR_row) ** 2 + (cols - BR_col) ** 2)
        TR_r = np.sqrt((rows - TR_row) ** 2 + (cols - TR_col) ** 2)
        inRectangle = (
            (rows >= bT) & (rows < numRows - bB) & (cols >= bL) & (cols < numCols - bR)
        )
        inCorners = (
            (in_TL & (TL_r > corner_radius))
            | (in_BL & (BL_r > corner_radius))
            | (in_BR & (BR_r > corner_radius))
            | (in_TR & (TR_r > corner_radius))
        )
        self.in_usuable_region = inRectangle & ~inCorners

    def find_clusters(self, threshold_factor=2.5):
        """ Cluster the image to identify isolated stars

        Find pixels over threshold, then group pixels sharing a common edge
        into clusters.  When cluster started, create a stack.  Pop pixels off
        the stack, look at their neighbours, push onto stack if over threshold,
        iterate until stack empty.
        """

        cluster_number = -1
        num_rows, num_cols = self.data_shape

        def is_valid_pixel(i, j):
            return 0 <= i < num_rows and 0 <= j < num_cols

        threshold = self.b_ij + threshold_factor * self.sigma_0
        for i in range(num_rows):
            for j in range(num_cols):
                start_of_cluster = (
                    self.n_ij[i, j] > threshold[i, j] and self.cluster_number[i, j] < 0
                )
                if start_of_cluster:
                    cluster_number += 1
                    self.cluster_number[i, j] = cluster_number
                    cluster = Cluster(cluster_number, (i, j), self)
                    stack = deque([(i, j)])
                    while stack:
                        k, l = stack.pop()
                        directions = [(k - 1, l), (k + 1, l), (k, l - 1), (k, l + 1)]
                        for u, v in directions:
                            if is_valid_pixel(u, v):
                                if (
                                    self.cluster_number[u, v] < 0
                                    and self.n_ij[u, v] > threshold[u, v]
                                ):
                                    stack.append((u, v))
                                    self.cluster_number[u, v] = cluster_number
                                    cluster.add_pixel((u, v))
                    cluster.set_cluster_number(cluster_number)
                    self.cluster_list.append(cluster)
        logger.info(f"Image {self.fits_file_basename} has {cluster_number + 1} clusters")

    def process_cluster_list(self, minimum_pixels=6, sigma_cl_min=2.5, sigma_cl_max=5, rho_cl_max=0.1):
        """ Process the cluster list

        Accept clusters as star candidates if numPix >= minPix and
        cluster width/shape within cut limits.

        """

        numRows, numCols = self.data_shape
        star_number = -1
        self.mux_list = []
        self.muy_list = []
        self.star_list = []
        for cluster in self.cluster_list:
            covariance = cluster.cov()
            sigma_x = -1.0
            sigma_y = -1.0
            rho_xy = 0.0
            if covariance[0, 0] > 0 and covariance[1, 1] > 0:
                sigma_x = np.sqrt(covariance[0, 0])
                sigma_y = np.sqrt(covariance[1, 1])
                rho_xy = covariance[0, 1] / (sigma_x * sigma_y)
            covariance_ok = (
                sigma_x >= sigma_cl_min
                and sigma_x <= sigma_cl_max
                and sigma_y >= sigma_cl_min
                and sigma_y <= sigma_cl_max
                and abs(rho_xy) <= rho_cl_max
            )
            mux, muy = cluster.mu()
            xy_ok = self.in_usuable_region[int(muy), int(mux)]
            accept_cluster = cluster.number_of_pixels() >= minimum_pixels and covariance_ok and xy_ok
            if accept_cluster:
                star_number += 1
                star = cluster
                star.set_cluster_number(star_number)
                mux, muy = star.mu()
                self.mux_list.append(mux)
                self.mux_list.append(muy)
                self.star_list.append(star)
        # Reorder in decreasing flux
        self.star_list.sort(key=lambda x: x.flux(), reverse=True)
        star_number = 0
        for star in self.star_list:
            star.set_cluster_number(star_number)
            star_number += 1
        self.mux_list = [star.mu()[0] for star in self.star_list]
        self.muy_list = [star.mu()[1] for star in self.star_list]
        # Show summary table and plot of stars found
        print("Stars for image", self.fits_file)
        print(
            "Number numPix   Flux     mu_x     mu_y      sigma_x  " + "sigma_y  rho_xy"
        )
        for star in self.star_list:
            flux = star.flux()
            number_of_pixels = star.number_of_pixels()
            mux, muy = star.mu()
            covariance = star.cov()
            sigma_x = np.sqrt(covariance[0, 0])
            sigma_y = np.sqrt(covariance[1, 1])
            rho_xy = covariance[0, 1] / (sigma_x * sigma_y)
            print(
                "{:4d}".format(star.cluster_number),
                "{:6d}".format(number_of_pixels),
                "{:10.1f}".format(flux),
                "{:8.2f}".format(mux),
                "{:8.2f}".format(muy),
                "{:8.3f}".format(sigma_x),
                "{:8.3f}".format(sigma_y),
                "{:8.3f}".format(rho_xy),
            )

    def plot_starfield(self, data, norm=None, vmin_frac=0.99, vmax_frac=1.01, cmap="Greys", use_wcs=True, figsize=(8, 4)):

        # Set up the kwargs
        ims_kw = dict(cmap=cmap, origin="upper")
        subplot_kw = {}
        subplot_set_kw = {}

        if use_wcs:
            wcs = WCS(self.header)
            subplot_kw.update(dict(projection=wcs))
            ims_kw.update(dict(origin=None))
            subplot_set_kw.update(xlabel="RA", ylabel="DEC")
        else:
            subplot_set_kw.update(xlabel="Pixels", ylabel="Pixels")

        fig, ax = plt.subplots(nrows=1, figsize=figsize, subplot_kw=subplot_kw)

        vminVal = vmin_frac * np.median(data)
        vmaxVal = vmax_frac * np.median(data)

        if norm == "lognorm":
            norm = matplotlib.colors.LogNorm(vmin=vminVal, vmax=vmaxVal)
            ims_kw["norm"] = norm
        else:
            ims_kw.update(dict(vmin=vminVal, vmax=vmaxVal, norm=norm))

        cbar = ax.imshow(data, **ims_kw)
        fig.colorbar(cbar, label="Counts")

        ifr = self.in_usuable_region
        boundary = np.zeros_like(ifr, dtype=np.uint8)
        boundary[ifr] = 255  # Set boundary pixels to white
        ax.set(title=f"File: {self.fits_file}", **subplot_set_kw)
        ax.contour(boundary, levels=[0.5], colors="dodgerblue", linestyles="dashed")

        if use_wcs:
            ax.coords.grid(color='white', alpha=0.5, linestyle='solid')
            lon = ax.coords[0]
            lat = ax.coords[1]

            lon.set_major_formatter('hh:mm:ss')
            lon.set_ticks(spacing=1. * units.arcmin)
            lon.set_ticklabel(exclude_overlapping=True)
            lon.set_axislabel("RA")

            lat.set_major_formatter('dd:mm')
            lat.set_ticks(spacing=1. * units.arcmin)
            lat.set_ticklabel(exclude_overlapping=True)
            lat.set_axislabel("DEC")

        image_name = self.fits_file_basename.replace(".fit", ".png")
        image_path = os.path.join(self.fits_file_dirname, image_name)

        fig.savefig(image_path)


def gaussian_background_fit(image_region):
    """Fast Gaussian fitter for background estimation"""
    nData = image_region.flatten()
    nMin = np.amin(nData)
    numBins = 400
    nMax = nMin + numBins
    nHist, bin_edges = np.histogram(nData, bins=numBins, range=(nMin, nMax))
    xLoFit = np.around(np.percentile(nData, 10))
    xHiFit = np.around(np.percentile(nData, 70))
    binLoFit = np.clip(
        np.searchsorted(bin_edges, xLoFit, side="right") - 1, 0, numBins - 1
    )
    binHiFit = np.clip(np.searchsorted(bin_edges, xHiFit, side="left"), 0, numBins - 1)
    xFit = np.array(bin_edges[binLoFit : binHiFit + 1])
    offset = xFit[0]
    xFit -= offset  # Improved numerical stability
    nFit = nHist[binLoFit : binHiFit + 1]
    doQuickFit = np.amin(nFit) > 0.0  # othewise use median

    if doQuickFit:
        yFit = np.log(nFit)
        a = np.empty([3, 3])
        a[0, 0] = np.sum(nFit)
        a[0, 1] = a[1, 0] = np.dot(nFit, xFit)
        a[1, 1] = a[0, 2] = a[2, 0] = np.dot(nFit, xFit**2)
        a[1, 2] = a[2, 1] = np.dot(nFit, xFit**3)
        a[2, 2] = np.dot(nFit, xFit**4)
        b = np.empty([3])
        b[0] = np.dot(nFit, yFit)
        b[1] = np.sum(nFit[:] * xFit[:] * yFit[:])
        b[2] = np.sum(nFit[:] * xFit[:] * xFit[:] * yFit[:])
        x, residuals, rank, s = np.linalg.lstsq(a, b, rcond=None)
        mu = -0.5 * x[1] / x[2] + offset
        sigma = np.sqrt(-0.5 / x[2]) if x[2] < 0 else 0
    else:
        mu = np.median(nData)
        sigma = np.std(nData)
    return mu, sigma
