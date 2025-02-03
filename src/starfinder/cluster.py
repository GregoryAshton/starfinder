import numpy as np


class Cluster:

    def __init__(self, cluster_number, pixel, starfield):
        self.cluster_number = cluster_number
        self.pixel_list = [pixel]
        self.sf = starfield

    def set_cluster_number(self, cluster_number):
        self.cluster_number = cluster_number

    def add_pixel(self, pixel):
        self.pixel_list.append(pixel)

    def number_of_pixels(self):
        return len(self.pixel_list)

    def flux(self):
        """Estimate of flux for all pixels in cluster"""
        flux = 0
        for pixel in self.pixel_list:
            i, j = pixel
            flux += self.sf.n_ij[i, j] - self.sf.b_ij[i, j]
        return flux

    def mu(self):
        """Barycentre"""
        sumx = 0
        sumy = 0
        denom = 0
        for pixel in self.pixel_list:
            i, j = pixel
            sumx += j * (self.sf.n_ij[i, j] - self.sf.b_ij[i, j])
            sumy += i * (self.sf.n_ij[i, j] - self.sf.b_ij[i, j])
            denom += self.sf.n_ij[i, j] - self.sf.b_ij[i, j]
        mux = sumx / denom
        muy = sumy / denom
        return mux, muy

    def cov(self):
        """ Covariance matrix """
        mux, muy = self.mu()
        jc = round(mux)
        ic = round(muy)
        nRow, nCol = self.sf.data_shape
        halfBox = 8
        minCol = max(jc - halfBox, 0)
        maxCol = min(jc + halfBox, nCol - 1)
        minRow = max(ic - halfBox, 0)
        maxRow = min(ic + halfBox, nRow - 1)
        vx = 0
        vy = 0
        covxy = 0
        denom = 0
        for i in range(minRow, maxRow):
            for j in range(minCol, maxCol):
                vx += (j - mux) ** 2 * (self.sf.n_ij[i, j] - self.sf.b_ij[i, j])
                vy += (i - muy) ** 2 * (self.sf.n_ij[i, j] - self.sf.b_ij[i, j])
                covxy += (j - mux) * (i - muy) * (self.sf.n_ij[i, j] - self.sf.b_ij[i, j])
                denom += self.sf.n_ij[i, j] - self.sf.b_ij[i, j]
        if denom > 0:
            vx = vx / denom
            vy = vy / denom
            covxy = covxy / denom
        else:
            vx = -1.0
            vy = -1.0
            covxy = -1.0
        cov = np.array([[vx, covxy], [covxy, vy]])
        return cov
