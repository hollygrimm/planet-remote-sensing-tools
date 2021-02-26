import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import rasterio

def get_bands(filename_img, dict_band_nums):
    bands = {}
    with rasterio.open(filename_img) as src:
        for band_label, band_num in dict_band_nums.items():
            bands[band_label] = src.read(band_num)
    return bands, src.meta

def _linear_scale(ndarray, old_min, old_max, new_min, new_max):
    """ 
    Linear scale from old_min to new_min, old_max to new_max.
  
    Values below min/max are allowed in input and output.
    Min/Max values are two data points that are used in the linear scaling.
    https://en.wikipedia.org/wiki/Normalization_(image_processing)
  
    Parameters: 
    ndarray : ndarray 3D array containing data with `double` type.
  
    Returns: 
    ndarray: normalized 3D array containing data with `double` type.
  
    """    
    return (ndarray - old_min)*(new_max - new_min)/(old_max - old_min) + new_min


def bands_to_display(bands, alpha=True):
    """Converts a list of bands to a 3-band rgb, normalized array for display."""
    rgb_bands = np.dstack(bands[:3])

    old_min = np.percentile(rgb_bands, 2)
    old_max = np.percentile(rgb_bands, 98)
    new_min = 0
    new_max = 1
    scaled = _linear_scale(rgb_bands.astype(np.double),
                           old_min, old_max, new_min, new_max)
    bands = np.clip(scaled, new_min, new_max)
    if alpha is True:
        bands = _add_alpha_mask(bands)
    return bands


"""
The NDVI values will range from -1 to 1. You want to use a diverging color scheme to visualize the data,
and you want to center the colorbar at a defined midpoint. The class below allows you to normalize the colorbar.
"""
class midpoint_normalize(colors.Normalize):
    """
    Normalise the colorbar so that diverging bars work there way either side from a prescribed midpoint value)
    e.g. im=ax1.imshow(array, norm=MidpointNormalize(midpoint=0.,vmin=-100, vmax=100))
    Credit: Joe Kington, http://chris35wills.github.io/matplotlib_diverging_colorbar/
    Credit: https://stackoverflow.com/a/48598564
    """
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)
    
    def __call__(self, value, clip=None):    
        # Note that I'm ignoring clipping and other edge cases here.
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        
        # REVIEW: Alernate method to calculate
        # result, is_scalar = self.process_value(value)
        # return np.ma.array(np.interp(value, x, y), mask=result.mask, copy=False)

        return np.ma.masked_array(np.interp(value, x, y), np.isnan(value))

def get_reflectance_coeffs(filename_metadata):
    from xml.dom import minidom

    xmldoc = minidom.parse(filename_metadata)
    nodes = xmldoc.getElementsByTagName('ps:bandSpecificMetadata')

    # XML parser refers to bands by numbers 1-4
    coeffs = {}
    for node in nodes:
        bn = node.getElementsByTagName('ps:bandNumber')[0].firstChild.data
        if bn in ['1', '2', '3', '4']:
            i = int(bn)
            value = node.getElementsByTagName('ps:reflectanceCoefficient')[0].firstChild.data
            coeffs[i] = float(value)
    return coeffs


def show_ndvi_fig(ndvi, filename_ndvi_fig, midpoint=0, figsize=(20, 10)):
    """
    set midpoint according to how NDVI is interpreted: https://earthobservatory.nasa.gov/Features/MeasuringVegetation/
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    # diverging color scheme chosen from https://matplotlib.org/users/colormaps.html
    cmap = plt.cm.RdYlGn 

    # Set min/max values from NDVI range for image (excluding NAN)
    mmin = np.nanmin(ndvi)
    mmax = np.nanmax(ndvi)

    cax = ax.imshow(ndvi, cmap=cmap, clim=(mmin, mmax),
                    norm=midpoint_normalize(midpoint=midpoint, vmin=mmin, vmax=mmax))

    ax.axis('off')
    ax.set_title('Normalized Difference Vegetation Index', fontsize=18, fontweight='bold')

    cbar = fig.colorbar(cax, orientation='horizontal', shrink=0.5)

    fig.savefig(filename_ndvi_fig, dpi=200, bbox_inches='tight', pad_inches=0.7)

    plt.show()

def show_ndvi_hist(ndvi, filename_ndvi_hist, figsize=(10, 10)):
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)

    plt.title("NDVI Histogram", fontsize=18, fontweight='bold')
    plt.xlabel("NDVI values", fontsize=14)
    plt.ylabel("# pixels", fontsize=14)

    x = ndvi[~np.isnan(ndvi)]
    numBins = 20
    ax.hist(x, numBins, color='green', alpha=0.8)

    fig.savefig(filename_ndvi_hist, dpi=200, bbox_inches='tight', pad_inches=0.7)

    plt.show()


# TODO:
# def plot_image(masked_bands, title=None, figsize=(10, 10)):
#     fig = plt.figure(figsize=figsize)
#     ax = fig.add_subplot(1, 1, 1)
#     show(ax, masked_bands)
#     if title:
#         ax.set_title(title)
#     ax.set_axis_off()


# def show(axis, bands, alpha=True):
#     """Show bands as image with option of converting mask to alpha.

#     Alters axis in place.
#     """
#     assert len(bands) in [1, 3]

#     mask = None
#     try:
#         mask = bands[0].mask
#     except AttributeError:
#         # no mask
#         pass

#     bands = [b for b in bands.copy()]  # turn into list
#     bands = _scale_bands(bands)

#     if alpha and len(bands) == 3 and mask is not None:
#         bands.append(_mask_to_alpha(mask))

#     if len(bands) >= 3:
#         dbands = np.dstack(bands)
#     else:
#         dbands = bands[0]

#     return axis.imshow(dbands)


# def _mask_bands(bands, mask):
#     return [np.ma.array(b, mask) for b in bands]


# def _scale_bands(bands):
#     def _percentile(bands, percentile):
#         all_pixels = np.concatenate([b.compressed() for b in bands])
#         return np.percentile(all_pixels, percentile)

#     old_min = _percentile(bands, 2)
#     old_max = _percentile(bands, 98)
#     new_min = 0
#     new_max = 1

#     def _linear_scale(ndarray, old_min, old_max, new_min, new_max):
#         # https://en.wikipedia.org/wiki/Normalization_(image_processing)
#         return (ndarray - old_min) * (new_max - new_min) / (old_max - old_min) + new_min

#     scaled = [np.clip(_linear_scale(b.astype(np.float),
#                                     old_min, old_max,
#                                     new_min, new_max),
#                       new_min, new_max)
#               for b in bands]

#     filled = [b.filled(fill_value=0) for b in scaled]
#     return filled


# def _mask_to_alpha(mask):
#     alpha = np.zeros_like(np.atleast_3d(mask))
#     alpha[~mask] = 1
#     return alpha

