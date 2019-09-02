from scipy.signal import convolve2d
from Bone import *
import copy
from numpy import cov, mean, std, sqrt
from numpy import sum as npsum
from numpy import abs as npabs
from numpy.linalg import svd
from pandas import DataFrame

NUM_FRAC_SEEDS = 10
FRACTION_DILATION = 1
SCAPHOID_COLOR = 1
FRACTURE_COLOR = 2
DURSAL_LUNAR = 5
DURSAL_RADIAL = 6
VOLAR_LUNAR = 7
VOLAR_RADIAL = 8


class Scaphoid(Bone):
    def __init__(self, original_path, bone_seeds, fracture_seeds, dilation):
        """
        :param original_path: string representing the path of the scan
        :param bone_seeds: an array of seeds to start the bone from
        :param fracture_seeds: an array of seeds to start the fracture from
        :param dilation: an integer, how many times should the bone use
        dilation"""
        Bone.__init__(self, original_path, bone_seeds, dilation)
        self.__possible_fractures = ndarray((0, 0, 0))
        self.__fraction_seeds = fracture_seeds
        self.__fraction_alone = ndarray((0, 0, 0))
        self.__bone_with_fracture = ndarray((0, 0, 0))
        self.__surface_area = ndarray((0, 0, 0))
        self.__bone_quarters = ndarray((0, 0, 0))
        self.__fracture_quarters = ndarray((0, 0, 0))

    def __del__(self):
        Bone.__del__(self)
        del self.__possible_fractures
        del self.__fraction_seeds
        del self.__fraction_alone
        del self.__bone_with_fracture
        del self.__surface_area
        del self.__bone_quarters
        del self.__fracture_quarters
        collect()

    def region_growing_from_input(self, color, bone_from_scan=None):
        """
        This function runs the region growing algorithm
        :param color: the color you want for the bone
        :param bone_from_scan: If the same scan were used inside another bone
        """
        collect()
        # initilize
        if not bone_from_scan:
            self.load_original_data()
        else:
            self.copy_original_from_bone(bone_from_scan)
        checked = zeros(self._original_img_data.shape)
        seg = zeros(self._original_img_data.shape)
        need_to_check = []
        # Color the seeds and check for neighbors
        for seed in self._seeds_points:
            seg[seed] = color
            checked[seed] = 1
            neighbors = self._get_neighbors(seed, checked, self.
                                            _original_img_data.shape)
            for neighbor in neighbors:
                if self._get_threshold(self._original_img_data[neighbor],
                                       BONE_BOUND_VALUES[0],
                                       BONE_BOUND_VALUES[1]):
                    need_to_check.append(neighbor)
        # Region Growing - while there's a neighbor, color it and keep going
        while need_to_check:
            pt = need_to_check.pop()
            if checked[pt] == 1:
                continue
            else:
                checked[pt] = 1
                neighbors = self._get_neighbors(pt, checked, self.
                                                _original_img_data.shape)
                for neighbor in neighbors:
                    if self._get_threshold(
                            self._original_img_data[neighbor],
                            BONE_BOUND_VALUES[0],
                            BONE_BOUND_VALUES[1]):
                        need_to_check.append(neighbor)
                seg[pt] = color
        # Closing holes
        del need_to_check, checked
        for i in range(self._dilation):
            seg = dilation(seg, cube(3, uint8))
        for i in range(self._dilation):
            seg = erosion(seg, cube(3, uint8))
        self._segmentation_data = seg
        del seg
        collect()

    def neighbors_for_fracture(self, pt, checked):
        """
        Get all of the neighbors of the pt that weren't checked yet
        :param pt: current point
        :param checked: 3D array of all the other points - it they were checked
        or not
        :return: an array of the relevant neighbors
        """
        seeds_std = std([self._original_img_data[x] for x in
                      self.__fraction_seeds])
        seeds_mean = mean([self._original_img_data[x] for x in
                        self.__fraction_seeds])
        neighbors = self._get_neighbors(pt, checked, self.
                                        _original_img_data.shape)
        relevant_neighbors = []
        value = self._original_img_data[pt]
        for neighbor in neighbors:
            if self._get_threshold(self._original_img_data[neighbor],
                                   VOID_VALUES[0], (value + 5 * seeds_mean) / 6 +
                                                   1.2 * seeds_std) and \
                    self._segmentation_data[neighbor] != 0:
                relevant_neighbors.append(neighbor)
        return relevant_neighbors

    def segment_fracture_region_growing_mean(self, color, bone_color=1):
        """
        Region growing for the fracture using different function to choose
        which neighbors are relevant
        :param color: color we want for the fracture
        :param bone_color: color we want for the bone
        :return:
        """
        if self._segmentation_data.size == 0:
            self.region_growing_from_input(bone_color)
        self.__possible_fractures = zeros(self._original_img_data.shape)
        self.__fraction_alone = zeros(self._original_img_data.shape)

        checked = zeros(self._original_img_data.shape)
        seg = zeros(self._original_img_data.shape)
        need_to_check = []
        for seed in self.__fraction_seeds:
            seg[seed] = color
            checked[seed] = 1
            need_to_check.extend(self.neighbors_for_fracture(seed, checked))
        # Region Growing - while there's a neighbor, color it and keep going
        while need_to_check:
            pt = need_to_check.pop()
            if checked[pt] == 1:
                continue
            else:
                checked[pt] = 1
                need_to_check.extend(self.neighbors_for_fracture(pt, checked))
                seg[pt] = color
        # Closing holes
        del need_to_check, checked
        for i in range(FRACTION_DILATION):
            seg = dilation(seg, cube(3, uint8))
        for i in range(FRACTION_DILATION):
            seg = erosion(seg, cube(3, uint8))
        self.__fraction_alone = seg
        del seg
        frac = nonzero(self.__fraction_alone)
        self.__bone_with_fracture = copy.deepcopy(self._segmentation_data)
        for i in range(len(frac[0]) - 1):
            self.__bone_with_fracture[
                (frac[0][i], frac[1][i], frac[2][i])] = color
        del frac

    def save_bone_with_fracture(self, output_path):
        """Saves the fracture alone to a file"""
        img = load(self._original_path)
        img_data = img.get_data()
        img_data[::] = self.__bone_with_fracture
        save(img, output_path)
        del img_data, img
        collect()

    def save_fracture_without_bone(self, output_path):
        """Saves the fracture and the bone to a file"""
        img = load(self._original_path)
        img_data = img.get_data()
        img_data[::] = self.__fraction_alone
        save(img, output_path)
        del img_data, img
        collect()

    def load_bone_fracture(self, path):
        """Loads a bone and fracture array"""
        img = load(path)
        img_data = img.get_data()
        self.__bone_with_fracture = zeros(img_data.shape)
        self.__bone_with_fracture[img_data == SCAPHOID_COLOR] = SCAPHOID_COLOR
        self.__bone_with_fracture[img_data == FRACTURE_COLOR] = FRACTURE_COLOR
        del img, img_data

    def load_fracture_from_bone_fracture(self):
        """gets the fraction from bone_fraction"""
        self.__fraction_alone = zeros(self.__bone_with_fracture.shape)
        self.__fraction_alone[self.__bone_with_fracture == FRACTURE_COLOR] \
            = FRACTURE_COLOR

    def get_fracture(self):
        """Getter for the fracture"""
        return self.__fraction_alone

    def get_fracture_with_bone(self):
        """Getter for the fracture with the bone"""
        return self.__bone_with_fracture

    def load_fracture(self, path):
        """Loads the fracture"""
        img = load(path)
        self.__fraction_alone = img.get_data()
        del img

    def divide_bone_into_quarters(self, pca):
        """Divides the bone into quarters where PCA is components from PCA"""
        self.__bone_quarters = self.divide_segmentation_into_quarters(
            pca, self.__bone_with_fracture)

    def divide_fracture_into_quarters(self, pca):
        """Divides the fractures into quarters where PCA is components from PCA"""
        self.__fracture_quarters = self.divide_segmentation_into_quarters(
            pca, self.__fraction_alone)

    def get_geometric_features(self):
        """
        gets segmentation and produce edges
        :param seg_scan:
        :return:
        """
        geo_features = dict()
        img = load(self._original_path)
        header = img.header
        del img
        zooms = header.get_zooms()
        del header
        voxel_area = zooms[0] * zooms[2]
        voxel_volume = zooms[0]*zooms[1]*zooms[2]
        geo_features["Fracture Volume"] = str(voxel_volume * npsum(
            self.__fraction_alone == FRACTURE_COLOR)) + " mm^3"

        x, y, z = self.__fraction_alone.shape
        self.__surface_area = copy.deepcopy(self.__fraction_alone)
        self.__surface_area = self.__surface_area.astype(int)
        for i in range(z):
            cur_im = self.__surface_area[:, :, i]
            edge = conv_der(cur_im)
            self.__surface_area[:, :, i] = edge
        self.__surface_area[self.__fraction_alone != FRACTURE_COLOR] = 0
        voxels_in_area = npsum(self.__surface_area == FRACTURE_COLOR)
        geo_features["Fracture Surface Area"] = str(voxels_in_area *
                                                    voxel_area) + " mm^2"

        more_features = self.longest_dist_fracture(zooms)
        for k, v in more_features.items():
            geo_features[k] = v
        # geo_features["Flatness Ratio"] = self.flatness()
        del zooms, more_features
        return geo_features

    def get_bone_quarters(self):
        """returns the bone's quarters"""
        return self.__bone_quarters

    def get_fracture_quarters(self):
        """returns the fracture's quarters"""
        return self.__fracture_quarters

    def divide_segmentation_into_quarters(self, pca_comp, segmentation):
        """
        Using PCA axes from RADIUS segmentation to partitioning the
        fracture into quarters
        :param pca_comp: PCA components from PCA
        :param seg: numpy array array - Bone & fracture segmentation
        :return: Fracture with colorize quarters
        """
        # finds centroid of the fracture
        centroid = self.find_centeroid_segmentation(segmentation)
        # extract pca components
        first_axis = pca_comp[2]
        second_axis = pca_comp[1]

        # Calculates the sum of the plane
        d_1 = centroid[0]*first_axis[0] + centroid[1]*first_axis[1] + \
              centroid[2]*first_axis[2]
        d_2 = centroid[0]*second_axis[0] + centroid[1]*second_axis[1] + \
              centroid[2]*second_axis[2]
        del centroid
        seg_nonzero = nonzero(segmentation)
        nonzero_x = seg_nonzero[0]
        nonzero_y = seg_nonzero[1]
        nonzero_z = seg_nonzero[2]
        del seg_nonzero
        quarters = zeros(segmentation.shape)

        for i, j, k in zip(nonzero_x, nonzero_y, nonzero_z):
            if (i*first_axis[0]+j*first_axis[1]+k*first_axis[2]) > d_1:
                if (i*second_axis[0]+j*second_axis[1]+k*second_axis[2]) > d_2:
                    quarters[i, j, k] = DURSAL_LUNAR
                else:
                    quarters[i, j, k] = DURSAL_RADIAL
            else:
                if (i * second_axis[0] + j * second_axis[1] +
                    k * second_axis[2]) > d_2:
                    quarters[i, j, k] = VOLAR_LUNAR
                else:
                    quarters[i, j, k] = VOLAR_RADIAL
        del first_axis, second_axis, nonzero_x, nonzero_y, nonzero_z
        return quarters

    def flatness(self):
        """Calculate the flatness of the fracture - The feacture currently
        doesn't work"""
        cov_mat = cov(self.__fraction_alone)
        u, s, v = svd(cov_mat)
        return s[1] / s[0]

    def longest_dist_fracture(self, zooms):
        """This function returns the longest distances of the fracture
        from https://python-graph-gallery.com/372-3d-pca-result/
        :param zooms: distances of each axis, from the file
        :return:
        """
        X = array(nonzero(self.__fraction_alone == FRACTURE_COLOR))
        X = X.T
        pca = PCA(n_components=3)
        pca.fit(X)
        result = DataFrame(pca.transform(X), columns=['PCA%i' % i for i in
                                                         range(3)])
        del X, pca
        shortest_pixel_dist = zooms[0]
        biggest_pixel_dist = zooms[2]
        max_first, max_sec, max_third = result['PCA0'].max(), result[
            'PCA1'].max(), result['PCA2'].max()
        min_first, min_sec, min_third = result['PCA0'].min(), result[
            'PCA1'].min(), result['PCA2'].min()
        del result
        length_along_axis_1 = max_first - min_first
        length_along_axis_2 = max_sec - min_sec
        length_along_axis_3 = max_third - min_third

        largest_possible_dist_along_axis_1 = biggest_pixel_dist * length_along_axis_1
        shortest_possible_dist_along_axis_1 = shortest_pixel_dist * length_along_axis_1
        largest_possible_dist_along_axis_2 = biggest_pixel_dist * length_along_axis_2
        shortest_possible_dist_along_axis_2 = shortest_pixel_dist * length_along_axis_2
        largest_possible_dist_along_axis_3 = biggest_pixel_dist * length_along_axis_3
        shortest_possible_dist_along_axis_3 = shortest_pixel_dist * length_along_axis_3

        features = {}
        features["Length in Axis 1"] = "between " + \
                                       str(shortest_possible_dist_along_axis_1) \
                                       + " and " + str(largest_possible_dist_along_axis_1) + "mm"
        features["Length in Axis 2"] = "between " + \
                                       str(shortest_possible_dist_along_axis_2) \
                                       + " and " + str(largest_possible_dist_along_axis_2) + "mm"
        features["Length in Axis 3"] = "between " + \
                                       str(shortest_possible_dist_along_axis_3) \
                                       + " and " + str(largest_possible_dist_along_axis_3) + "mm"

        return features


def conv_der(im):
    """
    calculate derivative of the image with convolution we learn in class
    :param im: image
    :return: the derived image
    """
    x_conv = array([[1, 0, -1]])
    x_der = convolve2d(im, x_conv, mode='same')
    y_der = convolve2d(im, x_conv.T, mode='same')
    magnitude = sqrt((npabs(x_der) ** 2) + (npabs(y_der) ** 2))
    del x_conv, x_der, y_der
    return magnitude
