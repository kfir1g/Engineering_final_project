from nibabel import load, save
from numpy import ndarray, zeros, nonzero, array, int, median, uint8
from gc import collect
from skimage.morphology import dilation, erosion, cube
from sklearn.decomposition import PCA

BONE_BOUND_VALUES = (400, 1000)
VOID_VALUES = (-100, 200)
NUM_BONE_SEEDS = 20


class Bone:
    """This class Represents bones, any bone can be expanded from it"""
    def __init__(self, original_path, seeds_array, dilation):
        """
        :param original_path: string representing the path of the scan
        :param seeds_array: an array of seeds to start from
        :param dilation: an integer, how many times should the bone use
        dilation
        """
        self._original_path = original_path
        self._seeds_points = seeds_array
        self._original_img_data = ndarray((0, 0, 0))
        self._segmentation_data = ndarray((0, 0, 0))
        self._dilation = dilation

    def __del__(self):
        del self._seeds_points
        del self._original_img_data
        del self._segmentation_data
        collect()

    @staticmethod
    def _get_neighbors(pt, checked, dimensions):
        """
        Get all of the neighbors of the pt that weren't checked yet
        :param pt: current point
        :param checked: 3D array of all the other points - it they were checked
        or not
        :param dimensions: The dimensions of the scan
        :return: an array of the relevant neighbors
        """
        neighbors = []
        if pt[0] - 1 > 0 and checked[pt[0] - 1, pt[1], pt[2]] == 0:
            neighbors.append((pt[0] - 1, pt[1], pt[2]))
        if pt[1] - 1 > 0 and checked[pt[0], pt[1] - 1, pt[2]] == 0:
            neighbors.append((pt[0], pt[1] - 1, pt[2]))
        if pt[2] - 1 > 0 and checked[pt[0], pt[1], pt[2] - 1] == 0:
            neighbors.append((pt[0] - 1, pt[1], pt[2] - 1))
        if pt[0] + 1 < dimensions[0] and checked[pt[0] + 1, pt[1], pt[2]] == 0:
            neighbors.append((pt[0] + 1, pt[1], pt[2]))
        if pt[1] + 1 < dimensions[1] and checked[pt[0], pt[1] + 1, pt[2]] == 0:
            neighbors.append((pt[0], pt[1] + 1, pt[2]))
        if pt[2] + 1 < dimensions[2] and checked[pt[0], pt[1], pt[2] + 1] == 0:
            neighbors.append((pt[0], pt[1], pt[2] + 1))
        return neighbors

    @staticmethod
    def _get_threshold(value, min_value, max_value):
        """This function returns true if the compared point is similar to the
        original point, false otherwise"""
        return min_value < value < max_value

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
        for i in range(self._dilation - 1):
            seg = erosion(seg, cube(3, uint8))
        self._segmentation_data = seg
        del seg
        collect()

    def save_segmentation(self, output_path):
        """This function saves the segmentation to a output_file"""
        img = load(self._original_path)
        img_data = img.get_data()
        img_data[::] = self._segmentation_data
        save(img, output_path)
        del img_data, img
        collect()

    def get_segmentation(self):
        """Getter for the segmentation"""
        return self._segmentation_data

    def get_original_data(self):
        """Getter for the original image data"""
        return self._original_img_data

    def copy_original_from_bone(self, bone):
        """Copy the original data from a different bone of type Bone"""
        self._original_img_data = bone.get_original_data()

    def load_original_data(self):
        """loads the original data from a file"""
        img = load(self._original_path)
        self._original_img_data = img.get_data()
        del img

    def load_segmentation(self, path):
        """loads the segmentation from a file"""
        img = load(path)
        self._segmentation_data = img.get_data()
        del img

    def get_original_path(self):
        """Getter for the path"""
        return self._original_path

    def extract_PCA_components(self):
        """Gets the PCA components of the bone"""
        segmentation_nonzero = nonzero(self._segmentation_data)
        X = array(segmentation_nonzero[:])
        X = X.T
        pca = PCA(n_components=3)
        pca.fit(X)
        del X, segmentation_nonzero
        return pca.components_

    def find_centeroid_segmentation(self, segmentation):
        """returns the centeroid of the segmentation, where segmentation is
        ndarray"""
        seg_nonzero = nonzero(segmentation)
        x_center = int(median(seg_nonzero[0]))
        y_center = int(median(seg_nonzero[1]))
        z_center = int(median(seg_nonzero[2]))
        del seg_nonzero
        return [x_center, y_center, z_center]
