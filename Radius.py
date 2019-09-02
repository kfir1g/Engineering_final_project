from Bone import *


RADIUS_VALUES = (400, 2000)


class Radius(Bone):
    def __init__(self, original_path, seeds_array, dilation):
        """
        :param original_path: string representing the path of the scan
        :param seeds_array: an array of seeds to start from
        :param dilation: an integer, how many times should the bone use
        dilation"""
        Bone.__init__(self, original_path, seeds_array, dilation)

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
                                       VOID_VALUES[0],
                                       VOID_VALUES[1]):
                    need_to_check.append(neighbor)
        # Region Growing - while there's a neighbor, color it and keep going
        bone_to_check = []
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
                            VOID_VALUES[0], VOID_VALUES[1]):
                        need_to_check.append(neighbor)
                    if self._get_threshold(
                            self._original_img_data[neighbor],
                            BONE_BOUND_VALUES[0], BONE_BOUND_VALUES[1]):
                        bone_to_check.append(neighbor)
                seg[pt] = color
        # Closing holes
        del need_to_check
        # check for Bone value - edge of the radius
        while bone_to_check:
            pt = bone_to_check.pop()
            if checked[pt] == 1:
                continue
            else:
                checked[pt] = 1
                neighbors = self._get_neighbors(pt, checked, self.
                                                _original_img_data.shape)
                for neighbor in neighbors:
                    if self._get_threshold(
                            self._original_img_data[neighbor],
                            RADIUS_VALUES[0], RADIUS_VALUES[1]):
                        bone_to_check.append(neighbor)
                seg[pt] = color
        del checked, bone_to_check
        for i in range(self._dilation):
            seg = dilation(seg, cube(3, uint8))
        for i in range(self._dilation - 1):
            seg = erosion(seg, cube(3, uint8))
        self._segmentation_data = seg
        del seg
        collect()
