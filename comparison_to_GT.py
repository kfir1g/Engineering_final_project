from nibabel import load
from numpy import logical_and, sum


def get_dc_bone_fracture(seg_obj, gt_obj, bone_color, frac_color):
    """
    Retruns the dice coefficient of the fracture
    :param seg_obj: Segmentation object from the algorithm
    :param gt_obj: Ground truth segmentation
    :param bone_color: Color of the bone
    :param frac_color: Color of the fracture
    :return:
    """
    gt_bone = gt_obj == bone_color
    gt_fracture = gt_obj == frac_color
    seg_bone = seg_obj == bone_color
    seg_fracture = seg_obj == frac_color
    bone_intersection = logical_and(gt_bone, seg_bone)
    bone_intersection = sum(bone_intersection)
    fracture_intersection = logical_and(gt_fracture, seg_fracture)
    fracture_intersection = sum(fracture_intersection)
    sum_gt_fracture = sum(gt_fracture)
    sum_gt_bone = sum(gt_bone)
    sum_seg_bone = sum(gt_bone)
    sum_seg_fracture = sum(seg_fracture)
    del gt_bone, gt_fracture, seg_bone, seg_fracture
    if (sum_seg_bone + sum_gt_bone) == 0:
        return [2, 2]
    bone_dc = 2 * bone_intersection / (sum_seg_bone + sum_gt_bone)
    if (sum_gt_fracture + sum_seg_fracture) == 0:
        fracture_dc = 2
    else:
        fracture_dc = 2 * fracture_intersection / (sum_gt_fracture
                                                   + sum_seg_fracture)
    return [bone_dc, fracture_dc]


def compare(algo_seg, ground_truth, bone_color, frac_color):
    """
    Loads segmentation and print the difference
    :param algo_seg: Segmentation object from the algorithm
    :param ground_truth: Ground truth segmentation
    :param bone_color: Color of the bone
    :param frac_color: Color of the fracture
    :return:
    """
    ground_truth = load(ground_truth)
    ground_truth_data = ground_truth.get_data()
    seg = load(algo_seg)
    seg_data = seg.get_data()
    del ground_truth, seg
    print(get_dc_bone_fracture(seg_data, ground_truth_data, bone_color,
                               frac_color))
    del seg_data, ground_truth_data


def compare_every_slice(algo_seg, gt_seg, bone_color, frac_color):
    """
    comapre dice coefficient per slice
    :param algo_seg: Segmentation object from the algorithm
    :param gt_seg: Ground truth segmentation
    :param bone_color: Color of the bone
    :param frac_color: Color of the fracture
    """
    file_name = algo_seg.split(".")[0]
    f = open(file_name + "_results.csv", "w")
    f.write("0000 \n bone DC , fracture DC \n")
    seg = load(algo_seg)
    gt = load(gt_seg)
    seg_data = seg.get_data()
    gt_data = gt.get_data()
    del seg, gt
    x, y, z = seg_data.shape

    for i in range(x):
        current_slice_seg = seg_data[i, :, :]
        current_slice_gt = gt_data[i, :, :]
        if (sum(current_slice_gt) + sum(current_slice_seg)) == 0:
            continue
        to_print = get_dc_bone_fracture(current_slice_seg, current_slice_gt,
                                        bone_color, frac_color)
        to_print = str(str(to_print[0]) + "," + str(to_print[1]) + "\n")
        f.write(str(to_print))
        del current_slice_seg, current_slice_gt
