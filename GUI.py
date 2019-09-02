from tkinter.ttk import Progressbar
from tkinter import filedialog, messagebox, Tk, Frame, Button, Label, Entry
from tkinter import HORIZONTAL, StringVar, N, S, W, E
from os.path import exists, join, abspath
from threading import Thread
from Scaphoid import *
from random import randint
from Radius import *
from numpy import dot, arccos, rad2deg, mod, where
# import comparison_to_GT as GT

# SCANS = ["0000021A.nii.gz", "0000009C.nii.gz", "000000C3.nii.gz"]
# HINTS = ["21hint.nii.gz", "9hint.nii.gz", "3hint.nii.gz"]
# GROUND_TRUTH = ["21seg.nii.gz", "009seg.nii.gz", "3seg.nii.gz"]
SAVE_FRACTURE_NAME = "FRACTION.nii.gz"
SAVE_SEGMENTATION_NAME_SC = "SCAPHOID_SEGMENT.nii.gz"
SAVE_SEGMENTATION_NAME_RA = "RADIUS_SEGMENT.nii.gz"
SAVE_BONE_FRACTURE = "BONE_FRACTURE.nii.gz"
SAVE_SEGMENTATION_NAME_CA = "CAPITATE_SEGMENT.nii.gz"
SAVE_FRACTURE_QUARTARS_FILE = "DIVIDED_FRACTURE.nii.gz"
SAVE_BONE_QUARTARS_FILE = "DIVIDED_BONE.nii.gz"
SAVE_GEOMETRICAL_FEATURES = "GEOMETRICAL_FEATURES.txt"
OUTPUT_PATH = ""
RADIUS_COLOR = 3
CAPITATE_UP_COLOR = 4
CAPITATE_DOWN_COLOR = 5


class GUI(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.user_frame = Frame()
        self.back_button = Button(self.user_frame, text="Back", command=self.
                                  back_to_home_gui)
        self.segmentation_button = Button(self.user_frame, text="Segmentation",
                                          command=self.segmentation_gui,
                                          height=2, width=10)
        self.analysis_button = Button(self.user_frame, text="Analysis",
                                      command=self.analysis_gui, height=2,
                                      width=10)
        self.progress = Progressbar(self.user_frame, orient=HORIZONTAL,
                                    length=200, mode='indeterminate')
        self.progress_label = Label(self.user_frame)
        # output folder
        self.output_path = StringVar("")
        self.output_label = Label(self.user_frame, text="please select "
                                                        "output path")
        self.output_input = Entry(self.user_frame,
                                  textvariable=self.output_path, width=40)
        self.select_output = Button(self.user_frame, text=" ", command=self.
                                    select_output_folder)

        # original image
        self.file_path = StringVar("")
        self.file_label = Label(self.user_frame, text="please select the CT "
                                                      "scan")
        self.file_input = Entry(self.user_frame, textvariable=self.file_path,
                                width=40)
        self.select_file = Button(self.user_frame, text=" ",
                                  command=lambda: self.select_nifti_file(
                                      self.file_path))
        # seeds
        self.seeds_path = StringVar("")
        self.seeds_label = Label(self.user_frame, text="please select the "
                                                       "marking")
        self.seeds_input = Entry(self.user_frame, textvariable=self.
                                 seeds_path, width=40)
        self.select_seeds = Button(self.user_frame, text=" ",
                                   command=lambda: self.select_nifti_file(
                                          self.seeds_path))

        # run buttons
        self.segmentation_run = Button(self.user_frame, text="Run "
                                                             "Segmentation",
                                       command=self.run_segmentation)
        self.analysis_run = Button(self.user_frame, text="Run Analysis",
                                       command=self.run_analysis)

        self.default_background = self.output_input.cget("background")
        self.scaphoid = None
        self.radius = None
        self.initilize_gui()

    def __del__(self):
        del self.user_frame
        del self.back_button
        del self.segmentation_button
        del self.analysis_button
        del self.progress
        del self.progress_label
        del self.output_path
        del self.output_label
        del self.output_input
        del self.select_output
        del self.file_path
        del self.file_label
        del self.file_input
        del self.select_file
        del self.seeds_path
        del self.seeds_label
        del self.seeds_input
        del self.select_seeds
        del self.segmentation_run
        del self.analysis_run
        del self.default_background
        del self.scaphoid
        del self.radius

    def select_output_folder(self):
        """Select an output folder"""
        path = filedialog.askdirectory()
        self.output_path.set(path)
        self.output_input.config(background=self.default_background)
        del path

    def select_nifti_file(self, var):
        """Select a nifti file"""
        input_path = filedialog.askopenfilename()
        var.set(input_path)
        self.user_frame.grid_slaves(int(str(var)[-1]) + 1, 1)[0].config(
            background=self.default_background)
        del input_path

    def back_to_home_gui(self):
        """forgets the other gui and reload the home gui"""
        self.forget_non_home_gui()
        self.seeds_path.set("")
        self.initilize_gui()

    def initilize_gui(self):
        """Initial GUI"""
        self.title("Scaphoid Fracture Segmentation and analysis")
        self.user_frame.grid()
        self.segmentation_button.grid(row=0, column=0, padx=(100, 50), pady=(
            30, 30))
        self.analysis_button.grid(row=0, column=1, padx=(50, 100),
                                  pady=(30, 30))

    def segmentation_gui(self):
        """Initial GUI of the segmentation"""
        self.segmentation_button.grid_forget()
        self.analysis_button.grid_forget()
        self.back_button.grid(row=0, pady=(2, 2))
        self.title("Scaphoid Fracture Segmentation")
        self.seeds_label.config(text="please select the marking")
        self.output_label.grid(row=1, column=0)
        self.output_input.grid(row=1, column=1)
        self.select_output.grid(row=1, column=2)
        self.file_label.grid(row=2, column=0)
        self.file_input.grid(row=2, column=1)
        self.select_file.grid(row=2, column=2)
        self.seeds_label.grid(row=3, column=0)
        self.seeds_input.grid(row=3, column=1)
        self.select_seeds.grid(row=3, column=2)
        self.segmentation_run.grid(row=4, columnspan=3, sticky=N + S + E + W)

    def analysis_gui(self):
        """Initial the analysis GUI"""
        self.segmentation_button.grid_forget()
        self.analysis_button.grid_forget()
        self.back_button.grid(row=0, pady=(2,2))
        self.title("Scaphoid Fracture analysis")
        self.seeds_label.config(text="please select the segmentation")
        self.output_label.grid(row=1, column=0)
        self.output_input.grid(row=1, column=1)
        self.select_output.grid(row=1, column=2)
        self.file_label.grid(row=2, column=0)
        self.file_input.grid(row=2, column=1)
        self.select_file.grid(row=2, column=2)
        self.seeds_label.grid(row=3, column=0)
        self.seeds_input.grid(row=3, column=1)
        self.select_seeds.grid(row=3, column=2)
        self.analysis_run.grid(row=4, columnspan=3, sticky=N + S + E + W)

    def forget_non_home_gui(self):
        """Forgets the grid of segmentation GUI"""
        collect()
        self.back_button.grid_forget()
        self.output_label.grid_forget()
        self.output_input.grid_forget()
        self.select_output.grid_forget()
        self.file_label.grid_forget()
        self.file_input.grid_forget()
        self.select_file.grid_forget()
        self.seeds_label.grid_forget()
        self.seeds_input.grid_forget()
        self.select_seeds.grid_forget()
        self.segmentation_run.grid_forget()
        self.analysis_run.grid_forget()

    def validate_data(self):
        """This function make sure that the data the user selected is valid"""
        valid = True
        if not exists(self.output_path.get()):
            valid = False
            self.output_input.config(background="tomato")
        file_path = self.file_path.get()
        seeds_path = self.seeds_path.get()
        if not (file_path.endswith(".nii.gz") and exists(file_path)):
            valid = False
            self.file_input.config(background="tomato")
        if not (seeds_path.endswith(".nii.gz") and exists(
                seeds_path)):
            valid = False
            self.seeds_input.config(background="tomato")
        return valid

    def run_segmentation(self):
        """Run the segmentation algorithm while updating the progressbar"""
        def threaded_prog():
            self.progress_label.grid(row=5, column=0)
            self.progress.grid(row=5, column=1, columnspan=2)
            self.progress.start()
            self.progress_label.config(text="Running Segmentation")
            self.segmentation_process()
            self.progress.stop()
            self.progress_label.grid_forget()
            self.progress.grid_forget()
            self.back_to_home_gui()
        if self.validate_data():
            Thread(target=threaded_prog).start()
        else:
            messagebox.showinfo("Error with the input", "Error with the input")

    def segmentation_process(self):
        """Creates the segmentation of the scaphoid and the fracture"""
        self.progress_label.config(text="Getting seeds")
        scaphoid_seeds, fracture_seeds = generate_scaphoid_seeds(
            self.seeds_path.get())
        self.progress_label.config(text="Isolating The Scaphoid")
        self.scaphoid = Scaphoid(self.file_path.get(), scaphoid_seeds,
                                 fracture_seeds, 6)
        self.scaphoid.region_growing_from_input(SCAPHOID_COLOR)
        self.progress_label.config(text="Isolating The Fracture")
        self.scaphoid.segment_fracture_region_growing_mean(FRACTURE_COLOR,
                                                           SCAPHOID_COLOR)

        self.progress_label.config(text="Saving Files")
        save_scaphoid_segmentations(self.scaphoid, self.output_path.get())
        self.progress_label.config(text="Finishing")
        self.scaphoid = None
        del scaphoid_seeds, fracture_seeds
        messagebox.showinfo("Process Finished Successfully",
                            "Process Finished Successfully")

    def run_analysis(self):
        """runs the analysis algorithm while updating the progress bar"""
        def threaded_prog():
            self.progress_label.grid(row=5, column=0)
            self.progress.grid(row=5, column=1, columnspan=2)
            self.progress.start()
            self.progress_label.config(text="Running Analysis")
            self.analysis_process()
            self.progress.stop()
            self.progress_label.grid_forget()
            self.progress.grid_forget()
            self.back_to_home_gui()
        if self.validate_data():
            Thread(target=threaded_prog).start()
        else:
            messagebox.showinfo("Error with the input", "Error with the input")

    def analysis_process(self):
        """The main analysis process"""
        self.progress_label.config(text="Getting the fracture")
        self.scaphoid = Scaphoid(self.file_path.get(), [], [], 1)
        self.scaphoid.load_bone_fracture(self.seeds_path.get())
        self.scaphoid.load_fracture_from_bone_fracture()
        self.progress_label.config(text="Getting seeds")
        radius_seeds, capitate_seg = generate_analysis_seeds(
            self.seeds_path.get())
        self.progress_label.config(text="Isolating The Radius")
        self.radius = Radius(self.file_path.get(), radius_seeds, 6)
        self.radius.region_growing_from_input(RADIUS_COLOR)
        self.progress_label.config(text="Getting PCA from radius")
        pca = self.radius.extract_PCA_components()
        self.progress_label.config(text="Dividing the bone")
        self.scaphoid.divide_bone_into_quarters(pca)
        self.progress_label.config(text="Dividing the fracture")
        self.scaphoid.divide_fracture_into_quarters(pca)
        self.progress_label.config(text="Getting geometric information")
        geo_features = self.scaphoid.get_geometric_features()
        geo_features["Angle between Radius and Capitate"] = \
            str(create_direction_vector_for_2_points_cloud(
                capitate_seg, pca[0]))
        self.progress_label.config(text="Saving Files")
        save_analysis_segmentation(self.scaphoid, self.output_path.get())

        file_name = str(abspath(self.scaphoid.get_original_path()).
                        split("\\")[-1].split(".")[0])
        save_geometric_features(geo_features, self.output_path.get(),
                                file_name)
        self.progress_label.config(text="Finishing")
        self.scaphoid = None
        self.radius = None
        del pca, geo_features
        messagebox.showinfo("Process Finished Successfully",
                            "Process Finished Successfully")


def generate_scaphoid_seeds(seeds_file):
    """Generate the seeds for the scaphoid and fracture"""
    collect()
    img = load(seeds_file)
    segmentation_data = img.get_data()
    del img
    # Scphoid Seeds
    seeds_counter = 0
    tuple_of_img_seg = where(segmentation_data == SCAPHOID_COLOR)
    scaphoid_seeds = []
    if len(tuple_of_img_seg[0]):
        while seeds_counter < NUM_BONE_SEEDS:
            pt = randomize_seed(tuple_of_img_seg)
            if pt not in scaphoid_seeds:
                scaphoid_seeds.append(pt)
                seeds_counter += 1
    # Fracture Seeds
    seeds_counter = 0
    tuple_of_img_seg = where(segmentation_data == FRACTURE_COLOR)
    fracture_seeds = []
    if len(tuple_of_img_seg[0]):
        for location in range(len(tuple_of_img_seg[0])):
            x_value = tuple_of_img_seg[0][location]
            y_value = tuple_of_img_seg[1][location]
            z_value = tuple_of_img_seg[2][location]
            pt = (x_value, y_value, z_value)
            if pt not in fracture_seeds:
                fracture_seeds.append(pt)
                seeds_counter += 1
    del tuple_of_img_seg, segmentation_data
    collect()
    return scaphoid_seeds, fracture_seeds


def generate_analysis_seeds(seeds_file):
    """Generate seeds for the analysis"""
    collect()
    img = load(seeds_file)
    segmentation_data = img.get_data()
    capitate_seg = zeros(segmentation_data.shape)
    del img
    seeds_counter = 0
    tuple_of_img_seg = where(segmentation_data == RADIUS_COLOR)
    radius_seeds = []
    if len(tuple_of_img_seg[0]):
        while seeds_counter < NUM_BONE_SEEDS:
            pt = randomize_seed(tuple_of_img_seg)
            if pt not in radius_seeds:
                radius_seeds.append(pt)
                seeds_counter += 1
    capitate_seg[segmentation_data == CAPITATE_UP_COLOR] = CAPITATE_UP_COLOR
    capitate_seg[segmentation_data == CAPITATE_DOWN_COLOR] = \
        CAPITATE_DOWN_COLOR
    del segmentation_data, tuple_of_img_seg
    collect()
    return radius_seeds, capitate_seg


def generate_radius_seeds(seeds_file):
    """Generate the seeds for the radius"""
    collect()
    img = load(seeds_file)
    segmentation_data = img.get_data()
    del img
    seeds_counter = 0
    tuple_of_img_seg = where(segmentation_data == RADIUS_COLOR)
    radius_seeds = []
    if len(tuple_of_img_seg[0]):
        while seeds_counter < NUM_BONE_SEEDS:
            pt = randomize_seed(tuple_of_img_seg)
            if pt not in radius_seeds:
                radius_seeds.append(pt)
                seeds_counter += 1
    del tuple_of_img_seg, segmentation_data
    collect()
    return radius_seeds


def randomize_seed(tuple_of_img_seg):
    """Get a random seed (pt) from the array"""
    location = randint(0, len(tuple_of_img_seg[0]) - 1)
    x_value = tuple_of_img_seg[0][location]
    y_value = tuple_of_img_seg[1][location]
    z_value = tuple_of_img_seg[2][location]
    pt = (x_value, y_value, z_value)
    return pt


def save_scaphoid_segmentations(scaphoid, output_path):
    """Saves the scaphoid segmentations"""
    output_path += "/"
    file_name = scaphoid.get_original_path()
    img = load(file_name)
    file_name = str(abspath(file_name).split("\\")[-1].split(".")[0])
    img_data = img.get_data()
    img_data[::] = scaphoid.get_segmentation()
    save(img, join(output_path, file_name + "_" +
                               SAVE_SEGMENTATION_NAME_SC))
    img_data[::] = scaphoid.get_fracture()
    save(img, join(output_path, file_name + "_" +
                               SAVE_FRACTURE_NAME))
    img_data[::] = scaphoid.get_fracture_with_bone()
    save(img, join(output_path, file_name + "_" +
                               SAVE_BONE_FRACTURE))
    del img, img_data
    collect()


def save_analysis_segmentation(scaphoid, output_path):
    """Saves all the files of the analysis part"""
    output_path += "/"
    file_name = scaphoid.get_original_path()
    img = load(file_name)
    file_name = str(abspath(file_name).split("\\")[-1].split(".")[0])
    img_data = img.get_data()
    img_data[::] = scaphoid.get_fracture()
    save(img, join(output_path, file_name + "_fixed_" +
                               SAVE_FRACTURE_NAME))
    img_data[::] = scaphoid.get_fracture_with_bone()
    save(img, join(output_path, file_name + "_fixed_" +
                   SAVE_BONE_FRACTURE))
    img_data[::] = scaphoid.get_bone_quarters()
    save(img, join(output_path, file_name + "_fixed_" +
                               SAVE_BONE_QUARTARS_FILE))
    img_data[::] = scaphoid.get_fracture_quarters()
    save(img, join(output_path, file_name + "_fixed_" +
                               SAVE_FRACTURE_QUARTARS_FILE))
    del img, img_data
    collect()


def save_geometric_features(features_dict, output_path, file_name):
    """Saves the geometrical features file"""
    output_path += "/"
    path = join(output_path, file_name + "_" +
                        SAVE_GEOMETRICAL_FEATURES)
    with open(path, 'w') as output_file:
        for k, v in features_dict.items():
            output_file.write(k + ": " + v + "\n")


def find_centeroid(seg_data):
    """returns the centeroid of the bone"""
    seg_nonzero = nonzero(seg_data)
    x_center = int(median(seg_nonzero[0]))
    y_center = int(median(seg_nonzero[1]))
    z_center = int(median(seg_nonzero[2]))
    del seg_nonzero
    return [x_center, y_center, z_center]


def substract_2_cetroids(first_center, sec_center):
    """
    substract 2 centroids in order to get the vector direction of them
    :param first_center: first centroid
    :param sec_center: second one
    :return: list of direction x,y,z
    """
    substracted = []
    for i in range(len(first_center)):
        substracted.append(first_center[i] - sec_center[i])
    return substracted


def create_direction_vector_for_2_points_cloud(two_clouds_for_vec,
                                               radius_main_comp):
    """Creates a direction vector from a cloud of dots"""
    point_cloud_above = two_clouds_for_vec == CAPITATE_UP_COLOR
    point_cloud_beneith = two_clouds_for_vec == CAPITATE_DOWN_COLOR

    above_centroid = find_centeroid(point_cloud_above)
    beneith_centroid = find_centeroid(point_cloud_beneith)
    del point_cloud_above, point_cloud_beneith
    new_vec = substract_2_cetroids(above_centroid, beneith_centroid)

    sum_of_list = sum(new_vec)
    new_vec[:] = [x / sum_of_list for x in new_vec]

    cosine_theta = dot(radius_main_comp, new_vec)
    to_div = npsum(radius_main_comp) * npsum(new_vec)
    cosine_theta /= to_div
    theta = arccos(cosine_theta)
    theta_deg = rad2deg(theta)
    return mod(theta_deg, 180)


if __name__ == "__main__":
    gui = GUI()
    gui.mainloop()
    del gui
    # diff_between_seg(SAVE_BONE_FRACTURE, "21seg.nii.gz")

