"""
The following is a simple example algorithm.

It is meant to run within a container.

To run it locally, you can call the following bash script:

  ./test_run.sh

This will start the inference and reads from ./test/input and outputs to ./test/output

To save the container and prep it for upload to Grand-Challenge.org you can call:

  ./save.sh

Any container that shows the same behavior will do, this is purely an example of how one COULD do it.

Happy programming!
"""

import torch
import os
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label
from glob import glob
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

def remove_small_objects_binary(binary_data, min_size=10):
    labeled_array, num_features = label(binary_data)
    sizes = np.bincount(labeled_array.ravel())
    remove = sizes < min_size
    remove[0] = False  # Ensure the background (label 0) is not removed
    labeled_array[remove[labeled_array]] = 0
    return labeled_array > 0
    
def run():
    # Process the inputs: any way you'd like
    _show_torch_cuda_info()
    
    nnUNet_results = "/opt/app/nnUNet/nnUNet_results"
    lowres_predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    lowres_predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset825_AortaSeg24_CTA_bin_50/nnUNetTrainer_CE_DC_CBDC_NoMirroring_500epochs__nnUNetPlans__3d_lowres'),
        use_folds=(0, 'all'),
        checkpoint_name='checkpoint_final.pth',
    )
    fullres_predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        perform_everything_on_device=True,
        device=torch.device('cuda', 0),
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True
    )
    # initializes the network architecture, loads the checkpoint
    fullres_predictor.initialize_from_trained_model_folder(
        join(nnUNet_results, 'Dataset824_AortaSeg24_CTA_50/nnUNetTrainer_Hierarchical_CBDC_AortaSeg23_NoMirroring_500epochs__nnUNetPlans__3d_fullres'),
        use_folds=(0, 1, 2, 3, 4, 'all'),
        checkpoint_name='checkpoint_final.pth',
    )
    
    input_folder = "/input/images/ct-angiography"
    output_folder = "/output/images/aortic-branches"
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    input_files = glob(str(input_folder + "/*.tiff")) + glob(str(input_folder + "/*.mha"))
    input_file_name = input_files[0]
    output_file_name = output_folder + "/output.mha"
    
    tmp_file_path = "/opt/app/tmp.mha"
    
    case_image = sitk.ReadImage(input_file_name)
    direction = case_image.GetDirection()
    origin = case_image.GetOrigin()
    spacing = case_image.GetSpacing()
    
    case_array = sitk.GetArrayFromImage(case_image)
    print("case_array.shape: ", case_array.shape)
    
    # x, y flip:
    case_array = np.flip(case_array, axis=1)
    case_array = np.flip(case_array, axis=2)
    
    new_case_image = sitk.GetImageFromArray(case_array)
    new_case_image.SetSpacing(spacing)
    new_case_image.SetOrigin(origin)
    new_case_image.SetDirection((1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(new_case_image, tmp_file_path)

    # predict a single numpy array
    img, props = SimpleITKIO().read_images([tmp_file_path])
    pred_array = lowres_predictor.predict_single_npy_array(img, props, None, None, False)
    pred_array = pred_array.astype(np.uint8)
    print("pred_array.shape: ", pred_array.shape)
    
    min_size = 1000
    # Remove small objects
    pred_array = remove_small_objects_binary(pred_array, min_size=min_size).astype(np.uint8)
    
    #print("props: ", props)
    
    a = 2.0 # scaling factor for roi

    non_zero_indices = np.argwhere(pred_array > 0)

    # Get the minimum and maximum values for x, y, z dimensions
    x_min, y_min, z_min = non_zero_indices.min(axis=0)
    x_max, y_max, z_max = non_zero_indices.max(axis=0)

    # Calculate the center of the current bounding box
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    z_center = (z_min + z_max) / 2

    # Calculate the new radius range after scaling by factor `a`
    x_range = (x_max - x_min + 1) * a / 2
    y_range = (y_max - y_min + 1) * a / 2
    z_range = (z_max - z_min + 1) * a / 2
    
    #print("x_range, y_range, z_range: ", x_range, y_range, z_range)
    
    # Calculate the new bounding box while ensuring it doesn't exceed the mask_array boundaries
    x_min_new = max(0, int(x_center - x_range))
    x_max_new = min(pred_array.shape[0] - 1, int(x_center + x_range))
    y_min_new = max(0, int(y_center - y_range))
    y_max_new = min(pred_array.shape[1] - 1, int(y_center + y_range))
    z_min_new = max(0, int(z_center - z_range))
    z_max_new = min(pred_array.shape[2] - 1, int(z_center + z_range))
    
    img_array = img[0]
    print("img_array.shape: ", img_array.shape)
    # Extract the scaled roi-array
    scaled_roi_array = img_array[x_min_new:x_max_new+1, y_min_new:y_max_new+1, z_min_new:z_max_new+1]
    print("scaled_roi_array.shape: ", scaled_roi_array.shape)
    
    roi_image = sitk.GetImageFromArray(scaled_roi_array)
    roi_image.SetDirection(props['sitk_stuff']['direction'])
    roi_image.SetOrigin(props['sitk_stuff']['origin'])
    roi_image.SetSpacing(props['sitk_stuff']['spacing'])
    
    roi_img_path = os.path.join("/opt/app", "roi_img")
    os.makedirs(roi_img_path, exist_ok=True)
    roi_img_mha_path = os.path.join(roi_img_path, "test_roi_img_0000.mha")
    sitk.WriteImage(roi_image, roi_img_mha_path)
    
    # predict a single numpy array
    roi_img, roi_props = SimpleITKIO().read_images([roi_img_mha_path])
    roi_pred_array = fullres_predictor.predict_single_npy_array(roi_img, roi_props, None, None, False)
    roi_pred_array = roi_pred_array.astype(np.uint8)
    print("roi_pred_array.shape: ", roi_pred_array.shape)
    
    pred_array[x_min_new:x_max_new+1, y_min_new:y_max_new+1, z_min_new:z_max_new+1] = roi_pred_array
    
    print("pred_array.shape: ", pred_array.shape)
    
    # x, y flip:
    pred_array = np.flip(pred_array, axis=1)
    pred_array = np.flip(pred_array, axis=2)
    
    image = sitk.GetImageFromArray(pred_array)
    image.SetDirection(direction)
    image.SetOrigin(origin)
    image.SetSpacing(spacing)
    sitk.WriteImage(
        image,
        output_file_name,
        useCompression=True,
    )
                                 
    print('Saved!!!')
    return 0

def _show_torch_cuda_info():

    print("=+=" * 10)
    print("Collecting Torch CUDA information")
    print(f"Torch CUDA is available: {(available := torch.cuda.is_available())}")
    if available:
        print(f"\tnumber of devices: {torch.cuda.device_count()}")
        print(f"\tcurrent device: { (current_device := torch.cuda.current_device())}")
        print(f"\tproperties: {torch.cuda.get_device_properties(current_device)}")
    print("=+=" * 10)


if __name__ == "__main__":
    raise SystemExit(run())
