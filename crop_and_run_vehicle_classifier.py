"""
Starting with a MD-formatted .json file, this notebook:

* Generates a folder of vehicle crops
* Runs the vehicle classifier
* Creates a new .json file with the vehicle results merged into the original .json file

"""

#%% Imports and constants

# pip install pytorch-lightning timm transformers

import os
import sys

import importlib.util

# Not used directly, just importing to make sure the classifier will run
import timm # type: ignore
import pytorch_lightning as pl # type: ignore
import transformers # type: ignore

from megadetector.utils.path_utils import insert_before_extension

image_folder = '/mnt/image/folder'
input_results_file = '/data/stuff.json'

custom_training_code_base = \
    os.path.expanduser('~/git/camera-trap-vehicle-classifier')
vehicle_best_checkpoint_file_stripped = \
    os.path.expanduser('~/models/camera-trap-vehicle-classifier/camera-trap-vehicle-classifier.2025.07.09.ckpt')

confidence_threshold_for_crops = 0.1
vehicle_classification_threshold = 0.5

assert os.path.isdir(custom_training_code_base)
assert os.path.isdir(image_folder)
assert os.path.isfile(input_results_file)
assert os.path.isfile(vehicle_best_checkpoint_file_stripped)

parallelization_defaults_to_threads = (os.name == 'nt')

task_name = os.path.basename(input_results_file).split('.')[0] + '_vehicles'
print('Task name: {}'.format(task_name))

preview_folder = os.path.join(os.path.dirname(input_results_file),task_name + '-preview')
os.makedirs(preview_folder,exist_ok=True)

crop_folder = os.path.join(os.path.expanduser('~/postprocessing/crops'),task_name)
os.makedirs(crop_folder,exist_ok=True)

detection_results_file_with_crop_ids = insert_before_extension(input_results_file,'with-crop-ids')
vehicle_classifier_output_file_crops_md_format = insert_before_extension(input_results_file,'vehicle-results-crops')
vehicle_classifier_output_file_image_level_md_format = insert_before_extension(input_results_file,'vehicle-results-image-level')

vehicle_class_names_file = os.path.join(os.path.dirname(vehicle_best_checkpoint_file_stripped),'classes.txt')
assert os.path.isfile(vehicle_class_names_file)

vehicle_merged_output_file_image_level_md_format = \
    insert_before_extension(input_results_file,'vehicle-merged')


#%% Load vehicle classifier modules

from megadetector.utils.path_utils import path_join

if custom_training_code_base not in sys.path:
    sys.path.append(custom_training_code_base)

load_with_importlib = False

if load_with_importlib:

    spec = importlib.util.spec_from_file_location('run_vehicle_classifier',
                                                  path_join(custom_training_code_base,
                                                  'run_vehicle_classifier.py'))
    run_custom_classifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_custom_classifier)

else:

    import run_vehicle_classifier as run_custom_classifier # type: ignore


#%% Generate crops

from megadetector.postprocessing.create_crop_folder import \
    CreateCropFolderOptions, create_crop_folder

create_crop_folder_options = CreateCropFolderOptions()
create_crop_folder_options.n_workers = 8
create_crop_folder_options.pool_type = 'process'
create_crop_folder_options.overwrite = False
create_crop_folder_options.confidence_threshold = confidence_threshold_for_crops

if parallelization_defaults_to_threads:
    create_crop_folder_options.pool_type = 'thread'
create_crop_folder_options.category_names_to_include = ['vehicle']

create_crop_folder(input_file=input_results_file,
                   input_folder=image_folder,
                   output_folder=crop_folder,
                   output_file=detection_results_file_with_crop_ids,
                   crops_output_file=None,
                   options=create_crop_folder_options)

assert os.path.isfile(detection_results_file_with_crop_ids)
assert os.path.isdir(crop_folder)


#%% Run the vehicle model

# When running in Windows, I hit some issues with using multiprocessing-based PyTorch dataloaders
# in IPython that I didn't want to fix; disabling multiprocessing is a workaround.
#
# TODO: fix this
#
# num_workers = 4
num_workers = 0
run_custom_classifier.run_inference(checkpoint_path=vehicle_best_checkpoint_file_stripped,
                                    input_dir=crop_folder,
                                    output_file=vehicle_classifier_output_file_crops_md_format,
                                    batch_size=128,
                                    num_workers=num_workers,
                                    input_size=None,
                                    class_file=vehicle_class_names_file,
                                    output_absolute_filenames=False)


##%% Validate vehicle model results

from megadetector.postprocessing.validate_batch_results import \
    ValidateBatchResultsOptions, validate_batch_results

validation_options = ValidateBatchResultsOptions()
validation_options.raise_errors = True
validation_options.verbose = True
validation_options.check_image_existence = True
validation_options.relative_path_base = crop_folder
_ = validate_batch_results(vehicle_classifier_output_file_crops_md_format, validation_options)


##%% Bring the vehicle classification results back to image level

from megadetector.postprocessing.create_crop_folder import crop_results_to_image_results

crop_results_to_image_results(
    image_results_file_with_crop_ids=detection_results_file_with_crop_ids,
    crop_results_file=vehicle_classifier_output_file_crops_md_format,
    output_file=vehicle_classifier_output_file_image_level_md_format,
    delete_crop_information=False,
    require_identical_detection_categories=False,
    restrict_to_top_n=1)

assert os.path.isfile(vehicle_classifier_output_file_image_level_md_format)


##%% Merge vehicle classifications

import json
from tqdm import tqdm
from copy import deepcopy

from megadetector.postprocessing.create_crop_folder import CreateCropFolderOptions
from megadetector.utils.ct_utils import invert_dictionary
from megadetector.utils.ct_utils import write_json

print('Loading vehicle classification results...')

# Load the vehicle classification results
with open(vehicle_classifier_output_file_image_level_md_format,'r') as f:
    vehicle_results = json.load(f)

n_images_with_classifications = 0

# im = vehicle_results['images'][0]
for im in vehicle_results['images']:
    if im['detections'] is None:
        continue
    for det in im['detections']:
        if 'classifications' in det:
            n_images_with_classifications += 1
            break

print('Loaded vehicle results for {} images ({} with classifications)'.format(
    len(vehicle_results['images']),n_images_with_classifications))

# Map filenames to results for the vehicle classification results
vehicle_filename_to_image = {im['file']:im for im in vehicle_results['images']}

# Load the input results (typically speciesnet results) with crop IDs attached
with open(detection_results_file_with_crop_ids,'r') as f:
    speciesnet_results = json.load(f)

print('Loaded input results for {} images'.format(len(speciesnet_results['images'])))

detection_category_name_to_id = invert_dictionary(speciesnet_results['detection_categories'])
assert 'vehicle' in detection_category_name_to_id
md_vehicle_category = detection_category_name_to_id['vehicle']

# Create classification categories for the vehicle classifier
vehicle_category_id_to_output_category_id = {}
next_category_id = max([int(c) for c in speciesnet_results['classification_categories'].keys()]) + 1

for vehicle_category_id in vehicle_results['classification_categories'].keys():
    vehicle_category_name = vehicle_results['classification_categories'][vehicle_category_id]
    vehicle_category_id_to_output_category_id[vehicle_category_id] = str(next_category_id)
    speciesnet_results['classification_categories'][str(next_category_id)] = vehicle_category_name
    if 'classification_category_descriptions' in speciesnet_results:
        speciesnet_results['classification_category_descriptions'][str(next_category_id)] = \
            'e2895ed5-780b-48f6-8a11-9e27cb594511;;;;;;' + vehicle_category_name
    next_category_id += 1

unknown_vehicle_category_id = str(next_category_id)
speciesnet_results['classification_categories'][unknown_vehicle_category_id] = 'unknown vehicle'
speciesnet_results['classification_category_descriptions'][unknown_vehicle_category_id] = \
    'e2895ed5-780b-48f6-8a11-9e27cb594511;;;;;;' + 'unknown vehicle'

speciesnet_category_name_to_id = invert_dictionary(speciesnet_results['classification_categories'])

n_classifications_modified = 0

# For each image in the speciesnet results
#
# im_speciesnet = speciesnet_results['images'][0]
for im_speciesnet in tqdm(speciesnet_results['images']):

    if ('detections' not in im_speciesnet) or (im_speciesnet['detections'] is None):
        continue

    # Get the corresponding image from vehicle results
    im_vehicle = vehicle_filename_to_image.get(im_speciesnet['file'])
    assert im_vehicle is not None

    # For each detection
    for det_speciesnet in im_speciesnet['detections']:

        # Is this a vehicle?  If not, don't touch it.
        if det_speciesnet['category'] != md_vehicle_category:
            continue

        # Is this above the threshold we used for creating crops?
        #
        #  If not, don't touch it.
        if det_speciesnet['conf'] < confidence_threshold_for_crops:
            continue

        # Find the vehicle classification result for this crop
        crop_id = det_speciesnet['crop_id']
        matching_det_vehicle = None

        for det_vehicle in im_vehicle['detections']:

            if 'crop_id' not in det_vehicle:
                continue

            if det_vehicle['crop_id'] == crop_id:
                matching_det_vehicle = det_vehicle
                break

        assert matching_det_vehicle is not None

        # Make sure these are really the same detection
        box_errors = [abs(det_vehicle['bbox'][i_coord] - det_speciesnet['bbox'][i_coord]) for i_coord in range(0,4)]
        box_error = sum(box_errors)
        assert box_error < 0.0000000001

        assert len(matching_det_vehicle['classifications']) == 1
        vehicle_category_id_in = matching_det_vehicle['classifications'][0][0]
        vehicle_category_id_out = vehicle_category_id_to_output_category_id[vehicle_category_id_in]
        vehicle_category_conf = matching_det_vehicle['classifications'][0][1]

        # Is it below our classification threshold?
        #
        # If so, skip it.
        if vehicle_category_conf < vehicle_classification_threshold:
            continue

        # Store the original classifications if necessary
        if 'original_classifications' in det_speciesnet:
            det_speciesnet['original_classifications'] = \
                deepcopy(det_speciesnet['classifications'])

        det_speciesnet['classifications'] = [[vehicle_category_id_out,vehicle_category_conf]]

        n_classifications_modified += 1

    # ...for each detection

# ...for each image

print('Modified {} classifications'.format(n_classifications_modified))

# Write merged .json file
write_json(vehicle_merged_output_file_image_level_md_format,speciesnet_results)

print('Wrote merged results to {}'.format(vehicle_merged_output_file_image_level_md_format))


##%% Preview (after merging vehicles)

from megadetector.postprocessing.postprocess_batch_results import PostProcessingOptions
from megadetector.postprocessing.postprocess_batch_results import process_batch_results
from megadetector.utils.path_utils import open_file

preview_options = PostProcessingOptions()
preview_options.image_base_dir = image_folder
preview_options.include_almost_detections = True
preview_options.num_images_to_sample = 7500
preview_options.confidence_threshold = 0.2
preview_options.almost_detection_confidence_threshold = \
    preview_options.confidence_threshold - 0.05
preview_options.ground_truth_json_file = None
preview_options.separate_detections_by_category = True
preview_options.sample_seed = 0
preview_options.max_figures_per_html_file = 2500
preview_options.sort_classification_results_by_count = True
preview_options.parallelize_rendering = True
preview_options.parallelize_rendering_n_cores = 20
preview_options.parallelize_rendering_with_threads = parallelization_defaults_to_threads
preview_options.additional_image_fields_to_display = \
    {'pre_smoothing_description':'pre-smoothing labels',
     'pre_filtering_description':'pre-filtering labels',
     'post_filtering_description':'post-filtering labels',
     'top_classification_common_name':'top class'}
preview_options.category_name_to_sort_weight = \
    {'animal':1,'blank':1,'unknown':1,'unreliable':1,'mammal':1}

preview_options.md_results_file = vehicle_merged_output_file_image_level_md_format
preview_options.output_dir = preview_folder

print('Generating post-classification preview in {}'.format(preview_folder))
ppresults = process_batch_results(preview_options)
open_file(ppresults.output_html_file,attempt_to_open_in_wsl_host=True,browser_name='chrome')
# import clipboard; clipboard.copy(ppresults.output_html_file)
