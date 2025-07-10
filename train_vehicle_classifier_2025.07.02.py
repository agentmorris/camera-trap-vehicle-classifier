"""

train_vehicle_classifier_2025.07.02.py

Notebook that drives train_vehicle_classifier.py, to train a vehicle classifier.  Don't be
fooled by the .py extension, this is a notebook in spirit.

Also:

* Runs the classifier on the val crops and previews the results
* Creates a val folder of whole images for processing with standard MegaDetector scripts

"""

#%% Imports and constants

import os
import importlib.util

from megadetector.utils.path_utils import path_join
from megadetector.utils.path_utils import insert_before_extension

training_output_folder = 'c:/temp/wwf-naidoo-training/wwf-naidoo-training_2025.07.02'
training_metadata_file = 'd:/data/wwf-naidoo/wwf_naidoo_training_images.json'
vehicle_crop_folder =  'd:/data/wwf_naidoo_training_crops'
vehicle_training_code_base = os.path.expanduser('~/git/agentmorrisprivate/camera-trap-jobs/wwf-naidoo')

assert os.path.isdir(vehicle_training_code_base)
assert not (vehicle_training_code_base.endswith('/') or vehicle_training_code_base.endswith('\\'))

class_names_file = os.path.join(training_output_folder,'classes.txt')
assert os.path.isfile(class_names_file)

val_output_file = os.path.join(training_output_folder,'val_results_md_format.json')
val_output_file_relative_pathnames = insert_before_extension(val_output_file,'relative_paths')
val_folder = os.path.join(vehicle_crop_folder,'val')
assert os.path.isdir(val_folder)

crop_preview_folder = 'c:/temp/wwf-naidoo-training/crop-preview'

# We just need this for the list of validation locations
training_images_json = 'd:/data/wwf-naidoo/wwf_naidoo_training_images.json'

# This has labels and locations for all the images
all_naidoo_images_json = 'd:/data/wwf-naidoo/wwf_naidoo_labels.json'

assert os.path.isfile(training_images_json)
assert os.path.isfile(all_naidoo_images_json)

whole_image_folder_base = 'd:/data/wwf-naidoo'
assert os.path.isdir(whole_image_folder_base)

val_image_folder_base = 'd:/data/wwf_naidoo_val_images'
os.makedirs(val_image_folder_base,exist_ok=True)


#%% Train the model

model_name = 'timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k'
os.makedirs(training_output_folder,exist_ok=True)
batch_size = 40
patience = 5
epochs = 100
freeze_layers = True

# mamba activate imagenet-inference
cmd = f'python {vehicle_training_code_base}/train_vehicle_classifier.py'
cmd += f' "{training_metadata_file}" "{vehicle_crop_folder}" "{training_output_folder}"'
cmd += f' --model {model_name} --batch-size {batch_size} --patience {patience} --epochs {epochs}'
if not freeze_layers:
    cmd += ' --no-freeze'

print(cmd)
import clipboard; clipboard.copy(cmd)

os.makedirs(training_output_folder,exist_ok=True)
with open(path_join(training_output_folder,'training_command.txt'),'w') as f:
    f.write(cmd + '\n')


#%% Find the best checkpoint

files = os.listdir(training_output_folder)
checkpoint_files = [fn for fn in files if fn.endswith('.ckpt')]
best_checkpoint_files = [fn for fn in checkpoint_files if (fn.startswith('best') and ('stripped' not in fn))]
assert len(best_checkpoint_files) == 1
best_checkpoint_file = os.path.join(training_output_folder,best_checkpoint_files[0])

print('Using best checkpoint: {}'.format(best_checkpoint_file))


#%% Strip optimizer state from the checkpoint (prep)

best_checkpoint_file_stripped = insert_before_extension(best_checkpoint_file,'stripped')

spec = \
    importlib.util.spec_from_file_location('train_vehicle_classifier',
                                           path_join(vehicle_training_code_base,
                                           'train_vehicle_classifier.py'))
train_vehicle_classifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_vehicle_classifier)

train_vehicle_classifier.strip_checkpoint(best_checkpoint_file,
                                         best_checkpoint_file_stripped,
                                         keep_hyperparams=True)

print('Wrote stripped checkpoint to {}'.format(best_checkpoint_file_stripped))


#%% Run the trained model (prep)

spec = importlib.util.spec_from_file_location('run_vehicle_classifier',
                                              path_join(vehicle_training_code_base,
                                                        'run_vehicle_classifier.py'))
run_vehicle_classifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_vehicle_classifier)


#%% Run the trained vehicle model (execution)

run_vehicle_classifier.run_inference(checkpoint_path=best_checkpoint_file_stripped,
                                     input_dir=val_folder,
                                     output_file=val_output_file,
                                     batch_size=128,
                                     num_workers=4,
                                     input_size=None,
                                     class_file=class_names_file,
                                     output_absolute_filenames=False)


#%% Validate results

from megadetector.postprocessing.validate_batch_results import \
    ValidateBatchResultsOptions, validate_batch_results

validation_options = ValidateBatchResultsOptions()
validation_options.raise_errors = True
validation_options.verbose = True
_ = validate_batch_results(val_output_file_relative_pathnames, validation_options)


#%% Preview results

from megadetector.postprocessing.postprocess_batch_results import PostProcessingOptions
from megadetector.postprocessing.postprocess_batch_results import process_batch_results
from megadetector.utils.path_utils import open_file

preview_options = PostProcessingOptions()

preview_options.image_base_dir = val_folder
preview_options.num_images_to_sample = 7500
preview_options.confidence_threshold = 0.2
preview_options.almost_detection_confidence_threshold = \
    preview_options.confidence_threshold / 2.0
preview_options.separate_detections_by_category = True
preview_options.sample_seed = 0
preview_options.max_figures_per_html_file = 1000
preview_options.sort_classification_results_by_count = True
preview_options.classification_confidence_threshold = 0.05
preview_options.md_results_file = val_output_file_relative_pathnames
preview_options.output_dir = crop_preview_folder
ppresults = process_batch_results(preview_options)

open_file(ppresults.output_html_file,attempt_to_open_in_wsl_host=True,browser_name='chrome')
# import clipboard; clipboard.copy(ppresults.output_html_file)


#%% Create a val folder of whole images (prep)

import json
from tqdm import tqdm

with open(training_images_json,'r') as f:
    d = json.load(f)
val_locations = d['val_locations']

val_images_relative = []

with open(all_naidoo_images_json,'r') as f:
    d = json.load(f)

for im in d['images']:

    if im['location'] in val_locations:
        val_images_relative.append(im['file_name'])

#...for each image

print('Copying {} of {} images to the val folder'.format(
    len(val_images_relative),len(d['images'])))

input_file_to_output_file = {}

for fn_relative in tqdm(val_images_relative):
    input_fn = os.path.join(whole_image_folder_base,fn_relative)
    output_fn = os.path.join(val_image_folder_base,fn_relative)
    input_file_to_output_file[input_fn] = output_fn
    assert os.path.isfile(input_fn)


#%% Create a val folder of whole images (execution)

from megadetector.utils.path_utils import parallel_copy_files

parallel_copy_files(input_file_to_output_file,
                    max_workers=8,
                    use_threads=True,
                    overwrite=True,
                    verbose=False,
                    move=False)
