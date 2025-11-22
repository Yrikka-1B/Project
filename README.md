# Project
<!-- run
    python convert_coco_to_yolo.py
        Gets two data sets (both under 'BTT_Data', with each one inside this folder under 'A' or 'B'),
        combines them and puts them in a folder labeled 'dataset',
        converts this combined coco file to yolo format
    python run_inference_and_score.py
        Runs inference on the datasets using the base yolo model, creates confidence scores,
        the 10% of images with the lowest confidence scores get added to a folder 'hardest_10_percent'
        (which is under 'inference_results') and the 'results.json' in 'inference_results' contains
        the confidence scores of every image in the total data set (not just the 10%)
    python clean_and_merge_dataset.py
        After cleaning up the labels in cvat, they should be uploaded to folder
        'corrected_images_labels' (under 'dataset' folder), this command will remove the original 10% of incorrect
        labels and replace them with the new ones from cvat, as well as the images. This new label and image set
        doesn't replace the og one, and instead is placed under 'clean_dataset' under 'dataset'
    *i lowkey don't remember which folders I created and which ones the files generated themselves,
    but the only ones I created might've been BTT_Data and dataset, as well as the .py files that are being run*
-->