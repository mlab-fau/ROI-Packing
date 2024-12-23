import cv2, os, time
import numpy as np
import argparse
import subprocess
import pandas as pd

from dicts import frame_rates
import region_boxes
import packer

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='Image path for processing')
parser.add_argument('--predictions', type=str, required=True, help='Path to detected objects CSV')
parser.add_argument('--output', type=str, required=True, help='Path to output folder')
parser.add_argument('--sequence', type=str, required=True, help='Should the images be processed as a sequence or set of individual images BOOL.')
parser.add_argument('--packing', type=str, required=True, help='Turn packing on or off BOOL')
parser.add_argument('--paddingAmount', type=str, required=True, help='prediction padding amount')
parser.add_argument('--size', type=str, required=True, help='The size that the image/sequence will be encoded at')
parser.add_argument('--scale', type=str, required=True, help='Apply region scaling BOOL')
parser.add_argument('--alignCTU', type=str, required=True, help='Align regions to CTU BOOL')
parser.add_argument('--reducedParameters', type=str, required=True, help='Reduce parameters to multiples of 16 BOOL')

args = parser.parse_args()

def get_regions(image, predictions, scale, ctu, size, padding_amount):
    predictions = region_boxes.prepare_predictions(predictions)
    region_polygon, current_image = region_boxes.create_box_region_polygon(predictions, image, padding_amount)
    region_grid_polygons = region_boxes.divide_box_region_with_grid(region_polygon, 16)
    merged_grid_polygon = region_boxes.combine_grid_region(region_grid_polygons)

    regions_dict = {}
    new_box_index = 0
    for region_poly, coords in merged_grid_polygon:
        split_region_boxes = region_boxes.cut_polygon_at_verticies(region_poly, coords)
        split_boxes = region_boxes.create_box_objects(split_region_boxes, image)
        reattached_region_boxes = region_boxes.reattach_shared_edges(split_boxes)

        for i, rb in enumerate(reattached_region_boxes):
            labels, scores = region_boxes.get_box_labels(rb, image)
            rb.label = labels
            rb.score = scores

            regions_dict[new_box_index] = rb
            new_box_index += 1
    
    if scale == True:
        regions_dict = region_boxes.OID_class_based_scaling(regions_dict, size)
    if ctu == True:
        regions_dict = region_boxes.CTU_scale(regions_dict, 16)
    return regions_dict

def main():
    input_path = args.input
    detected_objects = args.predictions
    processing_sequence = args.sequence == 'True' or args.sequence == 'true'
    packing_on = args.packing == 'True' or args.packing == 'true'
    apply_scaling = args.scale == 'True' or args.scale == 'true'
    padding_amount = int(args.paddingAmount)
    output_path = args.output
    apply_CTU_align = args.alignCTU == 'True' or args.alignCTU == 'true'
    size = int(args.size)
    reduce_parameters = args.reducedParameters == 'True' or args.reducedParameters == 'true'

    if not os.path.exists(f"{output_path}/csv/SIZE_{size}/"): os.makedirs(f"{output_path}/csv/SIZE_{size}/")
    if not os.path.exists(f"{output_path}/packed/SIZE_{size}/"): os.makedirs(f"{output_path}/packed/SIZE_{size}/")

    detected_objects = pd.read_csv(detected_objects)
    
    frame = 0
    max_w = 0
    max_h = 0
    
    pack_dims = (0, 0)
    print("input path:", input_path)
    for i, image_file in enumerate(sorted(os.listdir(input_path))): # for each image in the input path
        if image_file.endswith(".png") or image_file.endswith(".jpg"):
            
            print(image_file)
            start = time.time()
            
            current_image = region_boxes.Image(image_file, cv2.imread(input_path+image_file))
            current_objects = detected_objects[detected_objects['img'] == image_file]

            '''
            If processing a sequence, we attempt to keep the same regions between frames for as long as possible. As long as the intersection/area between
            the current and previous frame is above or equal to 0.95 we apply the same regions from the previous frame to the current frame.
            '''
            if processing_sequence: 
                if "_val" in current_image_name:
                    current_objects = current_objects[current_objects['score'] > 0.1].reset_index(drop=True)
                
                if i != 0:
                    binary_mask = np.zeros(current_image.shape).astype('int')
            
                    for _, row in current_objects.iterrows():
                        xmin, xmax = row['box_x1'], row['box_x2']
                        ymin, ymax = row['box_y1'], row['box_y2']
            
                        if row['label'] in reference_classes or row['score'] > 0.3:
                            binary_mask[int(ymin) : int(ymax), int(xmin) : int(xmax)] = 1
                    intersection = np.sum(np.logical_and(reference_mask, binary_mask).astype('int'))
                    area = np.sum(binary_mask)
                else: 
                    intersection = 0
                    area = 1
                
                if not (intersection/area) >= 0.95: 
                    reference_polygons = get_regions(current_image, current_objects, apply_scaling, apply_CTU_align, size, padding_amount)
                    reference_mask = np.zeros(current_image.shape).astype('int')
                    reference_classes = set()
                    for box in reference_polygons:
                        reference_mask[reference_polygons[box].ymin : reference_polygons[box].ymax,
                                    reference_polygons[box].xmin : reference_polygons[box].xmax] = 1
                        for label in reference_polygons[box].label:
                            reference_classes.add(label)
                                
                regions_dict = reference_polygons
                
            else: regions_dict = get_regions(current_image, current_objects, apply_scaling, apply_CTU_align, size, padding_amount)
           
            current_image_name = image_file[0:image_file.index(".")]
            
            if processing_sequence:
                if packing_on: image_packed, packed_parameters = packer.pack_image(frame, regions_dict, current_image, reduce_parameters, pack_dims)
                else: image_packed, packed_parameters = packer.no_pack_image(frame, regions_dict, current_image, reduce_parameters, pack_dims)
                
                if "TVD-01" in input_path: sequence_name = "TVD-01"
                elif "TVD-02" in input_path: sequence_name = "TVD-02"
                elif "TVD-03" in input_path: sequence_name = "TVD-03"
                else: sequence_name = current_image_name.split('_')[0]
                
                with open(f"{output_path}/csv/SIZE_{size}/{sequence_name}.csv", "a") as packed_csv:
                    packed_csv.write(packed_parameters)
            else:
                if packing_on: image_packed, packed_parameters = packer.pack_image(frame, regions_dict, current_image, reduce_parameters)
                else: image_packed, packed_parameters = packer.no_pack_image(frame, regions_dict, current_image, reduce_parameters)
                
                with open(f"{output_path}/csv/SIZE_{size}/{current_image_name}.csv", "w") as packed_csv:
                    packed_csv.write(packed_parameters)
            
            cv2.imwrite(f"{output_path}/packed/SIZE_{size}/{current_image_name}.png", image_packed)
            
            '''
            if processing a sequence we attempt to use the same packing dimensions for each frame
            '''
            if processing_sequence:
                packed_img_w = image_packed.shape[1]
                packed_img_h = image_packed.shape[0]
                if packed_img_w > max_w: max_w = packed_img_w
                if packed_img_h > max_h: max_h = packed_img_h
                pack_dims = (image_packed.shape[1], image_packed.shape[0]) 
                frame+=1
            else: pack_dims = (0, 0)
            
            end = time.time()
            print(f"processing time: {end - start}")
    
    if processing_sequence: 
        start = time.time()
        if not os.path.exists(f"{output_path}/yuv/padded/SIZE_{size}/"):
            os.makedirs(f"{output_path}/yuv/padded/SIZE_{size}/")
        
        for input_image in os.listdir(f"{output_path}/packed/SIZE_{size}/"):
            sequence_name = input_image.split('_')[0]
            
            if "TVD-01" in input_path: sequence_name = "TVD-01"
            elif "TVD-02" in input_path: sequence_name = "TVD-02"
            elif "TVD-03" in input_path: sequence_name = "TVD-03"
            
            im = cv2.imread(f"{output_path}/packed/SIZE_{size}/"+input_image)
            h, w = im.shape[:2]
            im = np.pad(im, ((0, max(max_h-h, 0)), (0, max(max_w-w, 0)), (0, 0)), constant_values=0)
            cv2.imwrite(f"{output_path}/yuv/padded/SIZE_{size}/{input_image}", im)
        
        try: 
            frames = frame_rates[sequence_name]
            print(sequence_name)
            
        except:
            raise Exception("error: could not create YUV output. Verify sequence name is correct.")
        end = time.time()
        print(f"additonal sequence processing time: {end - start}")    
    
if __name__ == "__main__":
    main()
    
    
    
