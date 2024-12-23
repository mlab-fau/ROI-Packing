
import cv2, os, sys, shutil
import numpy as np
import argparse
import dicts
import time

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, required=True, help='Image to unpack')
parser.add_argument('--csv', type=str, required=True, help='Packed parameters for image')
parser.add_argument('--output', type=str, required=True, help='Path to output folder')
parser.add_argument('--sequence', type=str, required=True, help='Processing images or sequence (True/False)')
parser.add_argument('--reducedParameters', type=str, required=True, help='Parameters are reduced to multiples of 16 (True/False)')
parser.add_argument('--encoded', type=str, required=True, help='were these images/csv files the result of decoding? (True/False)')
parser.add_argument('--rescaleSize', type=str, required=False, help='Rescale size for evaluation (100, 75, 50, 25)')

args = parser.parse_args()

image_path = args.input
csv_path = args.csv
output_path = args.output

if args.rescaleSize is not None:
    rescale_size = int(args.rescaleSize)
else: rescale_size = None

processing_sequence = args.sequence == 'True' or args.sequence == 'true'
reduced_parameters = args.reducedParameters == 'True' or args.reducedParameters == 'true'
encoded = args.encoded == 'True' or args.encoded == 'true'

def unpack_image(packed_matrix, parameters, reconstruct_coordinates):
    parameters = parameters.split(",")
    parameters = [parameters[0]] + [ int(x) for x in parameters[1::] ]
    
    original_image_width = parameters[2]
    original_image_height = parameters[3]
    
    unpacked_image = np.zeros((original_image_height, original_image_width, 3))
    binary_matrix = np.zeros((original_image_height, original_image_width)).astype('int')
    
    unpacked_image[unpacked_image == 0] = -1
    
    R = []
    G = []
    B = []
    
    num_objects = parameters[1]
    index = 4
    for i in range(num_objects):
        p_x = parameters[index+4]
        p_y = parameters[index+5]
        p_w = parameters[index+6]
        p_h = parameters[index+7]
        
        if reconstruct_coordinates:
           p_x = p_x * 16
           p_y = p_y * 16
           p_w = p_w * 16
           p_h = p_h * 16
        
        packed_box = packed_matrix[p_y : p_y + p_h, p_x : p_x + p_w]

        b_x = parameters[index]
        b_y = parameters[index+1]
        b_w = parameters[index+2]
        b_h = parameters[index+3]
        
        if reconstruct_coordinates:
           b_x = b_x * 16
           b_y = b_y * 16
           b_w = b_w * 16
           b_h = b_h * 16
           if b_w + b_x > original_image_width: b_w = b_w - ((b_w + b_x) - original_image_width)
           if b_h + b_y > original_image_height: b_h = b_h - ((b_h + b_y) - original_image_height)
        
        packed_box = cv2.resize(packed_box, (b_w, b_h), interpolation=cv2.INTER_CUBIC)
        
        unpacked_image[b_y:b_y+b_h, b_x:b_x+b_w] = packed_box
        
        R.append(np.mean(packed_box[:,:,2]))
        G.append(np.mean(packed_box[:,:,1]))
        B.append(np.mean(packed_box[:,:,0]))
        
        index+=8
    
    unpacked_image[:,:,0][unpacked_image[:,:,0] == -1] = int(np.mean(B))
    unpacked_image[:,:,1][unpacked_image[:,:,1] == -1] = int(np.mean(G))
    unpacked_image[:,:,2][unpacked_image[:,:,2] == -1] = int(np.mean(R))
        
    return unpacked_image.astype('uint8')

frame = 0
for image in sorted(os.listdir(image_path)):
    if image.endswith(".png"):
        start_time = time.time()
        
        ID = image.split("_")[0]
        ID = ID.replace(".png", "")
        
        if "FLIR" in ID: ID = image.split("_")[0] + "_" + image.split("_")[1]
        
        image_packed = cv2.imread(image_path+image)
        print(image)
        
        csv = image.replace(".png", ".csv")
        
        if encoded:
            if "TVD-01" in image_path: csv = "TVD-01_1920x1080_50.csv"
            if "TVD-02" in image_path: csv = "TVD-02_1920x1080_50.csv"
            if "TVD-03" in image_path: csv = "TVD-03_1920x1080_50.csv"
        else: 
            if "TVD-01" in image_path: csv = "TVD-01.csv"
            if "TVD-02" in image_path: csv = "TVD-02.csv"
            if "TVD-03" in image_path: csv = "TVD-03.csv"
        
        if "_val" in image: 
            if encoded: csv = ("_").join(image.split("_")[0:-2]) + ".csv"
            else: csv = image.split('_')[0] + ".csv"
        
        with open(csv_path+csv, "r") as file:
            if processing_sequence: 
                packed_params = file.readlines()
                packed_params = sorted(packed_params, key=lambda x: int(x.split(",")[0]))[frame]
            else: packed_params = file.readlines()[0]
        
        image_unpacked = unpack_image(image_packed, packed_params, reduced_parameters)
        
        if rescale_size is not None:
            if not os.path.exists(f"{output_path}/tmp/"): os.makedirs(f"{output_path}/tmp/")
            cv2.imwrite(f"{output_path}/tmp/{image}", image_unpacked)
            
            lookup_ID = ID
            
            if "_val" in image: lookup_ID = image.split("_")[0]
            if "TVD-01" in image_path: lookup_ID = "TVD-01"
            if "TVD-02" in image_path: lookup_ID = "TVD-02"
            if "TVD-03" in image_path: lookup_ID = "TVD-03"
            
            w, h = dicts.original_sizes[lookup_ID][0], dicts.original_sizes[lookup_ID][1]
            
            if rescale_size == 100:
                scale_cmd = f"crop={w}:{h}"
            else:
                scale_cmd = f"scale={w}:{h}"
            
            if "_val" in image: 
                ID = image.replace(".png", "")
                ID = ID.split("_")[0] + f"_{str(w)}x{str(h)}_" + f"{ID.split('_')[2]}_{ID.split('_')[3]}_{ID.split('_')[4]}"
            if "TVD-01" in image_path or "TVD-02" in image_path or "TVD-03" in image_path:
                ID = str(int(ID) + 1).zfill(6)
            
            cmd = f"ffmpeg -i {output_path}/tmp/{image} -vf \"{scale_cmd}\" {output_path}/{ID}.png"
            os.system(cmd)
        else: 
            if not os.path.exists(f"{output_path}"): os.makedirs(f"{output_path}")
            cv2.imwrite(f"{output_path}/{image}", image_unpacked)

        print("unpacking time: ", time.time() - start_time)
        frame+=1

shutil.rmtree(f"{output_path}/tmp/")
    
    
