import cv2
import numpy as np
from rectpack import newPacker, PackingBin, GuillotineBafSas, SORT_LSIDE
import math

'''
Helper function to format box data to be accepted by the Packer. Format is a list of tuples with
the following values (width, height, box id)

Intput: boxes_dict (dtype = Dict)
Output: List
'''
def prepare_for_packing(boxes_dict):
    dims = []
    for box_id in boxes_dict: dims.append((int(boxes_dict[box_id].aug_width),int(boxes_dict[box_id].aug_height), box_id))
    return dims

'''
Function to run the packing algorithm and obtain results. Output is a dictionary of new positions where
each key is a box id and each value a tuple with xmin, ymin, width, and height. The function also outputs 
the dimensions of the packed image (sum of widths and sum of heights of boxes). 

Input: boxes_dict (dtype = Dict)
Output: Dict, int, int
'''
def run_packer_ff(boxes_dict, frame_dims):
    dims = prepare_for_packing(boxes_dict)
    print(frame_dims)
    if frame_dims == (0, 0):
        print("initializing pack dims")
        side, long = 0, 0
        for r in dims:
            side+=r[0]
            long+=r[1]
    else: side, long = frame_dims[0], frame_dims[1]
    print(side, long)
    num_packed = 0
    while(num_packed != len(boxes_dict)):
        bins = [(side, long)]
        packer = newPacker(sort_algo=SORT_LSIDE, bin_algo=PackingBin.BFF,pack_algo=GuillotineBafSas, rotation=False)
        for r in dims: packer.add_rect(*r)
        for b in bins: packer.add_bin(*b)
        packer.pack()

        positions_dict = {}
        for abin in packer:
            for rect in abin:
                positions_dict[rect.rid] = (int(rect.x), int(rect.y), int(rect.width), int(rect.height))
        num_packed = len(positions_dict)
        print(len(boxes_dict), len(positions_dict))
        if num_packed != len(boxes_dict):
            side += 100
            long += 100
    return positions_dict, side, long
    
'''
Function to run the packing algorithm and obtain results. Output is a dictionary of new positions where
each key is a box id and each value a tuple with xmin, ymin, width, and height. The function also outputs 
the dimensions of the packed image (sum of widths and sum of heights of boxes). 

Input: boxes_dict (dtype = Dict)
Output: Dict, int, int
'''
def run_packer(boxes_dict):
    dims = prepare_for_packing(boxes_dict)
    #print(dims)
    
    side, long = 0, 0
    for r in dims:
        side+=r[0]
        long+=r[1]
        
    bins = [(side*10, long*10)]
    packer = newPacker(sort_algo=SORT_LSIDE, bin_algo=PackingBin.BFF,pack_algo=GuillotineBafSas, rotation=False)
    for r in dims: packer.add_rect(*r)
    for b in bins: packer.add_bin(*b)
    packer.pack()
    
    positions_dict = {}
    for abin in packer:
        for rect in abin:
            positions_dict[rect.rid] = (int(rect.x), int(rect.y), int(rect.width), int(rect.height))

    return positions_dict, side, long

'''
Function to create a packed image by calling the packer algorithm. Uses the output positions of 
run_packer to place boxes according to their packed positions. Additionally maintains DataFrame of 
positions used for unpacking. Uses image object to cut out original pixel values to be placed in the new
image. 

Input: boxes_dict (dtype = Dict), image (dtype = Image Object), df_output_positions (dtype = DataFrame)
Output: matrix, DataFrame
'''
def pack_image(frame_number, boxes_dict, image, simplify_parameters, frame_dims=None):
    if frame_dims is None:
        positions_dict, side, long = run_packer(boxes_dict)
    else:
        positions_dict, side, long = run_packer_ff(boxes_dict, frame_dims)
    
    packed_img = np.zeros((long, side, 3))
    
    packed_width = -1 #bottom right corner x coordinate
    packed_height = -1 #bottom left corner y coordinate

    num_objects = 0
    parameters = f"{frame_number},num_objects,{image.width},{image.height},"
    for box_id in positions_dict: 
        #get original box coordinates
        b_x, b_y = boxes_dict[box_id].xmin, boxes_dict[box_id].ymin
        b_w, b_h = boxes_dict[box_id].width, boxes_dict[box_id].height
        
        original_box = image.matrix[b_y:b_y+b_h, b_x:b_x+b_w] #get original box pixel values from image
        
        augmented_box = cv2.resize(original_box, (boxes_dict[box_id].aug_width, boxes_dict[box_id].aug_height)) 

        #get packed box coordinates
        p = positions_dict[box_id]
        p_x, p_y = p[0], p[1]
        p_w, p_h = p[2], p[3]
        
        if (p_x + p_w) > packed_width: packed_width = p_x + p_w   
        if (p_y + p_h) > packed_height: packed_height = p_y + p_h
            
        packed_img[p_y : p_y + p_h, p_x : p_x + p_w] = augmented_box #place original box pixels into packed position

        if simplify_parameters:
            b_x = int(math.ceil(b_x/16))
            b_y = int(math.ceil(b_y/16))
            b_w = int(math.ceil(b_w/16))
            b_h = int(math.ceil(b_h/16))
            p_x = int(math.ceil(p_x/16))
            p_y = int(math.ceil(p_y/16))
            p_w = int(math.ceil(p_w/16))
            p_h = int(math.ceil(p_h/16))
        parameters += f"{b_x},{b_y},{b_w},{b_h},{p_x},{p_y},{p_w},{p_h},"
        num_objects += 1

    parameters = parameters.replace("num_objects", str(num_objects))
    parameters = parameters[0:-1] + "\n"
    crop_packed = packed_img[0 : packed_height, 0 : packed_width] #crop the packed image
        
    return crop_packed.astype('uint8'), parameters

'''
Produce coordinates without packing
'''
def no_pack_image(frame_number, boxes_dict, image, simplify_parameters, frame_dims=None):
    #positions_dict, side, long = run_packer(boxes_dict, frame_dims)
    
    packed_img = np.zeros((image.height, image.width, 3))
    #packed_img[packed_img == 0] = -1

    #R = []
    #G = []
    #B = []
    
    num_objects = 0
    parameters = f"{frame_number},num_objects,{image.width},{image.height},"
    for box_id in boxes_dict: 
        #get original box coordinates
        b_x, b_y = boxes_dict[box_id].xmin, boxes_dict[box_id].ymin
        b_w, b_h = boxes_dict[box_id].width, boxes_dict[box_id].height
        
        original_box = image.matrix[b_y:b_y+b_h, b_x:b_x+b_w] #get original box pixel values from image
        
        augmented_box = cv2.resize(original_box, (boxes_dict[box_id].aug_width, boxes_dict[box_id].aug_height)) 

        p_x, p_y = b_x, b_y
        p_w, p_h = b_w, b_h
            
        packed_img[p_y : p_y + p_h, p_x : p_x + p_w] = augmented_box #place original box pixels into packed position

        if simplify_parameters:
            b_x = int(math.ceil(b_x/16))
            b_y = int(math.ceil(b_y/16))
            b_w = int(math.ceil(b_w/16))
            b_h = int(math.ceil(b_h/16))
            p_x = int(math.ceil(p_x/16))
            p_y = int(math.ceil(p_y/16))
            p_w = int(math.ceil(p_w/16))
            p_h = int(math.ceil(p_h/16))
        parameters += f"{b_x},{b_y},{b_w},{b_h},{p_x},{p_y},{p_w},{p_h},"
        num_objects += 1
        
        #R.append(np.mean(original_box[:,:,2]))
        #G.append(np.mean(original_box[:,:,1]))
        #B.append(np.mean(original_box[:,:,0]))

    parameters = parameters.replace("num_objects", str(num_objects))
    parameters = parameters[0:-1] + "\n"
    
    #packed_img[:,:,0][packed_img[:,:,0] == -1] = int(np.mean(B))
    #packed_img[:,:,1][packed_img[:,:,1] == -1] = int(np.mean(G))
    #packed_img[:,:,2][packed_img[:,:,2] == -1] = int(np.mean(R))
        
    return packed_img.astype('uint8'), parameters














