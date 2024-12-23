import cv2, os, time
import pandas as pd
from shapely.geometry import Polygon, MultiPolygon, LineString, MultiLineString, MultiPoint, MultiPolygon, Polygon
from shapely.ops import split, linemerge, polygonize, unary_union
import numpy as np
import math
import warnings
warnings.filterwarnings('ignore')

class Image:
    '''
    Image object contains information about the current image being processed.
    Initial attributes include ID, matrix, and shape data
    '''
    def __init__(self, ID, matrix):
        self.ID = ID
        self.matrix = matrix
        self.width = matrix.shape[1]
        self.height = matrix.shape[0]
        self.area = matrix.shape[0] * matrix.shape[1]
        self.shape = (matrix.shape[0], matrix.shape[1])
    
    '''
    set label information for image (enables mapping binary value to list of labels)
    matrix is a numpy matrix of binary values indicated classes present in a given pixel
    '''
    def set_label_info(self, label_list, matrix, score_matricies):
        self.label_matrix = matrix
        self.label_lookup = label_list
        self.score_matricies = score_matricies

        
class Box:
    '''
    Initialization of Box object requires minimum and maximum 
    xy coordinates, a prediction score, and class label. Initial
    attributes of a Box object are:
    
    xmin, ymin, xmax, ymax, score, label, width, height, area
    '''
    def __init__(self, xmin, ymin, xmax, ymax):
        self.xmin, self.ymin = (int(xmin), int(ymin))
        self.xmax, self.ymax = (int(xmax), int(ymax))
        
        self.width = int(xmax) - int(xmin)
        self.height = int(ymax) - int(ymin)
        self.area = (int(xmax) - int(xmin)) * (int(ymax) - int(ymin))
        self.label = ""
        self.score = 0.0
        
        self.aug_width = self.width
        self.aug_height = self.height
    
    '''
    Returns list of box coordinates in clockwise order.
    Can be used to create shapely Polygons
    '''
    def get_coordinate_list(self):
        return [[self.xmin, self.ymin], [self.xmax, self.ymin], 
                [self.xmax, self.ymax], [self.xmin, self.ymax]]
    
    '''
    When performing augmentations to boxes (i.e. scaling), set the new
    coordinates and box info by calling this function.
    '''
    def set_augmented_box(self, w, h):        
        self.aug_width = int(w)
        self.aug_height = int(h)
    
    '''
    Print individual box attributes.
    '''
    def print_box(self, ID):
        print(f"{ID}: ({self.xmin}, {self.ymin}, {self.width}, {self.height}, {self.label}, {self.score})")
        
    def print_box_augmentation(self, ID):
        print(f"{ID}: ({self.width} --> {self.aug_width}, {self.height} --> {self.aug_height})")

'''
Function to print attributes of boxes in a dictionary.

Input: box_dict (dtype = Dict)
Output: None
'''
def print_box_dict(boxes_dict):
    for box_id in boxes_dict: boxes_dict[box_id].print_box(box_id)

def print_box_aug_dict(boxes_dict):
    for box_id in boxes_dict: boxes_dict[box_id].print_box_augmentation(box_id)    
        
'''
Round x and y coordinates in Pandas DataFrame by conversion from float to int.

Input: df (dtype = DataFrame)
Output: DataFrame
'''
def prepare_predictions(df):
    df['box_x1'] = df['box_x1'].astype('int')
    df['box_x2'] = df['box_x2'].astype('int')
    df['box_y1'] = df['box_y1'].astype('int')
    df['box_y2'] = df['box_y2'].astype('int')
    df = df.reset_index(drop=True)
    df['label'] = df['label'].str.replace(' ', '_')
    return df

'''
Helper function used to convert a given value to a binary number based on lookup table. Binary values
are stored in numpy matrix. This is used to indicate classes present in a given pixel.
Example: lookup = ['background', 'person', 'handbag', 'bench']
A binary value of 0110 indicates that both a person and handbag are present in a given pixel

Input: matrix (dtype = np.array), value (dtype = string), lookup (dtype = list),
xmin, ymin, xmax, ymax (dtype = int)
Output: np.array (matrix)
'''
def set_binary_matrix_slice(matrix, value, lookup, xmin, ymin, xmax, ymax):
    value_index = lookup.index(value)
    binary_list = ['0'] * len(lookup)
    
    binary_list[value_index] = '1'
    binary_string = "".join(binary_list)
    binary_num = int(binary_string, 2)

    matrix[ymin : ymax, xmin : xmax] = np.bitwise_or(matrix[ymin : ymax, xmin : xmax], binary_num)
    return matrix
'''
Helper function used to set matrix values to maximum value. If a matrix value is greater than the input value, 
keep the matrix value, otherwise set the matrix value to the input value. 

Input: matrix (dtype = np.array), value (dtype = float), xmin, ymin, xmax, ymax (dtype = int)
Output: np.array(matrix)
'''
def set_max_matrix_slice(matrix, value, xmin, ymin, xmax, ymax):
    matrix_slice = matrix[ymin : ymax, xmin : xmax]
    matrix_slice[matrix_slice < value] = value
    matrix[ymin : ymax, xmin : xmax] = matrix_slice
    return matrix

'''
Helper function used to pad box coordinates by 15 pixels (within image bounds)

Input: xmin, ymin, xmax, ymax (dtype = int), image (dtype = Image)
Output: int, int, int, int
'''
def pad_box(xmin, ymin, xmax, ymax, image, amount):
    if xmax + amount <= image.width: xmax = xmax + amount
    else: xmax = image.width
    if ymax + amount <= image.height: ymax = ymax + amount
    else: ymax = image.height
    if xmin - amount > 0: xmin = xmin - amount
    else: xmin = 0
    if ymin - amount > 0: ymin = ymin - amount
    else: ymin = 0
    
    return xmin, ymin, xmax, ymax
'''
Create multipolygon box region using shapely library and predictions from deep learning model.
Additionally sets up label information and score information using matricies.

Input: df(dtype = DataFrame), image (dtype = Image)
Output: multipolygon (dtype = Shapely.MultiPolygon), image = (dtype = Image)
'''
def create_box_region_polygon(df, image, padding_amount):
    #p = []
    binary_mask = np.zeros(image.shape).astype('int')
    label_matrix = np.zeros(image.shape).astype('int')

    score_matricies = {}
    label_lookup = ['background'] + list(df['label'].unique())
    for label in label_lookup: score_matricies[label] = np.zeros(image.shape).astype('float')
    
    for _, row in df.iterrows():
        xmin, xmax, ymin, ymax = int(row['box_x1']), int(row['box_x2']), int(row['box_y1']), int(row['box_y2'])

        label_matrix = set_binary_matrix_slice(label_matrix, row['label'], label_lookup, xmin, ymin, xmax, ymax)
        score_matricies[row['label']] = set_max_matrix_slice(score_matricies[row['label']], row['score'], xmin, ymin, xmax, ymax)
        
        xmin, ymin, xmax, ymax = pad_box(xmin, ymin, xmax, ymax, image, padding_amount)
        #rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, edgecolor='r', facecolor="none", linewidth = 5)
        #p.append(rect)
        binary_mask[ymin : ymax, xmin : xmax] = 1
    
    image.set_label_info(label_lookup, label_matrix, score_matricies)
    
    contour, hier = cv2.findContours(binary_mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
    
    remove = []
    for i, c in enumerate(contour):
        if len(c) < 4: remove.append(i)
    if len(remove) != 0: contour = np.delete(contour, remove)
        
    contours = map(np.squeeze, contour)
    polygons = map(Polygon, contours)
    multipolygon = MultiPolygon(polygons)
    
    return multipolygon, image

'''
Functions to divide/pad multipolygon box region into squares of specified grid size

Input: multipolygon (dtype = Shapely.MultiPolygon), square_size (dtype = int)
Output: Shapely.MultiPolygon
'''
def create_square(x, y, s):
    return Polygon([(x, y), (x+s, y), (x+s, y+s), (x, y+s)])

def divide_box_region_with_grid(multipolygon, square_size):
    square_bounds = np.array(multipolygon.bounds)//square_size
    square_bounds[2:4] += 1
    xmin, ymin, xmax, ymax = square_bounds*square_size
    x_coords = np.arange(xmin, xmax, square_size)
    y_coords = np.arange(ymin, ymax, square_size)
    grid_polygons = MultiPolygon([create_square(x, y, square_size) for x in x_coords for y in y_coords])
    grid_polygons = MultiPolygon(list(filter(multipolygon.intersects, grid_polygons)))

    return grid_polygons


'''
Functions to merge the grid region into one or more polygons. Final output is a list of tuples with first 
element being the polygon and second element being the corresponding coordinates.

Input: grid_polygons (dtype = Shapely.MultiPolygon)
Output: List of tuples
'''
def rotate_coords(l, n):
    return l[-n:] + l[:-n]

def combine_grid_region(grid_polygons):
    final_polygon = unary_union(grid_polygons.geoms)
    if str(type(final_polygon)) != "<class 'shapely.geometry.multipolygon.MultiPolygon'>": 
        final_polygon = [final_polygon]
    
    final_poly_reduced_coords = []
    for poly in final_polygon:
        coords = (list(poly.exterior.coords))
        while (coords[0][0] == coords[-1][0]): coords = rotate_coords(coords, 1)
        reduced_coords = np.array(coords + [coords[0]])
        
        x_coords = reduced_coords[:,0]
        reduced_coords = np.vstack([(x[0], x[-1]) if len(x) > 1 else x[0] for x in np.split(reduced_coords, [i+1 for i, (x1, x2) in enumerate(zip(x_coords, x_coords[1:])) if x1 != x2])])
        y_coords = reduced_coords[:,1]
        reduced_coords = np.vstack([(y[0], y[-1]) if len(y) > 1 else y[0] for y in np.split(reduced_coords, [i+1 for i, (y1, y2) in enumerate(zip(y_coords, y_coords[1:])) if y1 != y2])])
        
        final_poly_reduced_coords.append((Polygon(reduced_coords), reduced_coords))
        
    return final_poly_reduced_coords

'''
Cut the polygon at verticies that are not aligned with polygon bounds. Creates new split MultiPolygon.

Input: polygon (dtype = Shapely.Polygon), poly_coords(dtype = np.array)
Output: Shapely.MultiPolygon
'''
def cut_polygon_at_verticies(polygon, poly_coords):
    x_lower = min(polygon.bounds[0], polygon.bounds[2])
    x_upper = max(polygon.bounds[0], polygon.bounds[2])
    y_lower = min(polygon.bounds[1], polygon.bounds[3])
    y_upper = max(polygon.bounds[1], polygon.bounds[3])
    lines = []
    for coords in poly_coords:
        x_line = LineString([(coords[0],y_lower), (coords[0],y_upper)])
        y_line = LineString([(x_lower,coords[1]), (x_upper,coords[1])])

        lines.append(x_line)
        lines.append(y_line)
    split_polygons = polygonize(polygon.intersection(MultiLineString(lines)))
    split_polygons = MultiPolygon(split_polygons)
    
    return split_polygons

'''
Returns valid coordinates of polygon. If any of the grid regions exceed image bound (which can occur after
applying grid padding), clip the coordinates to be at image bound. Input is list of x coords, list of y coords,
and image object.

Input: xs (dtype = List), ys (dtype = List)
Output: xmin, ymin, xmax, ymax (dtype = int)
'''
def get_valid_coordinates(xs, ys, image):
    if min(xs) < 0: xmin = 0
    else: xmin = min(xs)
    if min(xs) > image.width: xmin = image.width
    if max(xs) > image.width: xmax = image.width
    else: xmax = max(xs)

    if min(ys) < 0: ymin = 0
    else: ymin = min(ys)
    if min(ys) > image.height: ymin = image.height
    if max(ys) > image.height: ymax = image.height
    else: ymax = max(ys)
    
    return int(xmin), int(ymin), int(xmax), int(ymax)

'''
Creates list of box objects from split polygons. Assigned scores to each box based on the maximum score
present in that region. 

Input: polygons (dtype = Shapely.MultiPolygon), image (dtype = Image)
Output: List of box objects
'''
def create_box_objects(polygons, image):
    box_indicies = sorted([(i, b.area) for i, b in enumerate(polygons)], key=lambda x: -x[1])
    
    box_objects = []
    for position, _ in box_indicies:
        xmin, ymin, xmax, ymax = get_valid_coordinates(polygons[position].exterior.xy[0], polygons[position].exterior.xy[1], image)
        
        if xmax - xmin != 0.0 and ymax - ymin != 0.0: #if the box has width and height
            new_box = Box(xmin, ymin, xmax, ymax)
            box_objects.append(new_box)

    return box_objects


'''
Recursively reattaches new_box if it shares an edge with an existing box. Merge boxes that share an edge.

Input: boxes (dtype = List), image (dtype = Image)
Output: List of box objects, Box objects
'''
def process_reattach(boxes, new_box):
    for existing_box in boxes: #check existing dictionary elements
        share_edge = False

        if (existing_box.xmin, existing_box.ymin, existing_box.xmax, existing_box.ymin) == (new_box.xmin, new_box.ymax, new_box.xmax, new_box.ymax): share_edge = True
        elif (new_box.xmin, new_box.ymin, new_box.xmax, new_box.ymin) == (existing_box.xmin, existing_box.ymax, existing_box.xmax, existing_box.ymax): share_edge = True
        elif (existing_box.xmin, existing_box.ymin, existing_box.xmin, existing_box.ymax) == (new_box.xmax, new_box.ymin, new_box.xmax, new_box.ymax): share_edge = True
        elif (new_box.xmin, new_box.ymin, new_box.xmin, new_box.ymax) == (existing_box.xmax, existing_box.ymin, existing_box.xmax, existing_box.ymax): share_edge = True
            
        if share_edge:
            xmin, ymin = (min(new_box.xmin, existing_box.xmin), min(new_box.ymin, existing_box.ymin))
            xmax, ymax = (max(new_box.xmax, existing_box.xmax), max(new_box.ymax, existing_box.ymax))

            #check to see if the merge captured any of the existing boxes,
            boxes.remove(existing_box)
            boxes, new_box = process_reattach(boxes, Box(xmin, ymin, xmax, ymax))
            return boxes, new_box
    return boxes, new_box


'''
Reattach boxes that share an edge. A somewhat greedy approach with tries to reattach boxes with higher scores
first. Returns list of box objects.

Input: boxes (dtype = List of Box Objects
Output: List of Box Objects
'''
def reattach_shared_edges(boxes):
    final_boxes = []
    new_box_id = 0 #variable to track index of new boxes
    for new_box in boxes:
        final_boxes, new_box = process_reattach(final_boxes, new_box)
        final_boxes.append(new_box)

    return final_boxes

'''
Helper function to obtain box label for a region. Examines the unique labels present in the region and looks 
at how many pixels make up each label. Sort labels by area in region and preserve labels before the largest drop
in area. 

Input: box (dtype = Box), image (dtype = Image)
Output: List
'''
def get_box_labels(box, image):
    labels, counts = np.unique(image.label_matrix[box.ymin : box.ymax, box.xmin : box.xmax], return_counts=True)
    labels = np.column_stack((labels, counts)) 
    labels = labels[labels[:, 0] != 0]
    labels = sorted(labels, key=lambda k: -k[1])
    
    final_labels = []
    unique_labels = []
    for label in labels:
        binary_num = bin(label[0])
        binary_list = [int(x) for x in binary_num[2:]]

        if len(binary_list) < len(image.label_lookup):
            pad = [0] * (len(image.label_lookup) - len(binary_list))
            binary_list = pad + binary_list

        for index, bit in enumerate(binary_list):
            if bit: 
                l = image.label_lookup[index]
                s = 100 * np.mean(image.score_matricies[l][box.ymin : box.ymax, box.xmin : box.xmax][image.score_matricies[l][box.ymin : box.ymax, box.xmin : box.xmax] > 0])
                relative_size = label[1]/box.area
                if not l in unique_labels: 
                    final_labels.append((l, relative_size*s, s))
                    unique_labels.append(l)
    
    final_labels = sorted(final_labels, key=lambda k: -k[1])

    if len(final_labels) > 1:
        diff = [abs(final_labels[i][1] - final_labels[i+1][1]) for i in range(len(final_labels)-1)]
        final_labels = final_labels[0:np.argmax(diff)+1]
    
    if len(final_labels) > 0:
        labels, metrics, scores = zip(*final_labels)
    else: labels, scores = [], []
    return labels, scores


'''
Class based scaling using open images superset of sensitive classes
'''
def OID_class_based_scaling(boxes_dict, size):
    sensitive_classes = ['apple',  'orange',  'traffic_light',  'laptop',  'horse',  'car',  'zebra',  'person',  'sheep',  'cow',  'sports_ball',  'broccoli',  'airplane',  'book',  'spoon',  'vase',  'boat',  'carrot',  'skateboard',  'surfboard',  'bottle',  'donut',  'knife',  'suitcase',  'clock',  'backpack',  'tennis_racket',  'giraffe',  'couch',  'toilet',  'bus']
    highly_sensitive_classes = ['handbag']
    
    for box_id in boxes_dict:
        b = boxes_dict[box_id]
        
        if not set(b.label).isdisjoint(set(highly_sensitive_classes)): factor = 0
        elif set(b.label).isdisjoint(set(sensitive_classes)): factor = 50
        else: factor = 30

        scale = factor * (size/100)
            
        print(f"{b.label} permits {scale}% scaling based on size {size}")
        
        w_, h_ = math.ceil(boxes_dict[box_id].width * (100-scale)/100), math.ceil(boxes_dict[box_id].height * (100-scale)/100)
        boxes_dict[box_id].set_augmented_box(w_, h_)
    #print(print_box_aug_dict(boxes_dict))
    return boxes_dict

'''
Function to align boxes to n by n block size
'''
def closest_factor(x, n): #example: n = 16
    z = max(x - (x % n), n)
    return z

def CTU_scale(boxes_dict, factor):
    for box_id in boxes_dict:
        w, h = boxes_dict[box_id].aug_width, boxes_dict[box_id].aug_height
        
        long = max(w,h)
        short = min(w,h)
        
        short_ctu = closest_factor(short, factor)
        long_ctu = math.floor(long * (short_ctu / short))
        long_ctu = closest_factor(long_ctu, factor)
        
        if h == long: w_, h_ = (short_ctu, long_ctu)
        else: w_, h_ = (long_ctu, short_ctu)
        
        boxes_dict[box_id].set_augmented_box(w_, h_)
        
    return boxes_dict