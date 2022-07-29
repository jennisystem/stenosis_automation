import os
import sys
import math
import numpy as np
import sv
sys.path.append("D:\\1_CS\\jenn\\stenosis_research")
import helper_functions
sys.path.pop()

# Source: adapted from https://github.com/neilbalch/SimVascular-pythondemos/blob/master/contour_to_lofted_model.py
# Credit: adapted from original script by Jonathan Pham

def radius(radius_inlet, x, A, sigma, mu):
    # Reference: Sun, L., Gao, H., Pan, S. & Wang, J. X. Surrogate modeling for fluid flows based on physics-constrained deep learning without simulation data. Computer Methods in Applied Mechanics and Engineering 361, (2020).

    # controls severity of stenosis/aneurysm; positive for stenosis; negative for aneurysm
    return radius_inlet - A/np.sqrt(2.0*np.pi*sigma**0.5)*np.exp(-1.0*((x - mu)**2/2/sigma**0.5))

############################################################
############################################################
################## CHANGE THESE ENTRIES ####################
############################################################
############################################################
model_name = "model01"
x0 = -15.0
xf = 15.0
nx = 31 # make this an odd number, so that we will have a segmentation at the midpoint
distance = 0 # for removal of segmentations surrounding the midsection

midsection_percentage = 15.0 # midsection area as a percentage of the inlet area
radius_inlet = 1.0

sigma = 100.0 # controls the spread of the stenosis/aneurysm
mu = 0.0 # controls the center of the stenosis/aneurysm
path_name = model_name + "_path"
segmentations_name = model_name + "_segmentations"
############################################################
############################################################
############################################################
############################################################
############################################################


def generate_points_list(x0, xf, nx, distance, f=None, g=None, h=None):
    # make path points list
    # Create z axis
    z = np.linspace(x0, xf, nx)
    
    # Remove segmentations surrounding midsection based on distance provided
    mid_index = math.floor(len(z)/2.0)
    z = np.concatenate((z[0:mid_index - distance], np.array([z[mid_index]]), z[mid_index + 1 + distance:]))

    # Create x and y axis
    y = z.copy()
    x = z.copy()

    # Transform Curvature using functions, or default to straight path
    x = np.array([f(x_i) for x_i in x]) if (f is not None) else (x * 0)
    y = np.array([g(y_i) for y_i in y]) if (g is not None) else (y * 0)
    z = np.array([h(z_i) for z_i in z]) if (h is not None) else z

    path_points_array = np.column_stack((x, y, z))
    # print(path_points_array)

    path_points_list = path_points_array.tolist()

    return path_points_list, z

    
def generate_radii_list(midsection_percentage, radius_inlet, sigma, mu, z, radius=radius):
    # make radii list
    area_midsection = (midsection_percentage/100.0)*np.pi*radius_inlet**2.0
    radius_midsection = np.sqrt(area_midsection/np.pi)
    A = -1.0*(radius_midsection - radius_inlet)*np.sqrt(2.0*np.pi*sigma**0.5)/np.exp(-1.0*((0.0 - mu)**2/2/sigma**0.5))  # A = 4.856674471372556
    print("A = ", A)
    radius_array = radius(radius_inlet, z, A, sigma, mu)
    radii_list = radius_array.tolist()
    return radii_list


def generate_model(args):
    model_name, x0, xf, nx, distance, midsection_percentage, radius_inlet, sigma, mu, f, g, radius = args
    path_name = model_name + "_path"
    segmentations_name = model_name + "_segmentations"
    
    # make path points list
    path_points_list, z = generate_points_list(x0, xf, nx, distance, f, g)
    
    # make radii list
    radii_list = generate_radii_list(midsection_percentage, radius_inlet, sigma, mu, z, radius)
    
    if len(radii_list) != len(path_points_list):
        print("Error. Number of points in radius list does not match number of points in path list.")
    
    # create path and segmnetation objects
    path = helper_functions.create_path_from_points_list(path_points_list)
    segmentations = helper_functions.create_segmentations_from_path_and_radii_list(path, radii_list)

    # add path and segmentations to the SimVascular Data Manager (SV DMG) for visualization in GUI
    sv.dmg.add_path(name = path_name, path = path)
    sv.dmg.add_segmentation(name = segmentations_name, path = path_name, segmentations = segmentations)




# List of ranges in format (start, end, step), such that range is x in [start, end] and x = start+k*step, k in Z+
ranges = [
    (-15.0, -15.0, 1.0),    # x0
    (15.0, 15.0, 1.0),      # xf
    (31, 31, 2),            # nx - make this an odd number, so that we will have a segmentation at the midpoint
    (0, 0, 1),              # distance - for removal of segmentations surrounding the midsection
    10**np.arange(-1,2.1,0.5),      # midsection_percentage - midsection area as a percentage of the inlet area (Logarithmic scale between 0.1% and 100%)
    (1.0, 1.0, 0.1),        # radius_inlet
    (50, 100, 25),          # sigma - controls the spread of the stenosis/aneurysm
    (0.0, 0.0, 0.25)        # mu - controls the center of the stenosis/aneurysm
]


# Parametric function of path's x-axis component
def f(x):
    return 0  # Zero for straight path
    return 0 if x > 0 else (x/15) ** 2

# Parametric function of path's y-axis component
def g(y):
    return 0  # Zero for straight path
    return y/3 if y < 0 else y

# Radius function
def r(radius_inlet, x, A, sigma, mu):
    arr = np.array([(x_i if x_i > 0 else x_i**2) for x_i in x])
    return radius(radius_inlet, arr, A, sigma, mu)


def generate_simulation_inputs(range_index = (len(ranges)-1)):
    # Base case, no variables in list to generate combinations -- return empty list
    if(range_index < 0):
        yield []
    # Recursive case, combinations next variable in list and yield to callee
    else:
        # If a tuple is given, then (from, to, step) is given
        if isinstance(ranges[range_index], tuple):
            # Extract from, to and step of range
            fromVal, toVal, step = ranges[range_index]
            
            # For each prior sub-combinations, create one copy for each value in range
            for arr in generate_simulation_inputs(range_index-1):
                val = fromVal
                while val <= toVal:
                    arr_new = arr.copy()
                    arr_new.append(val)
                    yield arr_new
                    val += step  # Increment value by step
        # Otherwise, a range of values is given directly
        else:
            # For each prior sub-combinations, create one copy for each value in range
            for arr in generate_simulation_inputs(range_index-1):
                for val in ranges[range_index]:  # Extract values from range
                    arr_new = arr.copy()
                    arr_new.append(val)
                    yield arr_new
            
    
def run_simulation():
    i = 1
    # Generate inputs
    for input_arr in generate_simulation_inputs():
        # x0, xf, nx, distance, midsection_percentage, radius_inlet, sigma, mu = input_arr
        model_name = "model_{0}".format(i)
        
        args = (model_name, *input_arr, f, g, r)

        generate_model(args)
        
        i += 1


run_simulation()



