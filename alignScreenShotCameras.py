import argparse
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import shutil




def extract_transformation_matrices_instant(file_path):
    """
    Extract transformation matrices from a JSON file.
    
    :param file_path: str, the path to the JSON file
    :return: dict, a dictionary with file paths as keys and transformation matrices as values
    """
    # Load the JSON content
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract transformation matrices
    transformations = []
    indexes = []
    for frame in data['frames']:
        file_path = frame['file_path']
        transform_matrix = np.array(frame['transform_matrix'])
        transformations.append(transform_matrix)

        #extract index from file_path
        filename = file_path.split('/')[-1]

        # Extract the index part from the filename (remove the .png and take the last 4 characters)
        index_str = filename[:-4]

        # Convert the index to an integer
        index = int(index_str)



        indexes.append(index)
    
    #ordered_list = [transformations[key] for key in sorted(transformations)]
    return transformations, indexes


def save_corrected_dataset_to_json(corrected_dataset, original_json_path, output_json_path, output_folder):
    """
    Save the corrected dataset back to a JSON file, preserving the original structure.
    
    :param corrected_dataset: List of 4x4 numpy.ndarray transformation matrices (corrected).
    :param original_json_path: String, path to the original JSON file to preserve structure.
    :param output_json_path: String, path where the corrected JSON file will be saved.
    """
    # Load the original JSON file to preserve its structure
    with open(original_json_path, 'r') as file:
        data = json.load(file)
    
    # Assuming the structure includes a 'frames' list where transformation matrices are stored
    for i, frame in enumerate(data['frames']):
        if i < len(corrected_dataset):
            # Convert the numpy array back to a list for JSON serialization
            corrected_matrix_list = corrected_dataset[i].tolist()
            # Update the transformation matrix in the original data structure
            frame['transform_matrix'] = corrected_matrix_list
            

            file_path = frame['file_path']
       
            filename = file_path.split('/')[-1]

            new_file_path = os.path.join("./images", filename)

            frame['file_path'] = new_file_path

    
    # Write the updated data back to a new JSON file
    with open(output_json_path, 'w') as file:
        json.dump(data, file, indent=4)


def create_screenshot_cameras(transformation_matrices, input_json_file_path,output_folder, screenshot_limit=-1):
    """
    Create screenshot cameras from transformation matrices and save them to a folder.
    
    :param transformation_matrices: list of 4x4 numpy.ndarray, transformation matrices for the screenshot cameras.
    :param output_folder: str, the folder where to save the screenshot cameras.
    """
    

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)


    with open(input_json_file_path, 'r') as f:
        data = json.load(f)

    data.pop('frames', None)
    data['frames'] = []

    

    # Limit the number of screenshots if necessary
    if screenshot_limit != -1:
        transformation_matrices = transformation_matrices[:screenshot_limit]

    for i, mat in enumerate(transformation_matrices):
        # Create a new camera frame
        frame = {}
        #frame['file_path'] = f'frame_{i}.png'
        frame['transform_matrix'] = mat.tolist()
        #Zeropadded file path
        file_path = f'./images/{str(i).zfill(4)}.png'
        
        frame['file_path'] = file_path
        data['frames'].append(frame)

    # Write the data to a JSON file
        print(os.path.join(output_folder, 'screenshot_cams.json'))
    with open(os.path.join(output_folder, 'screenshot_cams.json'), 'w') as file:
        json.dump(data, file, indent=4)


def extract_transformation_matrices_from_yaml(file_path, indexes = []):
    """
    Extract and convert transformation matrices from a YAML file.
    
    :param file_path: str, the path to the YAML file
    :return: list, a list of 4x4 transformation matrices
    """
    # Load the YAML content
    with open(file_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Initialize a list to hold transformation matrices
    transformation_matrices = []

    #print(indexes)
    #print(data.keys())

    if(len(indexes) > 0):
        for i in indexes:
            # Extract the rotation matrix and reshape it to 3x3
            block= data[i]
            
            
            cam_R_m2c = np.array(block[0].get('cam_R_m2c')).reshape(3, 3)
            
            # Extract the translation vector
            cam_t_m2c = np.array(block[0].get('cam_t_m2c')).reshape(3, 1)
            
            # Combine into a 4x4 transformation matrix
            transformation_matrix = np.vstack((np.hstack((cam_R_m2c, cam_t_m2c)),
                                               [0, 0, 0, 1]))
            
            transformation_matrices.append(transformation_matrix)
    else:
        for item in data.values():
            # Extract the rotation matrix and reshape it to 3x3
            cam_R_m2c = np.array(item[0]['cam_R_m2c']).reshape(3, 3)
            
            # Extract the translation vector
            cam_t_m2c = np.array(item[0]['cam_t_m2c']).reshape(3, 1)
            
            # Combine into a 4x4 transformation matrix
            transformation_matrix = np.vstack((np.hstack((cam_R_m2c, cam_t_m2c)),
                                            [0, 0, 0, 1]))
            
            transformation_matrices.append(transformation_matrix)
    
    return transformation_matrices


def plot_transformation_matrices(transformation_matrices, ax, title="Transformation Matrices", limit = -1):
    """
    Plot transformation matrices as arrows in a 3D plot.
    
    :param transformation_matrices: list of 4x4 numpy.ndarray, transformation matrices to plot
    :param ax: matplotlib 3D axis object to plot on
    :param title: str, title of the plot
    """



    if limit != -1:
        transformation_matrices = transformation_matrices[:limit]


    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # Define colors for each axis
    colors = ['r', 'g', 'b']  # Red for X, Green for Y, Blue for Z

    all_positions = np.array([mat[:3, 3] for mat in transformation_matrices])
    #add origin
    all_positions_with_origin = np.vstack((all_positions, [0, 0, 0]))
    
    # Calculate the mean position
    mean_position = np.mean(all_positions_with_origin, axis=0)
    
    # Determine the overall range of positions (ptp) to set uniform axis ranges

    position_range = np.ptp(all_positions_with_origin, axis=0)
    max_range = np.max(position_range) / 2.0
    
    # Set axis limits centered around the mean position with uniform range
    
    
    # Plot transformation matrices (the plotting code remains unchanged)
    # Define colors for each axis
    colors = ['r', 'g', 'b']  # Red for X, Green for Y, Blue for Z
    arrow_scale = max_range / 5  # Adjust arrow_scale if necessary



    #Draw origin    
    ax.scatter(0, 0, 0, c='black', marker='o')
    
    

    for matrix in transformation_matrices:
        

        start_point = matrix[:3, 3]
        
        # Plot arrows for each axis, scaled by arrow_scale
        for i in range(3):
            direction = matrix[:3, i] * arrow_scale  # Scale the direction vector for visualization
            ax.quiver(start_point[0], start_point[1], start_point[2], 
                      direction[0], direction[1], direction[2], 
                      color=colors[i], length=np.linalg.norm(direction), normalize=True)

    ax.set_xlim(mean_position[0] - max_range, mean_position[0] + max_range)
    ax.set_ylim(mean_position[1] - max_range, mean_position[1] + max_range)
    ax.set_zlim(mean_position[2] - max_range, mean_position[2] + max_range)

def plot_and_save_transformation_datasets(dataset1, dataset2, save_path, limit=-1, angle_elevation=30, angle_azimuth=30):
    """
    Plot two sets of transformation matrices in 3D up to a given limit and save the figure.

    :param dataset1: list of 4x4 numpy.ndarray, the first set of transformation matrices.
    :param dataset2: list of 4x4 numpy.ndarray, the second set of transformation matrices.
    :param limit: int, the maximum number of transformations to display from each dataset.
    :param save_path: str, the path where to save the figure.
    """
    fig = plt.figure(figsize=(14, 7))

    # Ensure the limit does not exceed the dataset size
    if limit == -1:
        limit = max(len(dataset1), len(dataset2))

    limit1 = min(limit, len(dataset1))
    limit2 = min(limit, len(dataset2))

    limit = min(limit1, limit2)

    # Plot for the first dataset
    ax1 = fig.add_subplot(121, projection='3d')
    plot_transformation_matrices(dataset1[:limit], ax1, title="Dataset 1 Transformations")

    # Plot for the second dataset
    ax2 = fig.add_subplot(122, projection='3d')
    plot_transformation_matrices(dataset2[:limit], ax2, title="Dataset 2 Transformations")

    ax1.view_init(elev=angle_elevation, azim=angle_azimuth)
    ax2.view_init(elev=angle_elevation, azim=angle_azimuth)

    plt.savefig(save_path)



def compute_centroid(points):
    """Compute the centroid of a set of points."""
    return np.mean(points, axis=0)

def align_datasets(dataset1, dataset2):
    """
    Align the first dataset to the second using SVD for optimal rotation and translation.
    
    :param dataset1: list of 4x4 numpy.ndarray, transformation matrices from the first dataset.
    :param dataset2: list of 4x4 numpy.ndarray, transformation matrices from the second dataset.
    :return: R, s, t - Rotation matrix, scale, and translation vector to align dataset1 to dataset2.
    """

    #ensure both datasets have the same length
    min_len = min(len(dataset1), len(dataset2))
    dataset1 = dataset1[:5]
    dataset2 = dataset2[:5]

    # Extract positions (translations) from the transformation matrices
    
    positions1 = np.array([mat[:3, 3] for mat in dataset1])
    positions2 = np.array([mat[:3, 3] for mat in dataset2])
    
    # Compute centroids of both sets
    centroid1 = compute_centroid(positions1)
    centroid2 = compute_centroid(positions2)
    
    # Align positions to the centroids
    aligned_positions1 = positions1 - centroid1
    aligned_positions2 = positions2 - centroid2
    
    # Compute the scale factor 's' (assuming uniform scaling for simplicity)
    scale_factors = np.linalg.norm(aligned_positions2, axis=1) / np.linalg.norm(aligned_positions1, axis=1)
    s = np.mean(scale_factors)
    
    # Scale aligned_positions1
    aligned_positions1_scaled = aligned_positions1 * s
    
    # Compute the covariance matrix and perform SVD
    H = np.dot(aligned_positions1_scaled.T, aligned_positions2)
    U, _, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    
    # Ensure a right-handed coordinate system
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)
    
    # Compute the translation vector
    t = centroid2 - np.dot(R, centroid1 * s)
    
    
    return R, s, t

def construct_transformation_matrix(R, s, t):
    """Construct a 4x4 transformation matrix from rotation, scale, and translation."""
    T = np.identity(4)
    T[:3, :3] = R * s
    T[:3, 3] = t
    return T


def normalize_rotation_matrix(R):
    """
    Normalize a rotation matrix using SVD to ensure it remains orthogonal.
    
    :param R: The rotation matrix to normalize.
    :return: The normalized rotation matrix.
    """
    U, _, Vt = np.linalg.svd(R, full_matrices=False)
    R_normalized = np.dot(U, Vt)
    return R_normalized

def apply_and_normalize_transformation(dataset, T):
    """
    Apply a similarity transformation to all matrices in a dataset and normalize the rotation part.
    
    :param dataset: List of 4x4 numpy.ndarray transformation matrices.
    :param T: The 4x4 numpy.ndarray similarity transformation matrix.
    :return: A new list of transformed and normalized 4x4 numpy.ndarray transformation matrices.
    """
    transformed_dataset = []
    for mat in dataset:
        # Apply the transformation
        transformed_mat = np.dot(T, mat)
        # Normalize the rotation part
        R_normalized = normalize_rotation_matrix(transformed_mat[:3, :3])
        # Reconstruct the transformation matrix with the normalized rotation
        transformed_mat_normalized = np.eye(4)
        transformed_mat_normalized[:3, :3] = R_normalized
        transformed_mat_normalized[:3, 3] = transformed_mat[:3, 3]  # Keep the translation part unchanged
        transformed_dataset.append(transformed_mat_normalized)
    return transformed_dataset

def invert_transformation_matrices(dataset):
    """
    Invert transformation matrices in a dataset.
    
    :param dataset: list of 4x4 numpy.ndarray, transformation matrices to be inverted.
    :return: A new list of inverted 4x4 numpy.ndarray transformation matrices.
    """
    inverted_dataset = []
    for mat in dataset:
        # Extract the rotation and translation components
        R = mat[:3, :3]
        t = mat[:3, 3]
        
        # Invert the rotation by transposing
        R_inv = R.T
        # Invert the translation
        t_inv = -np.dot(R_inv, t)
        
        # Construct the inverted transformation matrix
        mat_inv = np.eye(4)  # Start with an identity matrix
        mat_inv[:3, :3] = R_inv
        mat_inv[:3, 3] = t_inv
        
        inverted_dataset.append(mat_inv)
    return inverted_dataset


def mirror_dataset(dataset, axis='x'):
    """
    Mirror transformation matrices in a dataset along a specified axis.
    
    :param dataset: List of 4x4 numpy.ndarray transformation matrices.
    :param axis: String, the axis along which to mirror ('x', 'y', or 'z').
    :return: A new list of mirrored 4x4 numpy.ndarray transformation matrices.
    """
    mirrored_dataset = []
    axis_indices = {'x': 0, 'y': 1, 'z': 2}
    axis_index = axis_indices[axis.lower()]
    
    for mat in dataset:
        # Copy the matrix to avoid altering the original
        mat_mirrored = np.copy(mat)
        
        # Mirror the translation component
        mat_mirrored[axis_index, 3] *= -1
        
        # If necessary, adjust the rotation component to maintain correct orientation after mirroring
        # This step depends on how your specific application handles rotations
        # Example for mirroring rotation along the X-axis:
        if axis.lower() == 'x':
            mat_mirrored[1:3, 0:3] *= -1  # Invert the Y and Z components of the rotation
        
        mirrored_dataset.append(mat_mirrored)
    
    return mirrored_dataset

def correctRotation(dataset):
    #Rotate 180 degrees around y axis then 180 degrees around z axis. dont change translation
    
    for mat in dataset:
        
        # Extract the rotation and translation components
        R = mat[:3, :3]
        t = mat[:3, 3]
        
        # Invert the rotation by transposing
        
        #Rotate 180 degrees around y axis
        R_Rot = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        R = np.dot(R, R_Rot)

        #Rotate 180 degrees around z axis
        R_Rot = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
        R = np.dot(R, R_Rot)

        # Construct the inverted transformation matrix
        mat[:3, :3] = R
    return dataset
        
        
def adjust_scale_and_translation(dataset, scale_factor, translation_vector):
    """
    Adjust the scale and translation of each transformation matrix in the dataset.
    
    :param dataset: List of 4x4 numpy.ndarray transformation matrices.
    :param scale_factor: float, the uniform scale factor to apply.
    :param translation_vector: numpy.ndarray, the translation vector to apply.
    :return: A new list of adjusted 4x4 numpy.ndarray transformation matrices.
    """
    adjusted_dataset = []
    for mat in dataset:
        # Copy the original matrix to preserve its rotation part
        adjusted_mat = np.copy(mat)
        
        # Adjust scale
        adjusted_mat[:3, 3] *= scale_factor
        
        # Adjust translation
        adjusted_mat[:3, 3] += translation_vector
        
        adjusted_dataset.append(adjusted_mat)
    
    return adjusted_dataset


    


# json_file_path = 'dataGenerator/data/transforms.json'
# json_file_path_corrected = 'dataGenerator/data/transforms_corrected.json'
# yaml_file_path = 'dataGenerator/Linemod_preprocessed/data/01/gt.yml'

# transformation_matrices_json = extract_transformation_matrices_instant(json_file_path)
# transformation_matrices_yaml = extract_transformation_matrices_from_yaml(yaml_file_path)


# #transformation_matrices_yaml_corrected = mirror_dataset(transformation_matrices_yaml_corrected, axis='x')

# dataset1=transformation_matrices_json
# dataset2=transformation_matrices_yaml


# #invert 
# #dataset1 = invert_transformation_matrices(dataset1)
# dataset2 = invert_transformation_matrices(dataset2)
# dataset2 = correctRotation(dataset2)



# angle_elevation = 30
# angle_azimuth = 90
# limit = 100
# plot_and_save_transformation_datasets(dataset1, dataset2, 'output/0beforeAlignment.png', limit=limit, angle_elevation=angle_elevation, angle_azimuth=angle_azimuth)


# R, s, t = align_datasets(dataset2, dataset1)
# T = construct_transformation_matrix(R, s, t)

# #transformed_dataset1 = apply_and_normalize_transformation(dataset1, T)

# dataset2= adjust_scale_and_translation(dataset2, s, [0, 0, 0])


# save_corrected_dataset_to_json(dataset2, json_file_path, json_file_path_corrected)

# plot_and_save_transformation_datasets(dataset1, dataset2, 'output/1afterAlignment.png', limit=limit, angle_elevation=angle_elevation, angle_azimuth=angle_azimuth)


# print("Done!")



if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Process and save corrected transformation matrices.')
    
    # Add arguments for input and output JSON file paths
    parser.add_argument('input_folder', type=str, help='Path to the input data location.')
    parser.add_argument('output_folder', type=str, help='Path to the output data location.')
    parser.add_argument('reference_yaml', type=str, help='Path to the reference YAML file.')
    parser.add_argument('full_yaml', type=str, help='Path to the full reference YAML file, to create the screenshot cams.')
    parser.add_argument('camera_location', type=str, help='Path to the camera location folder.')
    parser.add_argument('screenshot_limit', type=int, help='Limit the number of transformations to display in the plot.', default=-1)
    
    # Parse the command-line arguments
    args = parser.parse_args()
    
    
    #check if output folder exists
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
    shutil.rmtree(args.output_folder)  # Remove


    #Copy input folder to output folder
    shutil.copytree(args.input_folder, args.output_folder)


    # Load the transformation matrices from the JSON file
    json_file_path = os.path.join(args.output_folder, 'transforms.json')
    transformation_matrices_json, indexes = extract_transformation_matrices_instant(json_file_path)

    # Load the transformation matrices from the YAML file
    yaml_file_path = args.reference_yaml
    transformation_matrices_yaml = extract_transformation_matrices_from_yaml(yaml_file_path, indexes)


    screenshot_yaml_file_path = args.full_yaml
    transformation_matrices_yaml_screenshot = extract_transformation_matrices_from_yaml(screenshot_yaml_file_path)


    dataset1=transformation_matrices_json
    dataset2=transformation_matrices_yaml

    
    #invert 
    #dataset1 = invert_transformation_matrices(dataset1)
    dataset2 = invert_transformation_matrices(dataset2)
    dataset2 = correctRotation(dataset2)

    transformation_matrices_yaml_screenshot = invert_transformation_matrices(transformation_matrices_yaml_screenshot)
    transformation_matrices_yaml_screenshot = correctRotation(transformation_matrices_yaml_screenshot)



    
    R, s, t = align_datasets(dataset2, dataset1)
    
    # And a function to save the corrected dataset:
    
    dataset2= adjust_scale_and_translation(dataset2, s, [0, 0, 0])

    transformation_matrices_yaml_screenshot = adjust_scale_and_translation(transformation_matrices_yaml_screenshot, s, [0, 0, 0])


    save_corrected_dataset_to_json(dataset2, json_file_path, json_file_path, args.output_folder)

    #Create screenshot cameras
    create_screenshot_cameras(transformation_matrices_yaml_screenshot,json_file_path,  args.camera_location, args.screenshot_limit)