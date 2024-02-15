import os
import shutil
import random
import numpy as np
import yaml
import json

baseExperimentPath = "experiments3"
dataSourceBase = "dataGenerator/Linemod_preprocessed/data"
dataSources = ["01","08", "10"]

#imageCounts = [12,25,50,100,200]
imageCounts = [200,100,50,25,12]

class Experiment:
    def __init__(self, index,numberOfImages, dataSource):


        self.numberOfImages = numberOfImages
        self.dataSource = dataSource
        self.dataSourcePath = os.path.join(baseExperimentPath,"colmap", dataSource)

        #index name - but pad with zeros
        self.indexName = str(index).zfill(3)

        self.name = "experiment_" + self.indexName + "_dataSource_" + self.dataSource + "_images_" + str(self.numberOfImages)
        self.experimentPath = os.path.join(baseExperimentPath, self.name)


    def prepareData (self):
        # Create the experiment folder
        if not os.path.exists(self.experimentPath):
            os.makedirs(self.experimentPath)

        # Copy number of images from folder
        imageTargetPath = os.path.join(self.experimentPath, "precorrected/images")
        if not os.path.exists(imageTargetPath):
            os.makedirs(imageTargetPath)

        # Copy the images - get files and copy a certain number of them
            
        json_source_path = os.path.join(self.dataSourcePath, "transforms.json")
        json_output_path = os.path.join(self.experimentPath, "precorrected/transforms.json")

        # Load the JSON data
        with open(json_source_path, 'r') as infile:
            json_data = json.load(infile)
        
        # Randomly sample indexes
        frame_indexes = list(range(len(json_data['frames'])))  # All indexes
        random.shuffle(frame_indexes)  # Shuffle the indexes
        sampled_indexes = frame_indexes[:self.numberOfImages]  # Select the first num_frames indexes    
    
        frames = json_data['frames']
        sampled_frames = [frames[i] for i in sampled_indexes]  # Sample the frames
        
        #random.shuffle(files)
        

        i = 0

        imageIndexes = []


        for frame in sampled_frames:

            file_name = frame['file_path'].split("/")[-1]


        # Extract the index part from the filename (remove the .png and take the last 4 characters)
            index_str = file_name[:-4]

            # Convert the index to an integer
            index = int(index_str)

            newFilePath = os.path.join("./images", file_name)

            shutil.copy(os.path.join(self.dataSourcePath,"images", file_name), imageTargetPath)

            frame['file_path'] = newFilePath

            imageIndexes.append(index)
            #rename the files to have a consistent name

            #File name with leading zeros
            #new_file_name=str(i)
            #new_file_name = new_file_name.zfill(4) + ".png"

            #os.rename(os.path.join(imageTargetPath, file_name), os.path.join(imageTargetPath, new_file_name))

            i+=1


        #Copy the gt yaml file. but only for the transfered images
        gtSourcePath = os.path.join(self.dataSourcePath, "gt.yml")
        gtTargetPath = os.path.join(self.experimentPath, "gt.yml")

        with open(gtSourcePath, 'r') as gt_file:
            gt_data = yaml.safe_load(gt_file)

        #filtered_gt_data = [gt_data[i] for i in selectedIndexes if i < len(gt_data)]
        filtered_gt_data = {i: gt_data[i] for i in imageIndexes}

        #sorted_keys = sorted(filtered_gt_data.keys())

        #new_gt_data = {i: filtered_gt_data[key] for i, key in enumerate(sorted_keys)}

        new_gt_data = filtered_gt_data

        with open(gtTargetPath, 'w') as filtered_gt_file:
            yaml.dump(new_gt_data, filtered_gt_file, default_flow_style=False, width=float("inf"))


        # Create a new JSON structure with the sampled frames
            
        new_json_data = json_data.copy()  # Copy the original data
        new_json_data['frames'] = sampled_frames  # Replace the frames with the sampled ones

        # Save the new JSON data
        with open(json_output_path, 'w') as outfile:
            json.dump(new_json_data, outfile, indent=4)

        
    


    def sample_and_copy_json_frames(source_folder, output_folder, num_frames):
        """
        Samples a specific number of frames from the JSON data and copies the referred images.
        
        Args:
        - json_data: The original JSON data as a dictionary.
        - num_frames: Number of frames to randomly sample.
        - output_json_path: Path to save the new JSON file.
        - output_image_dir: Directory to copy the sampled images to.
        """
        
       
        
        # Create a new JSON structure with the sampled frames
        new_json_data = json_data.copy()  # Copy the original data
        new_json_data['frames'] = sampled_frames  # Replace the frames with the sampled ones
        
        # Save the new JSON data
        with open(output_json_path, 'w') as outfile:
            json.dump(new_json_data, outfile, indent=4)
        
        # Copy the images referred in the sampled frames
        for frame in sampled_frames:
            src_image_path = frame['file_path']
            dst_image_path = f"{output_image_dir}/{src_image_path.split('/')[-1]}"
            shutil.copy(src_image_path, dst_image_path)
        
        #print(f"Sampled {num_frames} frames and copied to {output_json_path}. Images copied to {output_image_dir}.")



    def runExperiment(self):
        # Prepare the data
        self.prepareData()

        # Run the experiment

        #Move resulting transforms.json into the experiment/precorected folder
        

        #Check if folder exists
        #if not os.path.exists(f"{self.experimentPath}/precorrected"):
        #    os.makedirs(f"{self.experimentPath}/precorrected")
        #shutil.copy(f"/volume/transforms.json", f"{self.experimentPath}/precorrected/transforms.json")
        #copy images to precorrected folder
        #shutil.copytree(f"{self.experimentPath}/images", f"{self.experimentPath}/precorrected/images")

        # parser.add_argument('input_folder', type=str, help='Path to the input data location.')
        # parser.add_argument('output_folder', type=str, help='Path to the output data location.')
        # parser.add_argument('reference_yaml', type=str, help='Path to the reference YAML file.')
        # parser.add_argument('full_yaml', type=int, help='Path to the full reference YAML file, to create the screenshot cams.')
        # parser.add_argument('camera_location', type=str, help='Path to the camera location folder.')
    

        os.system(f"python3 /volume/alignScreenShotCameras.py \
                  {self.experimentPath}/precorrected \
                  {self.experimentPath}/corrected \
                  {self.experimentPath}/gt.yml \
                  {self.dataSourcePath}/gt.yml \
                  {self.experimentPath}/ \
                  1000 ")

        
        #Run the nerf script
        #python3 scripts/run.py --scene dataGenerator/data --screenshot_transforms dataGenerator/data/transforms.json --screenshot_dir output --n_steps 100
        os.system(f"python3 /volume/scripts/run.py \
                    --scene {self.experimentPath}/corrected/ \
                    --screenshot_transforms {self.experimentPath}/screenshot_cams.json \
                    --screenshot_dir {self.experimentPath}/newOutput \
                    --n_steps 100000 \
                    --save_snapshot {self.experimentPath}/snapshot.msgpack \
                    --screenshot_spp 32\
                    --near_distance 0.33 \
                  ") 
                


def createColMapData():

    for dataSource in dataSources:

        #Create folder for the colmap data
        colmapDataPath = os.path.join(baseExperimentPath, f"colmap/{dataSource}")
        if not os.path.exists(colmapDataPath):
            os.makedirs(colmapDataPath)

        #Copy the images
        imageSourcePath = os.path.join(dataSourceBase, dataSource, "rgb")
        imageTargetPath = os.path.join(colmapDataPath, "images")
        if not os.path.exists(imageTargetPath):
            os.makedirs(imageTargetPath)
        
        files = os.listdir(imageSourcePath)
        files=files[:512]
        for file_name in files:
            print("Copying file: " + file_name)
            #if file does not exist in target folder copy it
            if not os.path.exists(os.path.join(imageTargetPath, file_name)):
                shutil.copy(os.path.join(imageSourcePath, file_name), imageTargetPath)



        if not os.path.exists(f"{colmapDataPath}/transforms.json"):
            os.system(f"python3 /volume/scripts/colmap2nerf.py --images {colmapDataPath}/images --run_colmap --overwrite")

            shutil.copy(f"/volume/transforms.json", f"{colmapDataPath}/transforms.json")

        #copy the gt.yml file
        
        shutil.copy(f"{dataSourceBase}/{dataSource}/gt.yml", f"{colmapDataPath}/gt.yml")
        #if trarget transforms.json exists, copy it to the colmap folder
        

        
        
            




def createExperiments():
    experiments = []

    index = 0
    for dataSource in dataSources:
        for imageCount in imageCounts:
            experiments.append(Experiment(index, imageCount, dataSource))
            index += 1

    return experiments


if __name__ == "__main__":
    print("Starting experiments")

    # Create the experiments
    experiments = createExperiments()
    


    if not os.path.exists(baseExperimentPath):
        os.makedirs(baseExperimentPath)

    #clear the base experiment folder
    for item in os.listdir(baseExperimentPath):
        # Construct the full path to the item
        item_path = os.path.join(baseExperimentPath, item)
        # Check if this is a folder and not the folder to keep
        if os.path.isdir(item_path) and "colmap" not in item_path:
            # Remove the folder and all its contents
            shutil.rmtree(item_path)
            print(f'Removed folder: {item_path}')

    createColMapData()
    
    # Run experiments
            
    for experiment in experiments:
        experiment.runExperiment()
        print("Experiment " + experiment.name + " done")
