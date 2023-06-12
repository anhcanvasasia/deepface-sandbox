import os

folder_path = "dataset/canvas-asia-face-img/Anh"  # Specify the path to the folder containing the images
file_extension = ".jpg"  # Specify the file extension of the images

# Get a list of all files in the folder
file_list = os.listdir(folder_path)

# Initialize a counter variable
counter = 1

# Loop through each file in the list
for file_name in file_list:
    if file_name.endswith(file_extension):
        # Construct the new file name with the counter
        new_file_name = folder_path.split("/")[-1] + "_" + str(counter) + file_extension

        # Get the current file path and the new file path
        current_path = os.path.join(folder_path, file_name)
        new_path = os.path.join(folder_path, new_file_name)

        # Rename the file
        os.rename(current_path, new_path)

        # Increment the counter
        counter += 1


