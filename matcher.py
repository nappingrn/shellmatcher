import os
import rembg
import torch
import open_clip
import cv2
from sentence_transformers import util
from PIL import Image


current_path_os = os.getcwd()

# image processing model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
model.to(device)

def process_files_in_directory(directory_path):
    """
    Iterates through each file in the specified directory and performs an action.

    Args:
        directory_path (str): The path to the directory.
    """

    paths = []
    try:
        # Check if the path is a valid directory
        if not os.path.isdir(directory_path):
            raise NotADirectoryError(f"'{directory_path}' is not a valid directory.")

        # Iterate through all items (files and subdirectories) in the directory
        for filename in os.listdir(directory_path):
            # Construct the full file path
            file_path = os.path.join(directory_path, filename)

            # Check if the item is a file
            if os.path.isfile(file_path):
                # Perform actions on the file (e.g., print the name)
                paths.append(file_path)
                # Add your file processing logic here
            else:
                 print(f"Skipping non-file item: {file_path}")
    
    except FileNotFoundError:
        print(f"Error: Directory '{directory_path}' not found.")
    except NotADirectoryError as e:
         print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

    return paths

def background_remover(image_input_path, image_output_path):

    input_image = Image.open(image_input_path)
    output_image = rembg.remove(input_image)
    new_output = output_image.convert("RGB")
    new_output.save(image_output_path)

    return image_output_path


def imageEncoder(img):
    img1 = Image.fromarray(img).convert('RGB')
    img1 = preprocess(img1).unsqueeze(0).to(device)
    img1 = model.encode_image(img1)
    return img1


def generateScore(image1, image2):

    test_img = cv2.imread(image1, cv2.IMREAD_UNCHANGED)
    data_img = cv2.imread(image2, cv2.IMREAD_UNCHANGED)
    img1 = imageEncoder(test_img)
    img2 = imageEncoder(data_img)
    cos_scores = util.pytorch_cos_sim(img1, img2)
    score = round(float(cos_scores[0][0])*100, 2)
    return score

parentPath = current_path_os + "\\shells_examples"
haulPath = current_path_os + "\\shell_inputs\\shell_haul"

image1 = current_path_os + "\\shells_examples\\auger_example.jpg"
image2 = current_path_os + "\\shell_inputs\\real_test.jpg"

output1 = current_path_os + "\\shell_inputs\\intermediary\\output1.jpg"
output2 = current_path_os + "\\shell_inputs\\intermediary\\output2.jpg"

current_max = 0
path = "none"

input_set = process_files_in_directory(parentPath)
haul_set = process_files_in_directory(haulPath)


#print(f"similarity Score: ", round(generateScore(image1, image2), 2))

for haul_item in haul_set:
    for file_item in input_set:
            
            path1 = background_remover(file_item, output1)
            path2 = background_remover(image2, output2)
        
            similarity_score = generateScore(path1, path2)

            if similarity_score > current_max:
                path = file_item
                current_max = similarity_score
    
    print(haul_item)
    print(path)
    print(current_max)
    print()

    current_max = 0
    path = "none"


           

#similarity Score: 76.77