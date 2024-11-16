from PIL import Image
import os

def create_directories(base_dir, num_dirs):
    for i in num_dirs:
        dir_path = os.path.join(base_dir, i)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

def main():
    image_dir = "/home/ahmad/ahmad_experiments/retinal_cond_diff/counterfactuals_samples_SPIE"
    final_dir = "/home/ahmad/ahmad_experiments/retinal_cond_diff/counterfactuals_samples_SPIE_split"
    os.makedirs(final_dir,exist_ok=True)
    num_splits = 7
    # num_dirs = ["Original","Normal", "Stage1", "Stage2", "Stage3", "laser_scars"]
    num_dirs = ["Original","AMD", "Cataracts", "DR", "Glaucoma", "Myopia", "Normal"]
    create_directories(base_dir=final_dir, num_dirs=num_dirs)

    for image_name in os.listdir(image_dir):
        image_path = os.path.join(image_dir, image_name)

        img = Image.open(image_path)
        width, height = img.size

        split_width, split_height = 224, 224

        step_width = width //num_splits

        for i, disease in enumerate(num_dirs):
            left = i * step_width
            right = left + split_width
            cropped_img = img.crop((left, 0, right, split_height))

            # Save in the corresponding directory
            subdirectory = os.path.join(final_dir, disease)
            output_path = os.path.join(subdirectory, f"{image_name[:-4]}_{disease}_{i+1}.png")
            cropped_img.save(output_path)
            print(f"Saved: {output_path}")


if __name__=="__main__":
    main()