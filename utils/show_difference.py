from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def compare_and_show_images(image_name, dir1, dir2, dir3):
    """
    현재 경로에서 이미지를 불러와서 3개의 이미지를 비교하여 보여줍니다.
    """
    image_path1 = os.path.join(dir1, image_name)
    image_path2 = os.path.join(dir2, image_name)
    image_path3 = os.path.join(dir3, image_name)

    image1 = np.array(Image.open(image_path1))
    image2 = np.array(Image.open(image_path2))
    image3 = np.array(Image.open(image_path3))

    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(image1)
    axs[0].set_title("Image from noaug150")
    axs[1].imshow(image2)
    axs[1].set_title("Image from aug80")
    axs[2].imshow(image3)
    axs[2].set_title("Image from aug105")

    # 경로 수정
    plt.savefig(
        f"/data/ephemeral/home/data/medical/img/comparison/noaug150-aug80-aug105/comparison_{image_name}.png"
    )

    plt.show()


image_names = os.listdir("/data/ephemeral/home/data/medical/img/myoutput")


for image_name in tqdm(image_names):
    if image_name.endswith(".jpg") or image_name.endswith(".png"):

        compare_and_show_images(
            image_name,
            "/data/ephemeral/home/data/medical/img/myoutput",
            "/data/ephemeral/home/data/medical/img/myoutput+aug_80",
            "/data/ephemeral/home/data/medical/img/myoutput+aug_105",
        )
