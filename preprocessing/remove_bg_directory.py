from rembg import remove, new_session
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from tqdm import tqdm


if __name__ == "__main__":
    parser = ArgumentParser(description="Remove background from images in a directory.")
    parser.add_argument("root_dir", type=str, help="Path to the root directory containing images.")
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"The directory {root_dir} does not exist.")
    
    imgs = sorted(root_dir.iterdir())
    model_name = "birefnet-general-lite"
    rembg_session = new_session(model_name)
    for i in range(0, len(imgs)):
        img_path = imgs[i]
        try:
            img_pil = Image.open(img_path)
            output = remove(img_pil, session=rembg_session)
            output_path = img_path.with_suffix('.png')
            output.save(output_path)
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
