from PIL import Image
from argparse import ArgumentParser


def paste_image(src_img: str, target_img: str, output: str):
    try:
        src_img = Image.open(src_img)
        target_img = Image.open(target_img)
    except:
        print("Image not found")
        return
    target_size = target_img.size
    
    # Calculate position to paste (center of target image)
    paste_x = (target_size[0] - src_img.size[0]) // 2
    paste_y = (target_size[1] - src_img.size[1]) // 2
    
    # We use the cow image itself as the mask to preserve transparency
    target_img.paste(src_img, (paste_x, paste_y), src_img)
    
    # Save the result
    target_img.save(output)

if __name__ == "__main__":
    parser = ArgumentParser(description="Paste a transparent image onto another image.")
    parser.add_argument("--src_img", type=str, required=True, help="Path to the img containing the subject.")
    parser.add_argument("--target_img", type=str, required=True, help="Path to the target image. The subject will be pasted onto this image.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the output image.")
    args = parser.parse_args()
    
    paste_image(args.src_img, args.target_img, args.output)
