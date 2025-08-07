import cv2 
import numpy as np
import pandas as pd


# change brightness and contrast of the image 
def change_brightness_contrast(image, brightness = 0, contrast = 0):
    img = np.int16(image) * (contrast / 128) - contrast + brightness 
    img = np.clip(img, 0, 255)
    return np.unit8(img)

# change color temperature of the image 
def change_color_temperature(image, temperature):
    temp_table = {
        -50: (255, 200, 150), 
        -25: (255, 220, 180),
        0 : (255, 255, 255),
        25: (220, 240, 255), 
        50: (180, 220, 255)
    }

    r, g, b = temp_table[temperature]
    matrix = np.array([[b/255.0, 0, 0], [0, g/255.0, 0], [0, 0, r/255.0]])
    image = cv2.transform(image, matrix)
    return np.clip(image, 0, 255).astype(np.uint8)

# split color channels into RGB
def split_color_channels(image):
    r, g, b = cv2.split(image)
    return r, g, b

# find line intensity ratio 
def line_intensity(image):
    r, g, b = split_color_channels(image)
    intensity_ratios = {}

    for channel, name in zip([r, g, b], ["Red", "Green", "Blue"]):
        h, w = channel.shape
        region = channel[int(0.3*h):int(0.7*h), int(0.5*w):int(0.8*w)]
        background = np.max(channel)
        test_line = np.min(region)

        ratio = background / test_line if test_line !=0 else np.nan
        intensity_ratios[name] = ratio

    return intensity_ratios

negative_test_path = ""
positive_test_path = ""

negative_img = cv2.imread(negative_test_path)
positive_img = cv2.imread(positive_test_path)

negative_img = cv2.cvtColor(negative_img. cv2.COLOR_BGR2RGB)
positive_img = cv2.cvtColor(positive_img, cv2.COLOR_BGR2RGB)

brightness_vals = list(range(-25, 25, 1))
contrast_vals = list(range(-30, 30, 1))
temp_vals = list(range(-25, 25, 1))

def save_excel_file(df, output_filename):
    excel_path=f""
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        for i in range(0, len(df), 1000000):
            df.iloc[i:i+1000000].to_excel(writer, sheet_name=f"")

def line_analysis(negative_img, positive_img, output_file):
    intensity_result = []
    for b in brightness_vals:
        for c in contrast_vals:
            for t in temp_vals:
                neg_img = change_brightness_contrast(negative_img, brightness = b, contrast = c)
                neg_img = change_color_temperature(neg_img, temperature = t)
                pos_img = change_brightness_contrast(positive_img, brightness = b, contrast = c)
                pos_img = change_color_temperature(pos_img, temperature = t)

                neg_intensity_ratio = line_intensity(neg_img)
                pos_inttensity_ratio = line_intensity(pos_img)

                for color in ["Red", "Green", "Blue"]:
                    neg_ratio = neg_intensity_ratio.get(color, np.nan)
                    pos_ratio = pos_inttensity_ratio.get(color, np.nan)
                    line_ratio = pos_ratio / neg_ratio if neg_ratio != 0 else np.nan
                    intensity_result.append((b, c, t, color, line_ratio))
    
    df = pd.DataFrame(intensity_result, columns=["Brightness", "Contrast", "Temperature", "Channel", "Ratio"])
    save_excel_file(df, output_file)

line_analysis(negative_img, positive_img, "intensity_ratio")