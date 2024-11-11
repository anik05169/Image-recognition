import matplotlib.pyplot as plt
import keras_ocr

# Initialize the keras-ocr pipeline, which downloads pretrained weights for detector and recognizer
pipeline = keras_ocr.pipeline.Pipeline()

# List of image URLs to process
urls = [
    #'https://upload.wikimedia.org/wikipedia/commons/b/bd/Army_Reserves_Recruitment_Banner_MOD_45156284.jpg',
    #'https://upload.wikimedia.org/wikipedia/commons/e/e8/FseeG2QeLXo.jpg',
    'https://upload.wikimedia.org/wikipedia/en/thumb/4/47/FC_Barcelona_%28crest%29.svg/800px-FC_Barcelona_%28crest%29.svg.png'
]

# Load images, handling any URL errors
images = []
for url in urls:
    try:
        images.append(keras_ocr.tools.read(url))
    except Exception as e:
        print(f"Error loading {url}: {e}")

# Proceed only if images are successfully loaded
if images:
    # Recognize text in images
    prediction_groups = pipeline.recognize(images)

    # Plot the predictions
    fig, axs = plt.subplots(nrows=len(images), figsize=(20, 20))

    # Ensure `axs` is always a list, even if there's only one image
    if len(images) == 1:
        axs = [axs]

    # Draw annotations for each image
    for ax, image, predictions in zip(axs, images, prediction_groups):
        keras_ocr.tools.drawAnnotations(image=image, predictions=predictions, ax=ax)

    plt.show()
else:
    print("No images could be loaded.")
