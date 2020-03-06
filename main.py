from sklearn.externals import joblib
import cv2
from image_related.image_constructor import get_mask, apply_mask
from image_related.image_predictor import extractDominantColor, plotColorBar
from matplotlib import pyplot as plt


image = cv2.imread("test/test_images/cabelo_claro/23_1_0_20170104165334721.jpg")
mask = get_mask(image)
builded_image = apply_mask(image, mask)
colorInformation = extractDominantColor(builded_image, hasThresholding=True)
dominantColors = colorInformation[0].get('color') + colorInformation[1].get('color') + colorInformation[2].get('color') + colorInformation[3].get('color') + colorInformation[4].get('color')

color_bar = plotColorBar(colorInformation)
plt.axis("off")
plt.imshow(color_bar)
plt.show()