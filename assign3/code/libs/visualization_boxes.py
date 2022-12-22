import torch
import numpy as np
import cv2
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

COCO_CLASSES = [
    "__background__",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "N/A",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "N/A",
    "backpack",
    "umbrella",
    "N/A",
    "N/A",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "N/A",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "N/A",
    "dining table",
    "N/A",
    "N/A",
    "toilet",
    "N/A",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "N/A",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

# Create different colors for each class.
np.random.seed(42)
COLORS = np.random.uniform(0, 255, size=(100, 3))

# Define the torchvision image transforms.
transform = transforms.Compose([transforms.ToTensor()])


def draw_boxes(bounding_boxes, class_names, label_indices, image):
    """
    It takes in a list of bounding boxes, a list of class names, a list of label indices, and an image,
    and returns an image with the bounding boxes drawn on it.

    :param bounding_boxes: A list of bounding boxes, each bounding box is a list of coordinates (x1, y1,
    x2, y2)
    :param class_names: A list of class names
    :param label_indices: A list of indices of the labels that you want to draw
    :param image: the image to draw the bounding boxes on
    """
    # Calculate line width and font thickness
    line_width = max(round(sum(image.shape) / 2 * 0.003), 2)
    font_thickness = max(line_width - 1, 1)

    # Iterate through the bounding boxes
    for i, box in enumerate(bounding_boxes):
        # Calculate the coordinates of the bounding box
        top_left, bottom_right = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))

        # Get the color and class name for this bounding box
        color = COLORS[label_indices[i]]
        class_name = class_names[i]

        # Draw the bounding box on the image
        cv2.rectangle(
            image,
            top_left,
            bottom_right,
            color[::-1],
            thickness=line_width,
            lineType=cv2.LINE_AA,
        )

        # Calculate the size of the text for this bounding box
        text_width, text_height = cv2.getTextSize(
            class_name, 0, fontScale=line_width / 3, thickness=font_thickness
        )[0]

        # Determine whether to draw the text above or below the bounding box
        text_outside = top_left[1] - text_height >= 3

        # Calculate the coordinates for the text background rectangle
        text_bg_bottom_right = (
            top_left[0] + text_width,
            top_left[1] - text_height - 3
            if text_outside
            else top_left[1] + text_height + 3,
        )

        # Draw the text background rectangle
        cv2.rectangle(
            image,
            top_left,
            text_bg_bottom_right,
            color=color[::-1],
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

        # Draw the class name on the image
        cv2.putText(
            image,
            class_name,
            (
                top_left[0],
                top_left[1] - 5 if text_outside else top_left[1] + text_height + 2,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=line_width / 3.8,
            color=(255, 255, 255),
            thickness=font_thickness,
            lineType=cv2.LINE_AA,
        )

    # Return the modified image
    return image


# Load the model and set it to eval mode
object_detection_model = torchvision.models.detection.fcos_resnet50_fpn(
    weights="DEFAULT"
)
object_detection_model.eval()
object_detection_model.to(device)

# Set the detection threshold
detection_threshold = 0.5

# Load the first image
image_path = sys.argv[1]  # '1.jpg'
bgr_image = cv2.imread(image_path)

# Transform the image to tensor
tensor_image = transform(bgr_image)
tensor_image = tensor_image.to(device)
tensor_image = tensor_image.unsqueeze(0)

# Get the predictions on the image
with torch.no_grad():
    model_outputs = object_detection_model(tensor_image)

# Get score for all the predicted objects
prediction_scores = model_outputs[0]["scores"].detach().cpu().numpy()

# Get all the predicted bounding boxes
prediction_bounding_boxes = model_outputs[0]["boxes"].detach().cpu().numpy()

# Get boxes above the threshold score
selected_boxes = prediction_bounding_boxes[
    prediction_scores >= detection_threshold
].astype(np.int32)
selected_labels = model_outputs[0]["labels"][prediction_scores >= detection_threshold]

# Get all the predicited class names
predicted_class_names = [COCO_CLASSES[i] for i in selected_labels.cpu().numpy()]

# Draw boxes on the image and display it
rgb_image = cv2.cvtColor(np.array(bgr_image), cv2.COLOR_BGR2RGB)
annotated_image = draw_boxes(
    selected_boxes, predicted_class_names, selected_labels, rgb_image
)
plt.imshow(annotated_image)
