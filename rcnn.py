import wget
import os
import cv2
import torch
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
import torchvision.ops as ops

import matplotlib.pyplot as plt
from PIL import Image

# pre-processing routine for torch's pre-trained models
preprocess = T.Compose([
    T.Resize((256,256)),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

net = models.resnet18(pretrained=True)
net.eval()

dirname = os.path.abspath(os.path.dirname(__file__))
# get imagenet classes if they aren't in the current directory
if not os.path.exists(os.path.join(dirname,"imagenet_classes.txt")):
    wget.download("https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt",out=dirname)

with open(os.path.join(dirname,"imagenet_classes.txt"), "r") as f:
    categories = [s.strip() for s in f.readlines()]

# image of a koala, taken from the internet
image_path = os.path.join(dirname,"koala.jpg")
image = cv2.imread(image_path)
image = cv2.resize(image, (256,256))

(H, W) = image.shape[:2]
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(image)
ss.switchToSelectiveSearchFast()
regions = ss.process()

proposals = []
boxes = []
for (x, y, w, h) in regions:
    # filter out very small bounding boxes
    if w / float(W) < 0.1 or h / float(H) < 0.1:
        continue
    # grab the bounded image using the box's location
    roi = image[y:y + h, x:x + w]
    # convert image from BGR to RBG (opencv uses BGR)
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    # covert back to PIL Image to create a torch tensor
    roi = preprocess(Image.fromarray(roi,"RGB")).numpy()
    # append cropped RoIs and bounding box to lists
    proposals.append(roi)
    boxes.append((x, y, w, h))

# convert the cropped images back into a pytorch tensor
proposals = torch.FloatTensor(proposals)

# move the input and model to GPU
if torch.cuda.is_available():
    proposals = proposals.to('cuda')
    net.to('cuda')

# Evaluate the entire batch of proposed regions in testing mode.
# Output is an (N x 1000) matrix where N is the
# number of proposed regions and 1000 is the amount
# of labels in the imagenet model
with torch.no_grad():
    output = net(proposals)
    probabilities = F.softmax(output, dim=1)
    probabilities,indices = torch.topk(probabilities,1,dim=1)

# reshape from (N x 1) into (N)
probabilities, indices = probabilities.reshape(-1), indices.reshape(-1)

labels = {}
preds = []
# loop over each proposed region
for index,prob in enumerate(probabilities):
    # grab the ID of the label from the imagenet classes
    labelID = indices[index]
    # grab the label using its ID
    label = categories[labelID]
    # skip over non-koala labels
    if label != "koala":
        continue
    # only add a region to the labels dictionary if
    # koala probability is greater than 90%
    if prob >= 0.9:
        preds.append(prob)
        (x, y, w, h) = boxes[index]
        box = (x, y, x+w, y+h)
        # add an empty list if the label isn't in the dictionary
        # otherwise grab the existing list
        L = labels.get(label, [])
        # append the bounding box and probability
        L.append((box, prob))
        labels[label] = L


nms_img = image.copy()
no_nms = image.copy()
# loop over the labels for each approved region in the image
for label in labels.keys():
    # loop over all bounding boxes for the current label
    for (box, prob) in labels[label]:
        # draw the bounding box on the image
        (startX, startY, endX, endY) = box
        cv2.rectangle(no_nms, (startX, startY), (endX, endY),(0, 0, 255), 1)
    # show all bounding box results before applying pytorch's non-maximum suppression
    cv2.imshow("Before NMS", no_nms)
    # extract the bounding boxes and probabilities
    boxes = torch.tensor([p[0] for p in labels[label]],dtype=torch.float)
    proba = torch.tensor([p[1] for p in labels[label]],dtype=torch.float)
    # non-maximum suppression to eliminate regions based on IoU > 0.4
    indices = ops.nms(boxes, proba,0.4)
    boxes = boxes[indices]
    # loop over bounding boxes that passed NMS check
    for b in boxes:
        startX = int(b[0].item())
        startY = int(b[1].item())
        endX = int(b[2].item())
        endY = int(b[3].item())
        # draw the bounding box and label on the image
        cv2.rectangle(nms_img, (startX,startY), (endX,endY),(0, 0, 255), 1)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(nms_img, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    # show the resulting image
    cv2.imshow("After NMS", nms_img)
    cv2.waitKey(0)