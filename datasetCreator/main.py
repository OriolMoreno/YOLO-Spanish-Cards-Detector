import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import Augmentor
from matplotlib import pyplot as plt
import numpy as np
import cv2
import random
import glob
from pascal_voc_tools import XmlParser


def pasteImageBackground(card, gt, background):
    result = background
    resultGT = np.zeros((background.shape[0], background.shape[1], 3), np.uint8)

    cardHeight = card.shape[0]
    cardWidth = card.shape[1]

    centerBackground = [background.shape[1] // 2, background.shape[0] // 2]
    startY = centerBackground[0] - cardHeight // 2
    startX = centerBackground[1] - cardWidth // 2

    result[startY:startY + cardHeight, startX:startX + cardWidth] = card
    resultGT[startY:startY + cardHeight, startX:startX + cardWidth] = gt

    return result, resultGT

def downscale(image, box, box2):
    # Downscale the image to width X
    width = random.randint(125, 200)
    scale_percent = width / image.shape[1]
    width = int(image.shape[1] * scale_percent)

    box[1] = float(box[1])
    box[2] = float(box[2])
    box[3] = float(box[3])
    box[4] = float(box[4])

    box2[1] = float(box2[1])
    box2[2] = float(box2[2])
    box2[3] = float(box2[3])
    box2[4] = float(box2[4])

    height = int(image.shape[0] * scale_percent)
    dim = (width, height) # apply this to the card label too
    # resize image
    image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return image, box, box2


def drawBox(image, box, box2):
    center = (int(box[1] * image.shape[1]), int(box[2] * image.shape[0]))
    center2 = (int(box2[1] * image.shape[1]), int(box2[2] * image.shape[0]))

    rectangle = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)

    xmin = int(center[0] - box[3] * image.shape[1] / 2)
    ymin = int(center[1] - box[4] * image.shape[0] / 2)
    xmax = int(center[0] + box[3] * image.shape[1] / 2)
    ymax = int(center[1] + box[4] * image.shape[0] / 2)

    xmin2 = int(center2[0] - box2[3] * image.shape[1] / 2)
    ymin2 = int(center2[1] - box2[4] * image.shape[0] / 2)
    xmax2 = int(center2[0] + box2[3] * image.shape[1] / 2)
    ymax2 = int(center2[1] + box2[4] * image.shape[0] / 2)

    im = cv2.rectangle(image.copy(), (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    im = cv2.rectangle(im, (xmin2, ymin2), (xmax2, ymax2), (0, 255, 0), 2)

    rectangle = cv2.rectangle(rectangle, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
    rectangle = cv2.rectangle(rectangle, (xmin2, ymin2), (xmax2, ymax2), (0, 255, 0), 2)

    plt.imshow(im)
    plt.show()
    plt.imshow(rectangle)
    plt.show()

    return rectangle

def getCardClass(cardNumber):
    return int(cardNumber) // 10

"""
For each card merge with 10 different backgrounds.
"""
def firstStep():
    # YOLO images must be multiples of 32 ex: 640x640

    counter = 0


    for nData in range(0, 48):
        for i in range(0, 10):
            nImage = random.randint(0, len(path) - 1)
            nCard1 = nData

            background = cv2.imread(path[nImage])
            bg = cv2.resize(background, (640, 640))

            # plt.imshow(bg)
            # plt.show()

            # read card 1 and its bounding box
            card1 = cv2.imread('croppedCards/images/' + str(nCard1) + '.jpg')
            cardLabel1 = open('croppedCards/tagYolo/' + str(nCard1) + '.txt')
            cardLabel1 = cardLabel1.read().split()
            box1 = cardLabel1[:5]
            box2 = cardLabel1[5:]

            # pipeline = Augmentor.Pipeline("croppedCards/images")
            # pipeline.ground_truth("croppedCards/tagYolo")

            card1, box1, box2 = downscale(card1, box1, box2)
            gt = drawBox(card1, box1, box2)

            result, resultGT = pasteImageBackground(card1, gt, bg)

            """
            plt.imshow(result)
            plt.show()
            plt.imshow(resultGT)
            plt.show()
            """

            cv2.imwrite(pathCard + str(counter) + ".jpg", result)
            cv2.imwrite(pathGT + str(counter) + ".jpg", resultGT)

            counter += 1

        nData += 1



def secondStep():
    pathCard = "dataset/before"
    pathGT = "dataset/beforeGT"

    p = Augmentor.Pipeline(pathCard)
    p.ground_truth(pathGT)

    p.rotate_random_90(probability=0.75)
    p.rotate(probability=1, max_left_rotation=10, max_right_rotation=10)
    p.skew(probability=0.7)
    p.sample(10000)

path = glob.glob('backgrounds/*')
path2 = glob.glob('dataset/imageGT/*')
nData = 0

def thirdStep():
    for i in range(0, len(path2)):
        rect = cv2.imread(path2[i], 0)
        fileName = path2[i].split(".jpg")
        # print(fileName)
        cardNumber = fileName[0].split("_")[-1]
        # print(cardNumber)
        _, rect2 = cv2.threshold(rect, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # plt.imshow(rect2)
        # plt.show()

        contours, hierarchy = cv2.findContours(rect2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        c = []
        text = []
        file_name = "dataset/yoloGT2/" + str(i) + '.txt'
        f = open(file_name, 'a+')
        for i in range(len(contours)):
            if hierarchy[0][i][3] == -1:
                c.append(contours[i])
                x, y, w, h = cv2.boundingRect(contours[i])
                wNew = w / rect.shape[1]
                hNew = h / rect.shape[0]
                xNew = x / rect.shape[1] + wNew / 2
                yNew = y / rect.shape[0] + hNew / 2
                cardC = getCardClass(cardNumber)
                box = [cardC, xNew, yNew, wNew, hNew]
                f.write(str(cardC) + " " + str(xNew) + " " + str(yNew) + " " + str(wNew) + " " + str(hNew) + "\n")
                # print(box)
                #rect = cv2.rectangle(rect, (x, y), (x+w, y+h), (255,255,0), 5)
                #  now we have yolo boxes, we have to write them into a the gtYolo folder
        f.close()
        # plt.imshow(rect)
        # plt.show()


thirdStep()