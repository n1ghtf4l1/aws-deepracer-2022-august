import argparse, time, torch
import numpy as np
import json, sys
from torch import nn, optim
from torchvision import datasets, models, transforms
from PIL import Image

def classify_image(image_path, topk=5):
    topk = int(topk)
    with torch.no_grad():
        
        im = Image.open(image_path)
        width, height = im.size
        picture_coords = [width, height]
        max_span = max(picture_coords)
        max_element = picture_coords.index(max_span)
        if (max_element == 0):
            min_element = 1
        else:
            min_element = 0
        aspect_ratio = picture_coords[max_element]/picture_coords[min_element]
        new_picture_coords = [0, 0]
        new_picture_coords[min_element] = 256
        new_picture_coords[max_element] = int(256 * aspect_ratio)
        im = im.resize(new_picture_coords)
        width, height = new_picture_coords
        left = (width - 244)/2
        top = (height - 244)/2
        right = (width + 244)/2
        bottom = (height + 244)/2
        im = im.crop((left, top, right, bottom))
        np_image = np.array(im)
        np_image = np_image.astype('float64')
        np_image = np_image / [255, 255, 255]
        np_image = (np_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        np_image = np_image.transpose((2, 0, 1))

        image = np_image
        image = torch.from_numpy(image)
        image.unsqueeze_(0)
        image = image.float()

        model_info = torch.load(args.model_checkpoint)
        model = model_info['transfer_model']
        model.classifier = model_info['classifier']
        model.load_state_dict(model_info['state_dict'])
        
        if (args.gpu):
           image = image.cuda()
           model = model.cuda()
        else:
            image = image.cpu()
            model = model.cpu()
        outputs = model(image)
        probs, classes = torch.exp(outputs).topk(topk)
        probs, classes = probs[0].tolist(), classes[0].add(1).tolist()
        results = zip(probs, classes)
        return results

def main():
    global args
    parser = argparse.ArgumentParser(description='use a neural network to classify an image!')
    parser.add_argument('image_input', help='image file to classifiy (required)')
    parser.add_argument('model_checkpoint', help='model used for classification (required)')
    parser.add_argument('--top_k', help='how many prediction categories to show [default 5].')
    parser.add_argument('--category_names', help='file for category names')
    parser.add_argument('--gpu', action='store_true', help='gpu option')
    args = parser.parse_args()
    if (args.gpu and not torch.cuda.is_available()):
        raise Exception("--gpu option enabled...but no GPU detected")
    if (args.top_k is None):
        top_k = 5
    else:
        top_k = args.top_k
    image_path = args.image_input

    prediction = classify_image(image_path, top_k)

    cat_file = args.category_names
    i = 0
    for p, c in prediction:
        i += 1
        p = str(round(p, 4) * 100.) + '%'
        if (cat_file):
            c = cat_file.get(str(c), 'None')
        else:
            c = ' class {}'.format(str(c))
        print("{}.{} ({})".format(i, c, p))
    return prediction

main()
