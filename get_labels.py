import json
import os

from PIL import Image
from torch.utils.data import Dataset

from vars import captions_file, img_folder, labels_file


class CocoCaptionsDataset(Dataset):
    def __init__(self, img_folder, captions_file, transform=None):
        self.img_folder = img_folder
        self.transform = transform

        with open(captions_file, 'r') as f:
            self.captions = json.load(f)

        # COCO stores images info in 'images' and annotations in 'annotations'
        self.images = {img['id']: img for img in self.captions['images']}
        self.annotations = {}
        for ann in self.captions['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann['caption'])

        self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_info = self.images[img_id]
        img_path = os.path.join(self.img_folder, img_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        captions = self.annotations[img_id]  # list of captions
        return image, captions


# Example usage

dataset = CocoCaptionsDataset(img_folder, captions_file)

labels = [captions[0].replace("\n", "") for image, captions in dataset]

with open(labels_file, "w") as file:
    file.writelines(s + '\n' for s in labels)
