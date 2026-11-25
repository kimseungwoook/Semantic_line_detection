import torch
import torch.nn.functional as F
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import normalize, pil_to_tensor
from pathlib import Path

from src.hough.hough_target import generate_hough_target

class Rgb_Dataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.data_path = self.root_dir / "data"
        self.annotation_path = self.root_dir / "annotations"

        self.annotations = {}
        annotation_paths = sorted(self.annotation_path.glob("o_*/annotations.xml"))
        for xml_path in annotation_paths:
            folder_name = xml_path.parent.name
            self.annotations[folder_name] = self.load_annotation(xml_path)

        self.rgb_path = []
        all_rgb_path = sorted(self.data_path.glob("o_*/rgb/dv_frame_*.png"))
        for rgb_path in all_rgb_path:
            folder_name = rgb_path.parent.parent.name
            time_stamp = rgb_path.stem.removeprefix("dv_frame_")

            if time_stamp in self.annotations.get(folder_name, {}):
                self.rgb_path.append(rgb_path)

    @staticmethod
    def load_annotation(xml_path):
        root = ET.parse(xml_path).getroot()
        annotations = {}

        for image in root.findall("image"):
            image_name = image.get("name")
            time_stamp = Path(image_name).stem.removeprefix("vis_event_")

            lines = []
            for polyline in image.findall("polyline"):
                label_str = polyline.get("label")
                points_str = polyline.get("points")

                label = float(label_str.removeprefix("line_"))
                points = []
                for point in points_str.split(";"):
                    x, y = map(float, point.split(","))
                    points.append([x, y])

                lines.append({"label" : label, "points" : points})
            annotations[time_stamp] = lines

        return annotations

    def __len__(self):
        return len(self.rgb_path)

    def __getitem__(self, idx):
        rgb_path = self.rgb_path[idx]
        folder_name = rgb_path.parent.parent.name
        time_stamp = rgb_path.stem.removeprefix("dv_frame_")

        with Image.open(rgb_path) as image:
            original_w, original_h = image.size
            rgb = pil_to_tensor(image.convert("RGB")).float() / 255.0

        target_h, target_w = 224, 224
        rgb_input = F.interpolate(
            rgb.unsqueeze(0),
            size = (target_h, target_w),
            mode = "area"
        ).squeeze(0)

        rgb_input = normalize(
            rgb_input,
            mean = (0.485, 0.456, 0.406),
            std = (0.229, 0.224, 0.225),
        )

        annotation = self.annotations[folder_name][time_stamp]
        gt_heatmap = generate_hough_target(
            annotations=annotation,
            original_w=original_w,
            original_h=original_h,
            target_w=target_w,
            target_h=target_h,
            R=112,
            THETA=100,
            target_label=1.0,
        )

        return {
            "gt_heatmap" : gt_heatmap,
            "annotation" : annotation,
            "rgb_input" : rgb_input,
            "folder_name" : folder_name
        }

def rgb_collate_fn(batch):
    return {
        "gt_heatmap" : torch.stack(
            [sample["gt_heatmap"]for sample in batch]
        ),
        "rgb_input" : torch.stack(
            [sample["rgb_input"] for sample in batch]
        ),
        "annotation" : [sample["annotation"] for sample in batch],
        "folder_name": [sample["folder_name"] for sample in batch],
    }