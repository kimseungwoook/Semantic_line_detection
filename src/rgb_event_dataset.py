import torch
import torch.nn.functional as F
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import normalize, pil_to_tensor
from pathlib import Path

from src.hough.hough_target import generate_hough_target

class Rgb_Event_Dataset(Dataset):
    def __init__(self, root_dir, event_clip):
        self.event_clip = event_clip
        self.root_dir = Path(root_dir)
        self.data_dir = self.root_dir / "data"
        self.annotation_dir = self.root_dir / "annotations"

        self.annotations_files = {}
        annotation_paths = sorted(self.annotation_dir.glob("o_*/annotations.xml"))
        for xml_path in annotation_paths:
            name = xml_path.parent.name
            self.annotations_files[name] = self.load_annotation(xml_path)

        self.event_paths = []
        all_event_paths = sorted(self.data_dir.glob("o_*/tensor_event/event_tensor_*.pt"))
        for event_path in all_event_paths:
            folder_name = event_path.parent.parent.name
            time_stamp = event_path.stem.removeprefix("event_tensor_")
            if time_stamp in self.annotations_files.get(folder_name, {}):
                self.event_paths.append(event_path)

        self.rgb_paths = []
        all_rgb_paths = sorted(self.data_dir.glob("o_*/rgb/dv_frame_*.png"))
        for rgb_path in all_rgb_paths:
            folder_name = rgb_path.parent.parent.name
            time_stamp = rgb_path.stem.removeprefix("dv_frame_")
            if time_stamp in self.annotations_files.get(folder_name, {}):
                self.rgb_paths.append(rgb_path)

        self.pairs = []
        for event_path in self.event_paths:
            scene_dir = event_path.parent.parent
            time_stamp = event_path.stem.removeprefix("event_tensor_")
            rgb_path = scene_dir / "rgb" / f"dv_frame_{time_stamp}.png"

            if not rgb_path.is_file():
                raise FileNotFoundError(f"짝이 되는 RGB 파일이 없습니다: {rgb_path}")

            self.pairs.append((event_path, rgb_path, scene_dir.name, time_stamp))

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
        return len(self.pairs)

    def __getitem__(self, idx):
        event_path, rgb_path, folder_name, time_stamp = self.pairs[idx]

        target_w, target_h = 224, 224
        R = 112
        THETA = 100
        target_label = 1.0

        # Event input
        event = torch.load(
            event_path,
            map_location="cpu",
            weights_only=True
        )

        count_map = event.sum(
            dim=0,
            keepdim=True
        )

        event_input = torch.clamp(
            count_map.float(),
            min=0,
            max=self.event_clip
        ) / self.event_clip

        event_input = F.interpolate(
            event_input.unsqueeze(0),
            size=(224, 224),
            mode="area"
        ).squeeze(0)

        # RGB input
        with Image.open(rgb_path) as image:
            original_w, original_h = image.size
            rgb = pil_to_tensor(image.convert("RGB")).float() / 255.0

        rgb_input = F.interpolate(
            rgb.unsqueeze(0),
            size=(target_h, target_w),
            mode="area"
        ).squeeze(0)

        rgb_input = normalize(
            rgb_input,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )

        annotation = self.annotations_files[folder_name][time_stamp]

        gt_heatmap = generate_hough_target(
            annotations=annotation,
            original_w=original_w,
            original_h=original_h,
            target_w=target_w,
            target_h=target_h,
            R=R, THETA=THETA,
            target_label=target_label
        )

        return {
            "event_input" : event_input,
            "rgb_input" : rgb_input,
            "gt_heatmap" : gt_heatmap,
            "annotation" : annotation
        }

def rgb_event_collate_fn(batch):
    return {
        "event_input" : torch.stack(
            [sample["event_input"] for sample in batch]
        ),
        "rgb_input" : torch.stack(
            [sample["rgb_input"] for sample in batch]
        ),
        "gt_heatmap" : torch.stack(
            [sample["gt_heatmap"] for sample in batch]
        ),
        "annotation": [sample["annotation"] for sample in batch]
    }