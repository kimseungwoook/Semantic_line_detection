import torch
import xml.etree.ElementTree as ET

from torch.utils.data import Dataset
from pathlib import Path

class EventDataset(Dataset):
    def __init__(self, root_dir, threshold):
        self.root_dir = Path(root_dir)
        self.threshold = threshold

        self.data = self.root_dir / "data"
        self.annotation = self.root_dir / "annotations"

        self.annotations = {}
        annotation_paths = sorted(self.annotation.glob("o_*/annotations.xml"))
        for xml_path in annotation_paths:
            folder_name = xml_path.parent.name
            self.annotations[folder_name] = self.load_annotations(xml_path)

        # annotation이 있는 event_tensor만 수집
        self.event_paths = []
        all_event_paths = sorted(self.data.glob(
            "o_*/tensor_event/event_tensor_*.pt"
        ))

        for event_path in all_event_paths:
            folder_name = event_path.parent.parent.name
            timestamp = event_path.stem.replace("event_tensor_", "")

            if (folder_name in self.annotations and timestamp in self.annotations[folder_name]):
                self.event_paths.append(event_path)
        print(f"{len(self.event_paths)}개의 event_tensor가 annotation과 매칭 and load 완료")

    @staticmethod
    def load_annotations(xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()

        annotations = {}
        for image in root.findall("image"):
            image_name = image.get("name")
            timestamp = image_name.replace("vis_event_", "").replace(".png", "")

            lines = []
            for polyline in image.findall("polyline"):
                label_str = polyline.get("label")
                points_str = polyline.get("points")

                label = float(label_str.replace("line_", ""))

                points = []
                for point in points_str.split(";"):
                    x, y = map(float, point.split(","))
                    points.append([x, y])

                lines.append({"label" : label, "points" : points})

            annotations[timestamp] = lines

        return annotations

    def __len__(self):
        return len(self.event_paths)

    def __getitem__(self, idx):
        event_path = self.event_paths[idx]
        folder_name = event_path.parent.parent.name
        timestamp = event_path.stem.replace("event_tensor_", "")

        # event tensor load
        event = torch.load(
            event_path,
            map_location="cpu",
            weights_only = True
        )

        # 4채널 합산 ---> [1, 260, 346]
        count_map = event.sum(
            dim = 0,
            keepdim = True
        )

        # 시각화용 이미지
        count_map_clipped = torch.clamp(
            count_map,
            min = 0,
            max = self.threshold
        )
        event_image = (count_map_clipped / self.threshold * 255).to(torch.uint8)

        # annotation 매칭
        lines = self.annotations[folder_name][timestamp]

        return {
            "count_map" : count_map,
            "event_image" : event_image,
            "annotation" : lines,
            "file_name" : event_path.name,
            "folder_name" : folder_name
        }