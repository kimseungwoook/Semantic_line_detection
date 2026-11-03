import torch
import xml.etree.ElementTree as ET
from src.dataset import EventDataset
from pathlib import Path

if __name__ == "__main__":
    root_dir = r"C:\Users\admin\Desktop\Semantic_line_detection_200"
    threshold = 3
    num = 100

    event_dataset = EventDataset(root_dir, threshold)

    print(f"event_count_map 개수 : {len(event_dataset)}개\n")

    sample = event_dataset[num]

    event_count_map = sample["count_map"]
    event_image = sample["event_image"]
    annotation = sample["annotation"]
    file_name = sample["file_name"]
    folder_name = sample["folder_name"]

    # event_count_map 검증
    print(f"Event_count_map Shape : {event_count_map.shape}")
    print(f"Event_count_map 자료형 : {type(event_count_map)}")
    print(f"Event_count_map Max : {event_count_map.max()}")

    # event_image 검증
    print(f"\nEvent_image Shape : {event_image.shape}")
    print(f"Event_image 자료형 : {type(event_image)}")
    print(f"Event_image Max : {event_image.max()}")

    # annotation
    print(f"\nevent_dataset annotation 자료형 : {type(event_dataset.annotations)}")
    print(f"Annotation 자료형 : {type(annotation)}\n")
    print(f"직선 개수 : {len(annotation)}개")

    for idx, line in enumerate(annotation):
        print(f"{idx+1}번 직선 중요도 : {line['label']}")
        print(f"{idx+1}번 직선 좌표 : {line['points']}")

    # file명과 folder명
    print("-----파일명과 폴더명 검증-----")
    original_path = event_dataset.event_paths[num]
    expected_file_name = original_path.name
    expected_folder_name = original_path.parent.parent.name

    print(f"file_name : {file_name}")
    print(f"folder_name : {folder_name}")

    assert expected_file_name == file_name
    assert expected_folder_name == folder_name
    print(f"\nfile, folder명 일치")

    # Event_count_map 4채널 합산 정확성 검증
    print("\n-----Event_count_map 4채널 합산 정확성 검증-----")
    event_path = event_dataset.event_paths[num]
    original_event_tensor = torch.load(
        event_path,
        map_location = "cpu",
        weights_only = True
    )
    expected_event_count_map = original_event_tensor.sum(dim=0, keepdim=True)
    print(f"Original Event_count_map shape :{expected_event_count_map.shape}")
    print(f"Event_dataset Event_count_map Shape : {event_count_map.shape}")

    assert torch.equal(expected_event_count_map, event_count_map)
    print(f"\nEvent_count_map 합산 정확성 검증 완료")\

    # XML Annotation 정확성 검증
    print(f"\n-----XML Annotation 정확성 검증-----")
    original_path = event_dataset.event_paths[num]
    time_stamp = file_name.replace("event_tensor_", "").replace(".pt", "")
    expected_annotation = event_dataset.annotations[folder_name][time_stamp]

    assert annotation == expected_annotation
    print(f"\nAnnotation 연결 검증 통과")

    # 원본 XML 직접 검증
    print("\n-----원본 XML Annotation 직접 검증-----")
    xml_path = event_dataset.annotation / folder_name / "annotations.xml"

    assert xml_path.is_file(), "XML 파일이 존재하지 않습니다."

    tree = ET.parse(xml_path)
    root = tree.getroot()

    target_timestamp = original_path.stem.replace("event_tensor_", "")
    raw_annotations = {}
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

            lines.append({
                "label": label,
                "points": points
            })

        raw_annotations[timestamp] = lines

    assert target_timestamp in raw_annotations, "XML에 해당 시간 정보가 없습니다."
    expected_raw_annotation = raw_annotations[target_timestamp]

    assert annotation == expected_raw_annotation, "Annotation이 원본 XML과 다릅니다."

    print("원본 XML Annotation 직접 검증 통과")


    # 전체 Dataset 예외 검사
    print("\n-----전체 Dataset 예외 검사-----")

    valid_labels = {1.0, 0.8, 0.6, 0.4, 0.2}

    total_samples = len(event_dataset)
    errors = []
    empty_annotations = []
    label_counts = {label: 0 for label in valid_labels}

    for idx in range(total_samples):
        event_path = event_dataset.event_paths[idx]

        try:
            # 1. 원본 Event Tensor 검사
            original_tensor = torch.load(
                event_path,
                map_location="cpu",
                weights_only=True
            )

            assert isinstance(original_tensor, torch.Tensor)
            assert original_tensor.shape == (4, 260, 346)
            assert torch.isfinite(original_tensor).all().item()

            # 2. Dataset 출력 검사
            current_sample = event_dataset[idx]

            current_count_map = current_sample["count_map"]
            current_event_image = current_sample["event_image"]
            current_annotation = current_sample["annotation"]

            assert current_count_map.shape == (1, 260, 346)
            assert current_event_image.shape == (1, 260, 346)

            assert torch.isfinite(current_count_map).all().item()
            assert current_event_image.dtype == torch.uint8

            # 3. Count Map 합산 검증
            expected_count_map = original_tensor.sum(
                dim=0,
                keepdim=True
            )

            assert torch.equal(current_count_map, expected_count_map)

            # 4. 파일명 및 폴더명 검증
            current_folder = event_path.parent.parent.name
            current_timestamp = event_path.stem.replace("event_tensor_", "")

            assert current_sample["file_name"] == event_path.name
            assert current_sample["folder_name"] == current_folder

            # 5. Annotation 연결 검증
            assert current_folder in event_dataset.annotations
            assert current_timestamp in event_dataset.annotations[current_folder]

            expected_annotation = event_dataset.annotations[current_folder][current_timestamp]

            assert current_annotation == expected_annotation

            # 6. Annotation 유효성 검사
            if len(current_annotation) == 0:
                empty_annotations.append(str(event_path))

            for line in current_annotation:
                label = line["label"]
                points = line["points"]

                assert label in valid_labels
                assert len(points) >= 2

                for x, y in points:
                    assert 0 <= x <= 346
                    assert 0 <= y <= 260

                # 모든 점이 같은 좌표인지 확인
                assert any(
                    point != points[0]
                    for point in points[1:]
                )

                label_counts[label] += 1

        except Exception as e:
            errors.append({
                "index": idx,
                "path": str(event_path),
                "error": str(e)
            })

        # 1,000개마다 진행 상황 출력
        if (idx + 1) % 1000 == 0 or (idx + 1) == total_samples:
            print(f"진행 상황 : {idx + 1}/{total_samples}")

    # 7. 전체 검사 결과 출력
    print("\n========== 전체 Dataset 검사 결과 ==========")
    print(f"전체 샘플 수 : {total_samples}")
    print(f"오류 발생 샘플 수 : {len(errors)}")
    print(f"Annotation 없는 샘플 수 : {len(empty_annotations)}")

    print("\n중요도별 직선 개수")
    for label in sorted(valid_labels, reverse=True):
        print(f"중요도 {label} : {label_counts[label]}개")

    if len(errors) == 0:
        print("\n전체 Dataset 예외 검사 통과")
    else:
        print("\n-----오류 발생 목록 (최대 20개)-----")
        for error in errors[:20]:
            print(f"Index : {error['index']}")
            print(f"Path : {error['path']}")
            print(f"Error : {error['error']}")
            print("-" * 40)

    if len(empty_annotations) > 0:
        print("\n-----Annotation 없는 샘플 (최대 10개)-----")
        for path in empty_annotations[:10]:
            print(path)