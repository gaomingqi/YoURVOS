import argparse
import os
import csv
from PIL import Image
import numpy as np


def read_ground_truth(file_path):
    ground_truth = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split(':')
            video_exp = parts[0]
            spans = [(int(x.split(',')[0]), int(x.split(',')[1])) for x in parts[1].split(';')]
            ground_truth[video_exp] = spans
    return ground_truth


def get_predicted_frames(annotations_path, video_name, exp):
    predicted_frames = []
    exp_path = os.path.join(annotations_path, video_name, exp)
    for file_name in os.listdir(exp_path):
        if file_name.endswith('.png'):
            file_path = os.path.join(exp_path, file_name)
            with Image.open(file_path) as img:
                if np.any(np.array(img)):
                    frame_number = int(file_name.split('.')[0])
                    predicted_frames.append(frame_number)
    return predicted_frames


def calculate_tiou(ground_truth, predicted_frames):
    intersection_frames = [frame for frame in predicted_frames if any(start <= frame <= end for start, end in ground_truth)]
    union_frames = list(set(predicted_frames + [frame for start, end in ground_truth for frame in range(start, end + 1, 5)]))
    
    tiou = len(intersection_frames) / len(union_frames) if union_frames else 0
    return tiou


def main(annotations_path, spans_file):
    ground_truth_data = read_ground_truth(spans_file)
    output_csv = os.path.join(os.path.dirname(annotations_path), 'tiou_results.csv')
    total_tiou = 0
    sample_count = 0
    results = []

    for video_exp in sorted(ground_truth_data):
        video_name, exp = video_exp.split(',')
        predicted_frames = get_predicted_frames(annotations_path, video_name, exp)
        tiou = calculate_tiou(ground_truth_data[video_exp], predicted_frames)
        total_tiou += tiou
        sample_count += 1
        results.append([video_name, exp, tiou])
        print(f"Video {video_name}, Exp {exp} - tIoU: {tiou}")
    
    average_tiou = total_tiou / sample_count if sample_count else 0
    results.insert(0, ["Average", "", average_tiou])

    with open(output_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Video Name', 'Exp', 'tIoU'])
        writer.writerows(results)

    print(f"Results written to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate tIoU for video annotations")
    parser.add_argument('annotations_path', type=str, help='Path to the annotations directory')
    parser.add_argument('spans_file', default='spans.txt', type=str, help='Path to the spans.txt file')
    args = parser.parse_args()

    main(args.annotations_path, args.spans_file)

