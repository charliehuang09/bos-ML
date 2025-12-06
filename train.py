from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
import torch
from autodistill_yolov8 import YOLOv8
from autodistill.utils import plot
import cv2
from ultralytics.nn.tasks import DetectionModel
from torch.nn.modules.container import Sequential
import ultralytics

def main():
    model = YOLOv8("yolov8n.pt")
    model .train("./labeled-images/data.yaml", epochs=200, device='0')
    model.export(format="onnx")


if __name__ == "__main__":
    main()
