from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology
from autodistill_yolov8 import YOLOv8
from autodistill.utils import plot
import cv2

def main():
    base_model = GroundedSAM(ontology=CaptionOntology({"fork": "fork"}))
    base_model.label(
        input_folder="./images",
        extension=".png",
        output_folder="./labeled-images"
    )

if __name__ == "__main__":
    main()
