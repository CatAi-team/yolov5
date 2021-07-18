import detector

detector.run_detect(weights= "D:\YazilimProgramlama\Teknofest\yolov5\models\kitten_learned_v0.2.pt",
                    imgsz=640, conf_thres=0.40, source="D:\YazilimProgramlama\Teknofest\yolov5\\test2.mp4", save_json=True,
                    project="D:\YazilimProgramlama\Teknofest\detected")
