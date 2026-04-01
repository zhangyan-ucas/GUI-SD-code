 # init 
anno_path = "/vlm-ssd/FoundationModel/dataset/ScreenSpot-Pro/annotations.json"
image_dir =  "/vlm-ssd/FoundationModel/dataset/ScreenSpot-Pro/images"

with open(anno_path, 'r') as f:
    total_data = json.load(f)

a = 1