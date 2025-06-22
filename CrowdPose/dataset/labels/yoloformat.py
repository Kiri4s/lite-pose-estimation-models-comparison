import json

if __name__ == "__main__":
    set = "val"
    with open("./annotations_json/crowdpose_"+set+".json", 'r') as f:
        file = json.load(f)
    
    w_and_h = {}
    for img in file['images']:
        with open(set+"/"+f"{img['id']}.txt", "w") as f:
            pass
        w_and_h[img['id']] = [img['width'], img['height']]

    for ann in file['annotations']:
        with open(set+"/"+f"{ann['image_id']}.txt", "a") as f:
            w = ann['bbox'][2]/w_and_h[ann['image_id']][0]
            h = ann['bbox'][3]/w_and_h[ann['image_id']][1]
            x_c = (ann['bbox'][0] + ann['bbox'][2] / 2) / w_and_h[ann['image_id']][0]
            y_c = (ann['bbox'][1] + ann['bbox'][3] / 2) / w_and_h[ann['image_id']][1]
            f.write(f"0 {x_c} {y_c} {w} {h} ")
            for i in range(14):
                f.write(f"{ann['keypoints'][3*i]/w_and_h[ann['image_id']][0]} {ann['keypoints'][3*i+1]/w_and_h[ann['image_id']][1]} {ann['keypoints'][3*i+2]} ")
            f.write("\n")
