import json
import os
import shutil

if __name__ == "__main__":
    with open("./annotations/crowdpose_val.json", 'r') as f:
        file = json.load(f)
    
    #path = "./images/"
    #moveto = "./images/val/"
    test_file_names = []
    for img in file['images']:
        test_file_names.append(img['file_name'])

    #for f in test_file_names:
    #    src = path+f
    #    dst = moveto+f
    #    shutil.move(src,dst)

    with open("./test_file_names.json", "w") as f:
        json.dump(test_file_names, f)