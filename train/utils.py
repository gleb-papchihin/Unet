from pathlib import Path
import ZipFile
import shutil
import json


def create_history():
    history = {
        "mIOU": [],
        "Loss": [],
        "PixelAccuracy": []
    }
    return history

def create_meta():
    meta = {
        "zip_slice_index": 0,
        "epoch": 0,
        "batch": 0
    }
    return meta

def save_json(path, data):
    with open(path, 'w') as file:
        json.dump(data, file)

def load_json(path):
    with open(path, 'r') as file:
        data = json.load(file)
    return data

def is_loadable(path: tp.Optional[str]) -> bool:
    if path is None:
        return False
    if Path(path).exists() is False:
        return False
    return True

def backgrounds_was_loaded(path: str):
    folder = Path(path)
    if not folder.exists():
        return False
    if len(list(folder.glob("*"))) == 0:
        return False
    return True

def create_folder(path):
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)

def delete_folder(path_to_folder: str):
    folder = Path(path_to_folder)
    if folder.exists() is True:
        shutil.rmtree(folder)

def split_on_batches(paths, batch_size):
    n_batches = len(paths) // batch_size
    batches = []
    for i in range(n_batches):
        start = i * batch_size
        stop = start + batch_size
        batch = paths[start:stop]
        batches.append(batch)
    return batches

def extract_zip(path_to_zip :str,
    slice: tp.Optional[tp.Tuple[int, int]]=None) -> None:
    with ZipFile(path_to_zip, 'r') as zip:
        namelist = zip.namelist()
        if slice is None:
            start, stop = (0, len(namelist))
        else:
            start, stop = slice
        zip.extractall(members=namelist[start: stop])

def create_heatmap(image, mask, alpha: float=0.4):
    convert_to_pil = transforms.ToPILImage()
    image = convert_to_pil(image_tensor).convert("RGB")
    mask = convert_to_pil(mask_tensor).convert("RGB")
    heatmap = Image.blend(image, mask, alpha)
    return heatmap

def save_heatmap(path, heatmap):
    heatmap.save(path)

def create_heatmap_from_folder(path_to_input, path_to_output, 
    model, preprocessor, device, alpha: float=0.2):

    folder = Path(path_to_input)

    if not folder.exists():
        raise Exception("Folder does not exists")

    for i, image_path in enumerate(folder.glob("*")):
        with torch.no_grad():
            image = Image.open(image_path).convert('RGB')
            x = preprocessor.x_preprocessor(image)
            x = x.unsqueeze(0)
            x = x.to(device).float()
            pred = model(x)
            mask = convert_prediction_to_mask(pred[0]).float()
            heatmap = create_heatmap(x[0], mask, alpha)
            path = path_to_output + "/" + str(i) + ".jpg"
            save_heatmap(path, heatmap)
