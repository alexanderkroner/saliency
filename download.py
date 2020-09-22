import io
import os
import zipfile
from urllib.request import urlretrieve

import h5py
import numpy as np
import requests
from matplotlib.pyplot import imread, imsave
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter


def download_salicon(data_path):
    """Downloads the SALICON dataset. Three folders are then created that
       contain the stimuli, binary fixation maps, and blurred saliency
       distributions respectively.

    Args:
        data_path (str): Defines the path where the dataset will be
                         downloaded and extracted to.

    .. seealso:: The code for downloading files from google drive is based
                 on the solution provided at [https://bit.ly/2JSVgMQ].
    """

    print(">> Downloading SALICON dataset...", end="", flush=True)

    default_path = data_path + "salicon/"
    fixations_path = default_path + "fixations/"
    saliency_path = default_path + "saliency/"

    os.makedirs(fixations_path, exist_ok=True)
    os.makedirs(saliency_path, exist_ok=True)

    ids = ["1g8j-hTT-51IG1UFwP0xTGhLdgIUCW5e5",
           "1P-jeZXCsjoKO79OhFUgnj6FGcyvmLDPj",
           "1PnO7szbdub1559LfjYHMy65EDC4VhJC8"]

    urls = ["https://drive.google.com/uc?id=" +
            i + "&export=download" for i in ids]

    save_paths = [default_path, fixations_path, saliency_path]

    session = requests.Session()

    for count, url in enumerate(urls):
        response = session.get(url, params={"id": id}, stream=True)
        token = _get_confirm_token(response)

        if token:
            params = {"id": id, "confirm": token}
            response = session.get(url, params=params, stream=True)

        _save_response_content(response, data_path + "tmp.zip")

        with zipfile.ZipFile(data_path + "tmp.zip", "r") as zip_ref:
            for file in zip_ref.namelist():
                if "test" not in file:
                    zip_ref.extract(file, save_paths[count])

    os.rename(default_path + "images", default_path + "stimuli")

    os.remove(data_path + "tmp.zip")

    print("done!", flush=True)


def download_mit1003(data_path):
    """Downloads the MIT1003 dataset. Three folders are then created that
       contain the stimuli, binary fixation maps, and blurred saliency
       distributions respectively.

    Args:
        data_path (str): Defines the path where the dataset will be
                         downloaded and extracted to.
    """

    print(">> Downloading MIT1003 dataset...", end="", flush=True)

    default_path = data_path + "mit1003/"
    stimuli_path = default_path + "stimuli/"
    fixations_path = default_path + "fixations/"
    saliency_path = default_path + "saliency/"

    os.makedirs(stimuli_path, exist_ok=True)
    os.makedirs(fixations_path, exist_ok=True)
    os.makedirs(saliency_path, exist_ok=True)

    url = "https://people.csail.mit.edu/tjudd/WherePeopleLook/ALLSTIMULI.zip"
    urlretrieve(url, data_path + "tmp.zip")

    with zipfile.ZipFile(data_path + "tmp.zip", "r") as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith(".jpeg"):
                file_name = os.path.split(file)[1]
                file_path = stimuli_path + file_name

                with open(file_path, "wb") as stimulus:
                    stimulus.write(zip_ref.read(file))

    url = "https://people.csail.mit.edu/tjudd/WherePeopleLook/ALLFIXATIONMAPS.zip"
    urlretrieve(url, data_path + "tmp.zip")

    with zipfile.ZipFile(data_path + "tmp.zip", "r") as zip_ref:
        for file in zip_ref.namelist():
            file_name = os.path.split(file)[1]

            if file.endswith("Pts.jpg"):
                file_path = fixations_path + file_name

                # this file is mistakenly included in the dataset and can be ignored
                if file_name == "i05june05_static_street_boston_p1010764fixPts.jpg":
                    continue

                with open(file_path, "wb") as fixations:
                    fixations.write(zip_ref.read(file))

            elif file.endswith("Map.jpg"):
                file_path = saliency_path + file_name

                with open(file_path, "wb") as saliency:
                    saliency.write(zip_ref.read(file))

    os.remove(data_path + "tmp.zip")

    print("done!", flush=True)


def download_cat2000(data_path):
    """Downloads the CAT2000 dataset. Three folders are then created that
       contain the stimuli, binary fixation maps, and blurred saliency
       distributions respectively.

    Args:
        data_path (str): Defines the path where the dataset will be
                         downloaded and extracted to.
    """

    print(">> Downloading CAT2000 dataset...", end="", flush=True)

    default_path = data_path + "cat2000/"

    os.makedirs(data_path, exist_ok=True)

    url = "http://saliency.mit.edu/trainSet.zip"
    urlretrieve(url, data_path + "tmp.zip")

    with zipfile.ZipFile(data_path + "tmp.zip", "r") as zip_ref:
        for file in zip_ref.namelist():
            if not("Output" in file or "allFixData" in file):
                zip_ref.extract(file, data_path)

    os.rename(data_path + "trainSet/", default_path)

    os.rename(default_path + "Stimuli", default_path + "stimuli")
    os.rename(default_path + "FIXATIONLOCS", default_path + "fixations")
    os.rename(default_path + "FIXATIONMAPS", default_path + "saliency")

    os.remove(data_path + "tmp.zip")

    print("done!", flush=True)


def download_dutomron(data_path):
    """Downloads the DUT-OMRON dataset. Three folders are then created that
       contain the stimuli, binary fixation maps, and blurred saliency
       distributions respectively.

    Args:
        data_path (str): Defines the path where the dataset will be
                         downloaded and extracted to.
    """

    print(">> Downloading DUTOMRON dataset...", end="", flush=True)

    default_path = data_path + "dutomron/"
    stimuli_path = default_path + "stimuli/"
    fixations_path = default_path + "fixations/"
    saliency_path = default_path + "saliency/"

    os.makedirs(stimuli_path, exist_ok=True)
    os.makedirs(fixations_path, exist_ok=True)
    os.makedirs(saliency_path, exist_ok=True)

    url = "http://saliencydetection.net/dut-omron/download/DUT-OMRON-image.zip"
    urlretrieve(url, data_path + "tmp.zip")

    with zipfile.ZipFile(data_path + "tmp.zip", "r") as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith(".jpg") and not "._" in file:
                file_name = os.path.basename(file)
                file_path = stimuli_path + file_name

                with open(file_path, "wb") as stimulus:
                    stimulus.write(zip_ref.read(file))

    url = "http://saliencydetection.net/dut-omron/download/DUT-OMRON-eye-fixations.zip"
    urlretrieve(url, data_path + "tmp.zip")

    with zipfile.ZipFile(data_path + "tmp.zip", "r") as zip_ref:
        for file in zip_ref.namelist():
            if file.endswith(".mat") and not "._" in file:
                file_name = os.path.basename(file)
                file_name = os.path.splitext(file_name)[0] + ".png"

                loaded_zip = io.BytesIO(zip_ref.read(file))

                fixations = loadmat(loaded_zip)["s"]
                sorted_idx = fixations[:, 2].argsort()
                fixations = fixations[sorted_idx]

                size = fixations[0, :2]

                fixations_map = np.zeros((size[1], size[0]))

                fixations_map[fixations[1:, 1],
                              fixations[1:, 0]] = 1

                saliency_map = gaussian_filter(fixations_map, 16)

                imsave(saliency_path + file_name, saliency_map, cmap="gray")
                imsave(fixations_path + file_name, fixations_map, cmap="gray")

    os.remove(data_path + "tmp.zip")

    print("done!", flush=True) 


def download_pascals(data_path):
    """Downloads the PASCAL-S dataset. Three folders are then created that
       contain the stimuli, binary fixation maps, and blurred saliency
       distributions respectively.

    Args:
        data_path (str): Defines the path where the dataset will be
                         downloaded and extracted to.
    """

    print(">> Downloading PASCALS dataset...", end="", flush=True)

    default_path = data_path + "pascals/"
    stimuli_path = default_path + "stimuli/"
    fixations_path = default_path + "fixations/"
    saliency_path = default_path + "saliency/"

    os.makedirs(stimuli_path, exist_ok=True)
    os.makedirs(fixations_path, exist_ok=True)
    os.makedirs(saliency_path, exist_ok=True)

    url = "http://cbs.ic.gatech.edu/salobj/download/salObj.zip"
    urlretrieve(url, data_path + "tmp.zip")

    with zipfile.ZipFile(data_path + "tmp.zip", "r") as zip_ref:
        for file in zip_ref.namelist():
            file_name = os.path.basename(file)

            if file.endswith(".jpg") and "imgs/pascal" in file:
                file_path = stimuli_path + file_name

                with open(file_path, "wb") as stimulus:
                    stimulus.write(zip_ref.read(file))

            elif file.endswith(".png") and "pascal/humanFix" in file:
                file_path = saliency_path + file_name

                with open(file_path, "wb") as saliency:
                    saliency.write(zip_ref.read(file))

            elif "pascalFix.mat" in file:
                loaded_zip = io.BytesIO(zip_ref.read(file))

                with h5py.File(loaded_zip, "r") as f:
                    fixations = np.array(f.get("fixCell"))[0]

                    fixations_list = []

                    for reference in fixations:
                        obj = np.array(f[reference])
                        obj = np.stack((obj[0], obj[1]), axis=-1)
                        fixations_list.append(obj)

            elif "pascalSize.mat" in file:
                loaded_zip = io.BytesIO(zip_ref.read(file))

                with h5py.File(loaded_zip, "r") as f:
                    sizes = np.array(f.get("sizeData"))
                    sizes = np.transpose(sizes, (1, 0))

    for idx, value in enumerate(fixations_list):
        size = [int(x) for x in sizes[idx]]
        fixations_map = np.zeros(size)

        for fixation in value:
            fixations_map[int(fixation[0]) - 1,
                          int(fixation[1]) - 1] = 1

        file_name = str(idx + 1) + ".png"
        file_path = fixations_path + file_name

        imsave(file_path, fixations_map, cmap="gray")

    os.remove(data_path + "tmp.zip")

    print("done!", flush=True)    


def download_osie(data_path):
    """Downloads the OSIE dataset. Three folders are then created that
       contain the stimuli, binary fixation maps, and blurred saliency
       distributions respectively.

    Args:
        data_path (str): Defines the path where the dataset will be
                         downloaded and extracted to.
    """

    print(">> Downloading OSIE dataset...", end="", flush=True)

    default_path = data_path + "osie/"
    stimuli_path = default_path + "stimuli/"
    fixations_path = default_path + "fixations/"
    saliency_path = default_path + "saliency/"

    os.makedirs(stimuli_path, exist_ok=True)
    os.makedirs(fixations_path, exist_ok=True)
    os.makedirs(saliency_path, exist_ok=True)

    url = "https://github.com/NUS-VIP/predicting-human-gaze-beyond-pixels/archive/master.zip"
    urlretrieve(url, data_path + "tmp.zip")

    with zipfile.ZipFile(data_path + "tmp.zip", "r") as zip_ref:
        for file in zip_ref.namelist():
            file_name = os.path.basename(file)

            if file.endswith(".jpg") and "data/stimuli" in file:
                file_path = stimuli_path + file_name

                with open(file_path, "wb") as stimulus:
                    stimulus.write(zip_ref.read(file))

            elif file_name == "fixations.mat":
                loaded_zip = io.BytesIO(zip_ref.read(file))

                loaded_mat = loadmat(loaded_zip)["fixations"]

                for idx, value in enumerate(loaded_mat):
                    subjects = value[0][0][0][1]

                    fixations_map = np.zeros((600, 800))

                    for subject in subjects:
                        x_vals = subject[0][0][0][0][0]
                        y_vals = subject[0][0][0][1][0]

                        fixations = np.stack((y_vals, x_vals), axis=-1)
                        fixations = fixations.astype(int)

                        fixations_map[fixations[:, 0],
                                      fixations[:, 1]] = 1

                    file_name = str(1001 + idx) + ".png"

                    saliency_map = gaussian_filter(fixations_map, 16)
                    
                    imsave(saliency_path + file_name, saliency_map, cmap="gray")
                    imsave(fixations_path + file_name, fixations_map, cmap="gray")
    
    os.remove(data_path + "tmp.zip")

    print("done!", flush=True)


def download_fiwi(data_path):
    """Downloads the FIWI dataset. Three folders are then created that
       contain the stimuli, binary fixation maps, and blurred saliency
       distributions respectively.

    Args:
        data_path (str): Defines the path where the dataset will be
                         downloaded and extracted to.
    """

    print(">> Downloading FIWI dataset...", end="", flush=True)

    default_path = data_path + "fiwi/"
    stimuli_path = default_path + "stimuli/"
    fixations_path = default_path + "fixations/"
    saliency_path = default_path + "saliency/"

    os.makedirs(stimuli_path, exist_ok=True)
    os.makedirs(fixations_path, exist_ok=True)
    os.makedirs(saliency_path, exist_ok=True)

    url = "https://www.dropbox.com/s/30nxg2uwd1wpb80/webpage_dataset.zip?dl=1"
    urlretrieve(url, data_path + "tmp.zip")

    with zipfile.ZipFile(data_path + "tmp.zip", "r") as zip_ref:
        for file in zip_ref.namelist():
            file_name = os.path.basename(file)

            if file.endswith(".png") and "stimuli" in file:
                file_path = stimuli_path + file_name

                with open(file_path, "wb") as stimulus:
                    stimulus.write(zip_ref.read(file))

            elif file.endswith(".png") and "all5" in file:
                loaded_zip = io.BytesIO(zip_ref.read(file))

                fixations = imread(loaded_zip)
                saliency = gaussian_filter(fixations, 30)

                imsave(saliency_path + file_name, saliency, cmap="gray")
                imsave(fixations_path + file_name, fixations, cmap="gray")

    os.remove(data_path + "tmp.zip")

    print("done!", flush=True)


def download_pretrained_weights(data_path, key):
    """Downloads the pre-trained weights for the VGG16 model when
       training or the MSI-Net when testing on new data instances.

    Args:
        data_path (str): Defines the path where the weights will be
                         downloaded and extracted to.
        key (str): Describes the type of model for which the weights will
                   be downloaded. This contains the device and dataset.

    .. seealso:: The code for downloading files from google drive is based
                 on the solution provided at [https://bit.ly/2JSVgMQ].
    """

    print(">> Downloading pre-trained weights...", end="", flush=True)

    os.makedirs(data_path, exist_ok=True)

    ids = {
        "vgg16_hybrid": "1ff0va472Xs1bvidCwRlW3Ctf7Hbyyn7p",
        "model_salicon_cpu": "1Xy9C72pcA8DO4CY0rc6B7wsuE9L9DDZY",
        "model_salicon_gpu": "1Th7fqVYx25ePMZz4LYsjNQWgAu8tJqwL",
        "model_mit1003_cpu": "1jsESjYtsTvkMqKftA4rdstfB7mSYw5Ec",
        "model_mit1003_gpu": "1P_tWxBl3igZlzcHGp5H3T3kzsOskWeG6",
        "model_cat2000_cpu": "1XxaEx7xxD6rHasQTa-VY7T7eVpGhMxuV",
        "model_cat2000_gpu": "1T6ChEGB6Mf02gKXrENjdeD6XXJkE_jHh",
        "model_dutomron_cpu": "14tuRZpKi8LMDKRHNVUylu6RuAaXLjHTa",
        "model_dutomron_gpu": "15LG_M45fpYC1pTwnwmArNTZw_Z3BOIA-",
        "model_pascals_cpu": "1af9IvBqFamKWx64Ror6ALivuKNioOVIf",
        "model_pascals_gpu": "1C-T-RQzX2SaiY9Nw1HmaSx6syyCt01Z0",
        "model_osie_cpu": "1JD1tvAqZGxj_gEGmIfoxb9dTe5HOaHj1",
        "model_osie_gpu": "1g8UPr1hGpUdOSWerRb751pZqiWBOZOCh",
        "model_fiwi_cpu": "19qj9nAjd5gVHLB71oRn_YfYDw5n4Uf2X",
        "model_fiwi_gpu": "12OpIMIi2IyDVaxkE2d37XO9uUsSYf1Ec"
    }

    url = "https://drive.google.com/uc?id=" + ids[key] + "&export=download"

    session = requests.Session()

    response = session.get(url, params={"id": id}, stream=True)
    token = _get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(url, params=params, stream=True)

    _save_response_content(response, data_path + "tmp.zip")

    with zipfile.ZipFile(data_path + "tmp.zip", "r") as zip_ref:
        for file in zip_ref.namelist():
            zip_ref.extract(file, data_path)

    os.remove(data_path + "tmp.zip")

    print("done!", flush=True)


def _get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def _save_response_content(response, file_path):
    chunk_size = 32768

    with open(file_path, "wb") as data:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                data.write(chunk)
