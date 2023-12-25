# Image-to-Captioned-Audio Synthesis

#### Project for CMSC 691 - Computer Vision

## Installing it

Works on Python 3.9 with system CUDA version 12.3. Tested on RTX 4060 (8 GB VRAM) and 16 GB RAM, so those are the minimum system reqs but may work on lower.
* Run setup.sh to create folders and fetch external libraries. 
* If I made a mistake in setup sh, just run the commands manually.
* Install required packages with `pip install -r requirements.txt`
* Download DeCap weights from [DeCap_CoCo.zip](https://drive.google.com/file/d/17FSoiiUg9emL3Y2TWLG9-HKUQ_J1H9WL/view?usp=sharing).
* Unzip and place inside custom_pipeline/pretrained weights/ (see gen_caption.py lines 32 and 58 for reference)

## Running it.

Just run main.py.

The existing pipeline can run inference in 2-3 minutes. The custom pipeline may take up to 30 minutes for inference.

## Credits

#### Shadab Hafiz Choudhury