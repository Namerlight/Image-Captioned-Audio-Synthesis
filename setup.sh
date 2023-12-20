cd custom_pipeline
mkdir external_libs
cd external_libs
git clone https://github.com/facebookresearch/ImageBind
git clone https://github.com/teticio/audio-diffusion.git
mv audio-diffusion AudioDiffusion
cd ..
mkdir pretrained_weights
