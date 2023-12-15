import os
import librosa
import gradio as gr
import existing_pipeline.scripts.image_captioning as img_to_audio

def img_audio_synth(image):
    caption = capgen.generate_captions(raw_image=image)
    audio_path = capgen.generate_audio(prompt_input=caption)
    data, sr = librosa.load(audio_path)
    return caption, (sr, data)


sample = os.path.join("existing_pipeline", "data", "inference_testing_images", "sample_concert.png")

demo = gr.Interface(img_audio_synth, gr.Image(type="pil", value=sample), ["text", "audio"],
                    title="Image to Text+Audio Synthesis.",
                    description="This version uses two separate models for generating text and audio rather than an "
                                "unified model, which is the goal. For now, its a proof-of-concept.",
                    )

if __name__ == "__main__":
    capgen = img_to_audio.pretrained_caption_audsynth()
    demo.launch(share=True)
