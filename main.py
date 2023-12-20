import os
import time
import librosa
import gradio as gr
import existing_pipeline.scripts.image_captioning as img_to_audio

import custom_pipeline

def pretr_img_audio_synth(image):
    caption = capgen.generate_captions(raw_image=image)
    audio_path = capgen.generate_audio(prompt_input=caption)
    data, sr = librosa.load(audio_path)
    return caption, (sr, data)


def cust_img_audio_synth(image):
    timestamp = time.time()
    sv_p = os.path.join("custom_pipeline", "output", "temp_image", f"{timestamp}.jpg")
    im_sv = image.save(sv_p)
    print("Generating Embeds now.")
    custom_capgen.gen_embeds(input_path=sv_p)
    print("Generating Captions now.")
    caption = custom_capgen.gen_caption()
    print("Generating Audio now.")
    audio_path = custom_capgen.gen_audio(timestamp=timestamp)
    data, sr = librosa.load(audio_path)
    return caption, (sr, data)



sample = os.path.join("existing_pipeline", "data", "inference_testing_images", "sample_concert.png")

demo = gr.Interface(pretr_img_audio_synth, gr.Image(type="pil", value=sample), ["text", "audio"],
                    title="Image to Text+Audio Synthesis.",
                    description="This version uses two separate models for generating text and audio rather than an "
                                "unified model, which is the goal. For now, its a proof-of-concept.",
                    )

demo2 = gr.Interface(cust_img_audio_synth, gr.Image(type="pil", value=sample), ["text", "audio"],
                    title="Image to Text+Audio Synthesis.",
                    description="This approach uses a single model to generate embeddings, then decodes those embeddings"
                                "to get a caption as well as to conditionally generate audio from a diffusion model.",
                     )

if __name__ == "__main__":
    capgen = img_to_audio.PretrainedCaptionAudSynth()
    custom_capgen = custom_pipeline.ImagebindDiffusionSynthesis()
    gr.TabbedInterface(
        [demo2, demo], ["Trimodal Approach", "Two Model Approach"]
    ).launch()


