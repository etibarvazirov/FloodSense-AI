"""FloodSense AI — Gradio research beta with hosted model checkpoints."""
import os
from pathlib import Path
import numpy as np
import gradio as gr
import torch
from PIL import Image
from scipy.special import softmax
from torchvision import models, transforms
from huggingface_hub import hf_hub_download

BASE = Path(__file__).resolve().parent
MODEL_REPO = 'etibarvazirov/FloodSense-AI-Models'
LEVELS = [f'Level{i}' for i in range(5)]
TEMPERATURE = 0.6826  # preliminary validation-derived parameter
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_b0 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
# Matches ImageClassification(crop_size=[384], resize_size=[384])
transform_v2 = transforms.Compose([
    transforms.Resize(384, interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def load_checkpoint(path):
    if not path.is_file():
        raise FileNotFoundError(f'Missing checkpoint: {path}. See README.md')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, dict):
        raise TypeError(f'Expected state_dict in {path.name}')
    if all(k.startswith('module.') for k in state):
        state = {k.removeprefix('module.'): v for k, v in state.items()}
    return state


def build_model(architecture, checkpoint):
    state = load_checkpoint(checkpoint)
    if architecture == 'b0':
        model = models.efficientnet_b0(weights=None)
    else:
        model = models.efficientnet_v2_s(weights=None)
    # Derive head shape from the saved model instead of assuming dropout config.
    final_key = 'classifier.1.weight'
    if final_key not in state:
        raise KeyError(f'{checkpoint.name} has no {final_key}; check architecture.')
    n_classes, in_features = state[final_key].shape
    if n_classes != 5:
        raise ValueError(f'{checkpoint.name} contains {n_classes} classes, expected 5.')
    model.classifier[1] = torch.nn.Linear(in_features, n_classes)
    model.load_state_dict(state, strict=True)
    return model.to(DEVICE).eval()


# Load once at server start; training is never performed here.
model_b0 = build_model('b0', Path(hf_hub_download(repo_id=MODEL_REPO, filename='floodsense_e2_advanced.pth')))
model_v2 = build_model('v2', Path(hf_hub_download(repo_id=MODEL_REPO, filename='floodsense_v2_finetuned.pth')))


def predict(image):
    if image is None:
        return {}, 'Please upload a photograph of a flood-affected vehicle.'
    image = image.convert('RGB')
    with torch.inference_mode():
        p0 = torch.softmax(model_b0(transform_b0(image).unsqueeze(0).to(DEVICE)), dim=1)[0].cpu().numpy()
        p1 = torch.softmax(model_v2(transform_v2(image).unsqueeze(0).to(DEVICE)), dim=1)[0].cpu().numpy()
    ensemble = 0.7 * p0 + 0.3 * p1
    calibrated = softmax(np.log(np.clip(ensemble, 1e-12, 1)) / TEMPERATURE)
    label = int(np.argmax(calibrated))
    return (
        {LEVELS[i]: float(calibrated[i]) for i in range(5)},
        f'Predicted class: Level{label}\n'
        f'Calibrated model score: {calibrated[label]:.1%}\n\n'
        'This is a research prototype, not a water-depth measurement. '
        'Do not use it to decide whether a road is safe to drive through.'
    )

CSS = '''
.gradio-container {max-width: 1120px !important; margin: auto !important;}
#photo-input {max-height: 380px !important;}
#photo-input .image-container {height: 340px !important; max-height: 340px !important;}
#photo-input img {object-fit: contain !important; width: 100% !important; height: 100% !important;}
#photo-input canvas {object-fit: contain !important;}
#main-title {margin-bottom: 0 !important;}
'''

with gr.Blocks(title='FloodSense AI') as demo:
    gr.Markdown('''
# 🌊 FloodSense AI
### Vehicle Flood-Level Assessment · Research Beta

Upload a photograph of a flood-affected vehicle to estimate its **relative flood-severity class** using two image-classification networks. The app combines EfficientNet-B0 (70%) and EfficientNetV2-S (30%) and applies preliminary confidence calibration.
''', elem_id='main-title')
    with gr.Row(equal_height=False):
        with gr.Column(scale=1, min_width=320):
            photo = gr.Image(label='Vehicle photograph', type='pil', sources=['upload', 'webcam'], height=340, elem_id='photo-input')
            run = gr.Button('Analyze Flood Level', variant='primary')
            clear = gr.ClearButton(value='Clear', components=[photo])
        with gr.Column(scale=1, min_width=320):
            probabilities = gr.Label(label='Estimated Class Probabilities', num_top_classes=5)
            assessment = gr.Textbox(label='AI Assessment', lines=5, interactive=False)
    run.click(predict, inputs=[photo], outputs=[probabilities, assessment])
    clear.click(lambda: ({}, ''), outputs=[probabilities, assessment])
    gr.Markdown('''
---
### Understanding the five levels

| Level | Interpretation |
|---|---|
| **Level0** | Lowest category in the dataset's ordered label system |
| **Level1** | Relatively low category |
| **Level2** | Intermediate category |
| **Level3** | Relatively high category |
| **Level4** | Highest category |

**Important:** These are *relative dataset categories*, **not verified water-depth intervals in centimeters**. Exact physical definitions should be added only after confirming the original STURM-FloodDepth annotation guidelines.

### How it works

1. Upload a photograph containing a vehicle and floodwater.
2. Two pretrained, fine-tuned image models produce five-class probability distributions.
3. The ensemble combines them and applies temperature scaling (**T = 0.6826**).
4. Review the predicted class and all five probabilities.

### Research results and limitations

- **Ensemble validation accuracy:** 73.49% on 381 group-separated validation images (the ensemble weight was selected on this same set).
- **Validation macro-F1:** 0.7340.
- **Preliminary calibration:** on a 191-image evaluation subset, ECE decreased from 0.1142 to 0.0687. This is **not a fully independent evaluation**, since ensemble weights were previously chosen using the full validation set.
- A confident prediction can still be wrong. Results on new internet photographs are not validated ground truth.
- Scenes with multiple vehicles, occlusion, reflections, unusual camera angles, and unfamiliar conditions can cause errors.
- **Not suitable for emergency response, vehicle operation, or driving safety decisions.**

*Future work: vehicle-focused explanations, physically grounded waterline checks, and independent test-set evaluation.*
''')

if __name__ == '__main__':
    demo.launch(server_name=os.environ.get('GRADIO_SERVER_NAME', '0.0.0.0'), server_port=int(os.environ.get('PORT', '7860')), share=os.environ.get('GRADIO_SHARE', 'false').lower() == 'true', css=CSS, theme=gr.themes.Soft())
