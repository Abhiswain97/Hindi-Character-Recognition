import gradio as gr
import torch
import json
import src.config as CFG
from src.model import HNet


def classify(model, image, mapping):
    image = torch.from_numpy(image).float()
    image = image.permute(2, 0, 1).unsqueeze(0)

    outputs = model(image)

    _, preds = torch.max(outputs, 1)

    return f"The predicted character is: {mapping[str(preds[0].item())]}"


def upload_and_clasify(image, option):

    mapping, model = None, None

    if option == "Digit":
        if CFG.BEST_MODEL_DIGIT.exists():
            model = HNet(num_classes=10)
            model.load_state_dict(
                torch.load(CFG.BEST_MODEL_DIGIT, map_location=CFG.DEVICE)
            )
            with open(CFG.INDEX_DIGIT, "r") as f:
                mapping = json.load(f)
            return classify(model, image, mapping)
    else:
        if CFG.BEST_MODEL_VYANJAN.exists():
            model = HNet(num_classes=36)
            model.load_state_dict(
                torch.load(CFG.BEST_MODEL_VYANJAN, map_location=CFG.DEVICE)
            )
            with open(CFG.INDEX_VYNAJAN, "r") as f:
                mapping = json.load(f)
            return classify(model, image, mapping)


demo = gr.Interface(
    fn=upload_and_clasify,
    inputs=["image", gr.Dropdown(["Digit", "Vyanjan"])],
    outputs="text",
)
demo.launch(server_port=8080)
