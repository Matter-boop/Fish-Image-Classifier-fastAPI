from __future__ import annotations

from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms


ROOT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ROOT_DIR / "model" / "fish_resnet18.pth"
WEB_PAGE = ROOT_DIR / "static" / "index.html"
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
TOP_K = 3


def load_classifier() -> tuple[torch.nn.Module, list[str], torch.device]:
	"""Load the trained ResNet checkpoint and its class-name mapping."""
	if not MODEL_PATH.exists():
		raise RuntimeError(
			f"Model checkpoint not found at {MODEL_PATH}. Run train.py first."
		)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	checkpoint: dict[str, Any] = torch.load(
		MODEL_PATH, map_location=device, weights_only=True
	)
	class_names = checkpoint.get("class_names")
	if not isinstance(class_names, list) or not class_names:
		raise RuntimeError("Checkpoint is missing its class_names list.")

	model = models.resnet18(weights=None)
	model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
	model.load_state_dict(checkpoint["model_state_dict"])
	model.to(device)
	model.eval()
	return model, class_names, device


@asynccontextmanager
async def lifespan(app: FastAPI):
	# Keep the web page available before training; prediction returns 503 until
	# a checkpoint has been created.
	app.state.classifier = None
	app.state.class_names = None
	app.state.device = None
	app.state.preprocess = transforms.Compose(
		[
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(
				mean=[0.485, 0.456, 0.406],
				std=[0.229, 0.224, 0.225],
			),
		]
	)
	if MODEL_PATH.exists():
		model, class_names, device = load_classifier()
		app.state.classifier = model
		app.state.class_names = class_names
		app.state.device = device
	yield


app = FastAPI(title="Fish Classifier", lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def home() -> FileResponse:
	return FileResponse(WEB_PAGE)


@app.get("/health")
async def health() -> dict[str, str | bool]:
	return {
		"status": "ok",
		"model_loaded": app.state.classifier is not None,
	}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
	if app.state.classifier is None:
		raise HTTPException(
			status_code=503,
			detail="The model is not trained yet. Run train.py, then restart the server.",
		)

	contents = await file.read(MAX_UPLOAD_BYTES + 1)
	if not contents:
		raise HTTPException(status_code=400, detail="Please select an image to upload.")
	if len(contents) > MAX_UPLOAD_BYTES:
		raise HTTPException(status_code=413, detail="Image must be smaller than 10 MB.")

	try:
		with Image.open(BytesIO(contents)) as uploaded_image:
			uploaded_image.verify()
		with Image.open(BytesIO(contents)) as uploaded_image:
			image = uploaded_image.convert("RGB")
	except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
		raise HTTPException(
			status_code=400,
			detail="That file could not be read as an image. Try a JPG, PNG, or WEBP.",
		) from None

	input_tensor = app.state.preprocess(image).unsqueeze(0).to(app.state.device)
	with torch.inference_mode():
		probabilities = torch.softmax(app.state.classifier(input_tensor)[0], dim=0)
		scores, indices = torch.topk(probabilities, k=min(TOP_K, len(app.state.class_names)))

	top_predictions = [
		{
			"name": app.state.class_names[index],
			"display_name": app.state.class_names[index].replace("_", " ").title(),
			"confidence": round(float(score) * 100, 2),
		}
		for score, index in zip(scores.cpu().tolist(), indices.cpu().tolist())
	]
	return {
		**top_predictions[0],
		"top_predictions": top_predictions,
	}
