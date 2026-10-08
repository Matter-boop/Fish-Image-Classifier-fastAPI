# Fish image classifier

A FastAPI app for uploading a fish image and seeing the model's best-match species. The training script uses the indexed cropped images in `Fish_Data/images/cropped` and fine-tunes an ImageNet-pretrained ResNet18.

## Setup (Windows)

1. Install the dependencies in your Python environment with `python -m pip install -r requirements.txt`. For a specific CUDA-enabled PyTorch build, choose the matching `torch` and `torchvision` install command at [pytorch.org](https://pytorch.org/get-started/locally/), then install the remaining app packages (`fastapi`, `uvicorn[standard]`, `python-multipart`, and `Pillow`).
2. From the project folder, train the model:

   ```powershell
   python train.py
   ```

   The first run downloads the ResNet18 ImageNet weights and may take a while. The best checkpoint is saved as `model/fish_resnet18.pth`.

3. Start the web app:

   ```powershell
   uvicorn main:app --reload
   ```

4. Open `http://127.0.0.1:8000` in your browser. The JSON prediction endpoint is `POST /predict` with a multipart field named `file`; API docs are at `http://127.0.0.1:8000/docs`.

You can adjust training duration and batch size, for example `python train.py --epochs 20 --batch-size 16`. Train the model before starting the app; without a checkpoint, the page loads but prediction returns a setup message.

## Notes

- The classifier predicts among labels present in the supplied index; it does not identify fish outside those classes.
- Confidence is the model's softmax score, not a calibrated probability or guarantee of correctness.
- Dataset labels have a long tail, with only a few examples for many species. Treat predictions as experimental and check accuracy on images not used for training.
- Uploaded images are processed in memory and are not saved by this app.