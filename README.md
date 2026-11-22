# Audio-Based Kannada Handwriting Learning Assistant

A beginner-friendly Python/ML project. The learner hears a Kannada character
pronunciation, writes the character on paper, uploads a photograph, and the
system compares the ML prediction with the hidden expected answer.

## 1. Features

- Audio quiz using recorded Kannada WAV files.
- One handwritten character photo per attempt.
- OpenCV preprocessing.
- HOG feature extraction.
- SVM classification.
- Correct / incorrect / uncertain feedback.
- SQLite learning history.
- Streamlit user interface.
- Accuracy, precision, recall and F1 report during training.

## 2. Requirements

- Python 3.10 or newer (Python 3.11 recommended).
- VS Code or another Python editor.
- Internet only to install packages; the application itself runs locally.
- A collection of labelled Kannada handwritten character images.
- One WAV pronunciation recording per supported character.

## 3. Installation

Windows PowerShell:

```powershell
cd Kannada_Audio_Handwriting_Assistant
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python prepare_dataset.py
```

If PowerShell blocks environment activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

## 4. Dataset preparation

The starter project lists 10 characters in `characters.json`:
ಅ, ಆ, ಇ, ಈ, ಉ, ಕ, ಖ, ಗ, ಚ, ಜ

Image directory layout:

```text
dataset/images/
  ಅ/
    writer01_01.jpg
    writer01_02.jpg
    writer02_01.jpg
  ಆ/
    writer01_01.jpg
    writer02_01.jpg
  ...
```

Each folder name is the label. Each image must contain ONE isolated character
written in dark ink on light paper. Use JPG or PNG.

Recommended initial collection target:
- 10 characters.
- 10 or more writers.
- 10 samples per character per writer.
- 1,000 images total.

This is a collection target, not a guarantee of good model performance.
Collect more examples if results are weak. Ask each writer to write naturally.
Keep a record of writer IDs separately so you can later test on writers who
were not used for training.

### Important split guidance

The included training script uses a stratified random split for a simple
first run. This can put samples from the same writer in both train and test
sets and overestimate real-world performance. For a stronger project report,
create a writer-independent split: train on some writers and test on entirely
different writers.

## 5. Audio dataset

Record one clear pronunciation per character with a fluent Kannada speaker.
Use a quiet room and save each recording as mono WAV, preferably 16-bit PCM,
16 kHz or 22.05 kHz.

Required filenames:

```text
dataset/audio/ಅ.wav
dataset/audio/ಆ.wav
dataset/audio/ಇ.wav
dataset/audio/ಈ.wav
dataset/audio/ಉ.wav
dataset/audio/ಕ.wav
dataset/audio/ಖ.wav
dataset/audio/ಗ.wav
dataset/audio/ಚ.wav
dataset/audio/ಜ.wav
```

Record the isolated character sound, not the English spelling. Have a Kannada
speaker verify the pronunciation, especially for aspirated sounds and vowels.
The audio is the quiz prompt; the expected label is determined by the filename
and the matching entry in `characters.json`.

## 6. Train the model

After adding images to the folders:

```bash
python train_model.py
```

The script:
- loads and preprocesses images,
- extracts HOG features,
- trains an SVM classifier,
- prints accuracy, precision, recall, F1 and confusion matrix,
- saves the model in `model/kannada_hog_svm.joblib`.

Each class needs at least five usable images for the included starter split.
Use substantially more for a meaningful evaluation.

## 7. Run the application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually:
http://localhost:8501

The quiz only includes characters that have BOTH:
1. a trained image class, and
2. a matching WAV file.

## 8. How a learner uses it

1. Enter a name or nickname.
2. Play the audio.
3. Listen and write the character on paper without seeing it.
4. Photograph ONE character in good lighting.
5. Upload the image.
6. Click Recognize my handwriting.
7. See the predicted character and compare it with the hidden answer.
8. Check the progress tab.

## 9. Limitations

- This is character-level recognition, not word or sentence OCR.
- The model checks character identity, not stroke order or handwriting quality.
- HOG + SVM is a baseline and may confuse visually similar Kannada characters.
- A photo containing multiple characters is outside the initial scope.
- The confidence score is not a calibrated probability of correctness.
- Image preprocessing assumes dark ink on light paper.
- The project does not include authentic handwriting images or voice recordings;
  those must be collected and labelled by the project team.

## 10. Suggested project evaluation

Report:
- class-wise image counts,
- writer-independent test accuracy,
- precision, recall and F1,
- confusion matrix,
- examples of correct, incorrect and uncertain predictions,
- limitations for similar-looking characters,
- a small usability test with beginner Kannada learners.

Do not claim a result before running the model on a held-out test set.
