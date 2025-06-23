import os
import zipfile
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import librosa

# === Constants ===
image_size = 224
audio_sample_length = 16000  # 1 second of audio at 16kHz

def unzip_if_needed(zip_path, extract_to):
    if not os.path.exists(extract_to):
        print(f"Extracting {zip_path} ...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Extracted to {extract_to}")
    else:
        print(f"{extract_to} already exists, skipping extraction.")

class BioIDDataset(Dataset):
    def __init__(self, face_dir, eye_dir, audio_dir=None, transform=None):
        self.face_dir = face_dir
        self.eye_dir = eye_dir
        self.audio_dir = audio_dir
        self.transform = transform

        self.samples = []
        for file in os.listdir(face_dir):
            if file.endswith('.pgm'):
                base_name = os.path.splitext(file)[0]
                face_path = os.path.join(face_dir, file)
                eye_path = os.path.join(eye_dir, f"{base_name}.eye")
                audio_path = os.path.join(audio_dir, f"{base_name}.wav") if audio_dir else None

                if os.path.exists(eye_path):
                    self.samples.append((face_path, eye_path, audio_path))

    def __len__(self):
        return len(self.samples)

    def parse_eye_file(self, path):
        coords = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip().startswith("#"):
                    continue  # skip comments
                parts = line.strip().split()
                for p in parts:
                    try:
                        coords.append(int(float(p)))
                    except:
                        continue
        if len(coords) >= 4:
            # First two = left eye coords, next two = right eye coords
            left_eye = torch.tensor(coords[:2], dtype=torch.float32)
            right_eye = torch.tensor(coords[2:4], dtype=torch.float32)
            return left_eye, right_eye
        else:
            # fallback
            return torch.zeros(2), torch.zeros(2)

    def __getitem__(self, idx):
        face_path, eye_path, audio_path = self.samples[idx]

        image = Image.open(face_path).convert('L')
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        left_eye, right_eye = self.parse_eye_file(eye_path)

        if audio_path and os.path.exists(audio_path):
            try:
                y, sr = librosa.load(audio_path, sr=16000)
                if len(y) > audio_sample_length:
                    y = y[:audio_sample_length]
                else:
                    y = np.pad(y, (0, audio_sample_length - len(y)))
            except Exception:
                y = np.zeros(audio_sample_length)
        else:
            y = np.zeros(audio_sample_length)

        audio = torch.tensor(y, dtype=torch.float32)
        label = torch.tensor(idx % 2, dtype=torch.long)  # Dummy label for now

        return image, left_eye, right_eye, audio, label


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.image_branch = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.eyes_fc = nn.Sequential(
            nn.Linear(4, 64),  # 4 because left_eye(2) + right_eye(2)
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.audio_fc = nn.Sequential(
            nn.Linear(audio_sample_length, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Calculate flattened image feature size:
        conv_out_size = 32 * (image_size // 4) * (image_size // 4)  # two MaxPool2d layers each with stride 2

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size + 32 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # binary classification
        )

    def forward(self, image, left_eye, right_eye, audio):
        img_feat = self.image_branch(image)

        # Concatenate eyes along dim=1, so both must be [batch_size, 2]
        eyes = torch.cat((left_eye, right_eye), dim=1)
        eye_feat = self.eyes_fc(eyes)

        audio_feat = self.audio_fc(audio)  # audio is already [batch, length]

        combined = torch.cat((img_feat, eye_feat, audio_feat), dim=1)
        output = self.fc(combined)
        return output


def main():
    # Paths to your zipped datasets
    zip_face_dir = r"C:\Users\akalo\OneDrive\Desktop\Dhyaan\saved_models\face_voice\BioID-FaceDatabase-V1.2.zip"
    zip_eye_dir = r"C:\Users\akalo\OneDrive\Desktop\Dhyaan\saved_models\face_voice\BioID-FD-Eyepos-V1.2.zip"
    zip_audio_dir = r"C:\Users\akalo\OneDrive\Desktop\Dhyaan\saved_models\face_voice\archive (3).zip"

    # Extracted folders (where data will be after extraction)
    face_dir = zip_face_dir[:-4]
    eye_dir = zip_eye_dir[:-4]
    audio_dir = zip_audio_dir[:-4]

    # Unzip datasets if not already extracted
    unzip_if_needed(zip_face_dir, face_dir)
    unzip_if_needed(zip_eye_dir, eye_dir)
    unzip_if_needed(zip_audio_dir, audio_dir)

    # Transformations for images
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = BioIDDataset(face_dir, eye_dir, audio_dir=audio_dir, transform=transform)

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check your face_dir, eye_dir, or audio_dir.")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleModel().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    num_epochs = 30
    prev_lr = optimizer.param_groups[0]['lr']

    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0

        for images, left_eye, right_eye, audio, labels in train_loader:
            images = images.to(device)
            left_eye = left_eye.to(device)
            right_eye = right_eye.to(device)
            audio = audio.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images, left_eye, right_eye, audio)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = 100 * correct / total

        model.eval()
        val_loss = 0.0
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, left_eye, right_eye, audio, labels in val_loader:
                images = images.to(device)
                left_eye = left_eye.to(device)
                right_eye = right_eye.to(device)
                audio = audio.to(device)
                labels = labels.to(device)

                outputs = model(images, left_eye, right_eye, audio)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = 100 * val_correct / val_total if val_total else 0

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        if current_lr != prev_lr:
            print(f"Learning rate reduced from {prev_lr:.6f} to {current_lr:.6f}")
            prev_lr = current_lr

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% "
              f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%")

    # Save model
    os.makedirs("saved_models", exist_ok=True)
    torch.save(model.state_dict(), "saved_models/face_voice_model.pth")
    print("Model saved to saved_models/face_voice_model.pth")


if __name__ == "__main__":
    main()