import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FairFaceDataset
from model import VisionIQ

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 1e-3


def main():

    train_dataset = FairFaceDataset(
        csv_file="dataset/FairFace/train_labels.csv",
        root_dir="dataset/FairFace",
        train=True
    )

    val_dataset = FairFaceDataset(
        csv_file="dataset/FairFace/val_labels.csv",
        root_dir="dataset/FairFace",
        train=False
    )

    print(f"Training Samples   : {len(train_dataset):,}")
    print(f"Validation Samples : {len(val_dataset):,}\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = VisionIQ().to(DEVICE)

    age_loss_fn = nn.CrossEntropyLoss()
    gender_loss_fn = nn.CrossEntropyLoss()

    optimizer = Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3
    )

    best_val_age = 0.0

    for epoch in range(EPOCHS):

        # =========================
        # Train
        # =========================

        model.train()

        running_loss = 0

        age_correct = 0
        gender_correct = 0
        total = 0

        progress = tqdm(train_loader)

        for images, ages, genders in progress:

            images = images.to(DEVICE, non_blocking=True)
            ages = ages.to(DEVICE, non_blocking=True)
            genders = genders.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(images)

            age_loss = age_loss_fn(outputs["age"], ages)
            gender_loss = gender_loss_fn(outputs["gender"], genders)

            loss = age_loss + gender_loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            age_pred = outputs["age"].argmax(dim=1)
            gender_pred = outputs["gender"].argmax(dim=1)

            age_correct += (age_pred == ages).sum().item()
            gender_correct += (gender_pred == genders).sum().item()

            total += images.size(0)

            progress.set_description(f"Epoch {epoch + 1}/{EPOCHS}")
            progress.set_postfix(
                Loss=f"{loss.item():.4f}",
                Age=f"{100 * age_correct / total:.2f}%",
                Gender=f"{100 * gender_correct / total:.2f}%"
            )

        train_loss = running_loss / len(train_loader)

        train_age_acc = 100 * age_correct / total
        train_gender_acc = 100 * gender_correct / total

        # =========================
        # Validation
        # =========================

        model.eval()

        val_age_correct = 0
        val_gender_correct = 0
        val_total = 0

        with torch.no_grad():

            for images, ages, genders in val_loader:

                images = images.to(DEVICE, non_blocking=True)
                ages = ages.to(DEVICE, non_blocking=True)
                genders = genders.to(DEVICE, non_blocking=True)

                outputs = model(images)

                age_pred = outputs["age"].argmax(dim=1)
                gender_pred = outputs["gender"].argmax(dim=1)

                val_age_correct += (age_pred == ages).sum().item()
                val_gender_correct += (gender_pred == genders).sum().item()

                val_total += images.size(0)

        val_age_acc = 100 * val_age_correct / val_total
        val_gender_acc = 100 * val_gender_correct / val_total

        scheduler.step(val_age_acc)

        print("\n" + "=" * 60)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("=" * 60)

        print(f"Train Loss                : {train_loss:.4f}")
        print(f"Train Age Accuracy        : {train_age_acc:.2f}%")
        print(f"Train Gender Accuracy     : {train_gender_acc:.2f}%")

        print(f"Validation Age Accuracy   : {val_age_acc:.2f}%")
        print(f"Validation Gender Accuracy: {val_gender_acc:.2f}%")

        print(f"Learning Rate             : {optimizer.param_groups[0]['lr']:.6f}")

        torch.save(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_val_age": best_val_age,
            },
            "models/last_model.pth"
        )

        if val_age_acc > best_val_age:

            best_val_age = val_age_acc

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "best_val_age": best_val_age,
                },
                "models/best_model.pth"
            )

            print("\n✅ Best model saved!")

        print("-" * 60)


if __name__ == "__main__":
    main()