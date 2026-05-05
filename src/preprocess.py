import pandas as pd
import re
import os

# ---------------------------
# Text cleaning function
# ---------------------------
def clean_text(text):
    text = str(text).lower()                     # lowercase
    text = re.sub(r"http\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-zA-Z ]", "", text)       # remove special chars
    text = re.sub(r"\s+", " ", text).strip()     # remove extra spaces
    return text

# ---------------------------
# Main preprocessing function
# ---------------------------
def preprocess():
    input_path = "data/train.csv"
    output_path = "data/processed.csv"

    print("📥 Loading dataset...")
    df = pd.read_csv(input_path)

    print("🔍 Dataset shape:", df.shape)

    # ---------------------------
    # Combine multiple toxicity labels into one
    # ---------------------------
    print("⚙️ Creating binary target...")

    df["toxic_combined"] = df[
        ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    ].max(axis=1)

    # ---------------------------
    # Keep only required columns
    # ---------------------------
    df = df[["comment_text", "toxic_combined"]]

    # ---------------------------
    # Drop missing values
    # ---------------------------
    df = df.dropna()

    # ---------------------------
    # Clean text
    # ---------------------------
    print("🧹 Cleaning text...")
    df["clean_text"] = df["comment_text"].apply(clean_text)

    # ---------------------------
    # Remove empty rows after cleaning
    # ---------------------------
    df = df[df["clean_text"].str.strip() != ""]

    print("✅ Final dataset shape:", df.shape)

    # ---------------------------
    # Create output directory if not exists
    # ---------------------------
    os.makedirs("data", exist_ok=True)

    # ---------------------------
    # Save processed dataset
    # ---------------------------
    df.to_csv(output_path, index=False)

    print(f"💾 Processed data saved to {output_path}")


# ---------------------------
# Run script
# ---------------------------
if __name__ == "__main__":
    preprocess()