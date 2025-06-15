
import argparse
import pandas as pd

def merge_predictions(emotion_csv, sarcasm_csv, output_csv):
    """
    Merge emotion and sarcasm (irony) predictions for visualization.

    The emotion CSV is expected to have:
        - comment
        - comment_emotion1, comment_emotion2, comment_emotion3,
          comment_conf1, comment_conf2, comment_conf3, etc.
    
    The sarcasm CSV is expected to have:
        - Comment, Most Relevant Transcript Chunk, sarcasm_label, sarcasm_prob

    This function will:
      - Standardize the key column names.
      - Merge the two dataframes on the comment text.
      - Rename 'sarcasm_label' to 'Irony' and 'sarcasm_prob' to 'Irony Probability'.
      - Output the merged dataframe to a new CSV.
    """
    # Load the emotion predictions CSV
    emotion_df = pd.read_csv(emotion_csv)
    # Load the sarcasm (irony) predictions CSV
    sarcasm_df = pd.read_csv(sarcasm_csv)
    
    # Standardize key column names:
    # In emotion_df, the key is 'comment'
    # In sarcasm_df, the key is 'Comment'
    if "comment" not in emotion_df.columns and "Comment" in emotion_df.columns:
        emotion_df.rename(columns={"Comment": "comment"}, inplace=True)
    
    if "Comment" not in sarcasm_df.columns and "comment" in sarcasm_df.columns:
        sarcasm_df.rename(columns={"comment": "Comment"}, inplace=True)
    
    # Merge the two DataFrames on the comment text.
    merged_df = pd.merge(emotion_df, sarcasm_df, left_on="comment", right_on="Comment", how="inner")
    
    # Optionally, drop the duplicate comment column from sarcasm_df
    merged_df.drop(columns=["Comment"], inplace=True)
    
    # Rename sarcasm columns for visualization purposes
    merged_df.rename(columns={
        "sarcasm_label": "Irony",
        "sarcasm_prob": "Irony Probability"
    }, inplace=True)
    
    # Save the merged DataFrame
    merged_df.to_csv(output_csv, index=False)
    print(f"Final merged predictions saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge emotion and sarcasm (irony) predictions for visualization."
    )
    parser.add_argument("--emotion_csv", type=str, required=True,
                        help="Path to the emotion predictions CSV file")
    parser.add_argument("--sarcasm_csv", type=str, required=True,
                        help="Path to the sarcasm (irony) predictions CSV file")
    parser.add_argument("--output_csv", type=str, required=True,
                        help="Path to output the merged CSV file")
    
    args = parser.parse_args()
    merge_predictions(args.emotion_csv, args.sarcasm_csv, args.output_csv)
'''
import argparse
import math
import pandas as pd
from typing import Dict

###########################################################################
# merge.py – combine emotion + sarcasm CSVs **and** compute Tone‑Variation
###########################################################################
# ‑ Relies only on Pandas + math (no heavyweight ML libs) so it stays fast.
# ‑ Outputs two NEW columns:
#       ToneVariationScore  (float 0‑1)
#       ToneVariationLabel  (categorical)
###########################################################################

VALENCE_MAP: Dict[str, int] = {
    "Joy": 1, "Love": 1,
    "Anger": -1, "Sadness": -1, "Fear": -1,
    "Surprise": 0,  # treat as neutral
}

# ---------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------

def entropy(p1: float, p2: float, p3: float, eps: float = 1e-9) -> float:
    """Shannon entropy of three probs, normalised to [0,1]."""
    total = p1 + p2 + p3 + eps
    probs = [p1 / total, p2 / total, p3 / total]
    h = -sum(p * math.log(p + eps) for p in probs)
    return h / math.log(3)  # divide by log|alphabet|


def tone_variation_components(row: pd.Series) -> pd.Series:
    """Compute the four core signals & weighted ToneVariationScore."""
    # 1) Irony probability -------------------------------------------
    irony_s = row["Irony Probability"]  # already 0‑1

    # 2) Mixed‑emotion signal ----------------------------------------
    mixed_s = entropy(row["comment_conf1"], row["comment_conf2"], row["comment_conf3"])

    # 3) Valence clash ------------------------------------------------
    c_val = VALENCE_MAP.get(row["comment_emotion1"], 0)
    t_val = VALENCE_MAP.get(row["transcript_emotion1"], 0)
    valence_clash = abs(c_val - t_val) * (row["comment_conf1"] + row["transcript_conf1"]) / 2
    clash_s = valence_clash / 2.0  # max possible is 2 → rescale

    # 4) Intensity difference ---------------------------------------
    intensity_s = max(0.0, row["comment_conf1"] - row["transcript_conf1"])  # 0‑1

    # Weighted sum ---------------------------------------------------
    score = (
        0.5 * irony_s +
        0.2 * mixed_s +
        0.2 * clash_s +
        0.1 * intensity_s
    )

    # Bucket into labels --------------------------------------------
    if score >= 0.70:
        label = "Likely Tone‑Shift"
    elif score >= 0.40:
        label = "Possible Tone‑Shift"
    else:
        label = "Straightforward"

    return pd.Series({"ToneVariationScore": round(score, 3), "ToneVariationLabel": label})


# ---------------------------------------------------------------------
# Main merge function
# ---------------------------------------------------------------------

def merge_predictions(emotion_csv: str, sarcasm_csv: str, output_csv: str) -> None:
    """Merge emotion & sarcasm CSVs, add tone‑variation columns, save."""

    # 1. Load CSVs ----------------------------------------------------
    emotion_df = pd.read_csv(emotion_csv)
    sarcasm_df = pd.read_csv(sarcasm_csv)

    # 2. Standardise key column names --------------------------------
    if "comment" not in emotion_df.columns and "Comment" in emotion_df.columns:
        emotion_df.rename(columns={"Comment": "comment"}, inplace=True)

    if "Comment" not in sarcasm_df.columns and "comment" in sarcasm_df.columns:
        sarcasm_df.rename(columns={"comment": "Comment"}, inplace=True)

    # 3. Merge on comment text ---------------------------------------
    merged = pd.merge(
        emotion_df,
        sarcasm_df,
        left_on="comment",
        right_on="Comment",
        how="inner",
    )

    merged.drop(columns=["Comment"], inplace=True)

    # 4. Rename sarcasm cols -----------------------------------------
    merged.rename(
        columns={
            "sarcasm_label": "Irony",
            "sarcasm_prob": "Irony Probability",
        },
        inplace=True,
    )

    # 5. Compute tone‑variation --------------------------------------
    merged[["ToneVariationScore", "ToneVariationLabel"]] = merged.apply(
        tone_variation_components,
        axis=1,
    )

    # 6. Save ---------------------------------------------------------
    merged.to_csv(output_csv, index=False)
    print(f"Final merged predictions saved to {output_csv}")


# ---------------------------------------------------------------------
# CLI entry‑point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Merge emotion & sarcasm CSVs and compute tone‑variation score/label.",
    )
    parser.add_argument("--emotion_csv", type=str, required=True, help="Path to emotion predictions CSV")
    parser.add_argument("--sarcasm_csv", type=str, required=True, help="Path to sarcasm predictions CSV")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to output merged CSV")

    args = parser.parse_args()
    merge_predictions(args.emotion_csv, args.sarcasm_csv, args.output_csv)
'''

