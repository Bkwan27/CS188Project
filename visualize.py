import pandas as pd
import matplotlib.pyplot as plt
from argparse import ArgumentParser

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True, help="Path to the CSV file containing training data")
    return parser.parse_args()

def main():
    args = parse_args()
    # Load the uploaded CSV file
    df = pd.read_csv(args.csv_path)

    # Display the first few rows to understand its structure
    df.head()

    # Re-plotting the data with a clean line (no markers)
    plt.figure(figsize=(8, 5))
    plt.plot(df["Step"], df["Value"], linestyle='-', linewidth=2)  # removed marker
    plt.title("Episode Reward Mean vs. Training Steps")
    plt.xlabel("Training Steps")
    plt.ylabel("Mean Episode Reward")
    plt.grid(True)
    plt.tight_layout()

    # Save the updated plot as a PDF
    clean_output_path = "plot.pdf"
    plt.savefig(clean_output_path, format="pdf")

    clean_output_path

if __name__ == "__main__":
    main()