import pandas as pd

data_path = '/Users/vaisiyabalakrishnan/techjam/data/clean/cleaned_combined_reviews.csv'
df = pd.read_csv(data_path)

# Skip the first 1000 reviews which are already seperately labeled by Qwen (in another csv)
df2 = df.iloc[1000:].copy()
df2["label"] = None

# Label categories
label_names = ["Ad", "Irrelevant", "Rant", "Valid"]
num_labels = len(label_names)

# Iterate through the DataFrame
for i, row in df2.iterrows():
    print(f"\nRow {i+1}/{len(df2)}: {row['review_text']}")
    for idx, label in enumerate(label_names, start=1):
        print(f"{idx}: {label}")
    
    while True:
        try:
            choice = int(input(f"Enter label number (1-{num_labels}): "))
            if 1 <= choice <= num_labels:
                df2.at[i, "label"] = label_names[choice-1]
                break
            else:
                print(f"Please enter a number between 1 and {num_labels}.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Save the labeled DataFrame to a new CSV file
data_path = "../data/label/hand_labeled_combined_reviews.csv"
df2.to_csv(data_path, index=False)

df2.head(10)