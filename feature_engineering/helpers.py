import pandas as pd
import json
import ast
import re
from huggingface_hub import InferenceClient
import os
import time

client = InferenceClient(provider="nscale")
api_key=os.getenv("HF_TOKEN")

def change_type(df: pd.DataFrame, type1, type2):
    """Change the dtype of all columns of type1 to type2, safer per-column conversion"""
    cols = df.select_dtypes(include=[type1]).columns
    for col in cols:
        try:
            df[col] = df[col].astype(type2)
        except Exception as e:
            print(f"Could not convert column '{col}': {e}")
    return df

def stream_and_combine_data(reviews_path, metadata_path):
    """Stream process large files without loading everything into memory"""
    
    # Load metadata into a dictionary - only need business_name
    metadata_dict = {}
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    business = json.loads(line.strip())
                    metadata_dict[business.get('gmap_id')] = business.get('name', 'Unknown Business')
                except:
                    try:
                        business = ast.literal_eval(line.strip())
                        metadata_dict[business.get('gmap_id')] = business.get('name', 'Unknown Business')
                    except:
                        continue # Skip malformed lines
    
    # Stream process reviews
    combined_data = []
    with open(reviews_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    review = json.loads(line.strip())
                    gmap_id = review.get('gmap_id')
                    business_name = metadata_dict.get(gmap_id, 'Unknown Business')
                    
                    combined_data.append({
                        'business_name': business_name,
                        'user_name': review.get('name', 'Anonymous User'),
                        'rating': review.get('rating'),
                        'text': review.get('text', '')
                    })
                except:
                    try:
                        review = ast.literal_eval(line.strip())
                        gmap_id = review.get('gmap_id')
                        business_name = metadata_dict.get(gmap_id, 'Unknown Business')
                        
                        combined_data.append({
                            'business_name': business_name,
                            'user_name': review.get('name', 'Anonymous User'),
                            'rating': review.get('rating'),
                            'text': review.get('text', '')
                        })
                    except:
                        continue # Skip malformed lines
    
    return pd.DataFrame(combined_data)

def clean_text(text):
    """Cleans texts by removing extra whitespaces and special characters"""
    if not isinstance(text, str):
        return ""
    
    # Clean review text
    text = text.strip()
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r"[^\w\s.,!?@#$%&*()\-\']", "", text) # Remove special chars but keep basic punctuation
    
    return text

def remove_duplicate_reviews(df):
    """ """
    initial_count = len(df)
    
    # Drop exact duplicates
    df = df.drop_duplicates(subset=['user_name', 'business_name', 'text_clean'])
    print(f"Removed {initial_count - len(df)} duplicate reviews")

    return df

def clean_review_data(df):
    """ """
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values
    df_clean['text'] = df_clean['text'].fillna('').astype(str)
    df_clean['user_name'] = df_clean['user_name'].fillna('Anonymous User')
    df_clean['business_name'] = df_clean['business_name'].fillna('Unknown Business')
    
    # Remove rows with missing ratings
    initial_count = len(df_clean)
    df_clean = df_clean.dropna(subset=['rating'])
    print(f"Removed {initial_count - len(df_clean)} rows with missing ratings")
    
    # Clean text data
    df_clean['text_clean'] = df_clean['text'].apply(clean_text)
    
    # Convert rating to integer and handle outliers
    df_clean['rating'] = pd.to_numeric(df_clean['rating'], errors='coerce')

    # Ensure ratings are between 1-5
    df_clean = df_clean[(df_clean['rating'] >= 1) & (df_clean['rating'] <= 5)]
    df_clean['rating'] = df_clean['rating'].astype(int)
    
    # Clean user and business names
    df_clean['user_name'] = df_clean['user_name'].str.strip()
    df_clean['business_name'] = df_clean['business_name'].str.strip()
    
    # Remove duplicates
    df_clean = remove_duplicate_reviews(df_clean)
    
    print(f"Final clean dataset shape: {df_clean.shape}")
    return df_clean


def create_classification_prompt(reviews):
    """ """
    reviews_text = "\n".join(f"- {r}" for r in reviews)
    return f"""
    You are an expert content moderator. Classify each restaurant review into one of:

    1. Advertisement
    2. Irrelevant
    3. Rant
    4. Valid

    Reviews to classify:
    {reviews_text}

    Return ONLY valid JSON in the form:
    [
    {{"review": "text", "label": "Advertisement|Irrelevant|Rant|Valid"}}
    ]
    """

def classify_batch(reviews):
    """ """
    prompt = create_classification_prompt(reviews)
    
    try:
        completion = client.chat.completions.create(
            model="Qwen/Qwen3-4B-Instruct-2507",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.1,
        )
        response = completion.choices[0].message.content.strip()
        return json.loads(response.replace("```json", "").replace("```", "").strip())
    
    except Exception as e:
        print(f"Classification error: {e}")
        return []

def label_data(df, batch_size=5, max_reviews=None):
    """ """
    if max_reviews:
        df = df.head(max_reviews).copy()
    
    df["llm_label"] = None

    for i in range(0, len(df), batch_size):
        batch = df["text"].iloc[i:i+batch_size].tolist()
        results = classify_batch(batch)
        for r in results:
            idx = df.index[df["text"] == r["review"]]
            if not idx.empty:
                df.at[idx[0], "llm_label"] = r["label"]
        time.sleep(1)  # avoid rate limit
        
    return df