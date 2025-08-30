# ML-Based System for Evaluating Google Local Reviews

#### Team Members: Balakrishnan Vaisiya, Atharshlakshmi Vijayakumar

## Project Overview
Our project is an ML-based system designed to evaluate the quality and relevancy of Google Local Reviews. The specific problem we address is the challenge of automatically detecting low-quality reviews—such as advertisements, irrelevant content, or rants—while preserving valid, informative reviews. 

This directly tackles the problem of ensuring trustworthiness and reliability in location-based reviews, which are often cluttered with spam or misleading content.
The system works by:
- **Classifying reviews into four categories:** Ad, Rant, Irrelevant, and Valid.
- **Feature engineering** to capture useful signals, such as review length, all-caps ratio (to detect rants), and a heuristic relevancy score (longer reviews with higher sentiment and lower caps ratio are more trustworthy).
- **Labeling data** using a combination of Qwen with **few-shot prompting** and **manual hand-labeling** of 1000 and 200 reviews respectively to improve data quality.
- Training a **Logistic Regression** to classify reviews into categories.
- Evaluating performance with **precision, recall, and F1-score** to measure how well the system identifies different types of low-quality reviews.
- This solution helps platforms **enforce content policies and provides more reliable information for users making location-based decisions**.


## Setup Instructions

#### 1. Clone Repository
Clone the repo using:
```
git clone https://github.com/atharshlakshmi/techjam
```

#### 2. Create Virtual Environment
Create your own virtual environment in the terminal
```
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

#### 3. Install dependencies
```
pip install -r requirements.txt
```

#### 4. Install data
> The dataset is **not committed** to Git (protected by `.gitignore`). 
1. Go to UCSD Google Local Dataset (https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/).
2. Download complete 'Other' reviews and metadata.
3. Add the downloaded files to your local repo under: 
```
data/raw
```


#### 5. Setup Environment Variables
Create a .env file in the project root with:
```
HF_TOKEN=yourhuggingfacetoken
```

## To Reproduce Results
1. Download the data as mentioned above.
2. Visit pre_processing/process_data.ipynb to explore the data.
3. Label the data by running both labelling/hand_label_data.py and labelling/qwen_label_data.ipynb.
4. Do feature engineering by running feature_engineering/extract_features.ipynb.
5. Train the model and get your results by running models/logistic_regression.ipynb. 
6. To classify a new set of data, clean and extract its features before inputting it into the model.

## Future Improvements
1. Fine-tune larger pre-trained models on more labeled reviews.
2. Add metadata features (e.g., GPS proximity, user history).
3. Deploy as a real-time API with explainability (XAI).