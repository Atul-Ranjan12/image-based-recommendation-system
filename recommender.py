import ast
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

# Load your DataFrame
df = pd.read_csv("./image_captioning_sentiment_results.csv")

# Convert objects_detected from string to actual Python objects (list of dicts)
df['objects_detected'] = df['objects_detected'].apply(ast.literal_eval)

# Split into train, test sets
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

# Handle the similarity by generated captions (on train set)
vectorizer = TfidfVectorizer()
# Vectorize the generated captions for the train set
tfidf_matrix = vectorizer.fit_transform(train_df['generated_caption'])

# Handle the similarity by object detection (on train set)
train_df['object_string'] = train_df['objects_detected'].apply(lambda objects: ' '.join([obj['class'] for obj in objects]))

object_vectorizer = CountVectorizer()
object_count_matrix = object_vectorizer.fit_transform(train_df['object_string'])

# Handle the similarity of captions and objects (on train set)
caption_similarity = cosine_similarity(tfidf_matrix)
object_similarity = cosine_similarity(object_count_matrix)

# Define weights for the final recommendation
weights = {
    'caption': 0.7,  # Higher weight for caption similarity
    'object': 0.3,
}


# Function to compute weighted similarity score (on train set)
def compute_weighted_similarity(input_index):
    caption_sim = np.array(caption_similarity[input_index])
    object_sim = np.array(object_similarity[input_index])

    # Weighted combination of the similarities
    weighted_similarity = (
        weights['caption'] * caption_sim +
        weights['object'] * object_sim
    )

    return weighted_similarity


# Function to get top N recommendations based on weighted similarity (from train set)
def recommend_weighted(input_index, N=5):
    weighted_sim = compute_weighted_similarity(input_index)
    sim_scores = list(enumerate(weighted_sim))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    return [train_df.iloc[i[0]] for i in sim_scores[1:N+1]]


# Now, randomly select an image from the test set and show 5 recommendations from the train set
random_test_index = random.randint(0, len(test_df) - 1)
selected_test_image = test_df.iloc[random_test_index]


print(f"Selected Test Image Path: {selected_test_image['image_path']}")
print(f"Generated Caption: {selected_test_image['generated_caption']}")
print(f"Objects Detected: {selected_test_image['objects_detected']}")

# Use the caption similarity from the selected test image to find the most similar image in the train set
# Step 1: Vectorize the selected test image's caption
selected_caption_tfidf = vectorizer.transform([selected_test_image['generated_caption']])

# Step 2: Compute the similarity of this test image caption with all the captions in the train set
test_caption_similarity = cosine_similarity(selected_caption_tfidf, tfidf_matrix).flatten()

# Step 3: Compute the similarity of the test image objects with all train set objects
selected_test_object_string = ' '.join([obj['class'] for obj in selected_test_image['objects_detected']])
selected_object_count_matrix = object_vectorizer.transform([selected_test_object_string])
test_object_similarity = cosine_similarity(selected_object_count_matrix, object_count_matrix).flatten()

# Step 4: Combine the similarities based on weights
test_weighted_similarity = (
    weights['caption'] * test_caption_similarity +
    weights['object'] * test_object_similarity
)

# Step 5: Get top 5 recommendations from the train set based on the weighted similarity
sim_scores = list(enumerate(test_weighted_similarity))
sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
top_recommendations = [train_df.iloc[i[0]] for i in sim_scores[:5]]

# Display the top 5 recommendations and the test image as a grid using matplotlib
images = [selected_test_image] + top_recommendations

fig, axs = plt.subplots(1, 6, figsize=(20, 5))  # Create a grid for 6 images (1 test + 5 recommendations)

# Display each image in the grid
for i, img_data in enumerate(images):
    img = mpimg.imread(img_data['image_path'])  # Read the image from file
    axs[i].imshow(img)
    if i == 0:
        axs[i].set_title(f"Selected image")
    else:
        axs[i].set_title(f"Recommendation {i}")
    axs[i].axis('off')  # Hide the axis

plt.show()
