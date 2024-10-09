from flask import Flask, render_template, request 
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

app = Flask(__name__)

file_path = './kaggle_pivot_min_descending.csv'
data = pd.read_csv(file_path)

# Preprocessing the data
college_pivot = data.pivot_table(index=['college_name', 'branch', 'seat_type'], 
                                 values='mean', 
                                 aggfunc=np.mean).reset_index()

college_matrix = college_pivot.pivot(index='college_name', columns=['branch', 'seat_type'], values='mean').fillna(0)

# KNN model
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(college_matrix.values)

# Function to recommend colleges based on a user's input score and branch
def recommend_colleges_based_on_score(score, score_type, branch, n_recommendations=5):
    
    filtered_data = data[(data['score_type'] == score_type) & (data['branch'] == branch)]
    filtered_data['score_diff'] = abs(filtered_data['mean'] - score)
    
    
    sorted_data = filtered_data.sort_values(by='score_diff').head(n_recommendations)
    
    
    return sorted_data[['college_name', 'branch', 'seat_type', 'mean']]

@app.route('/')
def home():
    branches = data['branch'].unique()  
    return render_template('index.html', branches=branches)

@app.route('/recommend', methods=['POST'])
def recommend():
    score = float(request.form['score'])
    score_type = request.form['score_type']
    branch = request.form['branch']
    recommendations = recommend_colleges_based_on_score(score, score_type, branch)
    
    # Rendering the recommendations in the HTML
    return render_template('recommend.html', recommendations=recommendations.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)
