# Gaussian Mixture Model Page

## Introduction

This page is dedicated to the Gaussian Mixture Model (GMM) algorithm. The GMM is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. It is a generalization of the k-means algorithm and is used for clustering and density estimation.

## Algorithm

The GMM algorithm is based on the Expectation-Maximization (EM) algorithm. The EM algorithm is an iterative method that estimates the parameters of a statistical model with unobserved variables. The GMM algorithm uses the EM algorithm to estimate the parameters of the Gaussian distributions in the mixture model.

The GMM algorithm has two main steps:

1. Expectation Step: In this step, the algorithm computes the probability that each data point belongs to each of the Gaussian distributions in the mixture model. This step is also known as the E-step.

2. Maximization Step: In this step, the algorithm updates the parameters of the Gaussian distributions in the mixture model based on the data points and their probabilities of belonging to each distribution. This step is also known as the M-step.

The algorithm iterates between the E-step and the M-step until the parameters converge.

## Applications

The GMM algorithm has several applications in machine learning and data analysis, including:

1. Clustering: The GMM algorithm can be used to cluster data points into groups based on their similarity.

2. Density Estimation: The GMM algorithm can be used to estimate the probability density function of the data.

3. Anomaly Detection: The GMM algorithm can be used to detect anomalies in the data by identifying data points that have low probability under the model.

## Running the Page

### Backend

1. Navigate to the `backend` directory:

```bash
cd backend
```

2. Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

3. Run the Flask server using the following command:

```bash
python app.py
```

### Frontend

1. Navigate to the `frontend` directory:

```bash
cd frontend
```

2. Install the required packages using the following command:

```bash
npm install
```

3. Run the React app using the following command:

```bash
npm start
```

4. Open [http://localhost:3000](http://localhost:3000) to view the app in the browser.
