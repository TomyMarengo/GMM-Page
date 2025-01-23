// src/types.ts

export interface GmmResponse {
  clusters: number[];
  probabilities: number[][];
  means: number[][];
  covariances: number[][][];
  score: number;
  bic: number;
  aic: number;
  converged: boolean;
  n_iter: number;
  data: number[][];
  labels_true: number[];
  silhouette_score: number | null;
  calinski_harabasz_score: number | null;
  davies_bouldin_score: number | null;
  adjusted_rand_index: number | null;
}

export interface GmmParams {
  n_centers: number;
  n_components: number;
  random_state?: number;
  covariance_type?: string;
  n_samples?: number;
  n_features?: number;
}

export interface KMeansResponse {
  clusters: number[];
  centers: number[][];
  inertia: number;
  data: number[][];
  labels_true: number[];
  silhouette_score: number | null;
  calinski_harabasz_score: number | null;
  davies_bouldin_score: number | null;
  adjusted_rand_index: number | null;
}

export interface KMeansParams {
  n_centers: number;
  n_clusters: number;
  random_state?: number;
  n_samples?: number;
  n_features?: number;
}

export interface AnomalyResponse {
  algorithm: string;
  contamination: number;
  predictions: boolean[];
  data: number[][];
  metrics: {
    precision: number;
    recall: number;
    f1_score: number;
    roc_auc: number | null;
  };
  labels_true: number[];
  scores: number[];
}

export interface AnomalyParams {
  algorithm: 'GMM' | 'IsolationForest';
  contamination: number;
  n_components?: number; // Solo para GMM
  random_state?: number;
  n_samples?: number;
  n_features?: number;
  n_estimators?: number; // Solo para Isolation Forest
  max_samples?: string | number; // Solo para Isolation Forest
}

export interface DensityParams {
  n_components: number; // Número de componentes del GMM
  covariance_type: string; // Tipo de covarianza: "full", "diag", etc.
  random_state?: number; // Semilla opcional para reproducibilidad
  dataset: string; // Nombre del dataset (e.g., "iris", "housing")
}

export interface DensityResponse {
  data: number[][]; // Coordenadas de los puntos en 2D
  density: number[]; // Valores de densidad para cada punto
  probabilities: number[][]; // Probabilidades de pertenencia a cada componente
  means: number[][]; // Coordenadas de los centros de los clusters/componentes
  covariances: number[][][]; // Matrices de covarianza para cada componente
  bic: number; // Bayesian Information Criterion
  aic: number; // Akaike Information Criterion
  converged: boolean; // Si el modelo convergió o no
  feature_names: string[]; // Nombres de las características del dataset original (opcional)
}
