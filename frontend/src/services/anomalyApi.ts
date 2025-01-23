// src/services/anomalyApi.ts
import { api } from '@/services/api';

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

export const anomalyApi = api.injectEndpoints({
  endpoints: (builder) => ({
    fetchAnomalyDetection: builder.mutation<AnomalyResponse, AnomalyParams>({
      query: (params) => ({
        url: 'anomaly',
        method: 'POST',
        body: params,
      }),
    }),
  }),
});

export const { useFetchAnomalyDetectionMutation } = anomalyApi;
