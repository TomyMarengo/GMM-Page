// src/services/clusteringApi.ts
import { api } from '@/services/api';

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

export const clusteringApi = api.injectEndpoints({
  endpoints: (builder) => ({
    fetchGmmClustering: builder.mutation<GmmResponse, GmmParams>({
      query: (params) => ({
        url: 'cluster',
        method: 'POST',
        body: params,
      }),
    }),
    fetchKMeansClustering: builder.mutation<KMeansResponse, KMeansParams>({
      query: (params) => ({
        url: 'kmeans',
        method: 'POST',
        body: params,
      }),
    }),
  }),
});

export const {
  useFetchGmmClusteringMutation,
  useFetchKMeansClusteringMutation,
} = clusteringApi;
