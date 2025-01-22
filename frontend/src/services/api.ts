import { createApi, fetchBaseQuery } from '@reduxjs/toolkit/query/react';

export interface GmmResponse {
  clusters: number[];
  probabilities: number[][];
  means: number[][];
  covariances: number[][][];
  score: number;
  feature_names: string[];
  data: number[][];
  bic: number;
  aic: number;
  converged: boolean;
  n_iter: number;
}

export interface GmmParams {
  n_components: number;
  random_state?: number;
  covariance_type?: string;
  n_samples?: number;
  n_features?: number;
}

export const api = createApi({
  reducerPath: 'api',
  baseQuery: fetchBaseQuery({ baseUrl: 'http://localhost:5000/' }),
  endpoints: (builder) => ({
    fetchGmmClustering: builder.mutation<GmmResponse, GmmParams>({
      query: (params) => ({
        url: 'cluster',
        method: 'POST',
        body: params,
      }),
    }),
  }),
});

export const { useFetchGmmClusteringMutation } = api;
