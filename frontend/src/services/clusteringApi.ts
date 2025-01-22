// src/services/clusteringApi.ts
import { api } from '@/services/api';
import { GmmParams, GmmResponse, KMeansParams, KMeansResponse } from '@/types';

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
