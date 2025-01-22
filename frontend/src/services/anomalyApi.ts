// src/services/anomalyApi.ts
import { api } from '@/services/api';
import { AnomalyParams, AnomalyResponse } from '@/types';

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
