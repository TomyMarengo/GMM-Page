// src/services/segmentationApi.ts
import { api } from '@/services/api';

export interface SegmentationParams {
  algorithm: 'GMM' | 'KMeans';
  n_components: number;
  resize_shape: [number, number];
  image_url: string;
}

export interface SegmentationResponse {
  segmented_image: number[][][];
  labels: number[][];
  cluster_centers: number[][];
  algorithm: 'GMM' | 'KMeans';
}

export const segmentationApi = api.injectEndpoints({
  endpoints: (builder) => ({
    fetchSegmentation: builder.mutation<
      SegmentationResponse,
      SegmentationParams
    >({
      query: (params) => ({
        url: '/segment',
        method: 'POST',
        body: params,
      }),
    }),
  }),
});

export const { useFetchSegmentationMutation } = segmentationApi;
