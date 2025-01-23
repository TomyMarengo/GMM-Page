// src/services/segmentationApi.ts
import { api } from '@/services/api';

export interface SegmentationParams {
  algorithm: 'GMM' | 'KMeans';
  n_components: number;
  resize_shape: [number, number];
  imageUrl?: string;
  imageFile?: File;
}

export interface SegmentationResponse {
  segmented_image: number[][][];
  labels: number[][];
  cluster_centers: number[][];
  algorithm: 'GMM' | 'KMeans';
}

export const segmentationApiSlice = api.injectEndpoints({
  endpoints: (builder) => ({
    fetchSegmentation: builder.mutation<
      SegmentationResponse,
      SegmentationParams
    >({
      query: ({
        algorithm,
        n_components,
        resize_shape,
        imageUrl,
        imageFile,
      }) => {
        const formData = new FormData();
        formData.append('algorithm', algorithm);
        formData.append('n_components', n_components.toString());
        formData.append('resize_shape', resize_shape.join(','));

        if (imageFile) {
          formData.append('image_file', imageFile);
        } else if (imageUrl) {
          formData.append('image_url', imageUrl);
        }

        return {
          url: 'segment',
          method: 'POST',
          body: formData,
        };
      },
    }),
  }),
});

export const { useFetchSegmentationMutation } = segmentationApiSlice;
