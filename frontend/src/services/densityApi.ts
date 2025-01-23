// src/services/densityApi.ts

import { api } from '@/services/api';
import { DensityParams, DensityResponse } from '@/types';

export const densityApi = api.injectEndpoints({
  endpoints: (builder) => ({
    fetchDensity: builder.mutation<DensityResponse, DensityParams>({
      query: (params) => ({
        url: 'density',
        method: 'POST',
        body: params,
      }),
    }),
  }),
});

export const { useFetchDensityMutation } = densityApi;
