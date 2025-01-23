// src/services/densityApi.ts

import { api } from '@/services/api';

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
