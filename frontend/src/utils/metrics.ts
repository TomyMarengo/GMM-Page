// src/utils/metrics.ts

export interface ROCPoint {
  fpr: number;
  tpr: number;
}

export interface PRPoint {
  recall: number;
  precision: number;
}

// Función para calcular la Curva ROC
export function calculateROCCurve(
  scores: number[],
  labels: boolean[]
): ROCPoint[] {
  const sortedIndices = scores
    .map((score, index) => ({ score, index }))
    .sort((a, b) => b.score - a.score);

  let tp = 0;
  let fp = 0;
  const totalPos = labels.filter((label) => label).length;
  const totalNeg = labels.length - totalPos;

  const rocPoints: ROCPoint[] = [];

  sortedIndices.forEach(({ index }) => {
    if (labels[index]) {
      tp += 1;
    } else {
      fp += 1;
    }
    rocPoints.push({
      fpr: totalNeg === 0 ? 0 : fp / totalNeg,
      tpr: totalPos === 0 ? 0 : tp / totalPos,
    });
  });

  // Añadir el punto final (1,1) si no está presente
  if (
    rocPoints.length === 0 ||
    rocPoints[rocPoints.length - 1].fpr !== 1 ||
    rocPoints[rocPoints.length - 1].tpr !== 1
  ) {
    rocPoints.push({ fpr: 1, tpr: 1 });
  }

  return rocPoints;
}

// Función para calcular la Curva PR
export function calculatePRCurve(
  scores: number[],
  labels: boolean[]
): PRPoint[] {
  const sortedIndices = scores
    .map((score, index) => ({ score, index }))
    .sort((a, b) => b.score - a.score);

  let tp = 0;
  let fp = 0;
  const totalPos = labels.filter((label) => label).length;

  const prPoints: PRPoint[] = [];

  sortedIndices.forEach(({ index }) => {
    if (labels[index]) {
      tp += 1;
    } else {
      fp += 1;
    }
    const precision = tp + fp === 0 ? 1 : tp / (tp + fp);
    const recall = totalPos === 0 ? 0 : tp / totalPos;
    prPoints.push({
      precision,
      recall,
    });
  });

  return prPoints;
}
