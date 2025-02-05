import JSZip from 'jszip';
import React, { useState } from 'react';

const SegmentationUploadPage: React.FC = () => {
  const [trainingFiles, setTrainingFiles] = useState<FileList | null>(null);
  const [predictionFile, setPredictionFile] = useState<File | null>(null);
  const [modelFile, setModelFile] = useState<File | null>(null);
  const [predictionImage, setPredictionImage] = useState<string | null>(null);
  const [message, setMessage] = useState<string>('');
  const [modelMessage, setModelMessage] = useState<string>('');
  const [nClusters, setNClusters] = useState<number>(2);
  const [isLoadingTrain, setIsLoadingTrain] = useState<boolean>(false);
  const [isLoadingPredict, setIsLoadingPredict] = useState<boolean>(false);
  const [isLoadingModelUpload, setIsLoadingModelUpload] =
    useState<boolean>(false);
  const [isLoadingModelDownload, setIsLoadingModelDownload] =
    useState<boolean>(false);
  const [isModelTrained, setIsModelTrained] = useState<boolean>(false);

  const API_URL = process.env.REACT_APP_API_URL;

  const handleTrainingFilesChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    setTrainingFiles(e.target.files);
  };

  const handlePredictionFileChange = (
    e: React.ChangeEvent<HTMLInputElement>
  ) => {
    if (e.target.files && e.target.files[0]) {
      setPredictionFile(e.target.files[0]);
    }
  };

  const handleModelFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setModelFile(e.target.files[0]);
    }
  };

  const handleClustersChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setNClusters(Number(e.target.value));
  };

  const handleTrainSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!trainingFiles) {
      alert('Please select training images');
      return;
    }
    setIsLoadingTrain(true);

    // Create a zip file of the training images using JSZip
    const zip = new JSZip();
    const folder = zip.folder('images');
    if (folder) {
      const filesArray = Array.from(trainingFiles);
      filesArray.forEach((file) => {
        folder.file(file.name, file);
      });
    }
    const zipBlob = await zip.generateAsync({ type: 'blob' });

    const formData = new FormData();
    formData.append('train_images', zipBlob, 'train_images.zip');
    formData.append('n_clusters', nClusters.toString());

    try {
      const response = await fetch(`${API_URL}/train`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setMessage(data.message);
      // Assume success if no error is returned
      setIsModelTrained(true);
    } catch (error) {
      console.error('Error during training:', error);
      setMessage('Error during training.');
      setIsModelTrained(false);
    } finally {
      setIsLoadingTrain(false);
    }
  };

  const handlePredictSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!predictionFile) {
      alert('Please select an image to predict');
      return;
    }
    setIsLoadingPredict(true);
    const formData = new FormData();
    formData.append('image', predictionFile);

    try {
      const response = await fetch(`${API_URL}/predict`, {
        method: 'POST',
        body: formData,
      });
      // The response is an image blob
      const blob = await response.blob();
      const imageUrl = URL.createObjectURL(blob);
      setPredictionImage(imageUrl);
    } catch (error) {
      console.error('Error during prediction:', error);
    } finally {
      setIsLoadingPredict(false);
    }
  };

  const handleModelUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!modelFile) {
      alert('Please select a model file to upload');
      return;
    }
    setIsLoadingModelUpload(true);
    const formData = new FormData();
    formData.append('model_file', modelFile);

    try {
      const response = await fetch(`${API_URL}/upload_model`, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      setModelMessage(data.message);
      // If model was loaded successfully, consider the model as trained
      setIsModelTrained(true);
    } catch (error) {
      console.error('Error during model upload:', error);
      setModelMessage('Error during model upload.');
    } finally {
      setIsLoadingModelUpload(false);
    }
  };

  const handleModelDownload = async () => {
    setIsLoadingModelDownload(true);
    try {
      const response = await fetch(`${API_URL}/download_model`, {
        method: 'GET',
      });
      if (!response.ok) {
        throw new Error('Failed to download model');
      }
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'model.pkl';
      document.body.appendChild(a);
      a.click();
      a.remove();
    } catch (error) {
      console.error('Error during model download:', error);
    } finally {
      setIsLoadingModelDownload(false);
    }
  };

  return (
    <div className="p-8 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-6 text-center">
        Image Segmentation Upload
      </h1>

      {/* Training Section */}
      <div className="bg-white shadow-md rounded-lg p-6 mb-8">
        <h2 className="text-2xl font-semibold mb-4">Train Model</h2>
        <form onSubmit={handleTrainSubmit} className="flex flex-col gap-4">
          <div>
            <label className="block text-gray-700 font-medium mb-2">
              Number of Clusters:
            </label>
            <input
              type="number"
              min="1"
              value={nClusters}
              onChange={handleClustersChange}
              className="border border-gray-300 p-2 rounded w-20"
            />
          </div>
          <div>
            <label className="block text-gray-700 font-medium mb-2">
              Training Images:
            </label>
            <input
              type="file"
              multiple
              accept="image/*"
              onChange={handleTrainingFilesChange}
              className="border border-gray-300 p-2 rounded"
            />
          </div>
          <button
            type="submit"
            className="bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 transition-colors"
            disabled={isLoadingTrain}
          >
            {isLoadingTrain ? (
              <div className="flex items-center justify-center">
                <div className="w-5 h-5 border-2 border-t-2 border-gray-200 rounded-full animate-spin mr-2" />
                Training...
              </div>
            ) : (
              'Train'
            )}
          </button>
        </form>
        {message && (
          <p className="mt-4 text-green-600 font-medium">{message}</p>
        )}

        {/* Model Download Section */}
        {isModelTrained && (
          <div className="bg-white shadow-md rounded-lg p-6">
            <h2 className="text-2xl font-semibold mb-4">
              Download Trained Model
            </h2>
            <button
              onClick={handleModelDownload}
              className="bg-orange-600 text-white py-2 px-4 rounded hover:bg-orange-700 transition-colors"
              disabled={isLoadingModelDownload}
            >
              {isLoadingModelDownload ? (
                <div className="flex items-center justify-center">
                  <div className="w-5 h-5 border-2 border-t-2 border-gray-200 rounded-full animate-spin mr-2" />
                  Downloading...
                </div>
              ) : (
                'Download Model'
              )}
            </button>
          </div>
        )}
      </div>

      {/* Model Upload Section */}
      <div className="bg-white shadow-md rounded-lg p-6 mb-8">
        <h2 className="text-2xl font-semibold mb-4">Upload Trained Model</h2>
        <form
          onSubmit={handleModelUploadSubmit}
          className="flex flex-col gap-4"
        >
          <div>
            <label className="block text-gray-700 font-medium mb-2">
              Select Model File (.pkl):
            </label>
            <input
              type="file"
              accept=".pkl"
              onChange={handleModelFileChange}
              className="border border-gray-300 p-2 rounded"
            />
          </div>
          <button
            type="submit"
            className="bg-purple-600 text-white py-2 px-4 rounded hover:bg-purple-700 transition-colors"
            disabled={isLoadingModelUpload}
          >
            {isLoadingModelUpload ? (
              <div className="flex items-center justify-center">
                <div className="w-5 h-5 border-2 border-t-2 border-gray-200 rounded-full animate-spin mr-2" />
                Uploading...
              </div>
            ) : (
              'Upload Model'
            )}
          </button>
        </form>
        {modelMessage && (
          <p className="mt-4 text-green-600 font-medium">{modelMessage}</p>
        )}
      </div>

      {/* Prediction Section */}
      <div className="bg-white shadow-md rounded-lg p-6 mb-8">
        <h2 className="text-2xl font-semibold mb-4">Predict Segmentation</h2>
        <form onSubmit={handlePredictSubmit} className="flex flex-col gap-4">
          <div>
            <label className="block text-gray-700 font-medium mb-2">
              Select Image:
            </label>
            <input
              type="file"
              accept="image/*"
              onChange={handlePredictionFileChange}
              className="border border-gray-300 p-2 rounded"
            />
          </div>
          <button
            type="submit"
            className="bg-green-600 text-white py-2 px-4 rounded hover:bg-green-700 transition-colors"
            disabled={isLoadingPredict}
          >
            {isLoadingPredict ? (
              <div className="flex items-center justify-center">
                <div className="w-5 h-5 border-2 border-t-2 border-gray-200 rounded-full animate-spin mr-2" />
                Predicting...
              </div>
            ) : (
              'Predict'
            )}
          </button>
        </form>
        {predictionImage && (
          <div className="mt-6">
            <h3 className="text-xl font-semibold mb-2">Segmented Image:</h3>
            <img
              src={predictionImage}
              alt="Segmented result"
              className="w-full rounded shadow-md"
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default SegmentationUploadPage;
