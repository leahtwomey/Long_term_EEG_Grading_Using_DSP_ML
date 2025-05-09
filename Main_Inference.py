#################################################
# Importing Libraries
#################################################   

import numpy as np
import pandas as pd
import torch
import os
import librosa
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import timm
from collections import defaultdict
from scipy import stats
import glob
import random 
from sklearn.metrics import accuracy_score
from functools import partial
import scipy as sp
import json
import matplotlib.pyplot as plt

#################################################
# Define Constants
#################################################   

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

BATCH_SIZE = 32
N_CLASSES = 1


#################################################
# Define Dataset
#################################################   

class SpectrogramDataset(Dataset):
    def __init__(self, df, audio_folder, transform=None):
        self.df = df
        self.audio_folder = audio_folder
        self.transform = transform
        
        # Group the DataFrame by 'file_ID'
        self.groups = df.groupby('file_ID')

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, idx):
        # Get the file_id for the current index
        file_id = list(self.groups.groups.keys())[idx]
        
        # Get all rows for the current file_id
        rows = self.groups.get_group(file_id)

        # Convert rows to a list of records, then shuffle the order of the channels
        rows_list = rows.to_dict('records')
        random.shuffle(rows_list)  # Shuffle the list of rows (channels)
        
        # Initialize lists to hold the spectrograms and labels
        spectrograms = []
        labels = []

        for row in rows_list:  # Iterate over the shuffled rows
            channel = row['channel']
            label = row['grade']
            y = torch.tensor(label, dtype=torch.long)
            labels.append(y)
            
            # Load audio file
            audio_path = os.path.join(self.audio_folder, f"{file_id}_channel{channel}.wav")
            audio, sr = librosa.load(audio_path, sr=None)  # Load the audio file
            audio = audio[0:72000]  # Trim or pad to fixed length if necessary
            
            # Compute the spectrogram
            stft = librosa.stft(audio, n_fft=128, hop_length=64, win_length=128, window='hann')
            spectrogram = np.abs(stft)
            S_dB = librosa.amplitude_to_db(spectrogram, ref=np.max)
            
            # Normalize the spectrogram
            S_dB = (S_dB - np.min(S_dB)) / (np.max(S_dB) - np.min(S_dB))
            
            spectrograms.append(S_dB)
        
        # Concatenate the spectrograms along the axis
        spectrogram_concat = np.concatenate(spectrograms, axis=0)
        
        # Apply transformations if any
        if self.transform:
            spectrogram_concat = np.array(spectrogram_concat, dtype=np.float32)
            augmented = self.transform(image=spectrogram_concat)
            spectrogram_concat = augmented['image']
        
        # Convert to torch tensor
        spectrogram_concat = torch.tensor(spectrogram_concat, dtype=torch.float32)
        
        # Return the concatenated spectrogram, the first label (or you can combine them), and the file_id
        return spectrogram_concat, labels[0], file_id


def plot_spectrogram(spectrogram, file_id, save_path='spectrograms'):
    """
    Plot and save a spectrogram
    
    Args:
        spectrogram: The spectrogram tensor to plot
        file_id: The ID of the file being processed
        save_path: Directory to save the plots
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Convert tensor to numpy array and reshape if needed
    if torch.is_tensor(spectrogram):
        spectrogram = spectrogram.cpu().numpy()
    
    # Squeeze out single-channel dimension for grayscale
    spectrogram = np.squeeze(spectrogram)
    
    # Create the plot
    plt.figure(figsize=(10, 2))
    plt.imshow(spectrogram, aspect='auto', origin='lower', cmap='gray')
    plt.colorbar(format='%+2.0f dB')
    plt.title(f'Spectrogram for file {file_id}')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    
    # Save the plot
    plt.savefig(os.path.join(save_path, f'spectrogram_{file_id}.png'))
    plt.close()

#################################################
# Define Transforms
#################################################   

basic_transforms = A.Compose([
    ToTensorV2(),
])

#################################################
# Define Post-Processing Optimised Rounder
#################################################   

class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds
    to maximize accuracy
    """
    def __init__(self):
        self.coef_ = 0
        self.is_trained = False

    def _accuracy_loss(self, coef, X, y):
        """
        Get loss according to
        using current coefficients
        
        :param coef: A list of coefficients that will be used for rounding
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        X_p = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [1, 2, 3, 4])

        return -accuracy_score(y, X_p)

    def fit(self, X, y):
        """
        Optimize rounding thresholds
        
        :param X: The raw predictions
        :param y: The ground truth labels
        """
        loss_partial = partial(self._accuracy_loss, X=X, y=y)
        initial_coef = [ 1.5, 2.5, 3.5]
        
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method='nelder-mead')
        self.is_trained = True

    def predict(self, X, coef):
        """
        Make predictions with specified thresholds
        
        :param X: The raw predictions
        :param coef: A list of coefficients that will be used for rounding
        """
        return pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = [1, 2, 3, 4])


    def coefficients(self):
        """
        Return the optimized coefficients
        """
        return self.coef_['x']   



#################################################
# Load Model Weights
#################################################   

def load_model_weights():
    # Path to the checkpoint directory
    checkpoint_dir = '/NestedCV_Model_CheckpointFolder'
    
    # Get all SWA model weights
    model_paths = glob.glob(os.path.join(checkpoint_dir, 'XDimConcat_outerfold*_innerFold*_checkpoint.pth'))
    return model_paths

#################################################
# Inference Function
#################################################   

def run_inference(model, dataloader, plot_spectrograms=False):
    model.eval()
    predictions = defaultdict(list)
    labels = defaultdict(list)
    
    with torch.no_grad():
        for inputs, target_labels, file_ids in dataloader:
            inputs = inputs.to(device).float()
            outputs = model(inputs)
            
            # Plot spectrograms if requested
            if plot_spectrograms:
                for idx, (input_spec, file_id) in enumerate(zip(inputs, file_ids)):
                    plot_spectrogram(input_spec, file_id)
            
            # Store predictions and labels for each file_id
            for idx, file_id in enumerate(file_ids):
                predictions[file_id].append(outputs[idx].cpu().numpy())
                labels[file_id].append(target_labels[idx].item())
    
    return predictions, labels

#################################################
# Main Inference Pipeline
#################################################   

def main():
    # Load test data
    test_df = pd.read_csv('combined_data.csv')
    
    # Initialize the dataset
    test_dataset = SpectrogramDataset(
        df=test_df,
        audio_folder='Dataset',
        transform=basic_transforms
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )
    
    # Get all model weights
    model_paths = load_model_weights()
    print(model_paths)
    # Store predictions from all models
    all_predictions = defaultdict(list)
    all_labels = defaultdict(list)
    
    # Run inference with each model
    for model_path in model_paths:
        print(f"Running inference with model: {model_path}")
        
        # Initialize model
        model = timm.create_model('convnext_nano', pretrained=True, 
                                num_classes=N_CLASSES, in_chans=1).to(device)
        model.load_state_dict(torch.load(model_path)['state_dict'])
        model.eval()
        
        # Get predictions
        predictions, labels = run_inference(model, test_loader, plot_spectrograms=True)
        print(predictions)
        # Store predictions
        for file_id in predictions:
            all_predictions[file_id].extend(predictions[file_id])
            all_labels[file_id].extend(labels[file_id])
    
    # Calculate final predictions
    final_regression_predictions = {}
    final_predictions = {}
    final_labels = {}
    
    rounder = OptimizedRounder()
    optimised_coefs = json.load(open('optimized_thresholds.json'))['thresholds']
    
    for file_id in all_predictions:
        # Average predictions for each file_id first
        mean_pred = np.mean(all_predictions[file_id])
        final_regression_predictions[file_id] = mean_pred
        
        # Use the mean prediction for rounding
        final_predictions[file_id] = rounder.predict(np.array([mean_pred]), optimised_coefs)[0]
        
        # Take mode of labels (should all be the same for a given file_id)
        final_labels[file_id] = stats.mode(all_labels[file_id])[0]
    
    # Save results
    results_df = pd.DataFrame({
        'file_id': list(final_predictions.keys()),
        'predicted_grade': list(final_predictions.values()),
        'true_grade': list(final_labels.values())
    })
    
    results_df.to_csv('inference_results.csv', index=False)
    print("Inference completed. Results saved to 'inference_results.csv'")


if __name__ == "__main__":
    main()


