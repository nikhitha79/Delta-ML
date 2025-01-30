import os
import re
import numpy as np
import torch
from ase import Atoms
import pandas as pd
from schnetpack.data import ASEAtomsData
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, train_test_split
import json
import torchmetrics
import schnetpack as spk
from sklearn.metrics import r2_score
import torch.nn.functional as F
# Atomic number mapping
atomic_number_map = {
    'H': 1, 'C': 6, 'N': 7
}

# Read XYZ files
def read_xyz(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        
    num_atoms = int(lines[0])
    atoms = []
    positions = []
    
    for line in lines[2:2 + num_atoms]:
        parts = line.split()
        element = parts[0]
        x, y, z = map(float, parts[1:4])
        atoms.append(element)
        positions.append([x, y, z])
    
    return atoms, positions

# Convert symbols to atomic numbers
def symbols_to_atomic_numbers(symbols):
    return [atomic_number_map[symbol] for symbol in symbols]

# Split Roman numerals and integers from filename for Dataset-I. For Dataset-II, use only integers for matching
def split_roman_and_integer(file_name):
    match = re.match(r'([IVXLCDM]+)_([0-9]+)\.xyz', file_name)
    if match:
        roman = match.group(1)
        integer = int(match.group(2))
        return roman, integer
    return None, None

# Extract XYZ files from a directory
def get_xyz_files(directory):
    xyz_files = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.xyz'):
            roman, integer = split_roman_and_integer(file_name)
            if roman is not None and integer is not None:
                xyz_files.append((roman, integer, os.path.join(directory, file_name)))
    
    xyz_files.sort(key=lambda x: (x[0], x[1]))
    return [file[2] for file in xyz_files]

# Process XYZ files into a format compatible with SchNetPack
def process_xyz_files(file_paths):
    dataset_entries = []
    
    for file_path in file_paths:
        atoms, positions = read_xyz(file_path)
        atomic_numbers = symbols_to_atomic_numbers(atoms)
        positions = np.array(positions)
        
        entry = {
            'atomic_numbers': torch.tensor(atomic_numbers, dtype=torch.long),
            'positions': torch.tensor(positions, dtype=torch.float32),
        }
        
        dataset_entries.append(entry)
    
    return dataset_entries

# Process the folder containing XYZ files
def process_xyz_folder_to_schnet_format(directory):
    xyz_files = get_xyz_files(directory)
    dataset_entries = process_xyz_files(xyz_files)
    
    all_atomic_numbers = []
    all_positions = []
    
    for entry in dataset_entries:
        all_atomic_numbers.append(entry['atomic_numbers'].tolist())
        all_positions.append(entry['positions'].tolist())
    
    return all_atomic_numbers, all_positions

folder_path = 'Dataset_1'
schnet_compatible_data = process_xyz_folder_to_schnet_format(folder_path)


atoms_list = schnet_compatible_data[0]
positions_list = schnet_compatible_data[1]

# Load the CSV file containing properties
data = pd.read_csv('Dataset_1.csv')

# Define the correction factor according to the target property (S1, T1, singlet-triplet energy gap, f)
target_ref = data['target_ref'].tolist() 
ppp_cis = data['ppp_cis'].tolist()
data['correction_factor'] = data['target_ref'] - data['ppp_cis']
correction_factor = data['correction_factor'].tolist()
actual_correction_factors_test=[]
all_predicted_correction_factors_test=[]
all_actual_correction_factors_train=[]
all_predicted_correction_factors_train=[]
all_ref_train=[]
all_ref_test=[]
all_x_values_train = []
all_x_values_test=[]
atoms_objects = []
property_list = []


for atomic_numbers, positions, correction_factor, target_ref, ppp_cis in zip(atoms_list, positions_list, correction_factor, target_ref, ppp_cis):
    atoms = Atoms(numbers=atomic_numbers, positions=positions)
    properties = {
        'target_ref': np.array([target_ref]),
        'ppp_cis': np.array([ppp_cis]),
        'correction_factor': np.array([correction_factor]), 
    }
    atoms_objects.append(atoms)
    property_list.append(properties)

# Create a new ASE dataset
new_datatut = './new_dataset67'
if not os.path.exists(new_datatut):
    os.makedirs(new_datatut)

new_dataset = spk.data.ASEAtomsData.create(
    './new_dataset67.db',
    distance_unit='Ang',
    property_unit_dict={
        'correction_factor': 'eV',
        'target_ref':'eV',
        'ppp_cis': 'eV'
    }
)

new_dataset.add_systems(property_list, atoms_objects)

print('Number of reference calculations:', len(new_dataset))
print('Available properties:')

# Check the available properties
for p in new_dataset.available_properties:
    print('-', p)
print()    

example = new_dataset[0]
print('Properties of molecule with id 0:')

for k, v in example.items():
    print('-', k, ':', v.shape)

total_samples = len(new_dataset)

# Initialize indices for train/test split
train_val_indices, test_indices = train_test_split(np.arange(total_samples), test_size=0.2, random_state=42)

# Initialize KFold with 5 splits for the train/validation set
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Create a dictionary to store indices for each fold
folds = {}
for fold, (train_indices, val_indices) in enumerate(kf.split(train_val_indices)):
    folds[fold] = {
        'train_idx': train_val_indices[train_indices].tolist(),
        'val_idx': train_val_indices[val_indices].tolist()
    }

# Create a split file
split_file_path = 'k_fold_split_with_test.json'
with open(split_file_path, 'w') as split_file:
    json.dump(folds, split_file)

print(f"Split file path {split_file_path}")

def check_no_overlap(indices1, indices2):
    return set(indices1).isdisjoint(indices2)

# Check for overlaps between train, validation, and test sets
for fold, indices in folds.items():
    train_idx = indices['train_idx']
    val_idx = indices['val_idx']
    if not check_no_overlap(train_idx, val_idx):
        print(f"Overlap detected between train and validation indices in fold {fold}")
    if not check_no_overlap(train_idx, test_indices):
        print(f"Overlap detected between train and test indices in fold {fold}")
    if not check_no_overlap(val_idx, test_indices):
        print(f"Overlap detected between validation and test indices in fold {fold}")


# Iterate through each fold for training and validation
for fold, indices in folds.items():
    print(f"Training on fold {fold}")

    newdata = spk.data.AtomsDataModule(
        './new_dataset67.db',
        batch_size=600,
        num_train=len(indices['train_idx']),
        num_val=len(indices['val_idx']),
        num_test=len(test_indices),  
        distance_unit='Ang',
        property_units={'correction_factor': 'eV'},
        transforms=[
            spk.transform.ASENeighborList(cutoff=5.),
            spk.transform.RemoveOffsets('correction_factor', remove_mean=True, remove_atomrefs=False),
            spk.transform.CastTo32()
        ],
        num_workers=8,
        pin_memory=True,
        split_file=os.path.join(new_datatut, split_file_path),
        load_properties=['correction_factor', 'target_ref', 'ppp_cis']
    )

    newdata.prepare_data()
    newdata.setup() 

    # Initialize the model. The values of the variables were obtained for each target property through hyperparameter optimization using Optuna
    cutoff = 9.
    n_atom_basis = 64

    pairwise_distance = spk.atomistic.PairwiseDistances()
    radial_basis = spk.nn.GaussianRBF(n_rbf=18, cutoff=cutoff)
    # For oscillator strength, use representation.PaiNN
    schnet = spk.representation.SchNet( 
        n_atom_basis=n_atom_basis, n_interactions=3,
        radial_basis=radial_basis,
        cutoff_fn=spk.nn.CosineCutoff(cutoff),
        activation=F.silu # For PaiNN model
    )

    pred_correction = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='correction_factor')

    nnpot = spk.model.NeuralNetworkPotential(
        representation=schnet,
        input_modules=[pairwise_distance],
        output_modules=[pred_correction],
        postprocessors=[
            spk.transform.CastTo64(),
            spk.transform.AddOffsets('correction_factor', add_mean=True, add_atomrefs=False)
        ]
    )

    output_corr = spk.task.ModelOutput(
        name='correction_factor',
        loss_fn=torch.nn.MSELoss(),
        loss_weight=1.,
        metrics={
            'MAE': torchmetrics.MeanAbsoluteError(),
            'RMSE': torchmetrics.MeanSquaredError()
        }
    )

    task = spk.task.AtomisticTask(
        model=nnpot,
        outputs=[output_corr],
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={'lr': 5e-4},
        scheduler_cls=spk.train.ReduceLROnPlateau,
        scheduler_monitor='val_loss',
        scheduler_args={'mode': 'min', 'factor': 0.5, 'patience': 11, 'threshold_mode': 'rel', 'cooldown': 5},
    )
    # TensorBoard Logger to visualize the model performance
    logger = pl.loggers.TensorBoardLogger(save_dir=new_datatut)
    callbacks = [
        spk.train.ModelCheckpoint(
            model_path=os.path.join(new_datatut, f'best_inference_model_fold_{fold}'),
            save_top_k=1,
            monitor='val_loss'
        ),
        pl.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=15, min_delta=0),
        pl.callbacks.LearningRateMonitor(logging_interval='step'),
        spk.train.ExponentialMovingAverage(decay=0.995)
    ]

    # Fit the model
    trainer = pl.Trainer(
        max_epochs=2000,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=new_datatut,
        accelerator='auto',
        devices='auto',
        accumulate_grad_batches= 1,                                                                                                                                                                          
        val_check_interval= 1.0 ,                                                                                                                                                                            
        check_val_every_n_epoch= 1 ,                                                                                                                                                                         
        num_sanity_val_steps= 0,                                                                                                                                                                             
        fast_dev_run= False,  
        enable_checkpointing=True,
        overfit_batches= 0 ,                                                                                                                                                                                 
        limit_train_batches= 1.0  ,                                                                                                                                                                          
        limit_val_batches= 1.0  ,                                                                                                                                                                            
        limit_test_batches= 1.0,
        log_every_n_steps=5

    )

    trainer.fit(task, datamodule=newdata)

    # Load the best model
    best_model_path = os.path.join(new_datatut, f'best_inference_model_fold_{fold}')
    best_model = torch.load(best_model_path, map_location='cpu')
    test_data = newdata.test_dataset
    train_data=newdata.train_dataset
    
    actual_correction_factors = []
    actual_correction_factors_train = []
    predicted_correction_factors_train = []
    actual_target_ref_train = []
    actual_ppp_cis_train = []
    predicted_correction_factors_train_2 = []
    all_x_values_train = []
    all_ref_train = []
    actual_target_ref_test = []
    actual_ppp_cis_test = []
    all_predicted_correction_factors_test = []
    all_x_values_test = []
    
    best_model.eval()  # Set to evaluation mode before prediction

    with torch.no_grad(): # Disable gradients for prediction
        # Prediction of correction factors in the train set
        for batch in newdata.train_dataloader():
            actual_batch_correction_factors = batch['correction_factor'].cpu().numpy().tolist()
            actual_correction_factors_train.extend(actual_batch_correction_factors)

            # Predict using the model
            result = best_model(batch)
            predicted_batch_correction_factors = result['correction_factor'].cpu().detach().numpy().tolist()
            predicted_correction_factors_train.extend(predicted_batch_correction_factors)

    r_train = r2_score(actual_correction_factors_train, predicted_correction_factors_train)
    print(f"Fold {fold} - Train Correlation coefficient for correction-factor: {r_train}")


    with torch.no_grad():
        # Prediction of ppp_cis + correction factor for the target property in the train set
        for batch in newdata.train_dataloader():
            actual_target_ref_train_batch = batch['target_ref'].cpu().numpy()  
            actual_ppp_cis_train_batch = batch['ppp_cis'].cpu().numpy() 

            # Use the trained model to predict the correction factors
            result = best_model(batch)  
            predicted_batch_correction_factors = result['correction_factor'].cpu().detach().numpy()  # Extract correction factor

            # Store the predicted correction factors and actual values for this batch
            predicted_correction_factors_train_2.extend(predicted_batch_correction_factors.tolist())
            actual_target_ref_train.extend(actual_target_ref_train_batch.tolist())
            actual_ppp_cis_train.extend(actual_ppp_cis_train_batch.tolist())

            # Calculate ppp_cis + correction_factor for this batch
            x_values_train_batch = [
                cis + corr for cis, corr in zip(actual_ppp_cis_train_batch, predicted_batch_correction_factors)
        ]

            all_x_values_train.extend(x_values_train_batch)

    r2_train = r2_score(actual_target_ref_train, all_x_values_train)
    print(f"Fold {fold} - Train Correlation coefficient for ppp_cis + correction-factor: {r2_train}")

    predicted_correction_factors_test = []
    actual_correction_factors_test=[]
    
    with torch.no_grad():
        for batch in newdata.test_dataloader():
            # Predict correction factors using the trained model for the test set
            actual_batch_correction_factors = batch['correction_factor'].cpu().numpy().tolist()
            actual_correction_factors_test.extend(actual_batch_correction_factors)

            result = best_model(batch)
            predicted_correction_factors_test.extend(result['correction_factor'].tolist())
        
    r_test = r2_score(actual_correction_factors_test, predicted_correction_factors_test)
    print(f"Fold {fold} - Test Correlation coefficient for correction-factor: {r_test}")


    with torch.no_grad():    
        for batch in newdata.test_dataloader():
        # Prediction of ppp_cis + correction factor for the target property in the test set
            actual_target_ref_test_batch = batch['target_ref'].cpu().numpy().tolist()  
            actual_ppp_cis_test_batch = batch['ppp_cis'].cpu().numpy().tolist()  

            # Use the trained model to predict the correction factors for the test set
            result = best_model(batch) 
            predicted_batch_correction_factors = result['correction_factor'].cpu().detach().numpy().tolist()  # Extract correction factor

            # Store the actual and predicted values for this batch
            actual_target_ref_test.extend(actual_target_ref_test_batch)
            actual_ppp_cis_test.extend(actual_ppp_cis_test_batch)
            all_predicted_correction_factors_test.extend(predicted_batch_correction_factors)

            # Calculate ppp_cis + correction_factor for this batch
            x_values_test_batch = [
                cis + corr for cis, corr in zip(actual_ppp_cis_test_batch, predicted_batch_correction_factors)
            ]

            all_x_values_test.extend(x_values_test_batch)

    r2_test = r2_score(actual_target_ref_test, all_x_values_test)
    print(f"Fold {fold} - Test Correlation coefficient for ppp_cis + correction-factor: {r2_test}")
    
print("Cross-validation and testing completed.")    
