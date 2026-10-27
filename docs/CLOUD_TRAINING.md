# Optional cloud training

Cloud training is an optional future path for larger NIfTI or 3D experiments. The local app does not upload scans, dataset folders, model weights, or credentials to a cloud service.

## Before a cloud run

1. Select a dataset whose license and access terms allow the planned use.
2. Budget storage, GPU hours, and exported checkpoint size. BraTS2020's listed training data is about 7.96 GB before extraction or intermediate caches.
3. Use de-identified public research data only. Do not upload patient-identifiable scans, private keys, API tokens, or credentials.
4. Write down the subject-level split, preprocessing, seed, model configuration, and exact software versions before training.
5. Keep training and validation subjects separate from the final test subjects. Do not choose checkpoints based on the test results.

## Suggested experiment record

Record the dataset release/version, source URL and license, manifest/checksum, subject split, target modality, model architecture, input dimensions, batch size, optimizer, schedule, loss, augmentation, seed, GPU type, software versions, elapsed time, peak memory, validation history, and final untouched-test metrics. Preserve this record next to the exported weights.

## Returning a checkpoint

Export only the model state dictionary and metadata required by the model card. Review the checkpoint source before installing it into assets/models. Test it locally on known, de-identified research examples. The app's readiness check confirms file/metadata compatibility only; it does not validate the model.

The included local training scripts do not configure a cloud provider or transfer data. Choose a service and access method separately if a later study requires one.
