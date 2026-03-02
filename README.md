
# Plant food protein allergenicity predictor. 🧬

This Python script predicts the allergenicity of a plant protein using a SVM model trained to distinguish plant proteins that are allergens from those that are not. Although focused on plant proteins, the performance of the model in other types of allergens is also good (manuscript submitted)

The input for the script is a protein embedding generated with ProtTrans (Elnaggar et al, 2022; https://pubmed.ncbi.nlm.nih.gov/34232869/) in HDF5 (.h5) format. You can use pre-generated ProtTrans embeddings available at resources such as UniProt (https://www.uniprot.org/help/embeddings) or genereate yours installing that package (https://github.com/agemagician/ProtTrans). In the last case, use the "--per_protein 1" option to generate embeddings at the whole-protein level.


---

## ⚙️ Requirements

Make sure you have the following libraries installed in your Python environment. You can install them with `pip`:

```bash
pip install numpy pandas joblib h5py scikit-learn
```

---

## 🚀 Usage

To run the script, use the following command in your terminal:

```bash
python PREDICTOR_PROTEINAS_H5.py <model_path.plk> <embedding_path.h5>
```

* `<model_path.plk>`: The path to the trained SVM model file. This file must be a serialized object created with `joblib`.
* `<embedding_path.h5>`: The path to the HDF5 (`.h5`) file containing the protein embeddings to be predicted.

### **Example**

If your model is named `model_weights.plk` and your embeddings are in `dataset.h5`, the command would be:

```bash
python PREDICTOR_PROTEINAS_H5.py model_weights.plk dataset.h5
```

---

## 📝 File Formats

### **SVM Model (`.plk`)**

The script expects a binary classification model from `scikit-learn` serialized with `joblib`. This model should have been previously trained on a similar dataset.

### **Embeddings File (`.h5`)**

The script is designed to read an HDF5 file where each "dataset" within the file represents a protein. The **dataset name** is used as the protein identifier, and the **dataset content** is its embedding vector (a NumPy array).

**Example structure of an `.h5` file:**

```
/
  - Dataset: 'ProteinName1', Shape: (1024,), Dtype: float32
  - Dataset: 'ProteinName2', Shape: (1024,), Dtype: float32
  - Dataset: 'ProteinName3', Shape: (1024,), Dtype: float32
  ...
```

---

## 📊 Output

The script prints its output directly to the terminal, showing the prediction results for each protein in the input file. For each protein, you’ll see the following information:

* **Name**: The protein identifier.
* **Prediction**: `1` if it is an allergen, `0` if it is not.
* **Class**: Text description of the prediction.
* **Distance to the Boundary**: The distance of the sample to the SVM decision hyperplane. A positive value indicates class `1`, and a negative value indicates class `0`. The larger the absolute value, the greater the model’s confidence.
* **Non-Allergen Probability**: The probability that the protein belongs to class `0`.
* **Allergen Probability**: The probability that the protein belongs to class `1`.



