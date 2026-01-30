# Log10 Approximation using Neural Network and Taylor Series

This project computes the base-10 logarithm `log10(x)` using:
1. A pretrained **Neural Network** trained on a given logarithm table (1–10 range)
2. A **Taylor Series approximation** for comparison

The objective is to compare **accuracy, generalization, and computational efficiency**.

---

## 📂 Project Structure
```
Log10_Project_Final/
├── main.py              # Main executable script
├── best_log10_nn.pth    # Pretrained neural network weights
├── logs.xlsx            # Provided logarithm table (training reference)
├── README.md
```

---

## ⚙️ Requirements
```bash
pip install torch numpy pandas
```

Python 3.8+ recommended.

---

## ▶️ Run the Project
```bash
python main.py
```

The script:
- Loads the pretrained neural network
- Computes `log10(x)` using NN and Taylor Series
- Compares both with the true `math.log10(x)`
- Prints accuracy and efficiency results

---

## 🧠 Neural Network
- Architecture: 3 hidden layers, 128 neurons each  
- Trainable parameters: 33,409  
- Achieved precision: **2 decimal places**  

Model loading snippet:
```python
model.load_state_dict(torch.load("best_log10_nn.pth", map_location="cpu"))
model.eval()
```

---

## 📊 Taylor Series
- Uses normalized Taylor expansion
- Minimum terms for 2-decimal precision: **8**
- Much fewer operations than NN

---

## ⚡ Efficiency Comparison
| Method | Total Operations |
|------|------------------|
| Neural Network | ~66,000 |
| Taylor Series | ~20 |

**Result:** Taylor Series is significantly more efficient.

---

## 📁 logs.xlsx
- Contains the provided logarithm table (1–10 range)
- Included for training reference and project compliance
- Not required at runtime since the model is pretrained

### Change Excel Path (if retraining)
```python
excel_path = r"C:\path\to\logs.xlsx"
```

---

## 🏁 Conclusion
Both methods achieve the same precision, but the **Taylor Series outperforms the Neural Network in efficiency**, making it the better approach for computing `log10(x)`.

---

## 👤 Author
Saad Farooqui

