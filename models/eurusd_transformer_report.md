# EURUSD Transformer — Training Analysis

## Config
```json
{
  "symbol": "EUR",
  "start_date": "2025-08-08",
  "end_date": "2025-08-08",
  "lookback": 120,
  "horizon": 10,
  "batch_size": 256,
  "epochs": 10,
  "val_frac": 0.1,
  "lr": 0.001,
  "weight_decay": 0.0001,
  "seed": 42,
  "in_dim": 5,
  "num_classes": 3,
  "label_names": [
    "down",
    "flat",
    "up"
  ]
}
```
## Best Validation
- Epoch: **9**
- Val Loss: **0.8808**
- Val Acc: **0.595**

## Evaluation
- Eval range: `2025-08-08` → `2025-08-08`
- Lookback: `120` | Horizon: `10`
- Samples: `75470`
- Accuracy: **0.539**

### Per-class metrics
| class | precision | recall | f1 |
|---|---:|---:|---:|
| down | 0.325 | 0.108 | 0.162 |
| flat | 0.625 | 0.794 | 0.700 |
| up | 0.308 | 0.291 | 0.299 |

### Confusion matrix
Saved to: `C:\Users\sion0\PycharmProjects\trading\algo\models\confusion_matrix.png`

### Training curves
- Loss: `C:\Users\sion0\PycharmProjects\trading\algo\models\training_loss.png`
- Accuracy: `C:\Users\sion0\PycharmProjects\trading\algo\models\training_acc.png`

### Confidence histogram
- Max-prob histogram: `C:\Users\sion0\PycharmProjects\trading\algo\models\prob_hist.png`