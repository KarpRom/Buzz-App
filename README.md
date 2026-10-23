# Buzz-App

Streamlit apps to listen to bee recordings and check labels.

```bash
pip install -r requirements.txt
```

## BumbleBuzz.py: false-alarm review

`streamlit run BumbleBuzz.py`: reviews clips labeled "no buzz" that the detector predicts as
a buzz (CSV with `flacfile`, `buzzlabel`, `pred_best`).

## ErrorReview.py: model error review

`streamlit run ErrorReview.py`: reviews a classifier's errors clip by clip, from a bbeedeep
cross-validation study.

1. In bbeedeep, collect the study and build the review table:
   ```bash
   python scripts/collect_cv.py logs/cv/<name>
   python scripts/aggregate_predictions.py logs/cv/<name>/collected --task buzz_type
   ```
2. Copy `logs/cv/<name>/collected/` to your machine (small: no checkpoints).
3. In the app sidebar: the `collected/` folder, the audio folder
   (`Cuts_Database_ALL_SPECIES`), and a verdicts CSV path outside any repo.

Each clip shows the label, every model's held-out prediction and class probabilities, the
spectrogram, the audio and the metadata. Filter the queue (majority vote wrong, all models
wrong on the same class, target/predicted class, species, recording), sort by suspicion
(1 − mean probability of the label), or jump to a clip by file name. Verdicts (label
correct / mislabeled → class / ambiguous / bad audio, plus a note) are saved on each click
and reloaded on start. The Summary tab shows the majority-vote confusion matrix and where
errors concentrate (recording, species, protocol, ...).

Made with ❤️ by Purval Bhude. For more about me and my work, visit my [Portfolio](https://purvalbhude.github.io/Portfolio-Website/) or [LinkedIn](https://www.linkedin.com/in/purvalbhude).
