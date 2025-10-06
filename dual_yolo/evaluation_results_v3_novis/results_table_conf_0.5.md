# Evaluation Results (Confidence = 0.5)

| Method | Serum/Plasma | | | | | Buffy Coat | | | | |
|--------|--------------|---|---|---|---|-----------|----|---|---|---|
| | mAP@0.5 | Recall | IOU | Diff_up | Diff_low | mAP@0.5 | Recall | IOU | Diff_up | Diff_low |
|--------|---------|--------|-----|---------|----------|---------|--------|-----|---------|----------|
| id-blue | \textbf{99.50} | \textbf{100.00} | \textbf{0.98$\pm$0.01} | \textbf{4.2$\pm$2.9} | \textbf{3.4$\pm$2.2} | \textbf{99.50} | \textbf{100.00} | \textbf{0.77$\pm$0.07} | 4.3$\pm$3.0 | \underline{2.4$\pm$1.7} |
| id-white | 99.50 | 100.00 | 0.97$\pm$0.01 | 4.3$\pm$3.5 | \underline{3.6$\pm$2.7} | 98.55 | \underline{98.33} | 0.75$\pm$0.08 | 4.6$\pm$3.6 | 2.6$\pm$2.2 |
| concat-compress | \underline{99.50} | \underline{100.00} | 0.97$\pm$0.01 | 4.4$\pm$3.0 | 4.4$\pm$4.3 | 95.13 | 90.28 | \underline{0.77$\pm$0.06} | \textbf{3.7$\pm$2.2} | 2.4$\pm$1.8 |
| weighted-fusion | 99.50 | 99.44 | 0.97$\pm$0.03 | 5.1$\pm$20.3 | 3.6$\pm$2.9 | \underline{98.58} | 97.22 | 0.75$\pm$0.07 | 4.0$\pm$2.4 | 2.7$\pm$1.9 |
| crossattn | 99.36 | 99.31 | \underline{0.97$\pm$0.01} | \underline{4.2$\pm$4.2} | 5.4$\pm$6.9 | 71.94 | 43.89 | 0.74$\pm$0.07 | 4.2$\pm$2.8 | 2.7$\pm$2.2 |
| crossattn-precise | 99.04 | 98.47 | 0.97$\pm$0.03 | 4.9$\pm$4.4 | 5.6$\pm$21.7 | 69.38 | 39.44 | 0.77$\pm$0.05 | \underline{3.8$\pm$2.8} | \textbf{2.3$\pm$1.5} |


**Note**: **Bold** = Best, _Italic_ = Second Best
