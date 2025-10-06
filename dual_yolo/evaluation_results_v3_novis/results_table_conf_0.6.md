# Evaluation Results (Confidence = 0.6)

| Method | Serum/Plasma | | | | | Buffy Coat | | | | |
|--------|--------------|---|---|---|---|-----------|----|---|---|---|
| | mAP@0.5 | Recall | IOU | Diff_up | Diff_low | mAP@0.5 | Recall | IOU | Diff_up | Diff_low |
|--------|---------|--------|-----|---------|----------|---------|--------|-----|---------|----------|
| id-blue | \textbf{99.50} | \textbf{100.00} | \textbf{0.98$\pm$0.01} | \textbf{4.2$\pm$2.9} | \textbf{3.4$\pm$2.2} | \textbf{99.50} | \textbf{100.00} | \underline{0.77$\pm$0.07} | 4.3$\pm$3.0 | \underline{2.4$\pm$1.7} |
| id-white | 99.50 | 100.00 | 0.97$\pm$0.01 | 4.3$\pm$3.5 | \underline{3.6$\pm$2.7} | \underline{98.30} | \underline{97.78} | 0.75$\pm$0.08 | 4.6$\pm$3.6 | 2.6$\pm$2.2 |
| concat-compress | \underline{99.50} | \underline{100.00} | 0.97$\pm$0.01 | 4.4$\pm$3.0 | 4.4$\pm$4.3 | 90.13 | 80.28 | 0.77$\pm$0.06 | \textbf{3.7$\pm$2.2} | 2.5$\pm$1.8 |
| weighted-fusion | 99.50 | 99.44 | 0.97$\pm$0.03 | 5.1$\pm$20.3 | 3.6$\pm$2.9 | 97.75 | 95.56 | 0.76$\pm$0.07 | \underline{3.9$\pm$2.3} | 2.7$\pm$1.9 |
| crossattn | 99.17 | 98.89 | \underline{0.97$\pm$0.01} | \underline{4.2$\pm$4.2} | 5.4$\pm$6.9 | 59.58 | 19.17 | 0.75$\pm$0.07 | 4.3$\pm$3.1 | 2.5$\pm$1.9 |
| crossattn-precise | 98.52 | 97.22 | 0.97$\pm$0.03 | 4.9$\pm$4.4 | 5.1$\pm$17.8 | 52.08 | 4.17 | \textbf{0.80$\pm$0.04} | 4.2$\pm$2.3 | \textbf{2.2$\pm$1.6} |


**Note**: **Bold** = Best, _Italic_ = Second Best
