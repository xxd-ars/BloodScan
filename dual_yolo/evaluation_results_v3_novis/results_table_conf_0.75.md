# Evaluation Results (Confidence = 0.75)

| Method | Serum/Plasma | | | | | Buffy Coat | | | | |
|--------|--------------|---|---|---|---|-----------|----|---|---|---|
| | mAP@0.5 | Recall | IOU | Diff_up | Diff_low | mAP@0.5 | Recall | IOU | Diff_up | Diff_low |
|--------|---------|--------|-----|---------|----------|---------|--------|-----|---------|----------|
| id-blue | \textbf{99.50} | \textbf{100.00} | \textbf{0.98$\pm$0.01} | \underline{4.2$\pm$2.9} | \textbf{3.4$\pm$2.2} | \textbf{94.85} | \textbf{89.72} | \textbf{0.78$\pm$0.06} | 4.2$\pm$2.8 | 2.4$\pm$1.7 |
| id-white | 99.50 | \underline{100.00} | 0.97$\pm$0.01 | 4.3$\pm$3.5 | \underline{3.6$\pm$2.7} | 86.92 | 74.72 | 0.76$\pm$0.07 | 4.4$\pm$3.4 | 2.6$\pm$2.3 |
| concat-compress | \underline{99.50} | 99.72 | 0.97$\pm$0.01 | 4.4$\pm$3.0 | 4.3$\pm$3.9 | 65.69 | 31.39 | \underline{0.78$\pm$0.05} | 3.7$\pm$2.2 | 2.6$\pm$1.7 |
| weighted-fusion | 99.50 | 99.31 | 0.97$\pm$0.01 | \textbf{4.0$\pm$3.0} | 3.6$\pm$2.9 | \underline{88.19} | \underline{76.39} | 0.77$\pm$0.06 | 4.0$\pm$2.3 | 2.7$\pm$1.9 |
| crossattn | 98.50 | 97.50 | \underline{0.97$\pm$0.01} | 4.2$\pm$4.2 | 5.4$\pm$6.9 | 0.00 | 0.00 | 0.00 | \textbf{0.0} | \textbf{0.0} |
| crossattn-precise | 97.46 | 95.00 | 0.97$\pm$0.05 | 4.8$\pm$4.4 | 7.7$\pm$33.1 | 0.00 | 0.00 | 0.00 | \underline{0.0} | \underline{0.0} |


**Note**: **Bold** = Best, _Italic_ = Second Best
