# Evaluation Results (Confidence = 0.7)

| Method | Serum/Plasma | | | | | Buffy Coat | | | | |
|--------|--------------|---|---|---|---|-----------|----|---|---|---|
| | mAP@0.5 | Recall | IOU | Diff_up | Diff_low | mAP@0.5 | Recall | IOU | Diff_up | Diff_low |
|--------|---------|--------|-----|---------|----------|---------|--------|-----|---------|----------|
| id-blue | \textbf{99.50} | \textbf{100.00} | \textbf{0.98$\pm$0.01} | \underline{4.2$\pm$2.9} | \textbf{3.4$\pm$2.2} | \textbf{97.90} | \textbf{95.83} | 0.77$\pm$0.07 | 4.3$\pm$2.9 | \underline{2.4$\pm$1.7} |
| id-white | 99.50 | \underline{100.00} | 0.97$\pm$0.01 | 4.3$\pm$3.5 | \underline{3.6$\pm$2.7} | 94.40 | 89.72 | 0.75$\pm$0.08 | 4.5$\pm$3.4 | 2.6$\pm$2.3 |
| concat-compress | \underline{99.50} | 99.72 | 0.97$\pm$0.01 | 4.4$\pm$3.0 | 4.3$\pm$3.9 | 77.36 | 54.72 | \underline{0.78$\pm$0.06} | \underline{3.7$\pm$2.2} | 2.5$\pm$1.7 |
| weighted-fusion | 99.50 | 99.31 | 0.97$\pm$0.01 | \textbf{4.0$\pm$3.0} | 3.6$\pm$2.9 | \underline{95.00} | \underline{90.00} | 0.76$\pm$0.07 | 4.0$\pm$2.4 | 2.7$\pm$1.9 |
| crossattn | 99.04 | 98.61 | \underline{0.97$\pm$0.01} | 4.2$\pm$4.2 | 5.4$\pm$6.9 | 50.14 | 0.28 | \textbf{0.79} | 6.0 | 3.6 |
| crossattn-precise | 98.05 | 96.25 | 0.97$\pm$0.04 | 4.8$\pm$4.4 | 5.9$\pm$24.0 | 0.00 | 0.00 | 0.00 | \textbf{0.0} | \textbf{0.0} |


**Note**: **Bold** = Best, _Italic_ = Second Best
