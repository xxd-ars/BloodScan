# Evaluation Results (Confidence = 0.7)

| Method | Serum/Plasma | | | | Buffy Coat | | | |
|--------|--------------|----|----|----|-----------|----|----|----|
| | Detection Rate | IOU | Diff_up | Diff_low | Detection Rate | IOU | Diff_up | Diff_low |
|--------|----------------|-----|---------|----------|----------------|-----|---------|----------|
| id-blue | 99.72 | \underline{0.98$\pm$0.01} | 4.1$\pm$3.2 | 3.6$\pm$2.6 | \textbf{97.50} | 0.77$\pm$0.07 | 4.3$\pm$2.9 | 2.5$\pm$1.9 |
| id-white | \underline{100.00} | 0.97$\pm$0.01 | \textbf{3.9$\pm$3.8} | 3.7$\pm$2.9 | \underline{95.56} | 0.75$\pm$0.07 | 4.6$\pm$3.3 | 2.5$\pm$1.9 |
| concat-compress | 99.44 | 0.97$\pm$0.01 | 4.4$\pm$3.1 | 4.3$\pm$4.0 | 54.72 | \underline{0.78$\pm$0.06} | \underline{3.7$\pm$2.2} | \underline{2.5$\pm$1.7} |
| weighted-fusion | 98.89 | 0.97$\pm$0.01 | \underline{4.0$\pm$2.9} | \textbf{3.5$\pm$2.6} | 90.00 | 0.76$\pm$0.07 | 4.0$\pm$2.4 | 2.7$\pm$1.9 |
| crossattn | 97.22 | 0.97$\pm$0.01 | 4.1$\pm$4.3 | 5.4$\pm$7.0 | 0.28 | \textbf{0.79} | 6.0 | 3.6 |
| crossattn-30epoch | \textbf{100.00} | \textbf{0.98$\pm$0.01} | 4.7$\pm$4.5 | \underline{3.6$\pm$2.8} | 94.72 | 0.76$\pm$0.07 | 4.7$\pm$3.1 | 2.6$\pm$2.0 |
| crossattn-precise | 90.83 | 0.97$\pm$0.01 | 4.8$\pm$4.4 | 3.9$\pm$5.1 | 0.00 | 0.00 | \textbf{0.0} | \textbf{0.0} |


**Note**: **Bold** = Best, _Italic_ = Second Best
