# Evaluation Results (Confidence = 0.65)

| Method | Serum/Plasma | | | | Buffy Coat | | | |
|--------|--------------|----|----|----|-----------|----|----|----|
| | Detection Rate | IOU | Diff_up | Diff_low | Detection Rate | IOU | Diff_up | Diff_low |
|--------|----------------|-----|---------|----------|----------------|-----|---------|----------|
| id | 100.00 | 0.97$\pm$0.01 | 5.4$\pm$5.8 | \underline{3.2$\pm$3.3} | 96.39 | 0.77$\pm$0.07 | 4.3$\pm$3.1 | 2.3$\pm$1.5 |
| id-blue | 100.00 | \underline{0.98$\pm$0.01} | 4.2$\pm$3.0 | 3.4$\pm$2.4 | \textbf{99.17} | 0.77$\pm$0.07 | 4.3$\pm$3.0 | 2.4$\pm$1.7 |
| id-white | 100.00 | 0.97$\pm$0.01 | 4.3$\pm$3.7 | 3.6$\pm$2.9 | 96.39 | 0.75$\pm$0.08 | 4.6$\pm$3.7 | 2.6$\pm$2.2 |
| id-blue-30 | 99.72 | 0.98$\pm$0.01 | 4.1$\pm$3.2 | 3.6$\pm$2.6 | \underline{99.17} | \textbf{0.78$\pm$0.07} | 4.3$\pm$2.9 | 2.5$\pm$1.9 |
| id-white-30 | \textbf{100.00} | 0.97$\pm$0.01 | \underline{3.9$\pm$3.8} | 3.7$\pm$2.9 | 98.61 | 0.75$\pm$0.07 | 4.6$\pm$3.3 | 2.5$\pm$1.9 |
| concat-compress | 98.89 | 0.97$\pm$0.02 | 4.3$\pm$3.9 | 4.3$\pm$6.0 | 65.56 | 0.76$\pm$0.06 | 4.4$\pm$2.5 | 2.1$\pm$1.4 |
| weighted-fusion | 0.00 | 0.00 | \textbf{0.0} | \textbf{0.0} | 21.67 | 0.76$\pm$0.06 | \underline{3.9$\pm$2.1} | 2.2$\pm$1.4 |
| crossattn | 40.83 | 0.97$\pm$0.01 | 7.0$\pm$13.6 | 4.4$\pm$5.8 | 73.06 | 0.77$\pm$0.07 | 4.4$\pm$2.6 | \underline{2.1$\pm$1.5} |
| crossattn-30epoch | \underline{100.00} | \textbf{0.98$\pm$0.01} | 4.7$\pm$4.5 | 3.6$\pm$2.8 | 97.50 | 0.76$\pm$0.07 | 4.6$\pm$3.0 | 2.6$\pm$2.0 |
| crossattn-precise | 49.44 | 0.96$\pm$0.03 | 7.5$\pm$16.0 | 4.4$\pm$4.9 | 51.94 | \underline{0.78$\pm$0.05} | \textbf{3.7$\pm$2.4} | \textbf{1.9$\pm$1.5} |


**Note**: **Bold** = Best, _Italic_ = Second Best
